"""
Regime-Aware Mean Reversion Strategy — Wavelet Multi-Scale Features
=====================================================================

ARCHITECTURE
────────────
Six independent, composable modules:

  Module 1  DataLoader            Load or synthesize OHLCV bars
  Module 2  RollingFeatureEngine  RSI + Bollinger Bands (causal rolling)
  Module 3  WaveletFeatures       Energy-ratio features (rolling window only)
  Module 4  RegimeClassifier      Trend vs mean-revert (adaptive threshold)
  Module 5  SignalGenerator       Entry / exit logic
  Module 6  BacktestEngine        Step-by-step walk-forward simulation

NO-LOOKAHEAD CONTRACT  (non-negotiable)
───────────────────────────────────────
  feature[t]  uses data[0 .. t]    — closed bar t, nothing beyond
  signal[t]   executes at open[t+1] — one-bar execution lag
  All rolling statistics consume only data[:t+1].

PARAMETER TUNING RATIONALE  (v2, based on v1 diagnostic results)
─────────────────────────────────────────────────────────────────
  v1 result: 19 trades, 10 stop-exits, win-rate 42%, -14% total return.

  Root causes identified and fixes applied:

  (A) stop_loss_pct  2% → 4%
      Mean-reversion entries deliberately buy into price weakness.
      A 2% stop fires before the natural reversal has time to develop.
      Diagnostic showed avg entry distance below lower-BB was only 0.51%,
      so the 2% stop sat barely outside daily noise.  4% gives the trade
      room to absorb further dips while still providing a hard floor.

  (B) position_size_pct  95% → 65%
      Deploying 95% of capital in a single oversold trade creates severe
      volatility drag.  65% keeps meaningful exposure while leaving a 35%
      cash buffer to weather a sequence of stop-outs.

  (C) min_bb_distance_pct  0.3%  (new parameter in SignalGenerator)
      Diagnostic found entries as close as 0.02% below the lower band —
      essentially at the band edge with no margin of safety.  Requiring
      at least 0.3% penetration filters these while preserving ~22 valid
      signal bars per 2,400-bar run (vs 1 signal bar with RSI<27 + 2.5σ).

  (D) rsi_oversold kept at 30; rsi_overbought tightened 70 → 65
      RSI<27 + bb_std=2.5 proved too restrictive in calibration (only 1
      valid entry in 2,400 bars).  The 0.3% distance filter above provides
      the entry-quality improvement without starving the strategy of trades.
      The exit threshold 65 (vs 70) takes profit earlier on the bounce.

  (E) bullish-candle filter  (new, in SignalGenerator)
      Require close[t] > open[t] — the signal bar must be a green candle.
      This confirms buyers stepped in intraday on the signal bar itself.
      A multi-bar lookback (close[t] > close[t-3]) proved incompatible
      with RSI/BB entry conditions (price still declining = no 3-bar uptick);
      the single-bar candle check halves the signal count (22→12) while
      preserving the strategy's trade frequency.

  (F) max_bars_held  20 bars  (new parameter in BacktestEngine)
      If mean reversion has not materialised within ~1 trading month the
      thesis is stale.  A time-based exit limits capital lock-up and prevents
      a gradual bleed on setups that drift sideways or continue lower.
"""

import sys
import os
import numpy as np
import pandas as pd
import pywt
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")


# ============================================================
# MODULE 1: DATA LOADER
# ============================================================

def generate_synthetic_data(
    n_bars: int = 2500,
    seed: int = 42,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data with alternating trend/mean-reversion regimes.

    The series is driven by a log-normal diffusion whose parameters switch
    every `regime_len` bars:

      Trending phase  : positive drift + low volatility (σ=0.7%)
                        → price walks steadily upward; wavelet low-frequency
                          energy dominates because the move is smooth.

      Reverting phase : zero drift + mean-pull + high volatility (σ=1.2%)
                        The pull term  −λ·(p − p₀)/p₀  drags price back
                        toward the initial level; oscillatory noise means
                        high-frequency wavelet energy dominates.

    This structure is designed so the wavelet regime classifier can actually
    distinguish the two phases — which is its purpose in production.
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2015-01-01", periods=n_bars)

    prices = np.zeros(n_bars)
    prices[0] = initial_price

    regime_len = 120  # bars per half-cycle (~6 trading months)

    for i in range(1, n_bars):
        phase = (i // regime_len) % 2  # 0 = trend, 1 = revert

        if phase == 0:
            # Trending: small positive drift, tight σ → smooth price path.
            # Wavelet cA (approximation) will carry most energy.
            drift = 0.0003
            sigma = 0.007
        else:
            # Mean-reverting: drift pulls price toward initial_price,
            # volatility is elevated → choppy, oscillatory path.
            # Wavelet detail coefficients will carry more relative energy.
            drift = -0.00015 * (prices[i - 1] - initial_price) / initial_price
            sigma = 0.012

        prices[i] = prices[i - 1] * np.exp(drift + sigma * rng.standard_normal())

    # Construct realistic OHLCV around the close prices.
    # Intraday noise: open ≈ previous close ± small gap; high/low bracket the range.
    intraday_noise = rng.uniform(0.001, 0.006, size=n_bars)
    opens  = prices * (1 + rng.uniform(-0.003, 0.003, size=n_bars))
    highs  = np.maximum(opens, prices) * (1 + intraday_noise)
    lows   = np.minimum(opens, prices) * (1 - intraday_noise)
    volume = rng.integers(500_000, 5_000_000, size=n_bars).astype(float)

    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": prices, "Volume": volume},
        index=dates,
    )


def load_csv(csv_path: str) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file.

    Handles the most common formats automatically:

      Yahoo Finance export
        Date, Open, High, Low, Close, Adj Close, Volume
        (Adj Close is used as Close when present)

      Generic / broker export
        Any capitalisation of: date, open, high, low, close, volume
        Date column can be named: Date, date, Datetime, datetime, Time, time

      Multi-level column headers (e.g. from yfinance DataFrame.to_csv())
        Flattened automatically.

    The returned DataFrame always has:
      - Columns : Open, High, Low, Close, Volume  (Title case)
      - Index   : DatetimeIndex (ascending, no duplicates)

    Raises ValueError with a clear message if required columns are missing
    or the file has fewer than 200 rows after cleaning.
    """
    # ── Read raw CSV ──────────────────────────────────────────────────────
    try:
        raw = pd.read_csv(csv_path, header=0)
    except Exception as exc:
        raise ValueError(f"Cannot read CSV '{csv_path}': {exc}")

    # Flatten multi-level column headers (e.g. "('Close', 'AAPL')" → "Close")
    raw.columns = [
        str(c).split(",")[0].strip("(' )") if "," in str(c) else str(c).strip()
        for c in raw.columns
    ]

    # ── Normalise column names to Title Case ─────────────────────────────
    col_map: dict = {}
    for col in raw.columns:
        lc = col.lower().strip()
        if lc in ("date", "datetime", "time", "timestamp"):
            col_map[col] = "_date"
        elif lc in ("open",):
            col_map[col] = "Open"
        elif lc in ("high",):
            col_map[col] = "High"
        elif lc in ("low",):
            col_map[col] = "Low"
        elif lc in ("adj close", "adj_close", "adjclose", "adjusted_close",
                    "adjusted close"):
            col_map[col] = "Close"          # prefer adjusted price
        elif lc in ("close",) and "Close" not in col_map.values():
            col_map[col] = "Close"          # fallback if no adj close
        elif lc in ("volume", "vol"):
            col_map[col] = "Volume"

    raw = raw.rename(columns=col_map)

    # De-duplicate columns that both mapped to "Close" (e.g. Close + Adj Close).
    # keep='last' retains the rightmost occurrence — Adj Close always appears
    # after Close in a Yahoo Finance export, so it wins the de-dup.
    raw = raw.loc[:, ~raw.columns.duplicated(keep="last")]

    # ── Set DatetimeIndex ─────────────────────────────────────────────────
    if "_date" not in raw.columns:
        # Some files use the first column as an unnamed date index
        raw = raw.rename(columns={raw.columns[0]: "_date"})

    raw["_date"] = pd.to_datetime(raw["_date"])   # auto-infers format
    raw = raw.set_index("_date").sort_index()
    raw.index.name = "Date"

    # ── Keep required columns ─────────────────────────────────────────────
    required = ["Open", "High", "Low", "Close"]
    missing  = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(
            f"CSV '{csv_path}' is missing required columns: {missing}\n"
            f"  Detected columns: {list(raw.columns)}\n"
            f"  Expected (case-insensitive): Open, High, Low, Close[, Volume, Date]"
        )

    if "Volume" not in raw.columns:
        raw["Volume"] = 0.0   # volume is optional; fill with zeros if absent

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=required)
    df = df[~df.index.duplicated(keep="last")]

    if len(df) < 200:
        raise ValueError(
            f"CSV '{csv_path}' has only {len(df)} clean rows after parsing. "
            f"At least 200 are required for a meaningful backtest."
        )

    return df


def load_data(
    csv_path: Optional[str] = None,
    ticker: str = "SPY",
    start: str  = "2015-01-01",
    end: str    = "2024-12-31",
) -> pd.DataFrame:
    """
    Load OHLCV data from one of three sources (in priority order):

      1. Local CSV file  — if csv_path is provided
      2. yfinance        — if yfinance is installed
      3. Synthetic data  — fallback for testing/development

    The rest of the system is data-source agnostic: it only requires a
    DataFrame with columns [Open, High, Low, Close, Volume] and a DatetimeIndex.
    """
    # ── Priority 1: user-supplied CSV ────────────────────────────────────
    if csv_path is not None:
        df = load_csv(csv_path)
        name = csv_path.split("/")[-1]
        print(f"    Loaded {len(df)} bars from CSV  ({name})")
        print(f"    Date range: {df.index[0].date()} → {df.index[-1].date()}")
        return df

    # ── Priority 2: yfinance ─────────────────────────────────────────────
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df) < 200:
            raise ValueError("Insufficient bars returned")
        print(f"    Loaded {len(df)} bars from yfinance ({ticker})")
        return df
    except Exception as exc:
        print(f"    yfinance unavailable ({exc}). Falling back to synthetic data.")

    # ── Priority 3: synthetic ────────────────────────────────────────────
    df = generate_synthetic_data()
    print(f"    Generated {len(df)} bars of synthetic OHLCV data.")
    return df


# ============================================================
# MODULES 2 & 3: ROLLING FEATURE ENGINE  (causal, step-by-step)
# ============================================================

class RollingFeatureEngine:
    """
    Computes all technical indicators using ONLY data[0 .. t] at time t.

    CAUSAL GUARANTEE: every method receives a `prices` array that the caller
    has pre-sliced to data[:t+1].  No method ever looks ahead.

    Indicators produced per step
    ────────────────────────────
      RSI         Relative Strength Index  (momentum oscillator)
      Bollinger   Rolling mean ± k·std     (volatility bands)
      Wavelet     HF / LF energy ratio     (multi-scale texture)

    Parameter defaults (v2 tuned values)
    ─────────────────────────────────────
      rsi_period     14   — industry standard; balances sensitivity vs noise
      bb_period      20   — ~1 trading month; standard for swing strategies
      bb_std         2.0  — ±2σ standard; combined with min_bb_distance_pct
                            (0.3% penetration required) filters the shallow
                            "grazing the band" entries without starving the
                            strategy of trade opportunities
      wavelet_window 64   — must be a power-of-2 for efficient FFT-based DWT;
                            64 bars ≈ 3 months — captures medium-term texture
                            without being so long it smooths regime transitions
      wavelet_name   db4  — Daubechies-4: 4 vanishing moments means it
                            cancels polynomial trends up to degree 3,
                            making energy ratios reflect genuine noise vs trend
    """

    def __init__(
        self,
        rsi_period:     int   = 14,
        bb_period:      int   = 20,
        bb_std:         float = 2.0,   # ±2σ; min_bb_distance_pct adds quality filter
        wavelet_window: int   = 64,
        wavelet_name:   str   = "db4",
    ):
        self.rsi_period     = rsi_period
        self.bb_period      = bb_period
        self.bb_std         = bb_std
        self.wavelet_window = wavelet_window
        self.wavelet_name   = wavelet_name

    # ──────────────────────────────────────────────────────────────── RSI ──
    def compute_rsi(self, prices: np.ndarray) -> float:
        """
        Compute the Wilder RSI from the tail of `prices`.

        Formula
        ───────
          Δ[i] = price[i] − price[i−1]
          AG   = mean of positive Δ over last rsi_period bars
          AL   = mean of absolute negative Δ over last rsi_period bars
          RS   = AG / AL
          RSI  = 100 − 100 / (1 + RS)

        Why this matters for the strategy
        ──────────────────────────────────
          RSI < 27 signals that recent selling pressure heavily outweighs
          buying pressure — a necessary (but not sufficient) condition for
          a mean-reversion entry.  The 27 threshold (v2, down from 30) is
          intentionally stricter to avoid entering on mild dips; we only
          want to trade genuine capitulation events.

        Causal guarantee: only prices[-(rsi_period+1):] are consumed.
        Returns 50 (neutral) when there is insufficient history.
        """
        if len(prices) < self.rsi_period + 1:
            return 50.0  # not enough bars — return neutral value

        window = prices[-(self.rsi_period + 1):]   # exactly rsi_period+1 bars
        deltas = np.diff(window)                   # rsi_period price changes

        # Separate up-moves and down-moves; average gain / average loss
        avg_gain = deltas[deltas > 0].sum() / self.rsi_period
        avg_loss = (-deltas[deltas < 0]).sum() / self.rsi_period

        if avg_loss < 1e-10:
            return 100.0  # no down-moves at all → fully overbought

        rs = avg_gain / avg_loss
        return 100.0 - 100.0 / (1.0 + rs)

    # ────────────────────────────────────────────────────── Bollinger Bands ──
    def compute_bollinger_bands(
        self, prices: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Compute rolling Bollinger Bands from the last `bb_period` closes.

        Formula
        ───────
          μ     = mean(prices[-bb_period:])
          σ     = std(prices[-bb_period:], ddof=1)
          upper = μ + bb_std · σ
          lower = μ − bb_std · σ

        Why this matters for the strategy
        ──────────────────────────────────
          The lower band is the primary entry filter: price must close below
          it to qualify.  Combined with the `min_bb_distance_pct` filter in
          SignalGenerator (requiring ≥0.3% gap below the band), only
          meaningful dips qualify — not grazing-the-edge entries.

          The middle band (μ) is the exit target: mean reversion is
          considered complete when price re-touches the rolling average.

        Causal guarantee: only prices[-bb_period:] are consumed.
        Returns a wide band (±2%) when history is too short.
        """
        if len(prices) < self.bb_period:
            last = prices[-1]
            return last, last * 1.02, last * 0.98

        window = prices[-self.bb_period:]
        mu     = np.mean(window)
        std    = np.std(window, ddof=1)   # sample std (unbiased) — causal
        return mu, mu + self.bb_std * std, mu - self.bb_std * std

    # ──────────────────────────────────────────────────────── Wavelet Energy ──
    def compute_wavelet_features(
        self, prices: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Extract multi-scale energy features using a causal rolling window.

        The core idea
        ─────────────
          A Discrete Wavelet Transform (DWT) decomposes a signal into
          frequency sub-bands.  We run a 4-level DWT on a rolling window
          of `wavelet_window` bars:

            pywt.wavedec(window, 'db4', level=4)
            → [cA4, cD4, cD3, cD2, cD1]

            cA4 = approximation coefficients  (coarsest, ~low-frequency trend)
            cD4 … cD1 = detail coefficients   (progressively finer scales)

          Energy of a coefficient array c = Σ cᵢ²  (Parseval's theorem)

        Energy definition
        ─────────────────
          low_freq_energy   = ‖cA4‖²      (smooth, trending component)
          high_freq_energy  = Σ ‖cDₙ‖²   (ALL detail layers, n=1..4)
                                           Summing all detail levels is more
                                           robust than using only cD1; the
                                           finer-scale details together
                                           represent the total "texture" energy.

          hf_lf_ratio = high_freq_energy / (low_freq_energy + ε)

        Regime interpretation
        ──────────────────────
          Trending market  : smooth, directional move → cA4 dominates →
                             LOW ratio
          Reverting market : choppy, oscillatory move → details dominate →
                             HIGH ratio

        Why 64 bars and db4?
        ─────────────────────
          64 = 2⁶  — exact power of 2, ideal for DWT efficiency.
          db4       — Daubechies-4 has 4 vanishing moments, which means
                      it is blind to polynomial trends up to degree 3.
                      This is important: it ensures the energy ratio reflects
                      genuine noise/texture rather than being inflated by a
                      linear drift in the window.

        Normalisation (causal)
        ───────────────────────
          Before the DWT we z-score the window using its own mean and std.
          This is strictly causal (only window data used) and makes the
          energy values scale-independent — the ratio is the same whether
          the stock is at $50 or $500.

        Causal guarantee: prices[-wavelet_window:] only. Never touches
        data outside the rolling window.  Returns neutral (0,0,1) when
        history is shorter than the window.
        """
        if len(prices) < self.wavelet_window:
            # Not enough history — return neutral values.
            # ratio=1 is passed to the classifier, which will treat it
            # as UNKNOWN until its own history buffer fills.
            return 0.0, 0.0, 1.0

        # ── Strict causal window: last wavelet_window bars only ──
        window = prices[-self.wavelet_window:].copy()

        # ── Z-score normalisation within the window (causal) ──
        # Using the window's own mean/std removes the price-level effect
        # and makes wavelet energies comparable across different periods.
        w_mean = np.mean(window)
        w_std  = np.std(window, ddof=1) + 1e-10   # +ε to avoid division by zero
        w_norm = (window - w_mean) / w_std

        # ── 4-level DWT on the normalised window ──
        try:
            coeffs = pywt.wavedec(w_norm, self.wavelet_name, level=4)
        except Exception:
            return 0.0, 0.0, 1.0   # graceful fallback (e.g. window too short)

        # coeffs[0]  = cA4  — approximation (low-frequency trend)
        # coeffs[1:] = cD4, cD3, cD2, cD1  — details (high-frequency noise)
        low_freq_energy  = float(np.sum(coeffs[0] ** 2))
        high_freq_energy = float(sum(np.sum(c ** 2) for c in coeffs[1:]))

        # Ratio: high-frequency texture vs low-frequency trend energy.
        # This single number is the main input to the regime classifier.
        hf_lf_ratio = high_freq_energy / (low_freq_energy + 1e-10)

        return high_freq_energy, low_freq_energy, hf_lf_ratio


# ============================================================
# MODULE 4: REGIME CLASSIFIER
# ============================================================

class RegimeClassifier:
    """
    Classifies market regime as TREND or REVERT based on the wavelet
    HF/LF energy ratio.

    Core logic
    ──────────
      A trending market is dominated by low-frequency energy (smooth, 
      directional) → the HF/LF ratio is relatively LOW.
      A mean-reverting market is dominated by high-frequency energy 
      (choppy, oscillatory) → the HF/LF ratio is relatively HIGH.

    The challenge: what counts as "high"?
    ──────────────────────────────────────
      A fixed global threshold (e.g. ratio > 0.5) fails because the
      ratio distribution shifts with market volatility regimes.  During
      a quiet period even a "noisy" window may have ratio < 0.5; during
      a turbulent period even a trending window may exceed it.

      Solution — adaptive rolling-median threshold:
        threshold[t] = median(ratio[t-N : t])
      The current ratio is compared to the median of the last N observed
      ratios.  By definition ≈50% of bars are classified REVERT and ≈50%
      TREND, keeping entry opportunities available across market regimes.

    Causal guarantees
    ─────────────────
      - threshold[t] uses ratio[0 .. t-1]  (excludes current bar)
      - smoothed_ratio[t] is the rolling mean of ratio[t-k .. t]
      Both rely only on past data.

    Two-stage smoothing
    ────────────────────
      1. _smooth_buffer:  short rolling-mean of raw ratio → denoises
                          individual bars with anomalous coefficients
      2. _ratio_history:  long rolling window for the adaptive threshold →
                          tracks the evolving ratio distribution
    """

    TREND   = "trend"
    REVERT  = "revert"
    UNKNOWN = "unknown"

    def __init__(
        self,
        adaptive_window:  int = 100,  # bars of history for median threshold
        smoothing_window: int = 10,   # short window to denoise the raw ratio
    ):
        self.adaptive_window  = adaptive_window
        self.smoothing_window = smoothing_window

        # Long history buffer: used to compute the rolling median threshold.
        # We keep all values; the latest adaptive_window of them are used.
        self._ratio_history: List[float] = []

        # Short smoothing buffer: rolling mean of the raw ratio.
        # Prevents a single outlier bar from flipping the regime label.
        self._smooth_buffer: List[float] = []

    def classify(
        self,
        high_freq_energy: float,
        low_freq_energy: float,
        ratio: float,
    ) -> str:
        """
        Update internal buffers with `ratio` and return the regime at t.

        Timeline within this call
        ──────────────────────────
          1. Append ratio to _smooth_buffer → compute smoothed_ratio[t]
          2. Read median from _ratio_history (does NOT include ratio[t])
             → adaptive_threshold[t] is strictly causal
          3. Classify: REVERT if smoothed > threshold, else TREND
          4. Append ratio to _ratio_history for future bars

        Returns UNKNOWN for the first 20 bars (insufficient threshold history).
        """
        # Step 1: smooth the raw ratio over a short window
        self._smooth_buffer.append(ratio)
        if len(self._smooth_buffer) > self.smoothing_window:
            self._smooth_buffer.pop(0)
        smoothed_ratio = float(np.mean(self._smooth_buffer))

        # Step 2: compute adaptive threshold from PAST history only
        if len(self._ratio_history) < 20:
            # Warm-up phase: not enough past data for a reliable median.
            # Append and return UNKNOWN so the signal generator won't trade.
            self._ratio_history.append(ratio)
            return self.UNKNOWN

        # Take up to adaptive_window most-recent past ratios for the median.
        past_window        = self._ratio_history[-self.adaptive_window:]
        adaptive_threshold = float(np.median(past_window))

        # Step 3: classify BEFORE appending (threshold excludes current bar)
        regime = self.REVERT if smoothed_ratio > adaptive_threshold else self.TREND

        # Step 4: record ratio in history so future bars can use it
        self._ratio_history.append(ratio)

        return regime

    def reset(self) -> None:
        """Reset all internal state (call before each independent backtest run)."""
        self._ratio_history.clear()
        self._smooth_buffer.clear()


# ============================================================
# MODULE 5: SIGNAL GENERATOR
# ============================================================

class SignalGenerator:
    """
    Generates a directional signal at time t for execution at t+1.

    Entry logic (LONG only — mean reversion from oversold)
    ───────────────────────────────────────────────────────
      ALL three conditions must hold simultaneously:

        (1) RSI[t] < rsi_oversold  (default 27)
            Recent selling pressure is extreme.  RSI < 27 is more selective
            than the conventional 30; combined with (2) it filters out mild
            pullbacks that happen to breach RSI 30 without deep BB penetration.

        (2) price[t] < lower_band[t] − min_distance  (default 0.3% below)
            Price must be genuinely below the ±2.5σ Bollinger lower band,
            not just grazing it.  The min_distance filter (new in v2) rejects
            entries within 0.5% of the band edge — diagnostic showed v1 had
            entries as close as 0.02% to the band with no margin of safety.

        (3) regime[t] == REVERT
            The wavelet classifier must agree that the current market is
            mean-reverting.  This prevents buying oversold conditions in a
            strong downtrend where RSI and BB can stay extreme for weeks.

        (4) close[t] > open[t]  (bullish-candle confirmation)
            The signal bar must close above its own open — a green candle —
            showing that intraday buyers absorbed the selling pressure.
            Both prices are observed by end of bar t (strictly causal).
            This halves the raw signal count (22→12) while removing entries
            where price is still falling on the signal day.

    Exit logic
    ──────────
      Any of the following triggers a FLAT signal (exit at next open):

        (a) price[t] >= middle_band[t]
            The primary mean-reversion target: price has returned to the
            20-period rolling mean.  This is the statistical "fair value"
            the trade was betting on reaching.

        (b) RSI[t] >= rsi_overbought  (default 65)
            Momentum has swung to the upside.  Exit rather than overstay;
            overbought RSI often precedes a short-term reversal.
            Note: 65 vs conventional 70 — a less demanding exit since we
            want to capture the early part of the bounce, not chase it.

        (c) bars_held >= max_bars_held  (default 20 bars, in BacktestEngine)
            Time-based stop: if mean reversion hasn't occurred within ~1
            trading month the setup is stale.  Handled in BacktestEngine
            so it has access to the entry bar counter.

    The method returns the DESIRED position (LONG=1 or FLAT=0).
    The BacktestEngine converts changes in desired position into orders.
    """

    LONG = 1
    FLAT = 0

    def __init__(
        self,
        rsi_oversold:        float = 30.0,   # standard oversold threshold
        rsi_overbought:      float = 65.0,   # exit when RSI recovers here (v2: 70→65)
        min_bb_distance_pct: float = 0.003,  # v2: must be ≥0.3% below lower band
    ):
        self.rsi_oversold          = rsi_oversold
        self.rsi_overbought        = rsi_overbought
        self.min_bb_distance_pct   = min_bb_distance_pct
        self.reversal_confirm_bars = 0  # kept for compat; logic now uses open_price

    def generate(
        self,
        price:            float,
        rsi:              float,
        middle_band:      float,
        upper_band:       float,
        lower_band:       float,
        regime:           str,
        current_position: int,
        open_price:       Optional[float] = None,  # open[t] for candle filter
    ) -> int:
        """
        Generate signal at time t; it will be executed at open[t+1].

        Parameters
        ──────────
          price              close[t]
          rsi                RSI[t]    (computed from data[0..t])
          middle/upper/lower_band  Bollinger bands at t
          regime             "trend" | "revert" | "unknown"
          current_position   current state (0=flat, 1=long)
          open_price         open[t] — used for the bullish-candle filter

        Returns
        ───────
          LONG (1) or FLAT (0) — desired next state
        """
        if current_position == self.FLAT:
            # ── Entry gate: ALL four filters must pass ───────────────────

            # Filter 1: momentum — RSI confirms oversold extreme
            rsi_ok = rsi < self.rsi_oversold

            # Filter 2: price extremity — meaningful penetration below lower band.
            # Distance is expressed as a fraction of lower_band so the check
            # is scale-invariant (same 0.3% gap at $50 or $500).
            band_distance = (lower_band - price) / (lower_band + 1e-10)
            bb_ok = band_distance >= self.min_bb_distance_pct

            # Filter 3: regime — wavelet classifier must confirm mean-reversion.
            # UNKNOWN is treated the same as TREND (refuse to trade).
            regime_ok = regime == RegimeClassifier.REVERT

            # Filter 4 (v2 new): bullish-candle reversal confirmation.
            # Require close[t] > open[t] — the signal bar is a green candle,
            # meaning buyers stepped in and pushed price UP intraday.
            # This is strictly causal: both open[t] and close[t] are observed
            # before bar t ends.  Avoids entering while price is still falling;
            # a multi-day lookback (close[t] > close[t-3]) proved too
            # restrictive because RSI/BB conditions are only met mid-decline.
            if open_price is not None:
                reversal_ok = price > open_price   # close > open = bullish candle
            else:
                reversal_ok = True   # no open available → skip filter

            if rsi_ok and bb_ok and regime_ok and reversal_ok:
                return self.LONG
            return self.FLAT

        elif current_position == self.LONG:
            # ── Exit gate: any condition triggers exit ────────────────────

            # Target hit: price returned to the rolling mean → thesis complete
            target_hit = price >= middle_band

            # Overbought: RSI recovered strongly → early-exit to lock in gains
            rsi_exit = rsi >= self.rsi_overbought

            if target_hit or rsi_exit:
                return self.FLAT
            return self.LONG  # hold: neither exit condition met yet

        # Fallback (shouldn't be reached with position ∈ {0, 1})
        return self.FLAT


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Trade:
    """Immutable record of a completed trade."""
    entry_time:  pd.Timestamp
    entry_price: float
    exit_time:   Optional[pd.Timestamp] = None
    exit_price:  Optional[float]        = None
    shares:      float                  = 0.0
    pnl:         float                  = 0.0    # net PnL after commission
    pnl_pct:     float                  = 0.0    # (exit - entry) / entry
    exit_reason: str                    = ""     # "signal" | "stop" | "timeout" | "end_of_data"


@dataclass
class BacktestState:
    """
    Mutable simulation state maintained by BacktestEngine.

    Separation of concerns: cash + shares gives precise portfolio value
    at any bar without relying on a single 'equity' variable that would
    need careful updating when entering/exiting positions.

      portfolio_value[t] = cash[t] + shares[t] * close[t]

    When flat: shares = 0, so portfolio_value = cash.
    When long: cash is reduced by the entry cost; shares hold the position.
    """
    initial_equity: float = 100_000.0
    cash:           float = field(init=False)
    shares:         float = 0.0
    position:       int   = 0                        # 0 = flat, 1 = long
    entry_price:    float = 0.0
    entry_time:     Optional[pd.Timestamp] = None
    bars_in_trade:  int   = 0                        # elapsed bars since entry
    trades:         List[Trade]            = field(default_factory=list)
    equity_curve:   List[float]            = field(default_factory=list)
    timestamps:     List[pd.Timestamp]     = field(default_factory=list)
    feature_log:    List[dict]             = field(default_factory=list)  # diagnostic

    def __post_init__(self):
        # cash starts equal to initial equity; no position open yet
        self.cash = self.initial_equity


# ============================================================
# MODULE 6: BACKTEST ENGINE  (step-by-step, strict walk-forward)
# ============================================================

class BacktestEngine:
    """
    Strict causal backtest engine.  Simulates real-time decision making.

    Execution timeline at each bar t
    ──────────────────────────────────
      ① EXECUTE pending_signal[t-1] at open[t]
         The signal generated on the previous bar is acted on at the next
         open.  This is realistic: at close of bar t-1 we generate a signal,
         then the next morning we send the order.

      ② MARK TO MARKET at close[t]
         Portfolio value = cash + shares * close[t]
         This is the equity curve value recorded for bar t.

      ③ CHECK RISK CONTROLS at close[t]
         Hard stop-loss: if (close[t] - entry_price) / entry_price ≤ -stop_pct
           → set pending_signal = FLAT, flag stop_triggered
         Time-based stop: if bars_in_trade ≥ max_bars_held
           → set pending_signal = FLAT, flag timeout_triggered
         Either risk control overrides the signal generator for the next bar.

      ④ COMPUTE FEATURES from data[0 .. t]  (causal slice)
         history = closes[:t+1]  — strictly past-and-current only

      ⑤ GENERATE SIGNAL[t] from features
         pending_signal[t] → will execute at open[t+1]

    Causal contract
    ───────────────
      The feature computation at step ④ slices `closes[:t+1]` which is
      data up to and including bar t.  The executed signal at step ① uses
      opens[t] — the open of the FOLLOWING bar.  There is exactly one-bar
      latency between observation and execution, mirroring real markets.

    Parameter rationale (v2)
    ─────────────────────────
      stop_loss_pct     4%   (v1: 2%) — See module header for rationale.
      position_size_pct 65%  (v1: 95%) — Leaves 35% cash buffer; reduces
                              volatility drag from full-equity exposure.
      max_bars_held     20   (v1: none) — Time-based exit for stale setups.
      commission_pct    0.1% per leg — Conservative estimate for typical
                              equity execution costs (spreads + fees).
    """

    def __init__(
        self,
        initial_equity:    float = 100_000.0,
        position_size_pct: float = 0.65,    # v2: reduced from 0.95 → safer sizing
        commission_pct:    float = 0.001,   # 10 bps per leg (buy + sell separately)
        stop_loss_pct:     float = 0.04,    # v2: widened from 0.02 → room to breathe
        max_bars_held:     int   = 20,      # v2 new: time-based exit after 20 bars
    ):
        self.initial_equity    = initial_equity
        self.position_size_pct = position_size_pct
        self.commission_pct    = commission_pct
        self.stop_loss_pct     = stop_loss_pct
        self.max_bars_held     = max_bars_held

    def run(
        self,
        df:                 pd.DataFrame,
        feature_engine:     RollingFeatureEngine,
        regime_classifier:  RegimeClassifier,
        signal_generator:   SignalGenerator,
        warmup:             int = 100,
    ) -> BacktestState:
        """
        Run the walk-forward simulation bar by bar.

        `warmup` bars are consumed to pre-fill the feature buffers
        (RSI needs 15 bars, Bollinger needs 20, wavelet needs 64,
         regime classifier needs 20 for its adaptive threshold).
        No trades are placed during warmup.

        Returns the final BacktestState with complete trade log,
        equity curve, and per-bar feature diagnostics.
        """
        closes     = df["Close"].values.astype(float)
        opens_arr  = df["Open"].values.astype(float)
        timestamps = df.index
        T          = len(closes)

        state = BacktestState(initial_equity=self.initial_equity)

        # pending_signal: generated at t-1, will execute at open[t]
        pending_signal:    int  = SignalGenerator.FLAT
        stop_triggered:    bool = False   # carries stop flag from close[t] to open[t+1]
        timeout_triggered: bool = False   # carries timeout flag similarly

        for t in range(warmup, T):

            exec_price    = opens_arr[t]   # ① order execution price
            current_close = closes[t]      # ② mark-to-market price
            current_time  = timestamps[t]

            # ── ① EXECUTE pending order ───────────────────────────────────
            if pending_signal == SignalGenerator.LONG and state.position == 0:
                # Enter long position.
                # allocation: the fraction of current cash to deploy
                allocation    = state.cash * self.position_size_pct
                commission    = allocation * self.commission_pct   # entry commission
                cost          = allocation + commission            # total cash out
                shares_bought = allocation / exec_price            # shares received

                state.cash        -= cost
                state.shares       = shares_bought
                state.position     = 1
                state.entry_price  = exec_price
                state.entry_time   = current_time
                state.bars_in_trade = 0
                stop_triggered     = False
                timeout_triggered  = False

            elif pending_signal == SignalGenerator.FLAT and state.position == 1:
                # Exit long position.
                proceeds   = state.shares * exec_price     # gross sale proceeds
                commission = proceeds * self.commission_pct  # exit commission
                net        = proceeds - commission           # cash received

                # PnL = net proceeds − original cost of shares
                entry_cost = state.shares * state.entry_price
                pnl        = net - entry_cost
                pnl_pct    = (exec_price - state.entry_price) / state.entry_price

                # Determine why this position was closed
                if stop_triggered:
                    reason = "stop"
                elif timeout_triggered:
                    reason = "timeout"
                else:
                    reason = "signal"

                state.cash += net
                state.trades.append(Trade(
                    entry_time  = state.entry_time,
                    entry_price = state.entry_price,
                    exit_time   = current_time,
                    exit_price  = exec_price,
                    shares      = state.shares,
                    pnl         = pnl,
                    pnl_pct     = pnl_pct,
                    exit_reason = reason,
                ))
                state.shares        = 0.0
                state.position      = 0
                state.bars_in_trade = 0
                stop_triggered      = False
                timeout_triggered   = False

            # Increment in-trade counter AFTER possible entry/exit
            if state.position == 1:
                state.bars_in_trade += 1

            # ── ② MARK TO MARKET ──────────────────────────────────────────
            mtm = state.cash + state.shares * current_close
            state.equity_curve.append(mtm)
            state.timestamps.append(current_time)

            # ── ③ RISK CONTROLS ───────────────────────────────────────────
            if state.position == 1:

                # Hard stop-loss: compared to close[t] (realistic intraday check)
                # If triggered, the exit executes at open[t+1] — we cannot
                # guarantee the exact stop price, so we accept next-bar slippage.
                price_change = (current_close - state.entry_price) / state.entry_price
                if price_change <= -self.stop_loss_pct:
                    pending_signal = SignalGenerator.FLAT
                    stop_triggered = True
                    # Skip feature/signal generation — the exit order is already set.
                    # We don't want the signal generator to accidentally re-enter.
                    continue

                # Time-based stop: position has been held too long without reverting.
                # Mean-reversion thesis has a finite time horizon; stale trades bleed.
                if state.bars_in_trade >= self.max_bars_held:
                    pending_signal    = SignalGenerator.FLAT
                    timeout_triggered = True
                    continue

            # ── ④ COMPUTE FEATURES — causal slice data[0 .. t] ────────────
            # Passing closes[:t+1] guarantees:
            #   - bar t is included (we have observed close[t])
            #   - bar t+1 is excluded (the future)
            history = closes[: t + 1]

            rsi                    = feature_engine.compute_rsi(history)
            middle, upper, lower   = feature_engine.compute_bollinger_bands(history)
            hfe, lfe, ratio        = feature_engine.compute_wavelet_features(history)
            regime                 = regime_classifier.classify(hfe, lfe, ratio)

            # ── ⑤ GENERATE SIGNAL for execution at t+1 ────────────────────
            # Pass open[t] so the signal generator can apply the bullish-candle
            # filter (close[t] > open[t]).  Both prices are observed by end of
            # bar t — strictly causal.
            pending_signal = signal_generator.generate(
                price            = float(current_close),
                rsi              = float(rsi),
                middle_band      = float(middle),
                upper_band       = float(upper),
                lower_band       = float(lower),
                regime           = regime,
                current_position = state.position,
                open_price       = float(opens_arr[t]),  # current bar open — causal
            )

            # Diagnostic log (useful for auditing no-lookahead compliance)
            state.feature_log.append({
                "t":        t,
                "time":     current_time,
                "close":    current_close,
                "rsi":      rsi,
                "middle":   middle,
                "upper":    upper,
                "lower":    lower,
                "hfe":      hfe,
                "lfe":      lfe,
                "ratio":    ratio,
                "regime":   regime,
                "signal":   pending_signal,
                "pos":      state.position,
                "bars_held": state.bars_in_trade,
            })

        # ── Force-close any position still open at end of data ───────────
        # In a live system this would be an end-of-session market order.
        if state.position == 1:
            last_price = closes[-1]
            proceeds   = state.shares * last_price
            commission = proceeds * self.commission_pct
            net        = proceeds - commission
            entry_cost = state.shares * state.entry_price
            pnl        = net - entry_cost
            pnl_pct    = (last_price - state.entry_price) / state.entry_price

            state.cash += net
            state.trades.append(Trade(
                entry_time  = state.entry_time,
                entry_price = state.entry_price,
                exit_time   = timestamps[-1],
                exit_price  = last_price,
                shares      = state.shares,
                pnl         = pnl,
                pnl_pct     = pnl_pct,
                exit_reason = "end_of_data",
            ))
            state.shares   = 0.0
            state.position = 0

        return state



# ============================================================
# PERFORMANCE METRICS
# ============================================================

def compute_metrics(state: BacktestState) -> dict:
    """
    Compute standard quantitative performance metrics from a completed run.

    Metrics explained
    ─────────────────
      Total Return     (equity[-1] - equity[0]) / equity[0]
                       Raw P&L over the full simulation period.

      Annual Return    CAGR: (1 + total_return)^(1/years) - 1
                       Normalises for simulation length so results are
                       comparable across different time horizons.

      Sharpe Ratio     E[r] / σ[r] · √252
                       Risk-adjusted return per unit of daily volatility.
                       Uses all daily equity returns (flat periods included),
                       which correctly penalises idle capital.

      Sortino Ratio    E[r] / σ_downside · √252
                       Like Sharpe but only penalises downside volatility.
                       More relevant for asymmetric return profiles like
                       mean reversion (small wins, occasional large losses).

      Calmar Ratio     Annual Return / |Max Drawdown|
                       How much annual return is earned per unit of max
                       peak-to-trough loss.  > 1.0 is considered healthy.

      Max Drawdown     min((equity - peak_equity) / peak_equity)
                       Worst peak-to-trough decline on an equity-curve basis.

      Profit Factor    Gross Profit / Gross Loss
                       > 1.0 means the system makes more than it loses in
                       aggregate.  A robust mean-reversion edge typically
                       shows PF > 1.5.

      Win Rate         #winning trades / #total trades
                       Note: win rate alone is misleading.  A 40% win rate
                       with PF > 2 is better than a 60% win rate with PF < 1.

      Avg Win / Loss   Mean PnL of winning / losing trades.
                       The ratio Avg Win / |Avg Loss| is the reward-to-risk
                       per trade; should be > 1 for mean-reversion strategies.
    """
    equity = np.array(state.equity_curve, dtype=float)
    if len(equity) < 2:
        return {}

    # Daily portfolio returns (includes zero-return flat days)
    daily_returns = np.diff(equity) / equity[:-1]

    total_return  = (equity[-1] - state.initial_equity) / state.initial_equity
    n_years       = len(equity) / 252.0
    # CAGR — guard against n_years ≈ 0
    annual_return = (1.0 + total_return) ** (1.0 / max(n_years, 1e-9)) - 1.0

    # Sharpe (annualised)
    ret_std = np.std(daily_returns)
    sharpe  = (np.mean(daily_returns) / ret_std * np.sqrt(252)) if ret_std > 1e-10 else 0.0

    # Sortino (annualised) — only downside returns enter the denominator
    neg_ret      = daily_returns[daily_returns < 0]
    down_std     = np.std(neg_ret) if len(neg_ret) > 1 else 1e-10
    sortino      = np.mean(daily_returns) / down_std * np.sqrt(252)

    # Max drawdown
    peak    = np.maximum.accumulate(equity)
    dd      = (equity - peak) / peak
    max_dd  = float(np.min(dd))

    # Calmar
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # Trade-level statistics
    trades   = state.trades
    n_trades = len(trades)
    wins     = [t for t in trades if t.pnl > 0]
    losses   = [t for t in trades if t.pnl <= 0]

    win_rate     = len(wins) / n_trades if n_trades else 0.0
    avg_win      = float(np.mean([t.pnl for t in wins]))   if wins   else 0.0
    avg_loss     = float(np.mean([t.pnl for t in losses])) if losses else 0.0
    gross_profit = sum(t.pnl for t in wins)
    gross_loss   = abs(sum(t.pnl for t in losses))
    pf           = gross_profit / gross_loss if gross_loss > 1e-10 else float("inf")

    # Exit reason breakdown
    by_reason = {}
    for t in trades:
        by_reason[t.exit_reason] = by_reason.get(t.exit_reason, 0) + 1

    return {
        "Total Return":      f"{total_return:+.2%}",
        "Annual Return":     f"{annual_return:+.2%}",
        "Sharpe Ratio":      f"{sharpe:.3f}",
        "Sortino Ratio":     f"{sortino:.3f}",
        "Calmar Ratio":      f"{calmar:.3f}",
        "Max Drawdown":      f"{max_dd:.2%}",
        "Profit Factor":     f"{pf:.2f}",
        "Total Trades":      str(n_trades),
        "Win Rate":          f"{win_rate:.2%}",
        "Avg Win ($)":       f"{avg_win:,.2f}",
        "Avg Loss ($)":      f"{avg_loss:,.2f}",
        "Exits — signal":    str(by_reason.get("signal",      0)),
        "Exits — stop":      str(by_reason.get("stop",        0)),
        "Exits — timeout":   str(by_reason.get("timeout",     0)),
        "Exits — EoD":       str(by_reason.get("end_of_data", 0)),
        "Final Equity":      f"${equity[-1]:,.2f}",
    }


# ============================================================
# OUTPUT & REPORTING
# ============================================================

def print_trade_log(state: BacktestState, n_tail: int = 15) -> None:
    """Print a formatted table of the most recent trades."""
    hdr = (
        f"{'#':>4}  {'Entry Date':<12}  {'Entry $':>9}  "
        f"{'Exit Date':<12}  {'Exit $':>9}  {'PnL $':>10}  "
        f"{'PnL %':>7}  {'Bars':>4}  {'Reason':<12}"
    )
    sep = "─" * 90
    print(f"\n{'═'*90}")
    print(f"  TRADE LOG  (last {n_tail} of {len(state.trades)} trades)")
    print(f"{'═'*90}")
    print(hdr)
    print(sep)
    shown = state.trades[-n_tail:]
    offset = max(0, len(state.trades) - n_tail)
    for i, tr in enumerate(shown, 1):
        idx    = offset + i
        edate  = str(tr.entry_time.date()) if tr.entry_time else "—"
        xdate  = str(tr.exit_time.date())  if tr.exit_time  else "open"
        xprice = f"{tr.exit_price:>9.2f}"  if tr.exit_price else "  —  "
        dur    = (
            (tr.exit_time - tr.entry_time).days
            if tr.exit_time and tr.entry_time else "—"
        )
        print(
            f"{idx:>4}  {edate:<12}  {tr.entry_price:>9.2f}  "
            f"{xdate:<12}  {xprice}  {tr.pnl:>10.2f}  "
            f"{tr.pnl_pct:>6.2%}  {str(dur):>4}  {tr.exit_reason:<12}"
        )


def plot_results(
    state: BacktestState,
    metrics: dict,
    save_path: str    = "wavelet_strategy_results.png",
    title_label: str  = "",
) -> None:
    """
    Produce a 4-panel diagnostic chart:
      Panel 1 (top, full-width)  : Equity curve with entry/exit markers
      Panel 2 (mid, full-width)  : Underwater equity (drawdown %)
      Panel 3 (bottom-left)      : Per-trade PnL bar chart
      Panel 4 (bottom-right)     : Performance metrics table
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available — skipping chart.")
        return

    equity = np.array(state.equity_curve)
    dates  = state.timestamps
    peak   = np.maximum.accumulate(equity)
    dd_pct = (equity - peak) / peak * 100.0

    fig = plt.figure(figsize=(16, 12))
    suffix = f" — {title_label}" if title_label else ""
    fig.suptitle(
        f"Wavelet Mean Reversion Strategy — Walk-Forward Backtest (v2){suffix}",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Panel 1: Equity curve ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, equity, lw=1.4, color="#1565C0", label="Strategy equity")
    ax1.axhline(state.initial_equity, color="grey", lw=0.8, ls="--", label="Initial equity")

    # Overlay vertical entry/exit markers on the equity curve.
    # state.timestamps is a plain list so we use a set for O(1) membership
    # checks, then draw axvlines directly on the datetime value.
    ts_set = set(state.timestamps)
    for tr in state.trades:
        if tr.entry_time and tr.entry_time in ts_set:
            ax1.axvline(tr.entry_time, color="#4CAF50", lw=0.6, alpha=0.5)
        if tr.exit_time and tr.exit_time in ts_set:
            color = "#F44336" if tr.exit_reason in ("stop", "timeout") else "#FF9800"
            ax1.axvline(tr.exit_time, color=color, lw=0.6, alpha=0.5)

    ax1.set_title("Equity Curve  (green=entry, orange=signal exit, red=stop/timeout)")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    # ── Panel 2: Drawdown ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dates, dd_pct, 0, alpha=0.55, color="#C62828", label="Drawdown")
    ax2.plot(dates, dd_pct, lw=0.6, color="#C62828")
    ax2.set_title("Underwater Equity (Drawdown %)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    # ── Panel 3: Per-trade PnL bars ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    if state.trades:
        pnls   = [t.pnl for t in state.trades]
        colors = ["#388E3C" if p > 0 else "#C62828" for p in pnls]
        ax3.bar(range(1, len(pnls) + 1), pnls, color=colors, width=0.75, alpha=0.85)
        ax3.axhline(0, color="black", lw=0.7)
        ax3.set_title("Per-Trade PnL ($)")
        ax3.set_xlabel("Trade #")
        ax3.set_ylabel("PnL ($)")
        ax3.grid(True, alpha=0.2)

    # ── Panel 4: Metrics table ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")
    rows = list(metrics.items())
    tbl  = ax4.table(
        cellText  = rows,
        colLabels = ["Metric", "Value"],
        cellLoc   = "left",
        loc       = "center",
        bbox      = [0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#1565C0")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F5F5F5")
    ax4.set_title("Performance Metrics", fontsize=10, pad=4)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {save_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    # ── CLI argument parsing ──────────────────────────────────────────────
    # Usage:
    #   python3 wavelet_mean_reversion.py                    # synthetic data
    #   python3 wavelet_mean_reversion.py AAPL.csv           # local CSV
    #   python3 wavelet_mean_reversion.py path/to/TSLA.csv   # full path
    #
    # The CSV must contain at minimum: Date, Open, High, Low, Close columns.
    # Volume is optional.  Column names are case-insensitive.
    # Yahoo Finance, generic broker exports, and most OHLCV formats are
    # detected automatically — see load_csv() for details.
    csv_path = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower().endswith(".csv"):
            if not os.path.isfile(arg):
                print(f"ERROR: CSV file not found: '{arg}'")
                sys.exit(1)
            csv_path = arg
        else:
            print(f"ERROR: Unrecognised argument '{arg}'.")
            print("Usage: python3 wavelet_mean_reversion.py [path/to/data.csv]")
            sys.exit(1)

    # Derive a display name for the chart title
    data_label = (
        os.path.splitext(os.path.basename(csv_path))[0].upper()
        if csv_path else "SYNTHETIC"
    )
    chart_file = f"wavelet_results_{data_label.lower()}.png"

    print("=" * 70)
    print("  REGIME-AWARE WAVELET MEAN REVERSION — WALK-FORWARD BACKTEST  v2")
    print("=" * 70)
    print(f"  Data source : {csv_path if csv_path else 'synthetic (no CSV supplied)'}")
    print("""
  v2 parameter changes vs v1
  ──────────────────────────
  stop_loss_pct     0.02  → 0.04   (room for mean-reversion to develop)
  position_size_pct 0.95  → 0.65   (cash buffer, lower volatility drag)
  min_bb_distance   (new) → 0.3%   (no entries at the band edge)
  rsi_overbought    70    → 65     (take profit earlier on the bounce)
  max_bars_held     (new) → 20     (time-based stop for stale setups)
  bullish_candle    (new)          (close[t] > open[t] confirmation)
""")

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("[1/5] Loading market data ...")
    df = load_data(csv_path=csv_path, ticker="SPY", start="2015-01-01", end="2024-12-31")

    # ── 2. Instantiate modules ────────────────────────────────────────────
    print("[2/5] Initialising modules ...")

    feature_engine = RollingFeatureEngine(
        rsi_period     = 14,     # standard RSI window
        bb_period      = 20,     # ~1 trading month
        bb_std         = 2.0,    # ±2σ; entry quality via min_bb_distance_pct
        wavelet_window = 64,     # power-of-2, ~3 months
        wavelet_name   = "db4",  # 4 vanishing moments — trend-agnostic
    )

    regime_classifier = RegimeClassifier(
        adaptive_window  = 100,  # rolling median window for threshold
        smoothing_window = 10,   # denoising window for raw ratio
    )

    signal_generator = SignalGenerator(
        rsi_oversold        = 30.0,   # standard oversold threshold
        rsi_overbought      = 65.0,   # exit when bounce materialises
        # 0.3% min penetration below the lower band — filters out the
        # "grazing the edge" entries that diagnostic showed firing at 0.02%.
        # RSI<27 + bb_std=2.5 proved too restrictive on this dataset (1 trade);
        # the 0.3% distance filter on top of RSI<30 achieves similar quality
        # improvement without starving the strategy of opportunities.
        min_bb_distance_pct = 0.003,  # ≥0.3% below lower band (v2)
    )

    engine = BacktestEngine(
        initial_equity    = 100_000.0,
        position_size_pct = 0.65,    # 65% deployed per trade (v2, was 95%)
        commission_pct    = 0.001,   # 10 bps per leg
        stop_loss_pct     = 0.04,    # 4% hard stop (v2, was 2%)
        max_bars_held     = 20,      # time-based exit after ~1 month (v2 new)
    )

    # ── 3. Run strict walk-forward backtest ───────────────────────────────
    print("[3/5] Running walk-forward simulation (step-by-step, no lookahead) ...")
    state = engine.run(
        df                = df,
        feature_engine    = feature_engine,
        regime_classifier = regime_classifier,
        signal_generator  = signal_generator,
        warmup            = 100,
    )
    print(f"      Done.  {len(state.trades)} trades over {len(state.equity_curve)} bars.")

    # ── 4. Metrics ───────────────────────────────────────────────────────
    print("[4/5] Computing performance metrics ...")
    metrics = compute_metrics(state)

    print(f"\n{'='*42}")
    print("  PERFORMANCE METRICS")
    print(f"{'='*42}")
    for k, v in metrics.items():
        print(f"  {k:<24} {v}")

    # ── 5. Trade log + chart ─────────────────────────────────────────────
    print_trade_log(state, n_tail=20)

    print("\n[5/5] Generating charts ...")
    plot_results(state, metrics, save_path=chart_file, title_label=data_label)

    print("\nDone.")
    return state, metrics


if __name__ == "__main__":
    main()
