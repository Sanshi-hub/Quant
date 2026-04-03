"""
Regime-Aware Mean Reversion Strategy with Wavelet Multi-Scale Features
=======================================================================
STRICT walk-forward simulation:
  - feature[t] uses ONLY data[0 .. t]
  - signal[t] is EXECUTED at t+1 (open of next bar)
  - All normalizations use rolling windows
  - Iteration is strictly sequential: for t in range(start, T)
"""

import numpy as np
import pandas as pd
import pywt
import warnings
import os
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
    Generate synthetic OHLCV data with alternating trend / mean-reversion regimes.
    Used when live market data is unavailable.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2015-01-01", periods=n_bars)

    prices = np.zeros(n_bars)
    prices[0] = initial_price

    # Alternate regimes every ~120 bars
    regime_len = 120
    for i in range(1, n_bars):
        regime_phase = (i // regime_len) % 2
        if regime_phase == 0:
            # Trending regime: small drift + low noise
            drift = 0.0003
            sigma = 0.007
        else:
            # Mean-reverting regime: no drift + higher noise
            drift = -0.00015 * (prices[i - 1] - initial_price) / initial_price
            sigma = 0.012
        prices[i] = prices[i - 1] * np.exp(
            drift + sigma * rng.standard_normal()
        )

    # Build OHLCV
    noise = rng.uniform(0.001, 0.006, size=n_bars)
    opens  = prices * (1 + rng.uniform(-0.003, 0.003, size=n_bars))
    highs  = np.maximum(opens, prices) * (1 + noise)
    lows   = np.minimum(opens, prices) * (1 - noise)
    volume = rng.integers(500_000, 5_000_000, size=n_bars).astype(float)

    df = pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": prices, "Volume": volume},
        index=dates,
    )
    return df


def load_data(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: str = "2024-12-31",
) -> pd.DataFrame:
    """Load OHLCV data; falls back to synthetic data if yfinance unavailable."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if len(df) < 200:
            raise ValueError("Insufficient data from yfinance")
        print(f"    Loaded {len(df)} bars from yfinance ({ticker})")
        return df
    except Exception as e:
        print(f"    yfinance unavailable ({e}). Using synthetic data.")
        df = generate_synthetic_data()
        print(f"    Generated {len(df)} bars of synthetic OHLCV data")
        return df


# ============================================================
# MODULE 2 & 3: ROLLING FEATURE ENGINE  (causal, step-by-step)
# ============================================================

class RollingFeatureEngine:
    """
    Computes all indicators using ONLY data[0 .. t] at time t.

    Three indicators are produced per step:
      - RSI      : standard 14-period RSI
      - Bollinger: 20-period bands (middle, upper, lower)
      - Wavelet  : energy features from the last `wavelet_window` bars
    """

    def __init__(
        self,
        rsi_period: int = 14,
        bb_period: int = 20,
        bb_std: float = 2.0,
        wavelet_window: int = 64,
        wavelet_name: str = "db4",
    ):
        self.rsi_period = rsi_period
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.wavelet_window = wavelet_window
        self.wavelet_name = wavelet_name

    # ------------------------------------------------------------------ RSI --
    def compute_rsi(self, prices: np.ndarray) -> float:
        """
        Simple (non-smoothed) RSI over the last rsi_period+1 bars.
        Returns 50 (neutral) when history is too short.
        """
        if len(prices) < self.rsi_period + 1:
            return 50.0
        window = prices[-(self.rsi_period + 1):]
        deltas = np.diff(window)
        gains  = deltas[deltas > 0].sum()
        losses = -deltas[deltas < 0].sum()
        if losses == 0:
            return 100.0
        rs = (gains / self.rsi_period) / (losses / self.rsi_period)
        return 100.0 - 100.0 / (1.0 + rs)

    # -------------------------------------------------------- Bollinger Bands --
    def compute_bollinger_bands(
        self, prices: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Rolling Bollinger Bands.
        Returns (middle, upper, lower).
        Uses rolling mean/std over last bb_period bars — strictly causal.
        """
        if len(prices) < self.bb_period:
            last = prices[-1]
            return last, last * 1.02, last * 0.98
        window = prices[-self.bb_period:]
        # rolling mean and std from ONLY this window (causal)
        mu  = np.mean(window)
        std = np.std(window, ddof=1)
        return mu, mu + self.bb_std * std, mu - self.bb_std * std

    # ------------------------------------------------------- Wavelet Features --
    def compute_wavelet_features(
        self, prices: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Apply wavelet decomposition to the last `wavelet_window` bars ONLY.
        Never touches data outside the rolling window.

        Returns:
            high_freq_energy : energy in finest detail layer (cD1)
            low_freq_energy  : energy in approximation layer (cA)
            hf_lf_ratio      : high_freq / (low_freq + eps)
        """
        if len(prices) < self.wavelet_window:
            return 0.0, 0.0, 1.0  # neutral defaults

        # --- Strict causal window ---
        window = prices[-self.wavelet_window:].copy()

        # Normalize within window (rolling — uses only window data)
        w_mean = np.mean(window)
        w_std  = np.std(window, ddof=1) + 1e-10
        w_norm = (window - w_mean) / w_std

        # Wavelet decomposition limited to this window
        try:
            coeffs = pywt.wavedec(w_norm, self.wavelet_name, level=4)
        except Exception:
            return 0.0, 0.0, 1.0

        # coeffs[0]  = cA4  (approximation — low-frequency smooth trend)
        # coeffs[1:] = cD4, cD3, cD2, cD1  (details — high-frequency noise)
        #
        # low_freq_energy  = energy of the approximation layer
        # high_freq_energy = TOTAL energy across ALL detail layers
        # This ratio captures how "noisy" the window is relative to its trend.
        low_freq_energy  = float(np.sum(coeffs[0] ** 2))
        high_freq_energy = float(sum(np.sum(c ** 2) for c in coeffs[1:]))
        hf_lf_ratio      = high_freq_energy / (low_freq_energy + 1e-10)

        return high_freq_energy, low_freq_energy, hf_lf_ratio


# ============================================================
# MODULE 4: REGIME CLASSIFIER
# ============================================================

class RegimeClassifier:
    """
    Classifies market regime from wavelet energy features.

    Logic (causal only):
      - High HF/LF ratio  →  noisy / mean-reverting market  ("revert")
      - Low  HF/LF ratio  →  smooth / trending market        ("trend")

    Threshold is ADAPTIVE: the rolling median of the last `adaptive_window`
    observed ratios is used as the split point.  This is fully causal —
    the median is computed only from past (already observed) ratios.
    A smoothing buffer reduces spurious regime switches.
    """

    TREND   = "trend"
    REVERT  = "revert"
    UNKNOWN = "unknown"

    def __init__(
        self,
        adaptive_window: int  = 100,   # bars of ratio history for median
        smoothing_window: int = 10,    # bars for smoothing current ratio
    ):
        self.adaptive_window  = adaptive_window
        self.smoothing_window = smoothing_window
        # _ratio_history keeps all past ratios (used for adaptive threshold)
        self._ratio_history: List[float] = []
        # _smooth_buffer is a short window for denoising the current ratio
        self._smooth_buffer: List[float] = []

    def classify(
        self,
        high_freq_energy: float,
        low_freq_energy: float,
        ratio: float,
    ) -> str:
        """
        Append `ratio` to the rolling buffers and return the current regime.

        adaptive_threshold[t] = median(ratio[t-N : t])   ← causal
        smoothed_ratio[t]     = mean(ratio[t-k : t])     ← causal

        regime[t] = REVERT if smoothed_ratio > adaptive_threshold else TREND
        """
        # Short smoothing buffer (denoises the current signal)
        self._smooth_buffer.append(ratio)
        if len(self._smooth_buffer) > self.smoothing_window:
            self._smooth_buffer.pop(0)
        smoothed_ratio = float(np.mean(self._smooth_buffer))

        # Adaptive threshold: median of past N ratios (excludes current step)
        if len(self._ratio_history) < 20:
            # Not enough history — fall back to neutral
            self._ratio_history.append(ratio)
            return self.UNKNOWN

        window = self._ratio_history[-self.adaptive_window:]
        adaptive_threshold = float(np.median(window))

        # Classify BEFORE appending so threshold is strictly from the past
        regime = self.REVERT if smoothed_ratio > adaptive_threshold else self.TREND

        # Now append to history for future steps
        self._ratio_history.append(ratio)

        return regime

    def reset(self) -> None:
        self._ratio_history.clear()
        self._smooth_buffer.clear()


# ============================================================
# MODULE 5: SIGNAL GENERATOR
# ============================================================

class SignalGenerator:
    """
    Generates a directional signal at time t for execution at t+1.

    Entry  (LONG):
        RSI[t]   < rsi_oversold      (default 30)
        price[t] < lower_band[t]     (below 2-sigma Bollinger)
        regime[t] == REVERT

    Exit:
        price[t] >= middle_band[t]   (mean reversion complete)
        OR RSI[t] >= rsi_overbought  (default 70)
    """

    LONG = 1
    FLAT = 0

    def __init__(
        self,
        rsi_oversold: float  = 30.0,
        rsi_overbought: float = 70.0,
    ):
        self.rsi_oversold  = rsi_oversold
        self.rsi_overbought = rsi_overbought

    def generate(
        self,
        price: float,
        rsi: float,
        middle_band: float,
        upper_band: float,
        lower_band: float,
        regime: str,
        current_position: int,
    ) -> int:
        """
        Returns the DESIRED position state (LONG=1 or FLAT=0).
        The backtest engine converts the desired state into an order.
        """
        if current_position == self.FLAT:
            # --- Entry conditions ---
            entry_ok = (
                rsi < self.rsi_oversold
                and price < lower_band
                and regime == RegimeClassifier.REVERT   # block UNKNOWN too
            )
            return self.LONG if entry_ok else self.FLAT

        elif current_position == self.LONG:
            # --- Exit conditions ---
            exit_ok = price >= middle_band or rsi >= self.rsi_overbought
            return self.FLAT if exit_ok else self.LONG

        return self.FLAT


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Trade:
    entry_time:  pd.Timestamp
    entry_price: float
    exit_time:   Optional[pd.Timestamp] = None
    exit_price:  Optional[float]        = None
    shares:      float                  = 0.0
    pnl:         float                  = 0.0
    pnl_pct:     float                  = 0.0
    exit_reason: str                    = ""


@dataclass
class BacktestState:
    initial_equity: float = 100_000.0
    cash:           float = field(init=False)
    shares:         float = 0.0
    position:       int   = 0          # 0=flat, 1=long
    entry_price:    float = 0.0
    entry_time:     Optional[pd.Timestamp] = None
    trades:         List[Trade]         = field(default_factory=list)
    equity_curve:   List[float]         = field(default_factory=list)
    timestamps:     List[pd.Timestamp]  = field(default_factory=list)
    # per-step diagnostic log
    feature_log:    List[dict]          = field(default_factory=list)

    def __post_init__(self):
        self.cash = self.initial_equity

    @property
    def portfolio_value(self) -> float:
        """Cash + unrealized position value (requires external price)."""
        return self.cash  # updated externally with mark-to-market


# ============================================================
# MODULE 6: BACKTEST ENGINE  (step-by-step, strict walk-forward)
# ============================================================

class BacktestEngine:
    """
    Strict causal backtest engine.

    Timeline at step t
    ──────────────────
    1. EXECUTE pending_signal from t-1 using open[t]
    2. Mark portfolio to market using close[t]
    3. Check stop-loss against close[t]
    4. COMPUTE features from prices[0 .. t]   ← only past data
    5. GENERATE signal[t]                     ← stored as pending
    6. pending_signal will execute at t+1

    No future data ever crosses the step boundary.
    """

    def __init__(
        self,
        initial_equity:    float = 100_000.0,
        position_size_pct: float = 0.95,    # fraction of cash to deploy
        commission_pct:    float = 0.001,   # round-trip half (per leg)
        stop_loss_pct:     float = 0.02,    # 2% hard stop on entry price
    ):
        self.initial_equity    = initial_equity
        self.position_size_pct = position_size_pct
        self.commission_pct    = commission_pct
        self.stop_loss_pct     = stop_loss_pct

    # ----------------------------------------------------------
    def run(
        self,
        df:                 pd.DataFrame,
        feature_engine:     RollingFeatureEngine,
        regime_classifier:  RegimeClassifier,
        signal_generator:   SignalGenerator,
        warmup:             int = 100,
    ) -> BacktestState:
        """
        Main simulation loop. Iterates bar by bar from `warmup` to T-1.
        """

        closes     = df["Close"].values.astype(float)
        opens_arr  = df["Open"].values.astype(float)
        timestamps = df.index
        T          = len(closes)

        state = BacktestState(initial_equity=self.initial_equity)

        # Signal generated at t-1, waiting to execute at t
        pending_signal: int = SignalGenerator.FLAT
        stop_triggered: bool = False   # carry stop flag across steps

        for t in range(warmup, T):

            exec_price    = opens_arr[t]     # execution price (next bar open)
            current_close = closes[t]        # mark-to-market price
            current_time  = timestamps[t]

            # ─────────────────────────────────────────────────────────────
            # STEP 1 — EXECUTE pending order from previous bar
            # ─────────────────────────────────────────────────────────────
            if pending_signal == SignalGenerator.LONG and state.position == 0:
                # Enter long
                allocation    = state.cash * self.position_size_pct
                commission    = allocation * self.commission_pct
                cost          = allocation + commission
                shares_bought = allocation / exec_price
                state.cash   -= cost
                state.shares  = shares_bought
                state.position    = 1
                state.entry_price = exec_price
                state.entry_time  = current_time
                stop_triggered    = False

            elif pending_signal == SignalGenerator.FLAT and state.position == 1:
                # Exit long
                proceeds   = state.shares * exec_price
                commission = proceeds * self.commission_pct
                net        = proceeds - commission
                entry_cost = state.shares * state.entry_price
                pnl        = net - entry_cost
                pnl_pct    = (exec_price - state.entry_price) / state.entry_price

                state.cash   += net
                state.trades.append(Trade(
                    entry_time  = state.entry_time,
                    entry_price = state.entry_price,
                    exit_time   = current_time,
                    exit_price  = exec_price,
                    shares      = state.shares,
                    pnl         = pnl,
                    pnl_pct     = pnl_pct,
                    exit_reason = "stop" if stop_triggered else "signal",
                ))
                state.shares   = 0.0
                state.position = 0
                stop_triggered = False

            # ─────────────────────────────────────────────────────────────
            # STEP 2 — Mark to market (close price)
            # ─────────────────────────────────────────────────────────────
            mtm = state.cash + state.shares * current_close
            state.equity_curve.append(mtm)
            state.timestamps.append(current_time)

            # ─────────────────────────────────────────────────────────────
            # STEP 3 — Hard stop-loss check
            # Evaluated at close[t]; exit executes at open[t+1]
            # ─────────────────────────────────────────────────────────────
            if state.position == 1:
                drawdown = (current_close - state.entry_price) / state.entry_price
                if drawdown <= -self.stop_loss_pct:
                    pending_signal = SignalGenerator.FLAT
                    stop_triggered = True
                    continue  # skip signal generation — force exit next bar

            # ─────────────────────────────────────────────────────────────
            # STEP 4 — COMPUTE FEATURES using data[0 .. t]  (causal!)
            # history = prices[0], prices[1], ... prices[t]
            # ─────────────────────────────────────────────────────────────
            history = closes[: t + 1]   # slice is [0, t] — strictly past + current

            rsi = feature_engine.compute_rsi(history)
            middle, upper, lower = feature_engine.compute_bollinger_bands(history)
            hfe, lfe, ratio = feature_engine.compute_wavelet_features(history)
            regime = regime_classifier.classify(hfe, lfe, ratio)

            # ─────────────────────────────────────────────────────────────
            # STEP 5 — GENERATE SIGNAL for execution at t+1
            # ─────────────────────────────────────────────────────────────
            pending_signal = signal_generator.generate(
                price           = current_close,
                rsi             = rsi,
                middle_band     = middle,
                upper_band      = upper,
                lower_band      = lower,
                regime          = regime,
                current_position= state.position,
            )

            # Store diagnostics (optional, zero overhead in production)
            state.feature_log.append({
                "t":       t,
                "time":    current_time,
                "close":   current_close,
                "rsi":     rsi,
                "middle":  middle,
                "upper":   upper,
                "lower":   lower,
                "hfe":     hfe,
                "lfe":     lfe,
                "ratio":   ratio,
                "regime":  regime,
                "signal":  pending_signal,
                "pos":     state.position,
            })

        # ── Close any still-open position at the last bar ──────────────
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
    """Compute standard quantitative performance metrics."""
    equity = np.array(state.equity_curve, dtype=float)
    if len(equity) < 2:
        return {}

    daily_returns = np.diff(equity) / equity[:-1]
    total_return  = (equity[-1] - state.initial_equity) / state.initial_equity
    n_years       = len(equity) / 252.0
    annual_return = (1 + total_return) ** (1.0 / max(n_years, 1e-9)) - 1

    sharpe = 0.0
    if np.std(daily_returns) > 1e-10:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)

    peak       = np.maximum.accumulate(equity)
    drawdowns  = (equity - peak) / peak
    max_dd     = float(np.min(drawdowns))

    # Calmar ratio
    calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-10 else 0.0

    # Trade stats
    trades = state.trades
    n_trades = len(trades)
    wins  = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    win_rate = len(wins) / n_trades if n_trades else 0.0

    avg_win  = float(np.mean([t.pnl for t in wins]))   if wins   else 0.0
    avg_loss = float(np.mean([t.pnl for t in losses])) if losses else 0.0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss   = abs(sum(t.pnl for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-10 else float("inf")

    # Sortino
    neg_returns = daily_returns[daily_returns < 0]
    downside_std = np.std(neg_returns) if len(neg_returns) > 1 else 1e-10
    sortino = np.mean(daily_returns) / downside_std * np.sqrt(252)

    return {
        "Total Return":    f"{total_return:+.2%}",
        "Annual Return":   f"{annual_return:+.2%}",
        "Sharpe Ratio":    f"{sharpe:.3f}",
        "Sortino Ratio":   f"{sortino:.3f}",
        "Calmar Ratio":    f"{calmar:.3f}",
        "Max Drawdown":    f"{max_dd:.2%}",
        "Profit Factor":   f"{profit_factor:.2f}",
        "Total Trades":    str(n_trades),
        "Win Rate":        f"{win_rate:.2%}",
        "Avg Win ($)":     f"{avg_win:,.2f}",
        "Avg Loss ($)":    f"{avg_loss:,.2f}",
        "Final Equity":    f"${equity[-1]:,.2f}",
    }


# ============================================================
# OUTPUT & REPORTING
# ============================================================

def print_trade_log(state: BacktestState, n_tail: int = 15) -> None:
    hdr = (
        f"{'#':>4}  {'Entry Date':<12}  {'Entry $':>9}  "
        f"{'Exit Date':<12}  {'Exit $':>9}  {'PnL $':>10}  "
        f"{'PnL %':>8}  {'Reason':<12}"
    )
    print("\n" + "=" * 80)
    print(f"TRADE LOG  (last {n_tail} of {len(state.trades)} trades)")
    print("=" * 80)
    print(hdr)
    print("-" * 80)
    for i, tr in enumerate(state.trades[-n_tail:], 1):
        idx = len(state.trades) - n_tail + i
        edate = str(tr.entry_time.date()) if tr.entry_time else "—"
        xdate = str(tr.exit_time.date())  if tr.exit_time  else "open"
        xprice = f"{tr.exit_price:>9.2f}" if tr.exit_price else "  open  "
        print(
            f"{idx:>4}  {edate:<12}  {tr.entry_price:>9.2f}  "
            f"{xdate:<12}  {xprice}  {tr.pnl:>10.2f}  "
            f"{tr.pnl_pct:>7.2%}  {tr.exit_reason:<12}"
        )


def plot_results(state: BacktestState, metrics: dict, save_path: str = "wavelet_strategy_results.png") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available — skipping chart.")
        return

    equity  = np.array(state.equity_curve)
    dates   = state.timestamps
    peak    = np.maximum.accumulate(equity)
    dd      = (equity - peak) / peak * 100.0

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Wavelet Mean Reversion Strategy — Walk-Forward Backtest", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── Equity curve ────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, equity, linewidth=1.4, color="#2196F3", label="Strategy equity")
    ax1.axhline(state.initial_equity, color="grey", linewidth=0.8, linestyle="--", label="Initial equity")
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # ── Drawdown ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(dates, dd, 0, alpha=0.55, color="#F44336", label="Drawdown")
    ax2.set_title("Drawdown (%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25)

    # ── Trade PnL bar chart ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2, 0])
    if state.trades:
        pnls   = [t.pnl for t in state.trades]
        colors = ["#4CAF50" if p > 0 else "#F44336" for p in pnls]
        ax3.bar(range(1, len(pnls) + 1), pnls, color=colors, width=0.8, alpha=0.8)
        ax3.axhline(0, color="black", linewidth=0.6)
        ax3.set_title("Trade PnL ($)")
        ax3.set_xlabel("Trade #")
        ax3.set_ylabel("PnL ($)")
        ax3.grid(True, alpha=0.25)

    # ── Metrics table ────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis("off")
    rows = list(metrics.items())
    tbl  = ax4.table(
        cellText   = rows,
        colLabels  = ["Metric", "Value"],
        cellLoc    = "left",
        loc        = "center",
        bbox       = [0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2196F3")
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
    print("=" * 68)
    print(" REGIME-AWARE WAVELET MEAN REVERSION — STRICT WALK-FORWARD BACKTEST")
    print("=" * 68)

    # ── 1. Load data ─────────────────────────────────────────────────────
    print("\n[1/5] Loading market data ...")
    df = load_data(ticker="SPY", start="2015-01-01", end="2024-12-31")

    # ── 2. Instantiate modules ────────────────────────────────────────────
    print("[2/5] Initialising modules ...")

    feature_engine = RollingFeatureEngine(
        rsi_period     = 14,    # RSI look-back
        bb_period      = 20,    # Bollinger look-back
        bb_std         = 2.0,   # Band width (sigma)
        wavelet_window = 64,    # Bars fed to wavelet transform
        wavelet_name   = "db4", # Daubechies-4 wavelet
    )

    regime_classifier = RegimeClassifier(
        adaptive_window  = 100,  # rolling window for median threshold
        smoothing_window = 10,   # short buffer for ratio denoising
    )

    signal_generator = SignalGenerator(
        rsi_oversold   = 30.0,
        rsi_overbought = 70.0,
    )

    engine = BacktestEngine(
        initial_equity    = 100_000.0,
        position_size_pct = 0.95,
        commission_pct    = 0.001,   # 10 bps per leg
        stop_loss_pct     = 0.02,    # 2% stop on entry price
    )

    # ── 3. Run strict walk-forward backtest ───────────────────────────────
    print("[3/5] Running walk-forward simulation (step-by-step, no lookahead) ...")
    state = engine.run(
        df                = df,
        feature_engine    = feature_engine,
        regime_classifier = regime_classifier,
        signal_generator  = signal_generator,
        warmup            = 100,   # bars before trading begins
    )
    print(f"      Done. {len(state.trades)} trades executed over {len(state.equity_curve)} bars.")

    # ── 4. Compute & print metrics ────────────────────────────────────────
    print("[4/5] Computing performance metrics ...")
    metrics = compute_metrics(state)

    print("\n" + "=" * 40)
    print("  PERFORMANCE METRICS")
    print("=" * 40)
    for k, v in metrics.items():
        print(f"  {k:<22} {v}")

    # ── 5. Trade log & chart ──────────────────────────────────────────────
    print_trade_log(state, n_tail=15)

    print("\n[5/5] Generating charts ...")
    plot_results(state, metrics, save_path="wavelet_strategy_results.png")

    print("\nDone.")
    return state, metrics


if __name__ == "__main__":
    main()
