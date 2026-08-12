"""
Autonomous Trader — $10,000 paper portfolio.
No fixed strategy. Detects market regime daily, scores a broad universe
across asset classes, and switches between named strategy modes
(momentum, mean_reversion, trend_following, defensive) based on which
has produced the best recent closed-trade P&L.
"""

from pathlib import Path
from datetime import date, datetime, timedelta
import json, math, sys, re as _re, urllib.request

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("pip install yfinance pandas numpy")

BASE_DIR      = Path(__file__).parent
STATE_FILE    = BASE_DIR / "state.json"
STARTING_CAP  = 10_000.0
MAX_POSITIONS = 10
MIN_CASH_PCT  = 0.05   # always keep 5% cash

# ── Universe ──────────────────────────────────────────────────────────────────
# GICS sector → regime affinity mapping
_SECTOR_AFFINITY = {
    "Information Technology":    "risk_on",
    "Communication Services":    "risk_on",
    "Consumer Discretionary":    "risk_on",
    "Financials":                "risk_on",
    "Industrials":               "risk_on",
    "Materials":                 "any",
    "Energy":                    "any",
    "Real Estate":               "any",
    "Health Care":               "any",
    "Consumer Staples":          "risk_off",
    "Utilities":                 "risk_off",
}

# Supplemental non-stock assets (ETFs, crypto, leveraged, commodities)
_SUPPLEMENT = {
    # Broad market & factor
    "SPY": "risk_on", "QQQ": "risk_on", "IWM": "risk_on",
    "VTV": "risk_on", "VUG": "risk_on", "IJR": "risk_on",
    "MTUM": "risk_on", "QUAL": "risk_on", "VLUE": "any",
    # Sector ETFs
    "XLK": "risk_on", "XLF": "risk_on", "XLE": "any",
    "XLV": "any", "XLI": "risk_on", "XLY": "risk_on",
    "XLP": "risk_off", "XLU": "risk_off", "XLRE": "any",
    "XLB": "any", "XLC": "risk_on",
    # Thematic
    "ARKK": "risk_on", "ARKG": "risk_on", "ARKW": "risk_on",
    "BOTZ": "risk_on", "SOXX": "risk_on", "CIBR": "risk_on",
    "CLOU": "risk_on", "XBI": "risk_on", "JETS": "risk_on",
    "IBB": "any",
    # International
    "EFA": "risk_on", "EEM": "risk_on", "VEA": "risk_on",
    "VWO": "risk_on", "FXI": "risk_on", "EWJ": "any",
    "EWZ": "risk_on", "INDA": "risk_on", "IEMG": "risk_on",
    # Leveraged long
    "TQQQ": "risk_on", "UPRO": "risk_on", "SOXL": "risk_on",
    "TECL": "risk_on", "FNGU": "risk_on",
    # Inverse / short
    "SQQQ": "risk_off", "SH": "risk_off", "SDS": "risk_off",
    "SPXS": "risk_off", "SOXS": "risk_off", "PSQ": "risk_off",
    # Fixed income
    "TLT": "risk_off", "IEF": "risk_off", "SHY": "risk_off",
    "BND": "risk_off", "AGG": "risk_off", "LQD": "any",
    "HYG": "risk_on", "JNK": "risk_on", "TIP": "any",
    # Commodities
    "GLD": "any", "SLV": "any", "IAU": "any",
    "GDX": "any", "GDXJ": "risk_on",
    "USO": "any", "UNG": "any", "DBC": "any",
    "CORN": "any", "WEAT": "any", "CPER": "any",
    # Real estate
    "VNQ": "any", "IYR": "any",
    # Crypto ETFs
    "IBIT": "risk_on", "FBTC": "risk_on", "ETHA": "risk_on",
    # Volatility
    "UVXY": "risk_off", "VXX": "risk_off", "VIXY": "risk_off",
}

_SP500_CACHE_FILE = BASE_DIR / "sp500_tickers.json"


def _fetch_sp500() -> dict:
    """Fetch S&P 500 constituents from Wikipedia. Returns {ticker: affinity}."""
    try:
        import urllib.request
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
        # Parse ticker + GICS sector from the first wikitable
        rows = _re.findall(
            r'<td[^>]*>\s*<a[^>]*>([A-Z]{1,5}(?:\.[A-Z])?)</a>\s*</td>'
            r'(?:.*?<td[^>]*>.*?</td>){2}.*?<td[^>]*>(.*?)</td>',
            html, _re.DOTALL)
        result = {}
        for ticker, sector_raw in rows:
            sector = _re.sub(r'<[^>]+>', '', sector_raw).strip()
            affinity = _SECTOR_AFFINITY.get(sector, "any")
            # yfinance uses hyphens not dots for BRK.B etc.
            ticker = ticker.replace(".", "-")
            result[ticker] = affinity
        if len(result) > 400:  # sanity check
            return result
    except Exception as e:
        print(f"  Warning: S&P 500 fetch failed ({e}) — using cache")
    return {}


def build_universe() -> dict:
    """Return full investible universe: S&P 500 + supplement ETFs/crypto/commodities."""
    # Try live fetch first
    sp500 = _fetch_sp500()

    if sp500:
        # Cache for fallback
        _SP500_CACHE_FILE.write_text(json.dumps(sp500))
        print(f"  S&P 500: {len(sp500)} tickers fetched")
    else:
        # Use cache
        if _SP500_CACHE_FILE.exists():
            sp500 = json.loads(_SP500_CACHE_FILE.read_text())
            print(f"  S&P 500: {len(sp500)} tickers from cache")
        else:
            print("  S&P 500: no data, supplement only")

    universe = {**sp500, **_SUPPLEMENT}  # supplement overrides (correct affinities for ETFs)
    print(f"  Universe: {len(universe)} total tickers")
    return universe


REGIME_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "^VIX"]

# Module-level universe — populated at runtime by build_universe()
UNIVERSE = _SUPPLEMENT  # fallback until first run

# ── Strategy modes ────────────────────────────────────────────────────────────
# Each mode is a fixed weighting of the 5 signals.
# The bot selects the active mode nightly based on which produced the best
# avg P&L on trades opened under that mode (last 10 closed trades).
STRATEGY_MODES = {
    "momentum": {
        "momentum":   0.50,
        "volume":     0.25,
        "trend":      0.15,
        "mean_rev":   0.05,
        "regime_fit": 0.05,
        "description": "Chase price strength and volume breakouts",
    },
    "trend_following": {
        "momentum":   0.20,
        "volume":     0.10,
        "trend":      0.50,
        "mean_rev":   0.05,
        "regime_fit": 0.15,
        "description": "Follow SMA alignment and regime-matched assets",
    },
    "mean_reversion": {
        "momentum":   0.05,
        "volume":     0.15,
        "trend":      0.10,
        "mean_rev":   0.55,
        "regime_fit": 0.15,
        "description": "Buy oversold dips in trending markets",
    },
    "defensive": {
        "momentum":   0.10,
        "volume":     0.05,
        "trend":      0.20,
        "mean_rev":   0.10,
        "regime_fit": 0.55,
        "description": "Prioritize regime-fit safe-haven assets",
    },
}
MODE_SIGNAL_KEYS = ["momentum", "volume", "trend", "mean_rev", "regime_fit"]


# ── State ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        s = json.loads(STATE_FILE.read_text())
        s.setdefault("pending_orders", [])
        s.setdefault("active_mode", "momentum")
        s.setdefault("mode_history", {})
        s.setdefault("mode_pnl", {m: [] for m in STRATEGY_MODES})
        s.setdefault("last_backtest", [])
        for m in STRATEGY_MODES:
            s["mode_pnl"].setdefault(m, [])
        return s
    return {
        "capital":        STARTING_CAP,
        "positions":      {},
        "closed":         [],
        "nav_history":    {},
        "signal_weights": {k: v for k, v in STRATEGY_MODES["momentum"].items() if k in MODE_SIGNAL_KEYS},
        "active_mode":    "momentum",
        "mode_history":   {},
        "mode_pnl":       {m: [] for m in STRATEGY_MODES},
        "last_backtest":  [],
        "regime_history": {},
        "daily_log":      [],
        "pending_orders": [],
        "inception_date": date.today().isoformat(),
    }


def save_state(s: dict):
    STATE_FILE.write_text(json.dumps(s, indent=2, default=str))


# ── Market data ───────────────────────────────────────────────────────────────

def fetch_prices(tickers: list, period: str = "3mo") -> pd.DataFrame:
    raw = yf.download(tickers, period=period, auto_adjust=True,
                      progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]] if "Close" in raw.columns else raw
    return closes.dropna(how="all")


def fetch_volume(tickers: list, period: str = "3mo") -> pd.DataFrame:
    raw = yf.download(tickers, period=period, auto_adjust=True,
                      progress=False, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        return raw["Volume"].dropna(how="all")
    return pd.DataFrame()


def fetch_opens(tickers: list) -> dict:
    """Fetch today's open price for each ticker. Returns {ticker: open_price}."""
    if not tickers:
        return {}
    raw = yf.download(tickers, period="2d", interval="1d", auto_adjust=True,
                      progress=False, threads=True)
    opens = {}
    for tk in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                series = raw["Open"][tk].dropna()
            else:
                series = raw["Open"].dropna()
            if len(series) > 0:
                opens[tk] = round(float(series.iloc[-1]), 4)
        except Exception:
            pass
    return opens


# ── Regime detection ──────────────────────────────────────────────────────────

def detect_regime(prices: pd.DataFrame) -> tuple[str, dict]:
    """
    Returns (regime, details) where regime is 'bull', 'bear', or 'neutral'.
    Uses SPY trend, QQQ vs TLT ratio, and VIX level.
    """
    details = {}

    # SPY 20/50 SMA trend
    if "SPY" in prices.columns and len(prices) >= 50:
        spy = prices["SPY"].dropna()
        sma20 = spy.rolling(20).mean().iloc[-1]
        sma50 = spy.rolling(50).mean().iloc[-1]
        spy_cur = spy.iloc[-1]
        spy_trend = "up" if spy_cur > sma20 > sma50 else ("down" if spy_cur < sma20 < sma50 else "mixed")
        details["spy_trend"] = spy_trend
        details["spy_price"] = round(float(spy_cur), 2)
    else:
        spy_trend = "mixed"

    # VIX level
    vix_level = 0.0
    if "^VIX" in prices.columns:
        vix = prices["^VIX"].dropna()
        if len(vix) > 0:
            vix_level = float(vix.iloc[-1])
            details["vix"] = round(vix_level, 1)

    # QQQ/TLT ratio trend (risk appetite)
    risk_appetite = "neutral"
    if "QQQ" in prices.columns and "TLT" in prices.columns:
        ratio = (prices["QQQ"] / prices["TLT"]).dropna()
        if len(ratio) >= 20:
            r20 = ratio.rolling(20).mean()
            if ratio.iloc[-1] > r20.iloc[-1] * 1.02:
                risk_appetite = "up"
            elif ratio.iloc[-1] < r20.iloc[-1] * 0.98:
                risk_appetite = "down"
        details["risk_appetite"] = risk_appetite

    # Regime decision
    bull_signals = sum([
        spy_trend == "up",
        vix_level < 20 and vix_level > 0,
        risk_appetite == "up",
    ])
    bear_signals = sum([
        spy_trend == "down",
        vix_level > 30,
        risk_appetite == "down",
    ])

    if bull_signals >= 2:
        regime = "bull"
    elif bear_signals >= 2:
        regime = "bear"
    else:
        regime = "neutral"

    details["regime"] = regime
    return regime, details


# ── Signal scoring ────────────────────────────────────────────────────────────

def score_ticker(tk: str, prices: pd.DataFrame, volumes: pd.DataFrame,
                 regime: str, weights: dict) -> dict:
    if tk not in prices.columns:
        return None
    px = prices[tk].dropna()
    if len(px) < 21:
        return None

    cur = float(px.iloc[-1])
    if cur <= 0:
        return None

    scores = {}

    # 1. Momentum: 20-day return
    ret20 = (cur / float(px.iloc[-21]) - 1) if len(px) >= 21 else 0
    ret5  = (cur / float(px.iloc[-6])  - 1) if len(px) >= 6  else 0
    scores["momentum"] = float(np.clip((ret20 * 3 + ret5) / 4 * 10, -1, 1))

    # 2. Volume breakout: today's vol vs 20-day avg
    scores["volume"] = 0.0
    if tk in volumes.columns:
        vol = volumes[tk].dropna()
        if len(vol) >= 21:
            avg_vol = float(vol.iloc[-21:-1].mean())
            cur_vol = float(vol.iloc[-1])
            ratio = cur_vol / avg_vol if avg_vol > 0 else 1.0
            scores["volume"] = float(np.clip((ratio - 1) * 0.5, -0.5, 1.0))

    # 3. Trend alignment: price vs 20/50 SMA
    sma20 = float(px.rolling(20).mean().iloc[-1])
    sma50 = float(px.rolling(50).mean().iloc[-1]) if len(px) >= 50 else sma20
    if cur > sma20 > sma50:
        scores["trend"] = 1.0
    elif cur > sma20:
        scores["trend"] = 0.5
    elif cur < sma20 < sma50:
        scores["trend"] = -1.0
    elif cur < sma20:
        scores["trend"] = -0.5
    else:
        scores["trend"] = 0.0

    # 4. Mean reversion: RSI(14) — score is higher when oversold in bull regime
    if len(px) >= 15:
        delta = px.diff().dropna()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 100
        rsi = 100 - 100 / (1 + rs)
        # In bull: favor oversold (RSI < 40) or mild pullback (40-55)
        # In bear: favor overbought (RSI > 60) as short signal
        if regime == "bull":
            scores["mean_rev"] = float(np.clip((50 - rsi) / 50, -1, 1))
        elif regime == "bear":
            scores["mean_rev"] = float(np.clip((rsi - 50) / 50, -1, 1))
        else:
            scores["mean_rev"] = float(np.clip((50 - abs(rsi - 50)) / 50, 0, 1))
    else:
        scores["mean_rev"] = 0.0

    # 5. Regime fit
    affinity = UNIVERSE.get(tk, "any")
    if affinity == "any":
        scores["regime_fit"] = 0.3
    elif (affinity == "risk_on" and regime == "bull") or \
         (affinity == "risk_off" and regime == "bear"):
        scores["regime_fit"] = 1.0
    elif (affinity == "risk_on" and regime == "bear") or \
         (affinity == "risk_off" and regime == "bull"):
        scores["regime_fit"] = -0.5
    else:
        scores["regime_fit"] = 0.0

    # Weighted total
    total = sum(weights.get(k, 0) * v for k, v in scores.items())

    return {
        "ticker":    tk,
        "price":     cur,
        "total":     round(total, 4),
        "scores":    {k: round(v, 3) for k, v in scores.items()},
        "ret20":     round(ret20 * 100, 2),
        "ret5":      round(ret5 * 100, 2),
    }


# ── Strategy mode selection ────────────────────────────────────────────────────

def backtest_mode(mode_name: str, prices: pd.DataFrame, volumes: pd.DataFrame,
                  regime: str, lookback: int = 5, top_n: int = 5) -> dict:
    """
    Paper-backtest a mode by:
      1. Score the universe using that mode's weights at prices[:-lookback]
      2. Take the top_n picks
      3. Measure their actual return over the last `lookback` days
    Returns {"mode": ..., "avg_return": ..., "picks": [...], "raw": [...]}
    """
    weights = {k: v for k, v in STRATEGY_MODES[mode_name].items() if k in MODE_SIGNAL_KEYS}

    if len(prices) < lookback + 21:
        return {"mode": mode_name, "avg_return": 0.0, "picks": [], "raw": []}

    # Prices at decision point (lookback days ago)
    past_prices  = prices.iloc[:-(lookback)]
    past_volumes = volumes.iloc[:-(lookback)] if not volumes.empty else volumes

    candidates = []
    for tk in UNIVERSE:
        s = score_ticker(tk, past_prices, past_volumes, regime, weights)
        if s and s["total"] > 0:
            candidates.append(s)

    candidates.sort(key=lambda x: x["total"], reverse=True)
    picks = candidates[:top_n]

    returns = []
    for p in picks:
        tk = p["ticker"]
        if tk not in prices.columns:
            continue
        col = prices[tk].dropna()
        # price at decision point vs today
        decision_idx = len(col) - lookback - 1
        if decision_idx < 0:
            continue
        p_then = float(col.iloc[decision_idx])
        p_now  = float(col.iloc[-1])
        if p_then > 0:
            ret = (p_now - p_then) / p_then * 100
            returns.append({"ticker": tk, "return": round(ret, 2), "score": p["total"]})

    avg_ret = sum(r["return"] for r in returns) / len(returns) if returns else 0.0
    return {
        "mode":       mode_name,
        "avg_return": round(avg_ret, 3),
        "picks":      [r["ticker"] for r in returns],
        "raw":        returns,
    }


def select_mode(prices: pd.DataFrame, volumes: pd.DataFrame,
                regime: str, state: dict) -> tuple[str, dict, list]:
    """
    Nightly paper backtest: score each mode's top picks against actual recent
    market returns. The mode whose hypothetical picks performed best over the
    last 5 days wins. No portfolio trade history needed — evaluated fresh nightly.
    Returns (mode_name, weights_dict, backtest_results).
    """
    results = []
    for mode_name in STRATEGY_MODES:
        r = backtest_mode(mode_name, prices, volumes, regime, lookback=5, top_n=5)
        results.append(r)

    results.sort(key=lambda x: x["avg_return"], reverse=True)
    best = results[0]["mode"]
    weights = {k: v for k, v in STRATEGY_MODES[best].items() if k in MODE_SIGNAL_KEYS}
    return best, weights, results


def record_mode_pnl(state: dict, closed_today: list):
    """Keep a record of closed trade P&L per mode (for display in scorecard)."""
    for t in closed_today:
        entry_mode = t.get("entry_mode", state.get("active_mode", "momentum"))
        if entry_mode in state["mode_pnl"]:
            state["mode_pnl"][entry_mode].append(t["pnl_pct"])
            state["mode_pnl"][entry_mode] = state["mode_pnl"][entry_mode][-20:]


# ── Position sizing ───────────────────────────────────────────────────────────

def size_positions(candidates: list, capital: float, n_open: int) -> list:
    """
    Allocate capital proportionally to score, capped at MAX_POSITIONS total.
    Higher conviction = larger position.
    """
    slots = MAX_POSITIONS - n_open
    if slots <= 0:
        return []

    # Only buy positive-scored candidates
    buys = [c for c in candidates if c["total"] > 0.15][:slots]
    if not buys:
        return []

    # Conviction-weighted allocation
    total_score = sum(c["total"] for c in buys)
    available = capital * (1 - MIN_CASH_PCT)

    sized = []
    for c in buys:
        alloc  = available * (c["total"] / total_score)
        shares = math.floor(alloc / c["price"])
        if shares < 1:
            continue
        cost = shares * c["price"]
        sized.append({**c, "shares": shares, "cost": round(cost, 2)})

    return sized


# ── Exit logic ────────────────────────────────────────────────────────────────

def check_exits(positions: dict, prices: pd.DataFrame,
                regime: str, today_str: str) -> tuple[list, dict]:
    """
    Returns (closed_trades, remaining_positions).
    Exit rules: -8% stop loss, +25% take profit, 15-day max hold, regime flip exit.
    """
    closed = []
    remaining = {}

    for tk, pos in positions.items():
        if tk not in prices.columns:
            remaining[tk] = pos
            continue

        cur = float(prices[tk].dropna().iloc[-1])
        cost_ps = pos["cost_per_share"]
        pnl_pct = (cur - cost_ps) / cost_ps

        entry_date = date.fromisoformat(pos["entry_date"])
        hold_days  = (date.today() - entry_date).days

        # Determine exit
        reason = None
        if pnl_pct <= -0.08:
            reason = f"stop loss ({pnl_pct:+.1%})"
        elif pnl_pct >= 0.25:
            reason = f"take profit ({pnl_pct:+.1%})"
        elif hold_days >= 15:
            reason = f"max hold {hold_days}d"
        elif UNIVERSE.get(tk, "any") == "risk_on" and regime == "bear":
            reason = f"regime flip → bear"
        elif UNIVERSE.get(tk, "any") == "risk_off" and regime == "bull":
            reason = f"regime flip → bull"

        if reason:
            pnl = round((cur - cost_ps) * pos["shares"], 2)
            closed.append({
                "ticker":     tk,
                "entry_date": pos["entry_date"],
                "exit_date":  today_str,
                "entry_price": round(cost_ps, 4),
                "exit_price":  round(cur, 4),
                "shares":     pos["shares"],
                "cost":       pos["cost"],
                "pnl":        pnl,
                "pnl_pct":    round(pnl_pct * 100, 2),
                "reason":     reason,
                "top_signal": pos.get("top_signal"),
            })
        else:
            remaining[tk] = pos

    return closed, remaining


# ── Portfolio valuation ───────────────────────────────────────────────────────

def portfolio_value(capital: float, positions: dict, prices: pd.DataFrame) -> float:
    val = capital
    for tk, pos in positions.items():
        if tk in prices.columns:
            cur = float(prices[tk].dropna().iloc[-1])
            val += cur * pos["shares"]
        else:
            val += pos["cost"]
    return round(val, 2)


# ── Daily thesis ──────────────────────────────────────────────────────────────

def build_thesis(regime: str, regime_details: dict,
                 buys: list, exits: list,
                 scores_top: list, weights: dict,
                 active_mode: str = "momentum",
                 prev_mode: str = "", **kwargs) -> str:
    lines = []
    lines.append(f"Regime: {regime.upper()}")
    if "vix" in regime_details:
        lines.append(f"VIX {regime_details['vix']} · SPY trend {regime_details.get('spy_trend','?')} · risk appetite {regime_details.get('risk_appetite','?')}")

    bt = kwargs.get("backtest_results", [])
    if prev_mode and prev_mode != active_mode:
        lines.append(f"Strategy mode switched: {prev_mode} → {active_mode} ({STRATEGY_MODES[active_mode]['description']})")
    else:
        lines.append(f"Strategy mode: {active_mode} — {STRATEGY_MODES[active_mode]['description']}")
    if bt:
        bt_line = "Backtest (5d): " + "  |  ".join(
            f"{r['mode'].replace('_',' ')} {r['avg_return']:+.2f}%" for r in bt)
        lines.append(bt_line)

    if exits:
        lines.append(f"Exited {len(exits)} position(s): " +
                     ", ".join(f"{t['ticker']} {t['pnl']:+.0f} ({t['reason']})" for t in exits))

    if buys:
        lines.append(f"Entered {len(buys)} position(s):")
        for b in buys:
            top_sig = max(b["scores"], key=b["scores"].get)
            lines.append(f"  {b['ticker']} ${b['price']:.2f} × {b['shares']} shares "
                         f"(score {b['total']:+.3f}, led by {top_sig})")

    if not buys and not exits:
        lines.append("No trades today — holding or waiting for better setups.")

    top = sorted(scores_top, key=lambda x: x["total"], reverse=True)[:3]
    if top:
        lines.append("Top candidates screened: " +
                     ", ".join(f"{t['ticker']} {t['total']:+.3f}" for t in top))

    w_sorted = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    lines.append("Signal weights: " + " · ".join(f"{k} {v:.0%}" for k, v in w_sorted))

    return "\n".join(lines)


# ── HTML report ───────────────────────────────────────────────────────────────

def build_report(state: dict, prices: pd.DataFrame,
                 regime: str, regime_details: dict,
                 thesis: str, today_str: str) -> str:

    capital   = state["capital"]
    positions = state["positions"]
    closed    = state["closed"]
    nav_hist  = state["nav_history"]

    port_val  = portfolio_value(capital, positions, prices)
    total_ret = (port_val - STARTING_CAP) / STARTING_CAP * 100
    nav_dates = sorted(nav_hist.keys())

    # Day P&L
    if len(nav_dates) >= 2:
        prev_val = nav_hist[nav_dates[-2]]
        day_pnl  = port_val - prev_val
    else:
        prev_val = STARTING_CAP
        day_pnl  = port_val - STARTING_CAP
    day_pct  = day_pnl / prev_val * 100 if prev_val else 0
    day_sign = "+" if day_pnl >= 0 else "-"
    day_col  = "#00e676" if day_pnl >= 0 else "#ff5252"
    ret_col  = "#00e676" if total_ret >= 0 else "#ff5252"

    # Positions table — embed data attributes for live JS price updates
    pos_rows = ""
    for tk, pos in sorted(positions.items()):
        cur = float(prices[tk].dropna().iloc[-1]) if tk in prices.columns else pos["cost_per_share"]
        upnl = (cur - pos["cost_per_share"]) * pos["shares"]
        upnl_pct = (cur - pos["cost_per_share"]) / pos["cost_per_share"] * 100
        col  = "#00e676" if upnl >= 0 else "#ff5252"
        hold = (date.today() - date.fromisoformat(pos["entry_date"])).days
        pos_rows += f"""<tr data-ticker="{tk}" data-cost="{pos['cost_per_share']:.4f}" data-shares="{pos['shares']}">
          <td class="tk">{tk}</td>
          <td>{pos['entry_date']}</td>
          <td>{hold}d</td>
          <td>${pos['cost_per_share']:.2f}</td>
          <td class="live-price">${cur:.2f}</td>
          <td>{pos['shares']}</td>
          <td>${pos['cost']:,.0f}</td>
          <td class="live-upnl" style="color:{col}">{'+' if upnl>=0 else ''}{upnl:,.0f} ({upnl_pct:+.1f}%)</td>
          <td style="font-size:11px;color:#aaa">{pos.get('top_signal','—')}</td>
        </tr>"""

    if not pos_rows:
        pos_rows = '<tr><td colspan="9" style="color:#666;text-align:center">No open positions</td></tr>'

    # Closed trades (last 20)
    closed_rows = ""
    for t in reversed(closed[-20:]):
        col = "#00e676" if t["pnl"] > 0 else "#ff5252"
        closed_rows += f"""<tr>
          <td class="tk">{t['ticker']}</td>
          <td>{t['entry_date']}</td>
          <td>{t['exit_date']}</td>
          <td>${t['entry_price']:.2f}</td>
          <td>${t['exit_price']:.2f}</td>
          <td>{t['shares']}</td>
          <td style="color:{col}">{'+' if t['pnl']>=0 else ''}{t['pnl']:,.0f} ({t['pnl_pct']:+.1f}%)</td>
          <td style="font-size:11px;color:#aaa">{t.get('reason','—')}</td>
        </tr>"""

    if not closed_rows:
        closed_rows = '<tr><td colspan="8" style="color:#666;text-align:center">No closed trades yet</td></tr>'

    # Equity sparkline
    spark_pts = ""
    if len(nav_dates) >= 2:
        nav_vals = [nav_hist[d] for d in nav_dates]
        lo, hi = min(nav_vals), max(nav_vals)
        rng = hi - lo or 1
        W, H = 300, 60
        pts = []
        for i, v in enumerate(nav_vals):
            x = i / (len(nav_vals) - 1) * W
            y = H - ((v - lo) / rng) * (H - 8) - 4
            pts.append(f"{x:.1f},{y:.1f}")
        spark_pts = " ".join(pts)

    spark_svg = ""
    if spark_pts:
        col_s = "#00e676" if total_ret >= 0 else "#ff5252"
        area = f"0,{H} {spark_pts} {W},{H}"
        spark_svg = f"""<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" style="display:block">
          <polygon points="{area}" fill="{col_s}18"/>
          <polyline points="{spark_pts}" fill="none" stroke="{col_s}" stroke-width="2"/>
        </svg>"""

    # Signal weight bars for active mode
    weight_bars = ""
    for sig, w in sorted(state["signal_weights"].items(), key=lambda x: x[1], reverse=True):
        pct = int(w * 100)
        weight_bars += f"""<div style="display:flex;align-items:center;gap:8px;margin:3px 0">
          <span style="width:90px;font-size:11px;color:#aaa">{sig}</span>
          <div style="flex:1;height:8px;background:#1e1e1e;border-radius:4px">
            <div style="width:{pct}%;height:8px;background:#7c4dff;border-radius:4px"></div>
          </div>
          <span style="width:32px;font-size:11px;color:#ccc;text-align:right">{pct}%</span>
        </div>"""

    # Mode scorecard — show last backtest results
    active_mode = state.get("active_mode", "momentum")
    mode_pnl    = state.get("mode_pnl", {})
    last_backtest = state.get("last_backtest", [])
    bt_by_mode = {r["mode"]: r for r in last_backtest}
    mode_colors = {
        "momentum":        "#ff9800",
        "trend_following": "#00bcd4",
        "mean_reversion":  "#ab47bc",
        "defensive":       "#66bb6a",
    }
    mode_cards = ""
    for m, cfg in STRATEGY_MODES.items():
        bt = bt_by_mode.get(m, {})
        bt_ret = bt.get("avg_return")
        bt_picks = ", ".join(bt.get("picks", [])[:3])
        # Portfolio trade record for this mode
        trades = mode_pnl.get(m, [])
        port_wins = sum(1 for p in trades if p > 0)
        port_total = len(trades)
        is_active = (m == active_mode)
        border = f"2px solid {mode_colors.get(m,'#555')}" if is_active else "1px solid #1e1e1e"
        label = "● ACTIVE" if is_active else ""
        bt_str = f"{bt_ret:+.2f}%" if bt_ret is not None else "—"
        bt_col = "#00e676" if bt_ret and bt_ret > 0 else "#ff5252" if bt_ret and bt_ret < 0 else "#888"
        port_str = f"{port_wins}/{port_total} trades" if port_total else "no trades yet"
        mode_cards += f"""<div style="background:#111;border-radius:8px;padding:12px 14px;border:{border}">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
            <span style="font-weight:700;color:{mode_colors.get(m,'#ccc')};font-size:13px">{m.replace('_',' ').title()}</span>
            <span style="font-size:10px;color:{'#00e676' if is_active else '#444'};font-weight:600">{label}</span>
          </div>
          <div style="font-size:11px;color:#555;margin-bottom:8px">{cfg['description']}</div>
          <div style="font-size:11px;color:#666;margin-bottom:4px">5-day paper backtest</div>
          <div style="display:flex;gap:12px;margin-bottom:4px">
            <span style="font-size:13px;font-weight:700;color:{bt_col}">{bt_str}</span>
            <span style="font-size:11px;color:#555;align-self:flex-end">{bt_picks or '—'}</span>
          </div>
          <div style="font-size:11px;color:#444">{port_str}</div>
        </div>"""

    # JS positions array for live price updates
    pos_js = json.dumps([
        {"ticker": tk, "cost": pos["cost_per_share"], "shares": pos["shares"]}
        for tk, pos in sorted(positions.items())
    ])

    # Recent log entries
    log_html = ""
    for entry in reversed(state["daily_log"][-7:]):
        log_html += f'<div style="margin:8px 0;padding:10px;background:#111;border-radius:6px;border-left:3px solid #7c4dff"><div style="font-size:11px;color:#666;margin-bottom:4px">{entry["date"]}</div><pre style="font-size:12px;color:#ccc;white-space:pre-wrap;margin:0">{entry["thesis"]}</pre></div>'

    today_fmt = date.fromisoformat(today_str).strftime("%b %-d, %Y")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Autonomous Trader — {today_str}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ background: #0a0a0a; color: #e0e0e0;
           font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', monospace; }}
    .page {{ max-width: 1100px; margin: 0 auto; padding: 20px; }}
    h1 {{ font-size: 20px; font-weight: 700; color: #fff; letter-spacing: -0.5px; }}
    h2 {{ font-size: 13px; font-weight: 600; color: #888; text-transform: uppercase;
          letter-spacing: 1px; margin: 20px 0 8px; }}
    .header {{ display:flex; justify-content:space-between; align-items:center;
               border-bottom: 1px solid #1e1e1e; padding-bottom: 12px; margin-bottom: 16px; }}
    .regime-badge {{ padding: 4px 10px; border-radius: 20px; font-size: 11px;
                     font-weight: 700; letter-spacing: 1px; text-transform: uppercase; }}
    .regime-bull   {{ background:#003300; color:#00e676; border:1px solid #00e676; }}
    .regime-bear   {{ background:#330000; color:#ff5252; border:1px solid #ff5252; }}
    .regime-neutral{{ background:#1a1a1a; color:#ffd740; border:1px solid #ffd740; }}
    .stats-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:16px; }}
    .stat-card {{ background:#111; border-radius:10px; padding:14px 16px;
                  border: 1px solid #1e1e1e; }}
    .stat-label {{ font-size:11px; color:#666; text-transform:uppercase; letter-spacing:.5px; }}
    .stat-val   {{ font-size:22px; font-weight:700; margin-top:4px; }}
    .two-col {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:16px; }}
    .card {{ background:#111; border-radius:10px; padding:16px;
             border:1px solid #1e1e1e; overflow-x:auto; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th {{ color:#555; font-weight:600; padding:6px 8px; text-align:left;
          border-bottom:1px solid #1e1e1e; font-size:11px; text-transform:uppercase; }}
    td {{ padding:6px 8px; border-bottom:1px solid #141414; }}
    tr:last-child td {{ border-bottom:none; }}
    .tk {{ font-weight:700; color:#7c4dff; }}
    .spark-wrap {{ margin: 4px 0 12px; }}
  </style>
</head>
<body>
<div class="page">
  <div class="header">
    <div>
      <h1>&#9889; Autonomous Trader</h1>
      <div style="font-size:12px;color:#555;margin-top:2px">{today_fmt} · $10k paper portfolio</div>
    </div>
    <div style="display:flex;gap:8px;align-items:center">
      <span style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:.5px">Mode</span>
      <span style="font-size:13px;font-weight:700;color:{mode_colors.get(active_mode,'#ccc')}">{active_mode.replace('_',' ').title()}</span>
      <span class="regime-badge regime-{regime}" style="margin-left:8px">{regime}</span>
    </div>
  </div>

  <div class="stats-row">
    <div class="stat-card">
      <div class="stat-label">Portfolio</div>
      <div class="stat-val" id="port-value">${port_val:,.0f}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Today</div>
      <div class="stat-val" id="day-pnl" data-pnl="{day_pnl:.2f}" style="color:{day_col}">{day_sign}${abs(day_pnl):,.0f}<br>
        <span style="font-size:14px">{day_sign}{abs(day_pct):.2f}%</span>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-label">All-time</div>
      <div class="stat-val" style="color:{ret_col}">
        {'+' if total_ret>=0 else ''}{total_ret:.2f}%<br>
        <span style="font-size:14px">{'+' if (port_val-STARTING_CAP)>=0 else ''}${abs(port_val-STARTING_CAP):,.0f}</span>
      </div>
    </div>
    <div class="stat-card">
      <div class="stat-label">Cash</div>
      <div class="stat-val">${capital:,.0f}
        <span style="font-size:14px;color:#666"> / {capital/port_val*100:.0f}%</span>
      </div>
    </div>
  </div>

  <div class="two-col">
    <div class="card">
      <h2>Equity Curve</h2>
      <div class="spark-wrap">{spark_svg or '<div style="color:#333;font-size:12px">Accumulating history…</div>'}</div>
      <div style="font-size:11px;color:#444">{len(nav_dates)} trading days tracked</div>
    </div>
    <div class="card">
      <h2>Active Mode · Signal Weights</h2>
      <div style="margin-top:8px">{weight_bars}</div>
    </div>
  </div>

  <h2>Strategy Mode Scorecard</h2>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:16px">
    {mode_cards}
  </div>

  <div style="display:flex;align-items:center;gap:10px;margin:20px 0 8px">
    <h2 style="margin:0">Open Positions</h2>
    <span id="live-badge" style="font-size:10px;padding:2px 8px;border-radius:10px;background:#1a1a1a;color:#555;border:1px solid #222">fetching…</span>
    <span id="live-port" style="font-size:12px;color:#888;margin-left:auto"></span>
  </div>
  <div class="card" style="margin-bottom:16px">
    <table>
      <thead><tr>
        <th>Ticker</th><th>Entry</th><th>Hold</th><th>Cost/sh</th>
        <th>Now</th><th>Shares</th><th>Invested</th><th>Unreal P&amp;L</th><th>Lead Signal</th>
      </tr></thead>
      <tbody>{pos_rows}</tbody>
    </table>
  </div>

  <h2>Closed Trades</h2>
  <div class="card" style="margin-bottom:16px">
    <table>
      <thead><tr>
        <th>Ticker</th><th>Entry</th><th>Exit</th><th>Entry $</th>
        <th>Exit $</th><th>Shares</th><th>P&amp;L</th><th>Reason</th>
      </tr></thead>
      <tbody>{closed_rows}</tbody>
    </table>
  </div>

  <h2>Daily Thesis Log</h2>
  <div style="margin-bottom:16px">{log_html or '<div style="color:#333;font-size:12px;padding:12px">No entries yet.</div>'}</div>

</div>

<script>
(function() {{
  // Tickers with open positions — baked in at report generation time
  const POSITIONS = {pos_js};
  const CASH = {capital:.2f};

  if (!POSITIONS.length) return;

  function isMarketHours() {{
    const et = new Date(new Date().toLocaleString('en-US', {{timeZone: 'America/New_York'}}));
    const d = et.getDay(), m = et.getHours() * 60 + et.getMinutes();
    return d >= 1 && d <= 5 && m >= 570 && m < 960;
  }}

  function fmt(n, prefix) {{
    const s = prefix + '$' + Math.abs(n).toLocaleString('en-US', {{minimumFractionDigits:0, maximumFractionDigits:0}});
    return s;
  }}

  function setStatus(msg, col) {{
    const b = document.getElementById('live-badge');
    if (b) {{ b.textContent = msg; b.style.color = col || '#555'; b.style.borderColor = col || '#222'; }}
  }}

  async function fetchQuotes() {{
    const symbols = POSITIONS.map(p => p.ticker).join(',');
    const url = `https://query1.finance.yahoo.com/v7/finance/quote?symbols=${{encodeURIComponent(symbols)}}&fields=regularMarketPrice,regularMarketChangePercent`;
    try {{
      const res = await fetch(url, {{headers: {{'Accept': 'application/json'}}}});
      if (!res.ok) throw new Error(res.status);
      const data = await res.json();
      const quotes = data?.quoteResponse?.result || [];
      const priceMap = {{}};
      quotes.forEach(q => {{ priceMap[q.symbol] = q.regularMarketPrice; }});
      return priceMap;
    }} catch(e) {{
      // Yahoo CORS blocks direct fetch in some browsers — try corsproxy fallback
      try {{
        const proxy = `https://corsproxy.io/?${{encodeURIComponent(url)}}`;
        const res2 = await fetch(proxy);
        if (!res2.ok) throw new Error(res2.status);
        const data2 = await res2.json();
        const quotes2 = data2?.quoteResponse?.result || [];
        const priceMap2 = {{}};
        quotes2.forEach(q => {{ priceMap2[q.symbol] = q.regularMarketPrice; }});
        return priceMap2;
      }} catch(e2) {{
        setStatus('price unavailable', '#555');
        return null;
      }}
    }}
  }}

  async function update() {{
    setStatus('fetching…', '#555');
    const prices = await fetchQuotes();
    if (!prices) return;

    let totalLive = CASH;
    let totalUnreal = 0;

    POSITIONS.forEach(pos => {{
      const price = prices[pos.ticker];
      if (price == null) return;
      const row = document.querySelector(`tr[data-ticker="${{pos.ticker}}"]`);
      if (!row) return;

      const upnl = (price - pos.cost) * pos.shares;
      const upnlPct = (price - pos.cost) / pos.cost * 100;
      const col = upnl >= 0 ? '#00e676' : '#ff5252';
      const sign = upnl >= 0 ? '+' : '';

      const priceCell = row.querySelector('.live-price');
      const upnlCell  = row.querySelector('.live-upnl');
      if (priceCell) priceCell.textContent = '$' + price.toFixed(2);
      if (upnlCell) {{
        upnlCell.textContent = sign + '$' + Math.abs(upnl).toLocaleString('en-US', {{maximumFractionDigits:0}}) +
          ' (' + sign + upnlPct.toFixed(1) + '%)';
        upnlCell.style.color = col;
      }}

      totalLive  += price * pos.shares;
      totalUnreal += upnl;
    }});

    // Update portfolio value card
    const pv = document.getElementById('port-value');
    if (pv) pv.textContent = '$' + Math.round(totalLive).toLocaleString('en-US');

    // Update live portfolio line
    const lp = document.getElementById('live-port');
    if (lp) {{
      const sign = totalUnreal >= 0 ? '+' : '';
      lp.textContent = 'Unrealized: ' + sign + '$' + Math.abs(totalUnreal).toLocaleString('en-US', {{maximumFractionDigits:0}});
      lp.style.color = totalUnreal >= 0 ? '#00e676' : '#ff5252';
    }}

    const now = new Date().toLocaleTimeString('en-US', {{hour:'2-digit', minute:'2-digit', timeZone:'America/New_York'}});
    setStatus('live · ' + now + ' ET', '#00e676');
  }}

  // Run immediately, then every 5 minutes during market hours
  update();
  setInterval(() => {{ if (isMarketHours()) update(); }}, 5 * 60 * 1000);
}})();
</script>
</body>
</html>"""

    return html


# ── Phase 1: Close (4pm ET) — score universe, queue orders ───────────────────

def run():
    """Run at 9:35am ET each trading day.
    Uses yesterday's close prices for signals, fills at today's open price.
    """
    global UNIVERSE
    today_str = date.today().isoformat()
    state = load_state()
    print(f"[Autonomous Trader] {today_str}")

    # Build universe fresh (S&P 500 + supplement)
    UNIVERSE = build_universe()

    # 1. Fetch yesterday's close prices for signals (6mo history)
    all_tickers = list(dict.fromkeys(list(UNIVERSE.keys()) + REGIME_TICKERS))
    print(f"  Fetching close prices for {len(all_tickers)} tickers…")
    prices  = fetch_prices(all_tickers, period="6mo")
    volumes = fetch_volume(list(UNIVERSE.keys()), period="6mo")
    print(f"  Got {len(prices.columns)} price series, {len(prices)} bars")

    if prices.empty:
        print("  No data — aborting")
        return

    # 2. Detect regime from yesterday's close
    regime, regime_details = detect_regime(prices)
    print(f"  Regime: {regime} | {regime_details}")
    state["regime_history"][today_str] = regime

    # 3. Select strategy mode via paper backtest on yesterday's close prices
    prev_mode = state.get("active_mode", "momentum")
    new_mode, new_weights, backtest_results = select_mode(prices, volumes, regime, state)
    state["active_mode"] = new_mode
    state["signal_weights"] = new_weights
    state["mode_history"][today_str] = new_mode
    state["last_backtest"] = backtest_results
    if new_mode != prev_mode:
        print(f"  Mode switch: {prev_mode} → {new_mode}  (backtest: {backtest_results[0]['avg_return']:+.2f}%)")
    else:
        print(f"  Mode: {new_mode}  (backtest: {backtest_results[0]['avg_return']:+.2f}%)")

    # 4. Fetch today's open prices for actual trade fills
    held_tickers = list(state["positions"].keys())
    # Score universe first to know what we want to buy
    weights = state["signal_weights"]
    all_scores = []
    for tk in UNIVERSE:
        s = score_ticker(tk, prices, volumes, regime, weights)
        if s:
            all_scores.append(s)
    all_scores.sort(key=lambda x: x["total"], reverse=True)

    # Identify exits based on yesterday's close
    exits_flagged = []
    for tk, pos in state["positions"].items():
        if tk not in prices.columns:
            continue
        cur = float(prices[tk].dropna().iloc[-1])
        cost_ps = pos["cost_per_share"]
        pnl_pct = (cur - cost_ps) / cost_ps
        hold_days = (date.today() - date.fromisoformat(pos["entry_date"])).days

        reason = None
        if pnl_pct <= -0.08:
            reason = f"stop loss ({pnl_pct:+.1%})"
        elif pnl_pct >= 0.25:
            reason = f"take profit ({pnl_pct:+.1%})"
        elif hold_days >= 15:
            reason = f"max hold {hold_days}d"
        elif UNIVERSE.get(tk, "any") == "risk_on" and regime == "bear":
            reason = "regime flip → bear"
        elif UNIVERSE.get(tk, "any") == "risk_off" and regime == "bull":
            reason = "regime flip → bull"
        if reason:
            exits_flagged.append({"ticker": tk, "reason": reason})

    exiting = {e["ticker"] for e in exits_flagged}
    held = set(state["positions"].keys()) - exiting
    candidates = [s for s in all_scores if s["ticker"] not in held and s["ticker"] not in exiting]

    freed_capital = sum(state["positions"][e["ticker"]]["cost"] for e in exits_flagged)
    est_capital = state["capital"] + freed_capital
    buys_planned = size_positions(candidates, est_capital, len(state["positions"]) - len(exits_flagged))

    # Fetch today's open prices for everything we plan to touch
    fill_tickers = list(dict.fromkeys(
        [e["ticker"] for e in exits_flagged] + [b["ticker"] for b in buys_planned]
    ))
    print(f"  Fetching today's open for {len(fill_tickers)} tickers…")
    open_prices = fetch_opens(fill_tickers)
    print(f"  Got {len(open_prices)} open prices")

    # 5. Execute exits at today's open
    closed_today = []
    for e in exits_flagged:
        tk = e["ticker"]
        pos = state["positions"].get(tk)
        if not pos:
            continue
        fill = open_prices.get(tk) or round(float(prices[tk].dropna().iloc[-1]), 4)
        pnl = round((fill - pos["cost_per_share"]) * pos["shares"], 2)
        pnl_pct = (fill - pos["cost_per_share"]) / pos["cost_per_share"]
        closed_today.append({
            "ticker":      tk,
            "entry_date":  pos["entry_date"],
            "exit_date":   today_str,
            "entry_price": pos["cost_per_share"],
            "exit_price":  round(fill, 4),
            "shares":      pos["shares"],
            "cost":        pos["cost"],
            "pnl":         pnl,
            "pnl_pct":    round(pnl_pct * 100, 2),
            "reason":      e["reason"],
            "top_signal":  pos.get("top_signal"),
            "entry_mode":  pos.get("entry_mode", new_mode),
        })
        state["capital"] = round(state["capital"] + fill * pos["shares"], 2)
        del state["positions"][tk]
        print(f"  EXIT  {tk} @ ${fill:.2f}  P&L {pnl:+.2f}  ({e['reason']})")

    state["closed"].extend(closed_today)

    # 6. Execute entries at today's open
    filled_entries = []
    for b in buys_planned:
        tk = b["ticker"]
        if tk in state["positions"]:
            continue
        fill = open_prices.get(tk)
        if not fill:
            print(f"  SKIP {tk} — no open price")
            continue
        shares = math.floor(b["shares"] * b["price"] / fill)
        if shares < 1:
            continue
        cost = round(fill * shares, 2)
        if cost > state["capital"] * 0.95:
            print(f"  SKIP {tk} — insufficient capital")
            continue
        top_signal = max(b["scores"], key=b["scores"].get)
        state["positions"][tk] = {
            "shares":         shares,
            "cost_per_share": round(fill, 4),
            "cost":           cost,
            "entry_date":     today_str,
            "top_signal":     top_signal,
            "entry_mode":     new_mode,
            "thesis":         f"score {b['total']:+.3f} · led by {top_signal} · mode {new_mode}",
        }
        state["capital"] = round(state["capital"] - cost, 2)
        filled_entries.append({**b, "price": fill, "shares": shares, "cost": cost,
                                "top_signal": top_signal})
        print(f"  ENTRY {tk} @ ${fill:.2f}  × {shares} shares  ${cost:.0f}  [{new_mode}]")

    # 7. Record mode P&L, update NAV
    record_mode_pnl(state, closed_today)
    port_val = portfolio_value(state["capital"], state["positions"], prices)
    state["nav_history"][today_str] = round(port_val, 2)
    print(f"  Portfolio: ${port_val:,.2f}")

    # 8. Build thesis and log
    thesis = build_thesis(regime, regime_details, filled_entries, closed_today,
                          all_scores[:10], state["signal_weights"], new_mode, prev_mode,
                          backtest_results=backtest_results)
    state["daily_log"].append({"date": today_str, "thesis": thesis})
    state["pending_orders"] = []

    # 9. Save state and write report
    save_state(state)
    html = build_report(state, prices, regime, regime_details, thesis, today_str)
    report_path = BASE_DIR / "reports" / f"autonomous_{today_str}.html"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(html)
    (BASE_DIR / "index.html").write_text(html)
    print(f"  Report → {report_path}")
    print(f"  Report → {report_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run()
