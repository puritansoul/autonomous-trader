"""
Autonomous Trader — $10,000 paper portfolio.
No fixed strategy. Detects market regime daily, scores a broad universe
across asset classes, adapts signal weights based on recent performance,
and writes a plain-English thesis each day.
"""

from pathlib import Path
from datetime import date, datetime, timedelta
import json, math, sys

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
except ImportError:
    sys.exit("pip install yfinance pandas numpy")

BASE_DIR      = Path(__file__).parent
STATE_FILE    = BASE_DIR / "state.json"
STARTING_CAP  = 10_000.0
MAX_POSITIONS = 6
MIN_CASH_PCT  = 0.05   # always keep 5% cash

# ── Universe ──────────────────────────────────────────────────────────────────
# Grouped by regime affinity: "risk_on", "risk_off", "any"
UNIVERSE = {
    # Mega-cap / large-cap equities
    "AAPL":  "risk_on",  "MSFT": "risk_on",  "NVDA": "risk_on",
    "GOOGL": "risk_on",  "AMZN": "risk_on",  "META": "risk_on",
    "TSLA":  "risk_on",  "JPM":  "risk_on",  "XOM":  "any",
    # Broad market ETFs
    "SPY":   "risk_on",  "QQQ":  "risk_on",  "IWM":  "risk_on",
    "VTV":   "risk_on",
    # Leveraged ETFs (regime-gated)
    "TQQQ":  "risk_on",  "UPRO": "risk_on",
    "SQQQ":  "risk_off", "SH":   "risk_off",
    # Sectors
    "XLK":   "risk_on",  "XLF":  "risk_on",  "XLE":  "any",
    "XLV":   "any",      "XLU":  "risk_off", "XLP":  "risk_off",
    # Fixed income / safe haven
    "TLT":   "risk_off", "IEF":  "risk_off", "SHY":  "risk_off",
    "GLD":   "any",      "SLV":  "any",
    # Commodities
    "USO":   "any",      "DBC":  "any",      "PDBC": "any",
    # Crypto (via ETFs)
    "IBIT":  "risk_on",  "FBTC": "risk_on",  "ETHA": "risk_on",
    # Volatility
    "UVXY":  "risk_off",
}

REGIME_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "^VIX"]


# ── State ─────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {
        "capital":        STARTING_CAP,
        "positions":      {},   # {ticker: {shares, cost_per_share, cost, entry_date, thesis}}
        "closed":         [],
        "nav_history":    {},
        "signal_weights": {     # adaptive weights, start equal
            "momentum":   0.25,
            "volume":     0.20,
            "trend":      0.25,
            "mean_rev":   0.15,
            "regime_fit": 0.15,
        },
        "regime_history": {},   # {date: regime}
        "daily_log":      [],
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


# ── Adaptive weight update ─────────────────────────────────────────────────────

def update_weights(state: dict, closed_today: list) -> dict:
    """Nudge signal weights based on which signals were highest for winning trades."""
    weights = state["signal_weights"].copy()
    if not closed_today:
        return weights

    wins  = [t for t in closed_today if t["pnl"] > 0]
    losses = [t for t in closed_today if t["pnl"] <= 0]

    lr = 0.03  # learning rate — small nudges

    for trade in wins:
        sig = trade.get("top_signal")
        if sig and sig in weights:
            weights[sig] = min(0.50, weights[sig] + lr)

    for trade in losses:
        sig = trade.get("top_signal")
        if sig and sig in weights:
            weights[sig] = max(0.05, weights[sig] - lr)

    # Renormalize
    total = sum(weights.values())
    weights = {k: round(v / total, 4) for k, v in weights.items()}
    return weights


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
                 scores_top: list, weights: dict) -> str:
    lines = []
    lines.append(f"Regime: {regime.upper()}")
    if "vix" in regime_details:
        lines.append(f"VIX {regime_details['vix']} · SPY trend {regime_details.get('spy_trend','?')} · risk appetite {regime_details.get('risk_appetite','?')}")

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

    # Positions table
    pos_rows = ""
    for tk, pos in sorted(positions.items()):
        cur = float(prices[tk].dropna().iloc[-1]) if tk in prices.columns else pos["cost_per_share"]
        upnl = (cur - pos["cost_per_share"]) * pos["shares"]
        upnl_pct = (cur - pos["cost_per_share"]) / pos["cost_per_share"] * 100
        col  = "#00e676" if upnl >= 0 else "#ff5252"
        hold = (date.today() - date.fromisoformat(pos["entry_date"])).days
        pos_rows += f"""<tr>
          <td class="tk">{tk}</td>
          <td>{pos['entry_date']}</td>
          <td>{hold}d</td>
          <td>${pos['cost_per_share']:.2f}</td>
          <td>${cur:.2f}</td>
          <td>{pos['shares']}</td>
          <td>${pos['cost']:,.0f}</td>
          <td style="color:{col}">{'+' if upnl>=0 else ''}{upnl:,.0f} ({upnl_pct:+.1f}%)</td>
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

    # Signal weight bars
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
    <span class="regime-badge regime-{regime}">{regime}</span>
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
      <h2>Signal Weights (adaptive)</h2>
      <div style="margin-top:8px">{weight_bars}</div>
    </div>
  </div>

  <h2>Open Positions</h2>
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
</body>
</html>"""

    return html


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    today_str = date.today().isoformat()
    state     = load_state()
    print(f"[Autonomous Trader] {today_str}")

    # 1. Fetch prices for regime + universe
    all_tickers = list(UNIVERSE.keys()) + REGIME_TICKERS
    all_tickers = list(dict.fromkeys(all_tickers))  # dedup

    print(f"  Fetching {len(all_tickers)} tickers…")
    prices  = fetch_prices(all_tickers, period="3mo")
    volumes = fetch_volume(list(UNIVERSE.keys()), period="3mo")
    print(f"  Got {len(prices.columns)} price series, {len(prices)} bars")

    if prices.empty:
        print("  No data — aborting")
        return

    # 2. Detect regime
    regime, regime_details = detect_regime(prices)
    print(f"  Regime: {regime} | {regime_details}")
    state["regime_history"][today_str] = regime

    # 3. Check exits on open positions
    closed_today, state["positions"] = check_exits(
        state["positions"], prices, regime, today_str)

    for t in closed_today:
        state["capital"] = round(state["capital"] + t["cost"] + t["pnl"], 2)
        state["closed"].append(t)
    print(f"  Exits: {[t['ticker'] for t in closed_today]}")

    # 4. Score universe
    weights = state["signal_weights"]
    all_scores = []
    for tk in UNIVERSE:
        s = score_ticker(tk, prices, volumes, regime, weights)
        if s:
            all_scores.append(s)

    all_scores.sort(key=lambda x: x["total"], reverse=True)

    # Skip tickers already held
    held = set(state["positions"].keys())
    candidates = [s for s in all_scores if s["ticker"] not in held]

    # 5. Size and enter new positions
    buys = size_positions(candidates, state["capital"], len(state["positions"]))

    for b in buys:
        top_signal = max(b["scores"], key=b["scores"].get)
        state["positions"][b["ticker"]] = {
            "shares":         b["shares"],
            "cost_per_share": round(b["price"], 4),
            "cost":           b["cost"],
            "entry_date":     today_str,
            "top_signal":     top_signal,
            "thesis":         f"score {b['total']:+.3f} · led by {top_signal}",
        }
        state["capital"] = round(state["capital"] - b["cost"], 2)

    print(f"  Entries: {[b['ticker'] for b in buys]}")

    # 6. Update adaptive weights
    state["signal_weights"] = update_weights(state, closed_today)

    # 7. Update NAV
    port_val = portfolio_value(state["capital"], state["positions"], prices)
    state["nav_history"][today_str] = round(port_val, 2)
    print(f"  Portfolio: ${port_val:,.2f}")

    # 8. Build thesis
    thesis = build_thesis(regime, regime_details, buys, closed_today, all_scores[:10], state["signal_weights"])
    state["daily_log"].append({"date": today_str, "thesis": thesis})
    print(f"\n--- THESIS ---\n{thesis}\n")

    # 9. Save state
    save_state(state)

    # 10. Build HTML report
    html = build_report(state, prices, regime, regime_details, thesis, today_str)
    report_path = BASE_DIR / "reports" / f"autonomous_{today_str}.html"
    report_path.write_text(html)

    # Latest alias
    (BASE_DIR / "index.html").write_text(html)
    print(f"  Report → {report_path}")


if __name__ == "__main__":
    run()
