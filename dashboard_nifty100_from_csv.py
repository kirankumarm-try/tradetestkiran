# paste this entire file into your app (keep your existing filename)
import os
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import threading

# -------------------------
# Defaults and strategy params
# -------------------------
DEFAULT_SMA_FAST = 50
DEFAULT_SMA_MED = 150
DEFAULT_EMA_SLOW = 220
DEFAULT_52W_WINDOW = 252
DEFAULT_LOOKBACK_DIP = 90
DEFAULT_STOP_LOSS_PCT = 0.07

# Internal backtest defaults
DEFAULT_MAX_HOLD_DAYS = 20
DEFAULT_INITIAL_CAPITAL = 100000
DEFAULT_ALLOC_PCT = 0.10
DEFAULT_TAKE_PROFIT_PCT = 0.20

st.set_page_config(layout="wide", page_title="NIFTY100 Backtest + Sell Trigger (top100.csv)")

PERSIST_FILE = "dashboard_state.json"

# -------------------------
# CSV loader + sanitizer
# -------------------------
def load_tickers_from_csv_path(path_or_buffer):
    try:
        df = pd.read_csv(path_or_buffer, dtype=str, keep_default_na=False)
        if df.empty:
            return []
        cols = {c.lower(): c for c in df.columns}
        for candidate in ("symbol", "ticker"):
            if candidate in cols:
                colname = cols[candidate]
                raw = df[colname].astype(str)
                cleaned = raw.str.strip().str.replace(r"\s+", "", regex=True)
                cleaned = cleaned[cleaned != ""]
                return list(dict.fromkeys(cleaned.tolist()))
        first_col = df.columns[0]
        raw = df[first_col].astype(str)
        cleaned = raw.str.strip().str.replace(r"\s+", "", regex=True)
        cleaned = cleaned[cleaned != ""]
        return list(dict.fromkeys(cleaned.tolist()))
    except Exception:
        return []

def sanitize_tickers(tickers):
    if not isinstance(tickers, (list, tuple, pd.Series)):
        return []
    cleaned = []
    for t in tickers:
        if not isinstance(t, str):
            continue
        s = t.strip()
        if s == "":
            continue
        s = s.replace(" ", "")
        s = s.upper()
        if "." not in s:
            s = f"{s}.NS"
        cleaned.append(s)
    return list(dict.fromkeys(cleaned))

def get_tickers():
    uploaded = st.sidebar.file_uploader("Upload top100.csv (optional)", type=["csv"])
    if uploaded is not None:
        tickers = load_tickers_from_csv_path(uploaded)
        tickers = sanitize_tickers(tickers)
        if tickers:
            st.sidebar.success(f"Loaded {len(tickers)} tickers from uploaded CSV")
            return tickers
        else:
            st.sidebar.error("Uploaded CSV could not be parsed or contained no tickers. Falling back to local file or built-in list.")
    local_path = "top100.csv"
    if os.path.exists(local_path):
        tickers = load_tickers_from_csv_path(local_path)
        tickers = sanitize_tickers(tickers)
        if tickers:
            st.sidebar.success(f"Loaded {len(tickers)} tickers from top100.csv (local)")
            return tickers
        else:
            st.sidebar.error("Found top100.csv but could not parse tickers. Falling back to built-in list.")
    st.sidebar.warning("No valid top100.csv found. Using built-in sample tickers.")
    return sanitize_tickers([
        "RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK",
        "LT","HINDUNILVR","ITC","AXISBANK","SBIN"
    ])

def load_persisted_state():
    if not os.path.exists(PERSIST_FILE):
        return {}
    try:
        with open(PERSIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_persisted_state(state):
    try:
        with open(PERSIST_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

# -------------------------
# Batched fetch with retries
# -------------------------
def batch_fetch_tickers(tickers, period="2y", interval="1d", batch_size=8, max_retries=3, backoff_base=1.5):
    results = {}
    diagnostics = {"failed": [], "empty": []}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        attempt = 0
        while attempt < max_retries:
            try:
                tickers_str = " ".join(batch)
                raw = yf.download(tickers_str, period=period, interval=interval, group_by='ticker', progress=False, threads=True)
                if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
                    for t in batch:
                        if t in raw.columns.get_level_values(0):
                            df_t = raw[t].copy()
                            if df_t.empty:
                                results[t] = pd.DataFrame()
                                diagnostics["empty"].append(t)
                            else:
                                df_t.index = pd.to_datetime(df_t.index)
                                cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df_t.columns]
                                results[t] = df_t[cols].copy()
                        else:
                            df_t = yf.download(t, period=period, interval=interval, progress=False)
                            if df_t is None or df_t.empty:
                                results[t] = pd.DataFrame()
                                diagnostics["empty"].append(t)
                            else:
                                if isinstance(df_t.columns, pd.MultiIndex):
                                    df_t.columns = df_t.columns.get_level_values(0)
                                df_t = df_t[['Open','High','Low','Close','Volume']].copy()
                                df_t.index = pd.to_datetime(df_t.index)
                                results[t] = df_t.sort_index()
                else:
                    for t in batch:
                        df_t = yf.download(t, period=period, interval=interval, progress=False)
                        if df_t is None or df_t.empty:
                            results[t] = pd.DataFrame()
                            diagnostics["empty"].append(t)
                        else:
                            if isinstance(df_t.columns, pd.MultiIndex):
                                df_t.columns = df_t.columns.get_level_values(0)
                            df_t = df_t[['Open','High','Low','Close','Volume']].copy()
                            df_t.index = pd.to_datetime(df_t.index)
                            results[t] = df_t.sort_index()
                break
            except Exception as e:
                attempt += 1
                wait = backoff_base ** attempt
                time.sleep(wait)
                if attempt >= max_retries:
                    for t in batch:
                        results[t] = pd.DataFrame()
                        diagnostics["failed"].append((t, str(e)))
        time.sleep(0.15)
    return results, diagnostics

# -------------------------
# Indicator computation
# -------------------------
def compute_indicators(df, sma_fast, sma_med, ema_slow, lookback_dip=DEFAULT_LOOKBACK_DIP, window_52w=DEFAULT_52W_WINDOW):
    df = df.copy()
    df['sma_50'] = df['Close'].rolling(sma_fast).mean()
    df['sma_150'] = df['Close'].rolling(sma_med).mean()
    df['ema_220'] = df['Close'].ewm(span=ema_slow, adjust=False).mean()
    df['52w_high'] = df['Close'].rolling(window_52w, min_periods=1).max()
    df['52w_low'] = df['Close'].rolling(window_52w, min_periods=1).min()

    dipped = df['Low'] < df['ema_220']
    df['recent_dip'] = dipped.rolling(lookback_dip, min_periods=1).max().fillna(0)

    df[['sma_50','sma_150','ema_220','52w_high','52w_low']] = (
        df[['sma_50','sma_150','ema_220','52w_high','52w_low']].ffill().bfill()
    )

    for col in ['sma_50','sma_150','ema_220','52w_high','52w_low']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].isna().all():
            df[col] = 0.0
        else:
            df[col] = df[col].ffill().bfill()

    df['cond1_sma150_gt_ema220'] = (df['sma_150'] > df['ema_220']).fillna(False).astype(bool)
    df['cond2_close_gt_sma50']   = (df['Close'] > df['sma_50']).fillna(False).astype(bool)
    df['cond3_sma50_gt_sma150']  = (df['sma_50'] > df['sma_150']).fillna(False).astype(bool)
    df['cond4_close_gt_1.25_low'] = (df['Close'] > (1.25 * df['52w_low'])).fillna(False).astype(bool)
    df['cond5_recent_dip']       = (df['recent_dip'] == 1.0).fillna(False).astype(bool)

    df['trend_eligible'] = df[['cond1_sma150_gt_ema220','cond2_close_gt_sma50','cond3_sma50_gt_sma150','cond4_close_gt_1.25_low','cond5_recent_dip']].all(axis=1)
    df['breakout'] = (df['Close'] >= df['52w_high'].shift(1)).fillna(False).astype(bool)
    df['buy_signal'] = (df['trend_eligible'] & df['breakout']).astype(bool)

    return df

def evaluate_current_from_latest(latest_row, override_price):
    r = latest_row.copy()
    if override_price is not None:
        r['Close'] = float(override_price)
    sma50 = r.get('sma_50', np.nan)
    low52 = r.get('52w_low', np.nan)
    high52 = r.get('52w_high', np.nan)
    close = r.get('Close', np.nan)

    c1 = bool(r.get('cond1_sma150_gt_ema220', False))
    c2 = False if np.isnan(sma50) else bool(close > sma50)
    c3 = bool(r.get('cond3_sma50_gt_sma150', False))
    c4 = False if np.isnan(low52) else bool(close > (1.25 * low52))
    c5 = bool(r.get('cond5_recent_dip', False))
    breakout = False if np.isnan(high52) else bool(close >= high52)
    buy_signal = c1 and c2 and c3 and c4 and c5 and breakout
    return {
        'Close': float(close),
        'cond1': c1,
        'cond2': c2,
        'cond3': c3,
        'cond4': c4,
        'cond5': c5,
        'breakout': breakout,
        'buy_signal': buy_signal
    }

# -------------------------
# Backtest engine (per-ticker)
# -------------------------
def run_backtest(df, alloc_pct, stop_loss_pct, take_profit_pct, max_hold_days, initial_capital):
    if df.empty or len(df) < 10:
        return pd.DataFrame(), pd.Series(dtype=float), {'total_trades':0,'win_rate':0.0,'total_pnl':0.0,'avg_return_per_trade':0.0,'max_drawdown':0.0}
    for col in ['Open','High','Low','Close']:
        if col in df.columns:
            df = df[(df[col].notna()) & (np.isfinite(df[col])) & (df[col] > 0) & (df[col] <= 1_000_000)]
    if df.empty or len(df) < 10:
        return pd.DataFrame(), pd.Series(dtype=float), {'total_trades':0,'win_rate':0.0,'total_pnl':0.0,'avg_return_per_trade':0.0,'max_drawdown':0.0}

    trades = []
    equity = pd.Series(dtype=float)
    cash = initial_capital
    position = None
    dates = df.index

    for i, date in enumerate(dates):
        row = df.loc[date]
        if position is None:
            equity.loc[date] = cash
        else:
            equity.loc[date] = cash + position['shares'] * row['Close']

        if position is None and row.get('buy_signal', False):
            if i + 1 < len(dates):
                entry_date = dates[i + 1]
                entry_open = df.loc[entry_date]['Open']
                try:
                    entry_open = float(entry_open)
                except Exception:
                    continue
                if np.isnan(entry_open) or np.isinf(entry_open) or entry_open <= 0 or entry_open > 1_000_000:
                    continue
                allocation = cash * alloc_pct
                try:
                    shares = int(allocation // entry_open)
                except Exception:
                    continue
                if shares <= 0:
                    continue
                entry_price = entry_open
                stop_price = entry_price * (1 - stop_loss_pct)
                tp_price = entry_price * (1 + take_profit_pct)
                position = {'entry_date': entry_date,'entry_price': entry_price,'shares': shares,'stop_price': stop_price,'tp_price': tp_price,'max_hold_days': max_hold_days}
                cash -= shares * entry_price

        if position is not None:
            holding_days = (date - position['entry_date']).days
            exit_now = False
            exit_price = None
            reason = None
            if row['Low'] <= position['stop_price']:
                exit_now = True
                exit_price = position['stop_price']
                reason = 'stop_loss'
            elif row['High'] >= position['tp_price']:
                exit_now = True
                exit_price = position['tp_price']
                reason = 'take_profit'
            elif row['Close'] < row['sma_50']:
                exit_now = True
                exit_price = float(row['Close'])
                reason = 'sma50_breach'
            elif holding_days >= position['max_hold_days']:
                exit_now = True
                exit_price = float(row['Open']) if not np.isnan(row['Open']) else float(row['Close'])
                reason = 'max_hold_days'
            if exit_now:
                exit_date = date
                pnl = (exit_price - position['entry_price']) * position['shares']
                ret = pnl / (position['entry_price'] * position['shares'])
                trades.append({'entry_date': position['entry_date'],'entry_price': position['entry_price'],'exit_date': exit_date,'exit_price': float(exit_price),'shares': position['shares'],'pnl': float(pnl),'return': float(ret),'reason': reason})
                cash += position['shares'] * exit_price
                position = None
                equity.loc[date] = cash

    if position is not None:
        last_date = dates[-1]
        exit_price = float(df.loc[last_date]['Close'])
        exit_date = last_date
        pnl = (exit_price - position['entry_price']) * position['shares']
        ret = pnl / (position['entry_price'] * position['shares'])
        trades.append({'entry_date': position['entry_date'],'entry_price': position['entry_price'],'exit_date': exit_date,'exit_price': float(exit_price),'shares': position['shares'],'pnl': float(pnl),'return': float(ret),'reason': 'end_of_data'})
        cash += position['shares'] * exit_price
        position = None
        equity.loc[last_date] = cash

    trades_df = pd.DataFrame(trades)
    total_trades = len(trades_df)
    wins = trades_df[trades_df['pnl'] > 0] if total_trades > 0 else pd.DataFrame()
    win_rate = (len(wins) / total_trades) if total_trades > 0 else 0.0
    total_pnl = trades_df['pnl'].sum() if total_trades > 0 else 0.0
    avg_return = trades_df['return'].mean() if total_trades > 0 else 0.0

    equity = equity.sort_index().ffill().fillna(DEFAULT_INITIAL_CAPITAL)
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min() if not drawdown.empty else 0.0

    perf = {'total_trades': total_trades,'win_rate': win_rate,'total_pnl': float(total_pnl),'avg_return_per_trade': float(avg_return) if not np.isnan(avg_return) else 0.0,'max_drawdown': float(max_dd)}
    return trades_df, equity, perf

# -------------------------
# Streamlit UI
# -------------------------
st.title("NIFTY100 Strategy Backtest + Sell Trigger (top100.csv)")

tickers = get_tickers()
period = "2y"  # fixed
persisted_state = load_persisted_state()

# Initialize session state with persisted values
if "results_rows" not in st.session_state:
    st.session_state.results_rows = persisted_state.get("results_rows", [])
    st.session_state.diagnostics = persisted_state.get("diagnostics", {"failed_downloads": [], "too_short": [], "exceptions": []})
    st.session_state.trades_summary = persisted_state.get("trades_summary", {})
    st.session_state.equity_curves = persisted_state.get("equity_curves", {})
    st.session_state.last_refresh = persisted_state.get("last_refresh")

# Get persisted filter states
default_bought = [t for t in persisted_state.get("bought_tickers", []) if t in tickers]
default_sma_fast = persisted_state.get("sma_fast", DEFAULT_SMA_FAST)
default_sma_med = persisted_state.get("sma_med", DEFAULT_SMA_MED)
default_ema_slow = persisted_state.get("ema_slow", DEFAULT_EMA_SLOW)
default_window_52w = persisted_state.get("window_52w", DEFAULT_52W_WINDOW)
default_lookback_dip = persisted_state.get("lookback_dip", DEFAULT_LOOKBACK_DIP)
default_stop_loss_pct = persisted_state.get("stop_loss_pct", DEFAULT_STOP_LOSS_PCT)
default_view_mode = persisted_state.get("view_mode", "Historical")
default_run_backtest = persisted_state.get("run_backtest_flag", False)
default_auto_refresh = persisted_state.get("auto_refresh", False)
default_show_only_matches = persisted_state.get("show_only_matches", False)
default_show_recent = persisted_state.get("show_recent", False)
default_show_c1 = persisted_state.get("show_c1", False)
default_show_c2 = persisted_state.get("show_c2", False)
default_show_c3 = persisted_state.get("show_c3", False)
default_show_c4 = persisted_state.get("show_c4", False)
default_show_c5 = persisted_state.get("show_c5", False)
default_show_breakout = persisted_state.get("show_breakout", False)
default_keep_bought = persisted_state.get("keep_bought", True)
default_hide_bought = persisted_state.get("hide_bought", False)

with st.sidebar:
    st.header("Inputs (kept)")
    st.markdown("**Indicator settings**")
    sma_fast = st.number_input("SMA Fast", min_value=5, max_value=200, value=default_sma_fast, step=1)
    sma_med = st.number_input("SMA Medium", min_value=20, max_value=400, value=default_sma_med, step=1)
    ema_slow = st.number_input("EMA Slow (for sell trigger)", min_value=50, max_value=500, value=default_ema_slow, step=1)
    window_52w = st.number_input("52W Window (days)", min_value=100, max_value=365, value=default_window_52w, step=1)
    lookback_dip = st.number_input("Lookback dip (days)", min_value=10, max_value=365, value=default_lookback_dip, step=1)

    st.markdown("**View Mode**")
    view_mode = st.radio("Select view mode", ["Historical","Current"], index=0 if default_view_mode == "Historical" else 1)

    st.markdown("**Portfolio**")
    bought_tickers = st.multiselect("Bought tickers (mark those you own)", options=tickers, default=default_bought)

    st.markdown("**Optional: per-ticker entry prices (for stop-loss checks)**")
    st.markdown("Enter as comma-separated pairs: SYMBOL:ENTRY_PRICE, e.g. RELIANCE.NS:2500, TCS.NS:3200")
    default_entry_text = persisted_state.get("entry_prices_text", "")
    entry_prices_text = st.text_area("Entry prices (optional)", value=default_entry_text, height=80)

    recent_price_override = None
    if view_mode == "Current":
        st.markdown("**Current mode options**")
        price_source = st.selectbox("Price source for Current mode", ["Use recent Close","Override recent traded price"])
        if price_source == "Override recent traded price":
            recent_price_override = st.number_input("Override price (applies to all tickers)", min_value=0.0, value=0.0, step=0.01, format="%.2f")
            if recent_price_override <= 0:
                recent_price_override = None

    st.markdown("**Filters (enable/disable)**")
    use_c1 = st.checkbox("Require C1: 150 SMA > 220 EMA", value=True)
    use_c2 = st.checkbox("Require C2: Close > 50 SMA", value=True)
    use_c3 = st.checkbox("Require C3: 50 SMA > 150 SMA", value=True)
    use_c4 = st.checkbox("Require C4: Close > 1.25 * 52w Low", value=True)
    use_c5 = st.checkbox("Require C5: Recent dip below EMA occurred", value=True)
    use_breakout = st.checkbox("Require Breakout (Close >= prior 52w high)", value=True)

    st.markdown("**Risk / Backtest**")
    stop_loss_pct = st.slider("Stop loss % (used for stop-loss check vs entry price)", min_value=1, max_value=50, value=int(default_stop_loss_pct*100)) / 100.0
    run_backtest_flag = st.checkbox("Run backtest for matching tickers", value=default_run_backtest)

    st.markdown("**Refresh controls**")
    auto_refresh = st.checkbox("Auto-refresh every hour", value=default_auto_refresh)
    run_btn_manual = st.button("🔄 Run Analysis")

    # Use st_autorefresh with proper parameters for v1.0.1
    if auto_refresh:
        st_autorefresh(interval=3600)
        run_btn = True
    else:
        run_btn = run_btn_manual

    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = None
    if run_btn:
        st.session_state.last_refresh = datetime.now()
    if st.session_state.last_refresh is not None:
        st.sidebar.markdown(f"**Last refresh:** {st.session_state.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")

    # Background server-side refresh options (persists results even when no client connected)
    bg_default = persisted_state.get("background_refresh_enabled", False)
    bg_interval_default = persisted_state.get("background_refresh_interval", 60)
    background_refresh_enabled = st.checkbox("Enable background server refresh (persist refreshed results)", value=bg_default)
    background_refresh_interval = st.number_input("Background refresh interval (minutes)", min_value=1, max_value=1440, value=int(bg_interval_default), step=1)

# Start background thread if requested (single per-process)
try:
    if persisted_state.get("background_refresh_enabled") and not BACKGROUND_THREAD_STARTED:
        def _make_params_fn():
            def fn():
                return {
                    "tickers": tickers,
                    "period": period,
                    "sma_fast": sma_fast,
                    "sma_med": sma_med,
                    "ema_slow": ema_slow,
                    "lookback_dip": lookback_dip,
                    "window_52w": window_52w,
                    "bought_tickers": persisted_state.get("bought_tickers", []),
                    "entry_prices_map": parse_entry_prices(persisted_state.get("entry_prices_text", "")),
                    "stop_loss_pct": persisted_state.get("stop_loss_pct", stop_loss_pct),
                    "run_backtest_flag": persisted_state.get("run_backtest_flag", run_backtest_flag),
                    "entry_prices_text": persisted_state.get("entry_prices_text", "")
                }
            return fn

        interval = int(persisted_state.get("background_refresh_interval", 60))
        t = threading.Thread(target=_background_loop, args=(interval, _make_params_fn()), daemon=True)
        t.start()
        BACKGROUND_THREAD_STARTED = True
except Exception:
    pass

# parse entry prices text into dict
def parse_entry_prices(text):
    mapping = {}
    if not text or not isinstance(text, str):
        return mapping
    parts = [p.strip() for p in text.split(",") if p.strip()]
    for p in parts:
        if ":" in p:
            sym, val = p.split(":", 1)
            sym = sym.strip().upper().replace(" ", "")
            if "." not in sym:
                sym = f"{sym}.NS"
            try:
                price = float(val.strip())
                mapping[sym] = price
            except Exception:
                continue
    return mapping

entry_prices_map = parse_entry_prices(entry_prices_text)


def process_all_tickers(tickers, period, sma_fast, sma_med, ema_slow, lookback_dip, window_52w,
                        bought_tickers, entry_prices_map, stop_loss_pct, run_backtest_flag):
    """Run the full fetch/compute/backtest loop without Streamlit UI calls.
    Returns dict with results_rows, diagnostics, trades_summary, equity_curves, last_refresh
    """
    results_rows = []
    trades_summary = {}
    equity_curves = {}
    diagnostics = {"failed_downloads": [], "too_short": [], "exceptions": []}

    batch_results, batch_diag = batch_fetch_tickers(tickers, period=period, interval="1d", batch_size=8, max_retries=3)
    diagnostics["failed_downloads"].extend(batch_diag.get("failed", []))
    diagnostics["failed_downloads"].extend(batch_diag.get("empty", []))

    for ticker in tickers:
        try:
            df = batch_results.get(ticker, pd.DataFrame())
            if df is None or df.empty:
                results_rows.append({
                    "Symbol": ticker,
                    "Date": None,
                    "Close": None,
                    "Buy Signal": False,
                    "C1": False, "C2": False, "C3": False, "C4": False, "C5": False,
                    "Breakout": False,
                    "Bought": (ticker in bought_tickers),
                    "MatchesFilters": False,
                    "DownloadStatus": "no_data",
                    "Sell Trigger": False,
                    "Sell Reason": None
                })
                continue

            if len(df) < 10:
                diagnostics["too_short"].append(ticker)
                results_rows.append({
                    "Symbol": ticker,
                    "Date": None,
                    "Close": None,
                    "Buy Signal": False,
                    "C1": False, "C2": False, "C3": False, "C4": False, "C5": False,
                    "Breakout": False,
                    "Bought": (ticker in bought_tickers),
                    "MatchesFilters": False,
                    "DownloadStatus": "too_short",
                    "Sell Trigger": False,
                    "Sell Reason": None
                })
                continue

            df = compute_indicators(df, sma_fast, sma_med, ema_slow, lookback_dip, window_52w)

            if view_mode == "Historical":
                buy_idx = df.index[df["buy_signal"] == True]
                if len(buy_idx) > 0:
                    chosen_date = buy_idx[-1]
                    row = df.loc[chosen_date]
                    date_val = chosen_date.date()
                else:
                    row = df.iloc[-1]
                    date_val = df.index[-1].date()
            else:
                row = df.iloc[-1]
                date_val = df.index[-1].date()

            if view_mode == "Current":
                evals = evaluate_current_from_latest(row, recent_price_override)
                c1 = evals["cond1"]; c2 = evals["cond2"]; c3 = evals["cond3"]
                c4 = evals["cond4"]; c5 = evals["cond5"]; breakout = evals["breakout"]
                buy_signal = evals["buy_signal"]; close_val = evals["Close"]
            else:
                c1 = bool(row.get("cond1_sma150_gt_ema220", False))
                c2 = bool(row.get("cond2_close_gt_sma50", False))
                c3 = bool(row.get("cond3_sma50_gt_sma150", False))
                c4 = bool(row.get("cond4_close_gt_1.25_low", False))
                c5 = bool(row.get("cond5_recent_dip", False))
                breakout = bool(row.get("breakout", False))
                buy_signal = bool(row.get("buy_signal", False))
                close_val = float(row.get("Close", np.nan)) if not np.isnan(row.get("Close", np.nan)) else None

            cond_checks = {
                "C1": c1,
                "C2": c2,
                "C3": c3,
                "C4": c4,
                "C5": c5,
                "Breakout": breakout,
            }
            matches = all(cond_checks.values())

            sell_trigger = False
            sell_reasons = []
            if ticker in bought_tickers:
                ema_val = float(row.get("ema_220", np.nan)) if not np.isnan(row.get("ema_220", np.nan)) else None
                if ema_val is not None and close_val is not None and not np.isnan(ema_val):
                    if close_val < ema_val:
                        sell_trigger = True
                        sell_reasons.append("below_ema")
                entry_price = entry_prices_map.get(ticker)
                if entry_price is not None and close_val is not None:
                    stop_price = entry_price * (1 - stop_loss_pct)
                    if close_val <= stop_price:
                        sell_trigger = True
                        sell_reasons.append("stop_loss")

            sell_reason_str = ";".join(sell_reasons) if sell_reasons else None

            results_rows.append({
                "Symbol": ticker,
                "Date": date_val,
                "Close": close_val,
                "Buy Signal": buy_signal,
                "C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5,
                "Breakout": breakout,
                "Bought": (ticker in bought_tickers),
                "MatchesFilters": matches,
                "DownloadStatus": "ok",
                "Sell Trigger": sell_trigger,
                "Sell Reason": sell_reason_str
            })

            if run_backtest_flag and matches:
                trades_df, equity_series, perf = run_backtest(df, DEFAULT_ALLOC_PCT, stop_loss_pct, DEFAULT_TAKE_PROFIT_PCT, DEFAULT_MAX_HOLD_DAYS, DEFAULT_INITIAL_CAPITAL)
                trades_summary[ticker] = perf
                equity_curves[ticker] = equity_series

        except Exception as e:
            diagnostics["exceptions"].append((ticker, str(e)))
            results_rows.append({
                "Symbol": ticker,
                "Date": None,
                "Close": None,
                "Buy Signal": False,
                "C1": False, "C2": False, "C3": False, "C4": False, "C5": False,
                "Breakout": False,
                "Bought": (ticker in bought_tickers),
                "MatchesFilters": False,
                "DownloadStatus": "error",
                "Sell Trigger": False,
                "Sell Reason": None
            })
            continue

    return {
        "results_rows": results_rows,
        "diagnostics": diagnostics,
        "trades_summary": trades_summary,
        "equity_curves": equity_curves,
        "last_refresh": datetime.now()
    }


# Background thread control
BACKGROUND_THREAD_STARTED = False

def _background_loop(interval_minutes, get_params_fn):
    while True:
        try:
            params = get_params_fn()
            res = process_all_tickers(**params)
            # persist results to file
            state = load_persisted_state() or {}
            state.update(res)
            state["bought_tickers"] = params.get("bought_tickers", [])
            state["entry_prices_text"] = params.get("entry_prices_text", state.get("entry_prices_text", ""))
            state["last_refresh"] = res.get("last_refresh")
            save_persisted_state(state)
        except Exception:
            pass
        time.sleep(max(60, int(interval_minutes) * 60))

st.write(f"📊 Processing {len(tickers)} tickers | Mode: **{view_mode}**")
st.markdown("---")

# -------------------------
# Main processing
# -------------------------
if run_btn:
    # Call non-UI processing function and persist results
    with st.spinner("Running analysis..."):
        params = {
            "tickers": tickers,
            "period": period,
            "sma_fast": sma_fast,
            "sma_med": sma_med,
            "ema_slow": ema_slow,
            "lookback_dip": lookback_dip,
            "window_52w": window_52w,
            "bought_tickers": bought_tickers,
            "entry_prices_map": entry_prices_map,
            "stop_loss_pct": stop_loss_pct,
            "run_backtest_flag": run_backtest_flag
        }
        res = process_all_tickers(**params)
        st.session_state.results_rows = res["results_rows"]
        st.session_state.diagnostics = res["diagnostics"]
        st.session_state.trades_summary = res["trades_summary"]
        st.session_state.equity_curves = res["equity_curves"]
        st.session_state.last_refresh = res["last_refresh"]
        # persist results and settings for next visits
        state = load_persisted_state() or {}
        state.update(res)
        state["bought_tickers"] = bought_tickers
        state["entry_prices_text"] = entry_prices_text
        state["sma_fast"] = sma_fast
        state["sma_med"] = sma_med
        state["ema_slow"] = ema_slow
        state["window_52w"] = window_52w
        state["lookback_dip"] = lookback_dip
        state["stop_loss_pct"] = stop_loss_pct
        state["view_mode"] = view_mode
        state["run_backtest_flag"] = run_backtest_flag
        state["auto_refresh"] = auto_refresh
        state["background_refresh_enabled"] = background_refresh_enabled
        state["background_refresh_interval"] = background_refresh_interval
        state["show_only_matches"] = default_show_only_matches
        state["show_recent"] = default_show_recent
        state["show_c1"] = default_show_c1
        state["show_c2"] = default_show_c2
        state["show_c3"] = default_show_c3
        state["show_c4"] = default_show_c4
        state["show_c5"] = default_show_c5
        state["show_breakout"] = default_show_breakout
        state["keep_bought"] = default_keep_bought
        state["hide_bought"] = default_hide_bought
        save_persisted_state(state)
        st.success("Analysis complete")

if st.session_state.results_rows:
    results_df = pd.DataFrame(st.session_state.results_rows)
    diagnostics = st.session_state.diagnostics
    trades_summary = st.session_state.trades_summary
    equity_curves = st.session_state.equity_curves

    # Ensure latest bought selections are reflected even when cached rows exist
    if "Bought" in results_df.columns:
        results_df["Bought"] = results_df["Symbol"].isin(bought_tickers)
    else:
        results_df["Bought"] = results_df["Symbol"].isin(bought_tickers)

    # Display diagnostics
    st.markdown("### Diagnostics summary")
    st.write({
        "Tickers loaded": len(tickers),
        "Failed downloads (batch)": len(diagnostics.get("failed_downloads", [])),
        "Too-short data": len(diagnostics.get("too_short", [])),
        "Exceptions": len(diagnostics.get("exceptions", []))
    })
    if diagnostics.get("failed_downloads"):
        st.warning("Failed downloads (first 20):")
        st.write(diagnostics["failed_downloads"][:20])
    if diagnostics.get("exceptions"):
        st.error("Some tickers raised exceptions (first 10):")
        st.write(diagnostics["exceptions"][:10])

    # Results table with extra filters
    show_only_matches = st.checkbox("Show only tickers that match enabled filters (MatchesFilters == True)", value=default_show_only_matches)
    show_recent = st.checkbox("Show only tickers with Date within last 30 days", value=default_show_recent)
    show_c1 = st.checkbox("Show only tickers satisfying C1 (150 SMA > 220 EMA)", value=default_show_c1)
    show_c2 = st.checkbox("Show only tickers satisfying C2 (Close > 50 SMA)", value=default_show_c2)
    show_c3 = st.checkbox("Show only tickers satisfying C3 (50 SMA > 150 SMA)", value=default_show_c3)
    show_c4 = st.checkbox("Show only tickers satisfying C4 (Close > 1.25 * 52w Low)", value=default_show_c4)
    show_c5 = st.checkbox("Show only tickers satisfying C5 (Recent dip below EMA occurred)", value=default_show_c5)
    show_breakout = st.checkbox("Show only tickers satisfying Breakout (Close >= prior 52w high)", value=default_show_breakout)
    keep_bought = st.checkbox("Keep bought tickers visible even if other filters fail", value=default_keep_bought)
    hide_bought = st.checkbox("Hide bought tickers entirely", value=default_hide_bought)

    # Save filter states
    state = load_persisted_state() or {}
    state["show_only_matches"] = show_only_matches
    state["show_recent"] = show_recent
    state["show_c1"] = show_c1
    state["show_c2"] = show_c2
    state["show_c3"] = show_c3
    state["show_c4"] = show_c4
    state["show_c5"] = show_c5
    state["show_breakout"] = show_breakout
    state["keep_bought"] = keep_bought
    state["hide_bought"] = hide_bought
    save_persisted_state(state)

    display_df = results_df.copy()
    filter_mask = pd.Series(True, index=display_df.index)

    if show_only_matches:
        filter_mask &= display_df["MatchesFilters"] == True

    if show_recent and "Date" in display_df.columns:
        from datetime import datetime, timedelta
        cutoff = datetime.now().date() - timedelta(days=30)
        filter_mask &= display_df["Date"].notna() & (display_df["Date"] >= cutoff)

    if show_c1:
        filter_mask &= display_df["C1"] == True
    if show_c2:
        filter_mask &= display_df["C2"] == True
    if show_c3:
        filter_mask &= display_df["C3"] == True
    if show_c4:
        filter_mask &= display_df["C4"] == True
    if show_c5:
        filter_mask &= display_df["C5"] == True
    if show_breakout:
        filter_mask &= display_df["Breakout"] == True

    if keep_bought:
        filter_mask |= display_df["Bought"] == True
    if hide_bought:
        filter_mask &= display_df["Bought"] == False

    display_df = display_df[filter_mask]

    # Final display
    if display_df.empty:
        st.info("No tickers matched the selected filters or no data available.")
    else:
        cols_order = [
            "Symbol","Date","Close","Bought","Sell Trigger","Sell Reason",
            "MatchesFilters","Buy Signal","C1","C2","C3","C4","C5","Breakout","DownloadStatus"
        ]
        cols_present = [c for c in cols_order if c in display_df.columns]

        if view_mode == "Current":
            st.subheader(f"Results — {view_mode} mode (showing {len(display_df)} rows)")
            st.dataframe(display_df.set_index("Symbol")[cols_present[2:]])
        else:
            st.subheader(f"Results — {view_mode} mode (showing {len(display_df)} rows)")
            st.dataframe(display_df[cols_present].sort_values(["Symbol","Date"], ascending=[True, False]))

    if run_backtest_flag and trades_summary:
        st.subheader("Backtest summaries")
        for t, perf in trades_summary.items():
            st.markdown(f"### {t}")
            st.write({
                "Total trades": perf["total_trades"],
                "Win rate": f"{perf['win_rate']*100:.1f}%",
                "Total PnL": f"{perf['total_pnl']:.2f}",
                "Avg return per trade": f"{perf['avg_return_per_trade']*100:.2f}%",
                "Max drawdown": f"{perf['max_drawdown']*100:.2f}%"
            })
        if equity_curves:
            combined = pd.DataFrame({t: s for t, s in equity_curves.items()})
            st.line_chart(combined.ffill().fillna(0))
