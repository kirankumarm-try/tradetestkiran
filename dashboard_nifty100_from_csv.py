# dashboard_nifty100_buy_sell.py
import os
import time
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(layout="wide", page_title="NIFTY100 Buy/Sell Dashboard")

# -------------------------
# Defaults and strategy params
# -------------------------
DEFAULT_SMA_FAST = 50
DEFAULT_SMA_MED = 150
DEFAULT_EMA_SLOW = 220
DEFAULT_52W_WINDOW = 252
DEFAULT_LOOKBACK_DIP = 90
DEFAULT_STOP_LOSS_PCT = 0.07

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
            json.dump(state, f, indent=2, default=str)
    except Exception:
        pass

# -------------------------
# Fetch helpers
# -------------------------
def fetch_ticker(ticker, period="2y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open','High','Low','Close','Volume']].copy()
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()

def batch_fetch_tickers_simple(tickers, period="2y", interval="1d"):
    results = {}
    diagnostics = {"failed": [], "empty": []}
    for t in tickers:
        df = fetch_ticker(t, period=period, interval=interval)
        if df is None or df.empty:
            results[t] = pd.DataFrame()
            diagnostics["empty"].append(t)
        else:
            results[t] = df
    return results, diagnostics

# -------------------------
# Indicators
# -------------------------
def compute_indicators(df, sma_fast=DEFAULT_SMA_FAST, sma_med=DEFAULT_SMA_MED,
                       ema_slow=DEFAULT_EMA_SLOW, lookback_dip=DEFAULT_LOOKBACK_DIP,
                       window_52w=DEFAULT_52W_WINDOW, rsi_period=14):
    if df is None or df.empty:
        return df
    df = df.copy()

    # Existing indicators...
    df['sma_50'] = df['Close'].rolling(sma_fast).mean()
    df['sma_150'] = df['Close'].rolling(sma_med).mean()
    df['ema_220'] = df['Close'].ewm(span=ema_slow, adjust=False).mean()
    df['52w_high'] = df['Close'].rolling(window_52w, min_periods=1).max()
    df['52w_low'] = df['Close'].rolling(window_52w, min_periods=1).min()

    # RSI calculation
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # (rest of your existing indicator logic)
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

# -------------------------
# Sell finder
# -------------------------
def find_sell_after_buy(df, buy_date, entry_price, stop_loss_pct):
    """
    Returns (sell_date (date), sell_price (float), reason) or (None, None, None)
    """
    if df is None or df.empty or buy_date is None:
        return None, None, None
    df = df.sort_index()
    try:
        start_ts = pd.to_datetime(buy_date)
    except Exception:
        start_ts = pd.to_datetime(buy_date)
    future = df[df.index > start_ts]
    if future.empty:
        return None, None, None
    stop_threshold = None
    if entry_price is not None:
        try:
            stop_threshold = float(entry_price) * (1 - float(stop_loss_pct))
        except Exception:
            stop_threshold = None
    for idx, row in future.iterrows():
        close = row.get("Close", np.nan)
        ema = row.get("ema_220", np.nan)
        if not pd.isna(ema) and not pd.isna(close) and close < ema:
            return idx.date(), float(close), "below_ema"
        if stop_threshold is not None and not pd.isna(close) and close <= stop_threshold:
            return idx.date(), float(close), "stop_loss"
    return None, None, None

# -------------------------
# Processing pipeline (produces Buy Date/Price, Sell Date/Price, Recent Price)
# -------------------------
def process_all_tickers(tickers, period, sma_fast, sma_med, ema_slow, lookback_dip, window_52w,
                        entry_prices_map, stop_loss_pct, run_backtest_flag, view_mode="Historical", recent_price_override=None,buy_rsi = None, sell_rsi = None, recent_rsi_override = None, recent_rsi = None):
    results_rows = []
    diagnostics = {"failed_downloads": [], "too_short": [], "exceptions": []}

    batch_results, batch_diag = batch_fetch_tickers_simple(tickers, period=period, interval="1d")
    diagnostics["failed_downloads"].extend(batch_diag.get("failed", []))
    diagnostics["failed_downloads"].extend(batch_diag.get("empty", []))

    for ticker in tickers:
        try:
            df = batch_results.get(ticker, pd.DataFrame())
            if df is None or df.empty:
                results_rows.append({
                    "Symbol": ticker,
                    "Buy Date": None,
                    "buy_rsi" : None,
                    "Buy Price": None,
                    "Recent Price": None,
                    "Recent_Rsi": None,
                    "Buy Signal": False,
                    "C1": False, "C2": False, "C3": False, "C4": False, "C5": False,
                    "Breakout": False,
                    "MatchesFilters": False,
                    "DownloadStatus": "no_data",
                    "Sell Date": None,
                    "sell_rsi": None,
                    "Sell Price": None,
                    "Sell Reason": None
                })
                continue

            if len(df) < 10:
                diagnostics["too_short"].append(ticker)
                results_rows.append({
                    "Symbol": ticker,
                    "Buy Date": None,
                    "buy_rsi" : None,
                    "Buy Price": None,
                    "Recent Price": None,
                    "Recent_Rsi": None,
                    "Buy Signal": False,
                    "C1": False, "C2": False, "C3": False, "C4": False, "C5": False,
                    "Breakout": False,
                    "MatchesFilters": False,
                    "DownloadStatus": "too_short",
                    "Sell Date": None,
                    "sell_rsi": None,
                    "Sell Price": None,
                    "Sell Reason": None
                })
                continue

            df = compute_indicators(df, sma_fast, sma_med, ema_slow, lookback_dip, window_52w)

            # Determine buy date and buy price
            if view_mode == "Historical":
                buy_idx = df.index[df["buy_signal"] == True]
                if len(buy_idx) > 0:
                    buy_ts = buy_idx[-1]
                    buy_row = df.loc[buy_ts]
                    buy_date = buy_ts.date()
                    if buy_date is not None:
                        try:
                            buy_rsi = float(df.loc[pd.to_datetime(buy_date), "RSI"])
                        except Exception:
                            buy_rsi = None
                    # prefer Close on buy date; if NaN use next day's Open if available
                    buy_price = buy_row.get("Close", np.nan)
                    if pd.isna(buy_price):
                        # try next trading day open
                        next_idx = df.index[df.index > buy_ts]
                        if len(next_idx) > 0:
                            try:
                                buy_price = float(df.loc[next_idx[0]]['Open'])
                            except Exception:
                                buy_price = None
                        else:
                            buy_price = None
                else:
                    # no historical buy signal — use latest row as "Buy Date" (for display consistency)
                    buy_ts = df.index[-1]
                    buy_row = df.iloc[-1]
                    buy_date = buy_ts.date()
                    buy_price = float(buy_row.get("Close", np.nan)) if not pd.isna(buy_row.get("Close", np.nan)) else None
            else:
                # Current mode: use latest row as buy date (display)
                buy_ts = df.index[-1]
                buy_row = df.iloc[-1]
                buy_date = buy_ts.date()
                if buy_date is not None:
                        try:
                            buy_rsi = float(df.loc[pd.to_datetime(buy_date), "RSI"])
                        except Exception:
                            buy_rsi = None
                # If user provided recent_price_override, use that as recent price; buy price use Close
                buy_price = float(buy_row.get("Close", np.nan)) if not pd.isna(buy_row.get("Close", np.nan)) else None

            # Recent price (latest Close or override)
            if recent_price_override is not None:
                recent_price = float(recent_price_override)
                recent_rsi = float(recent_rsi_override)
            else:
                latest_row = df.iloc[-1]
                recent_price = float(latest_row.get("Close", np.nan)) if not pd.isna(latest_row.get("Close", np.nan)) else None
            if recent_price is not None:
                    try:
                    recent_rsi = float(latest_row.get("RSI", np.nan)) if not pd.isna(latest_row.get("RSI", np.nan)) else None
                    except Exception:
                        recent_rsi = None

            # Compute boolean conditions from the chosen row (buy_ts)
            chosen_row = buy_row if 'buy_row' in locals() else df.iloc[-1]
            c1 = bool(chosen_row.get("cond1_sma150_gt_ema220", False))
            c2 = bool(chosen_row.get("cond2_close_gt_sma50", False))
            c3 = bool(chosen_row.get("cond3_sma50_gt_sma150", False))
            c4 = bool(chosen_row.get("cond4_close_gt_1.25_low", False))
            c5 = bool(chosen_row.get("cond5_recent_dip", False))
            breakout = bool(chosen_row.get("breakout", False))
            buy_signal = bool(chosen_row.get("buy_signal", False))

            cond_checks = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "Breakout": breakout}
            matches = all(cond_checks.values())

            # Find sell date/price by scanning forward from buy_date if buy_date exists
            sell_date = None
            sell_price = None
            sell_reason = None
            entry_price = entry_prices_map.get(ticker)
            if buy_date is not None:
                sd, sp, reason = find_sell_after_buy(df, buy_date, entry_price, stop_loss_pct)
                if sd is not None:
                    sell_date = sd
                    if sell_date is not None:
                        try:
                            sell_rsi = float(df.loc[pd.to_datetime(sell_date), "RSI"])
                        except Exception:
                            sell_rsi = None
                    sell_price = sp
                    sell_reason = reason

            results_rows.append({
                "Symbol": ticker,
                "Buy Date": buy_date,
                "Buy Price": buy_price,
                "Recent Price": recent_price,
                "Recent_Rsi": recent_rsi,
                "Buy RSI": buy_rsi,
                "Buy Signal": buy_signal,
                "C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5,
                "Breakout": breakout,
                "MatchesFilters": matches,
                "DownloadStatus": "ok",
                "Sell Date": sell_date,
                "Sell Price": sell_price,
                "Sell RSI": sell_rsi,
                "Sell Reason": sell_reason
            })

        except Exception as e:
            diagnostics["exceptions"].append((ticker, str(e)))
            results_rows.append({
                "Symbol": ticker,
                "Buy Date": None,
                "Buy Price": None,
                "Recent Price": None,
                "Recent_Rsi": None,
                "Buy Signal": False,
                "C1": False, "C2": False, "C3": False, "C4": False, "C5": False,
                "Breakout": False,
                "MatchesFilters": False,
                "DownloadStatus": "error",
                "Sell Date": None,
                "Sell Price": None,
                "Sell RSI": None,
                "Sell Reason": None
            })
            continue

    return {
        "results_rows": results_rows,
        "diagnostics": diagnostics,
        "last_refresh": datetime.now().isoformat()
    }

# -------------------------
# Helper: parse entry prices text
# -------------------------
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

# -------------------------
# Streamlit UI
# -------------------------
st.title("NIFTY100 Buy / Sell Dashboard")

tickers = get_tickers()
period = "2y"
persisted_state = load_persisted_state()

# Initialize session state
if "results_rows" not in st.session_state:
    st.session_state.results_rows = persisted_state.get("results_rows", [])
    st.session_state.diagnostics = persisted_state.get("diagnostics", {"failed_downloads": [], "too_short": [], "exceptions": []})
    st.session_state.last_refresh = persisted_state.get("last_refresh")

# Sidebar inputs
with st.sidebar:
    st.header("Inputs")
    sma_fast = st.number_input("SMA Fast", min_value=5, max_value=200, value=DEFAULT_SMA_FAST, step=1)
    sma_med = st.number_input("SMA Medium", min_value=20, max_value=400, value=DEFAULT_SMA_MED, step=1)
    ema_slow = st.number_input("EMA Slow (for sell trigger)", min_value=50, max_value=500, value=DEFAULT_EMA_SLOW, step=1)
    window_52w = st.number_input("52W Window (days)", min_value=100, max_value=365, value=DEFAULT_52W_WINDOW, step=1)
    lookback_dip = st.number_input("Lookback dip (days)", min_value=10, max_value=365, value=DEFAULT_LOOKBACK_DIP, step=1)

    view_mode = st.radio("Select view mode", ["Historical","Current"], index=0)
    entry_prices_text = st.text_area("Entry prices (optional)", value="", height=80)
    entry_prices_map = parse_entry_prices(entry_prices_text)

    st.markdown("**Filters (enable/disable)**")
    use_c1 = st.checkbox("Require C1: 150 SMA > 220 EMA", value=False)
    use_c2 = st.checkbox("Require C2: Close > 50 SMA", value=False)
    use_c3 = st.checkbox("Require C3: 50 SMA > 150 SMA", value=False)
    use_c4 = st.checkbox("Require C4: Close > 1.25 * 52w Low", value=False)
    use_c5 = st.checkbox("Require C5: Recent dip below EMA occurred", value=False)
    use_breakout = st.checkbox("Require Breakout (Close >= prior 52w high)", value=False)
    show_only_matches = st.checkbox("Show only matches", value=False)
    show_recent = st.checkbox("Show recent (30 days)", value=False)

    st.markdown("**Risk**")
    stop_loss_pct = st.slider("Stop loss % (for Sell Date calc)", min_value=1, max_value=50, value=int(DEFAULT_STOP_LOSS_PCT*100)) / 100.0

    st.markdown("**Controls**")
    run_btn_manual = st.button("🔄 Run Analysis")
    debug_force = st.checkbox("DEBUG: Force refresh now", value=False)

refresh_minutes = st.sidebar.slider("Auto refresh interval (minutes)", 1, 30, 5)
from streamlit_autorefresh import st_autorefresh

auto_refresh_counter = st_autorefresh(interval=refresh_minutes * 60 * 1000, key="auto_refresh")

# Auto refresh every 5 minutes (300000 ms)
# auto_refresh_counter = st.experimental_autorefresh(interval=300000, key="auto_refresh")

# Run processing when requested OR auto refresh triggered
if debug_force or run_btn_manual or auto_refresh_counter > 0:
    res = process_all_tickers(
        tickers=tickers,
        period=period,
        sma_fast=sma_fast,
        sma_med=sma_med,
        ema_slow=ema_slow,
        lookback_dip=lookback_dip,
        window_52w=window_52w,
        entry_prices_map=entry_prices_map,
        stop_loss_pct=stop_loss_pct,
        run_backtest_flag=False,
        view_mode=view_mode,
        recent_price_override=None
    )
    persisted = load_persisted_state()
    persisted.update({
        "results_rows": res.get("results_rows", []),
        "diagnostics": res.get("diagnostics", {}),
        "last_refresh": res.get("last_refresh", datetime.now().isoformat())
    })
    save_persisted_state(persisted)
    st.session_state.results_rows = persisted["results_rows"]
    st.session_state.diagnostics = persisted.get("diagnostics", {})
    st.session_state.last_refresh = persisted.get("last_refresh")

# Load display DataFrame from session or persisted
display_rows = st.session_state.get("results_rows", [])
display_df = pd.DataFrame(display_rows)

# Apply UI filters (only if display_df not empty)
if not display_df.empty:
    # show_recent filter (applies to Buy Date)
    if show_recent and "Buy Date" in display_df.columns:
        try:
            date_col = pd.to_datetime(display_df["Buy Date"], errors="coerce").dt.date
            cutoff = datetime.now().date() - timedelta(days=30)
            display_df = display_df[date_col.notna() & (date_col >= cutoff)]
        except Exception:
            pass

    # apply rule filters if requested
    if show_only_matches:
        mask = pd.Series(True, index=display_df.index)
        if use_c1 and "C1" in display_df.columns:
            mask &= display_df["C1"].astype(bool)
        if use_c2 and "C2" in display_df.columns:
            mask &= display_df["C2"].astype(bool)
        if use_c3 and "C3" in display_df.columns:
            mask &= display_df["C3"].astype(bool)
        if use_c4 and "C4" in display_df.columns:
            mask &= display_df["C4"].astype(bool)
        if use_c5 and "C5" in display_df.columns:
            mask &= display_df["C5"].astype(bool)
        if use_breakout and "Breakout" in display_df.columns:
            mask &= display_df["Breakout"].astype(bool)
        display_df = display_df[mask]

# --- Render results with ticks and color indicators (compatible) ---
if display_df is None:
    display_df = pd.DataFrame([])

if display_df.empty:
    st.warning("No tickers matched the selected filters or no data available.")
else:
    display_copy = display_df.copy()

    # Remove any 'Bought' column if present (user requested no Bought)
    if "Bought" in display_copy.columns:
        display_copy = display_copy.drop(columns=["Bought"])

    # Ensure boolean-like columns exist and are booleans
    bool_cols = ["Buy Signal", "C1", "C2", "C3", "C4", "C5", "Breakout", "MatchesFilters"]
    for c in bool_cols:
        if c in display_copy.columns:
            display_copy[c] = display_copy[c].astype(bool)

    # Convert boolean columns to tick or empty string for display
    rule_cols = [c for c in bool_cols if c in display_copy.columns]
    for c in rule_cols:
        display_copy[c] = display_copy[c].apply(lambda v: "✓" if bool(v) else "")

    # Ensure numeric formatting
    for col in ["Buy Price", "Recent Price", "Sell Price", "Close"]:
        if col in display_copy.columns:
            display_copy[col] = pd.to_numeric(display_copy[col], errors="coerce")

    # Format Buy Date and Sell Date for display
    if "Buy Date" in display_copy.columns:
        try:
            display_copy["Buy Date"] = pd.to_datetime(display_copy["Buy Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        except Exception:
            pass
    if "Sell Date" in display_copy.columns:
        try:
            display_copy["Sell Date"] = pd.to_datetime(display_copy["Sell Date"], errors="coerce").dt.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Reorder columns for nicer layout
    cols_order = [
        "Symbol", "Buy Date", "Buy Price", "Buy RSI",
        "Recent Price", "Sell Date", "Sell Price", "Sell RSI", "Sell Reason",
        "MatchesFilters", "Buy Signal", "C1", "C2", "C3", "C4", "C5", "Breakout", "DownloadStatus"
    ]

    cols_present = [c for c in cols_order if c in display_copy.columns]
    display_copy = display_copy[cols_present]

    # Create Styler
    styler = display_copy.style

    # Elementwise highlight function (returns DataFrame of CSS strings)
    def _highlight_ticks(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for col in rule_cols:
            if col in df.columns:
                styles.loc[df[col] == "✓", col] = "background-color: #e6f9e6"
        return styles

    if rule_cols:
        styler = styler.apply(_highlight_ticks, axis=None)

    # Color the Symbol cell green and bold when Breakout is ticked
    def _symbol_color(row):
        styles = [""] * len(row)
        try:
            if "Breakout" in row.index and row["Breakout"] == "✓":
                if "Symbol" in row.index:
                    idx = list(row.index).index("Symbol")
                    styles[idx] = "color: #0b6623; font-weight: 600"
        except Exception:
            pass
        return styles

    styler = styler.apply(_symbol_color, axis=1)

    # Optionally highlight entire row lightly when MatchesFilters is ticked
    if "MatchesFilters" in display_copy.columns:
        def _row_highlight(row):
            if row.get("MatchesFilters") == "✓":
                return ["background-color: #f0fff0"] * len(row)
            return [""] * len(row)
        styler = styler.apply(_row_highlight, axis=1)

    # Numeric formatting for price columns
    fmt = {}
    for c in ["Buy Price", "Recent Price", "Sell Price", "Buy RSI", "Sell RSI"]:
        if c in display_copy.columns:
            display_copy[c] = pd.to_numeric(display_copy[c], errors="coerce")
            fmt[c] = "{:,.2f}"
    if fmt:
        styler = styler.format(fmt)

    styler = styler.set_table_styles([
        {"selector": "th", "props": [("text-align", "left")]},
        {"selector": "td", "props": [("text-align", "left"), ("padding", "6px 8px")]}
    ])

    st.subheader(f"Results — {view_mode} mode (showing {len(display_copy)} rows)")
    st.dataframe(styler)

# Diagnostics
st.sidebar.markdown("### Diagnostics")
ps = load_persisted_state()
st.sidebar.write("Last refresh:", ps.get("last_refresh"))
diag = st.session_state.get("diagnostics", {})
st.sidebar.write("Diagnostics:", diag)
st.sidebar.write("yfinance version:", yf.__version__)

# --- RSI Chart Section ---
if not display_df.empty and "Symbol" in display_df.columns:
    selected_symbol = st.selectbox(
    "Select ticker to view RSI chart",
    display_df["Symbol"].unique(),
    key="rsi_symbol_select"
)

    if selected_symbol:
        # Recompute indicators for the selected ticker to ensure RSI is present
        df_selected = fetch_ticker(selected_symbol, period="2y", interval="1d")
        df_selected = compute_indicators(df_selected)

        if "RSI" in df_selected.columns:
            st.subheader(f"RSI Trend for {selected_symbol}")
            st.line_chart(df_selected[["RSI"]])

import matplotlib.pyplot as plt

# --- RSI Chart Section ---
if not display_df.empty and "Symbol" in display_df.columns:
    selected_symbol = st.selectbox("Select ticker to view RSI chart", display_df["Symbol"].unique())

    if selected_symbol:
        # Fetch and compute indicators for the selected ticker
        df_selected = fetch_ticker(selected_symbol, period="2y", interval="1d")
        df_selected = compute_indicators(df_selected)

        import pandas as pd
        import matplotlib.pyplot as plt
        
        if "RSI" in df_selected.columns:
            fig, ax1 = plt.subplots(figsize=(10,5))
        
            # Price on left axis
            ax1.plot(df_selected.index, df_selected["Close"], color="blue", label="Close Price")
            ax1.set_ylabel("Price", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")
        
            # RSI on right axis
            ax2 = ax1.twinx()
            ax2.plot(df_selected.index, df_selected["RSI"], color="red", label="RSI")
            ax2.axhline(70, color="gray", linestyle="--")
            ax2.axhline(30, color="gray", linestyle="--")
            ax2.set_ylabel("RSI", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
        
            # --- Add Buy/Sell markers safely ---
            buy_date = display_df.loc[display_df["Symbol"] == selected_symbol, "Buy Date"].values[0]
            sell_date = display_df.loc[display_df["Symbol"] == selected_symbol, "Sell Date"].values[0]
        
            buy_dt = pd.to_datetime(buy_date) if pd.notna(buy_date) else None
            sell_dt = pd.to_datetime(sell_date) if pd.notna(sell_date) else None
        
            if buy_dt is not None and buy_dt in df_selected.index:
                ax2.scatter(buy_dt, df_selected.loc[buy_dt, "RSI"], color="green", marker="^", s=100, label="Buy")
        
            if sell_dt is not None and sell_dt in df_selected.index:
                ax2.scatter(sell_dt, df_selected.loc[sell_dt, "RSI"], color="red", marker="v", s=100, label="Sell")
        
            fig.legend(loc="upper left", bbox_to_anchor=(0.1,0.9))
            fig.tight_layout()
            st.pyplot(fig)

            fig.tight_layout()
            st.pyplot(fig)
