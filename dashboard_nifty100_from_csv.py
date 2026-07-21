# dashboard_nifty100_buy_sell_styled_sidebar_diag.py
import os
import json
from datetime import datetime
from matplotlib import pyplot as plt
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
# Indicators (Smoothed RSI)
# -------------------------
def compute_indicators(df, sma_fast=DEFAULT_SMA_FAST, sma_med=DEFAULT_SMA_MED,
                       ema_slow=DEFAULT_EMA_SLOW, lookback_dip=DEFAULT_LOOKBACK_DIP,
                       window_52w=DEFAULT_52W_WINDOW, rsi_period=14):
    if df is None or df.empty:
        return df
    df = df.copy()

    df['sma_50'] = df['Close'].rolling(sma_fast).mean()
    df['sma_150'] = df['Close'].rolling(sma_med).mean()
    df['ema_220'] = df['Close'].ewm(span=ema_slow, adjust=False).mean()
    df['52w_high'] = df['Close'].rolling(window_52w, min_periods=1).max()
    df['52w_low'] = df['Close'].rolling(window_52w, min_periods=1).min()

    # --- Smoothed RSI calculation (EMA smoothing) ---
    # --- Smoothed RSI calculation (EMA smoothing) ---
    # Work only on rows with valid Close values to avoid carrying RSI into NaN rows
    close = df['Close'].copy()
    valid_mask = close.notna()

    # Prepare a series that has NaN where Close is NaN
    close_valid = close.where(valid_mask)

    # Compute delta only on the valid series
    delta = close_valid.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # EMA smoothing for gains/losses; this will produce NaNs at the start until min_periods satisfied
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))

    # Assign RSI back to df but ensure rows with invalid Close remain NaN
    df['RSI'] = rsi_series
    df.loc[~valid_mask, 'RSI'] = np.nan


    # recent dip logic
    dipped = df['Low'] < df['ema_220']
    df['recent_dip'] = dipped.rolling(lookback_dip, min_periods=1).max().fillna(0)

    # Fill other indicators but do NOT ffill/bfill RSI
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

    df['trend_eligible'] = df[['cond1_sma150_gt_ema220','cond2_close_gt_sma50',
                               'cond3_sma50_gt_sma150','cond4_close_gt_1.25_low',
                               'cond5_recent_dip']].all(axis=1)
    df['breakout'] = (df['Close'] >= df['52w_high'].shift(1)).fillna(False).astype(bool)
    df['buy_signal'] = (df['trend_eligible'] & df['breakout']).astype(bool)

    return df

# -------------------------
# Helper to safely get RSI at or after a date/timestamp
# -------------------------
def get_rsi_at_or_after(df, dt):
    if df is None or df.empty or dt is None:
        return None
    try:
        ts = pd.to_datetime(dt)
    except Exception:
        return None
    if "RSI" not in df.columns:
        return None
    # exact match
    if ts in df.index:
        val = df.at[ts, "RSI"]
        return float(val) if not pd.isna(val) else None
    # find first index >= ts
    idx = df.index[df.index >= ts]
    if len(idx) > 0:
        val = df.at[idx[0], "RSI"]
        return float(val) if not pd.isna(val) else None
    # fallback: last available non-NaN RSI
    try:
        rsi_non_na = df["RSI"].dropna()
        if len(rsi_non_na) > 0:
            return float(rsi_non_na.iloc[-1])
    except Exception:
        return None
    return None

# -------------------------
# Sell finder
# -------------------------
def find_sell_after_buy(df, buy_date, entry_price, stop_loss_pct):
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
            return idx.strftime("%Y-%m-%d"), float(close), "below_ema"
        if stop_threshold is not None and not pd.isna(close) and close <= stop_threshold:
            return idx.strftime("%Y-%m-%d"), float(close), "stop_loss"
    return None, None, None

# -------------------------
# Processing pipeline
# -------------------------
def process_all_tickers(tickers, period, sma_fast, sma_med, ema_slow, lookback_dip, window_52w,
                        entry_prices_map, stop_loss_pct, run_backtest_flag, view_mode="Historical",
                        recent_price_override=None, recent_rsi_override=None, debug_show_rsi=False):
    results_rows = []
    diagnostics = {"failed_downloads": [], "too_short": [], "exceptions": []}

    batch_results, batch_diag = batch_fetch_tickers_simple(tickers, period=period, interval="1d")
    diagnostics["failed_downloads"].extend(batch_diag.get("failed", []))
    diagnostics["failed_downloads"].extend(batch_diag.get("empty", []))

    for ticker in tickers:
        buy_rsi = None
        sell_rsi = None
        r_rsi = None
        prev_rsi = None
        buy_price = None
        sell_price = None
        buy_date_str = None
        sell_date_str = None
        sell_reason = None

        try:
            df = batch_results.get(ticker, pd.DataFrame())
            if df is None or df.empty:
                results_rows.append({
                    "Symbol": ticker,
                    "Buy Date": None,
                    "Buy Price": None,
                    "Recent Price": None,
                    "R_RSI": None,
                    "Prev_RSI": None,
                    "Buy RSI": None,
                    "Buy Signal": False,
                    "C1": False, "C2": False, "C3": False, "C4": False, "C5": False,
                    "Breakout": False,
                    "MatchesFilters": False,
                    "DownloadStatus": "no_data",
                    "Sell Date": None,
                    "Sell Price": None,
                    "Sell RSI": None,
                    "Sell Reason": None
                })
                continue

            if len(df) < 10:
                diagnostics["too_short"].append(ticker)
                results_rows.append({
                    "Symbol": ticker,
                    "Buy Date": None,
                    "Buy Price": None,
                    "Recent Price": None,
                    "R_RSI": None,
                    "Prev_RSI": None,
                    "Buy RSI": None,
                    "Buy Signal": False,
                    "C1": False, "C2": False, "C3": False, "C4": False, "C5": False,
                    "Breakout": False,
                    "MatchesFilters": False,
                    "DownloadStatus": "too_short",
                    "Sell Date": None,
                    "Sell Price": None,
                    "Sell RSI": None,
                    "Sell Reason": None
                })
                continue

            # compute indicators (do not ffill/bfill RSI)
            df = compute_indicators(df, sma_fast, sma_med, ema_slow, lookback_dip, window_52w)

            # Determine buy date and buy price
            if view_mode == "Historical":
                buy_idx = df.index[df["buy_signal"] == True]
                if len(buy_idx) > 0:
                    buy_ts = buy_idx[-1]
                    buy_row = df.loc[buy_ts]
                    buy_date_str = buy_ts.strftime("%Y-%m-%d")
                    buy_rsi = get_rsi_at_or_after(df, buy_ts)
                    buy_price = buy_row.get("Close", np.nan)
                    if pd.isna(buy_price):
                        next_idx = df.index[df.index > buy_ts]
                        if len(next_idx) > 0:
                            try:
                                buy_price = float(df.at[next_idx[0], 'Open'])
                            except Exception:
                                buy_price = None
                        else:
                            buy_price = None
                else:
                    buy_ts = df.index[-1]
                    buy_row = df.iloc[-1]
                    buy_date_str = buy_ts.strftime("%Y-%m-%d")
                    buy_price = float(buy_row.get("Close", np.nan)) if not pd.isna(buy_row.get("Close", np.nan)) else None
                    buy_rsi = get_rsi_at_or_after(df, buy_ts)
            else:
                buy_ts = df.index[-1]
                buy_row = df.iloc[-1]
                buy_date_str = buy_ts.strftime("%Y-%m-%d")
                buy_rsi = get_rsi_at_or_after(df, buy_ts)
                buy_price = float(buy_row.get("Close", np.nan)) if not pd.isna(buy_row.get("Close", np.nan)) else None

            # Recent price (last non-NaN Close)
            if recent_price_override is not None:
                recent_price = float(recent_price_override) if pd.notna(recent_price_override) else None
                r_rsi = float(recent_rsi_override) if recent_rsi_override is not None and pd.notna(recent_rsi_override) else None
            else:
                try:
                    recent_price = float(df["Close"].ffill().bfill().iloc[-1])
                except Exception:
                    recent_price = None

                # r_rsi and prev_rsi: last two non-NaN RSI values (do NOT use ffill/bfill for RSI)
                # r_rsi and prev_rsi: last two RSI values from rows that have a valid Close
                r_rsi = None
                prev_rsi = None
                if "RSI" in df.columns:
                    # consider only rows where Close is valid (not NaN) and RSI is not NaN
                    mask = df['Close'].notna() & df['RSI'].notna()
                    rsi_valid = df.loc[mask, 'RSI']
                    if len(rsi_valid) >= 1:
                        r_rsi = float(rsi_valid.iloc[-1])
                    if len(rsi_valid) >= 2:
                        prev_rsi = float(rsi_valid.iloc[-2])


            # If buy_price is empty, show recent_price in Buy Price
            buy_price_display = buy_price if buy_price is not None else recent_price

            # Compute boolean conditions from the chosen row
            if df is not None and not df.empty:
                chosen_row = buy_row if 'buy_row' in locals() else df.iloc[-1]
            else:
                chosen_row = {}
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
            entry_price = entry_prices_map.get(ticker)
            if buy_date_str is not None:
                sd, sp, reason = find_sell_after_buy(df, buy_date_str, entry_price, stop_loss_pct)
                if sd is not None:
                    sell_date_str = sd
                    sell_reason = reason
                    sell_price = sp
                    sell_rsi = get_rsi_at_or_after(df, pd.to_datetime(sd))
                else:
                    sell_rsi = None

            # Optional debug output (controlled by sidebar checkbox)
            if debug_show_rsi:
                st.write(f"DEBUG df tail for {ticker}")
                st.write(df.tail(8)[['Close','RSI']])
                st.write(f"DEBUG RSI {ticker}: last={r_rsi} prev={prev_rsi}")


            # Convert numeric numpy types to native Python floats for JSON serialization
            def _to_float_safe(x):
                try:
                    return float(x) if x is not None and not (isinstance(x, float) and np.isnan(x)) else None
                except Exception:
                    return None

            results_rows.append({
                "Symbol": ticker,
                "Buy Date": buy_date_str,
                "Buy Price": _to_float_safe(buy_price_display),
                "Recent Price": _to_float_safe(recent_price),
                "R_RSI": _to_float_safe(round(r_rsi, 3) if r_rsi is not None else None),
                "Prev_RSI": _to_float_safe(round(prev_rsi, 3) if prev_rsi is not None else None),
                "Buy RSI": _to_float_safe(buy_rsi),
                "Buy Signal": bool(buy_signal),
                "C1": bool(c1), "C2": bool(c2), "C3": bool(c3), "C4": bool(c4), "C5": bool(c5),
                "Breakout": bool(breakout),
                "MatchesFilters": bool(matches),
                "DownloadStatus": "ok",
                "Sell Date": sell_date_str,
                "Sell Price": _to_float_safe(sell_price),
                "Sell RSI": _to_float_safe(sell_rsi),
                "Sell Reason": sell_reason
            })

        except Exception as e:
            diagnostics["exceptions"].append((ticker, str(e)))
            results_rows.append({
                "Symbol": ticker,
                "Buy Date": None,
                "Buy Price": None,
                "Recent Price": None,
                "R_RSI": None,
                "Prev_RSI": None,
                "Buy RSI": None,
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
# Retry helper for failed tickers
# -------------------------
def retry_failed_downloads(session_rows, tickers_to_retry, sma_fast, sma_med, ema_slow, lookback_dip, window_52w, stop_loss_pct, entry_prices_map, view_mode, debug_show_rsi=False):
    diagnostics = {"retried": [], "still_failed": []}
    idx_map = {r["Symbol"]: i for i, r in enumerate(session_rows)}
    for t in tickers_to_retry:
        try:
            df = fetch_ticker(t, period="2y", interval="1d")
            if df is None or df.empty or len(df) < 2:
                diagnostics["still_failed"].append(t)
                continue
            df = compute_indicators(df, sma_fast, sma_med, ema_slow, lookback_dip, window_52w)

            recent_price = None
            r_rsi = None
            prev_rsi = None
            try:
                recent_price = float(df["Close"].ffill().bfill().iloc[-1])
            except Exception:
                recent_price = None

            # r_rsi and prev_rsi: last two RSI values from rows that have a valid Close
            r_rsi = None
            prev_rsi = None
            if "RSI" in df.columns:
                # consider only rows where Close is valid (not NaN) and RSI is not NaN
                mask = df['Close'].notna() & df['RSI'].notna()
                rsi_valid = df.loc[mask, 'RSI']
                if len(rsi_valid) >= 1:
                    r_rsi = float(rsi_valid.iloc[-1])
                if len(rsi_valid) >= 2:
                    prev_rsi = float(rsi_valid.iloc[-2])

            buy_price = None
            buy_date_str = None
            buy_rsi = None
            if view_mode == "Historical":
                buy_idx = df.index[df["buy_signal"] == True]
                if len(buy_idx) > 0:
                    buy_ts = buy_idx[-1]
                    buy_row = df.loc[buy_ts]
                    buy_date_str = buy_ts.strftime("%Y-%m-%d")
                    buy_rsi = get_rsi_at_or_after(df, buy_ts)
                    buy_price = buy_row.get("Close", np.nan)
                    if pd.isna(buy_price):
                        next_idx = df.index[df.index > buy_ts]
                        if len(next_idx) > 0:
                            try:
                                buy_price = float(df.at[next_idx[0], 'Open'])
                            except Exception:
                                buy_price = None
                        else:
                            buy_price = None
                else:
                    buy_ts = df.index[-1]
                    buy_row = df.iloc[-1]
                    buy_date_str = buy_ts.strftime("%Y-%m-%d")
                    buy_price = float(buy_row.get("Close", np.nan)) if not pd.isna(buy_row.get("Close", np.nan)) else None
                    buy_rsi = get_rsi_at_or_after(df, buy_ts)
            else:
                buy_ts = df.index[-1]
                buy_row = df.iloc[-1]
                buy_date_str = buy_ts.strftime("%Y-%m-%d")
                buy_rsi = get_rsi_at_or_after(df, buy_ts)
                buy_price = float(buy_row.get("Close", np.nan)) if not pd.isna(buy_row.get("Close", np.nan)) else None

            sell_date_str = None
            sell_price = None
            sell_rsi = None
            entry_price = entry_prices_map.get(t)
            if buy_date_str is not None:
                sd, sp, reason = find_sell_after_buy(df, buy_date_str, entry_price, stop_loss_pct)
                if sd is not None:
                    sell_date_str = sd
                    sell_price = sp
                    sell_rsi = get_rsi_at_or_after(df, pd.to_datetime(sd))

            if debug_show_rsi:
                st.write(f"RETRY DEBUG RSI {t}: last={r_rsi} prev={prev_rsi}")

            if t in idx_map:
                i = idx_map[t]
                session_rows[i].update({
                    "Buy Date": buy_date_str,
                    "Buy Price": float(buy_price) if buy_price is not None else (float(recent_price) if recent_price is not None else None),
                    "Buy RSI": float(buy_rsi) if buy_rsi is not None else None,
                    "Recent Price": float(recent_price) if recent_price is not None else None,
                    "R_RSI": float(round(r_rsi, 3)) if r_rsi is not None else None,
                    "Prev_RSI": float(round(prev_rsi, 3)) if prev_rsi is not None else None,
                    "DownloadStatus": "ok",
                    "Sell Date": sell_date_str,
                    "Sell Price": float(sell_price) if sell_price is not None else None,
                    "Sell RSI": float(sell_rsi) if sell_rsi is not None else None,
                    "Sell Reason": reason if sd is not None else None
                })
            else:
                session_rows.append({
                    "Symbol": t,
                    "Buy Date": buy_date_str,
                    "Buy Price": float(buy_price) if buy_price is not None else (float(recent_price) if recent_price is not None else None),
                    "Buy RSI": float(buy_rsi) if buy_rsi is not None else None,
                    "Buy Signal": False,
                    "Recent Price": float(recent_price) if recent_price is not None else None,
                    "R_RSI": float(round(r_rsi, 3)) if r_rsi is not None else None,
                    "Prev_RSI": float(round(prev_rsi, 3)) if prev_rsi is not None else None,
                    "Sell Date": sell_date_str,
                    "Sell Price": float(sell_price) if sell_price is not None else None,
                    "Sell RSI": float(sell_rsi) if sell_rsi is not None else None,
                    "Sell Reason": reason if sd is not None else None,
                    "C1": False, "C2": False, "C3": False, "C4": False, "C5": False,
                    "Breakout": False,
                    "MatchesFilters": False,
                    "DownloadStatus": "ok",
                })
            diagnostics["retried"].append(t)
        except Exception:
            diagnostics["still_failed"].append(t)
    return session_rows, diagnostics

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

# Sidebar inputs and diagnostics
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

    # Single debug checkbox for RSI display
    debug_show_rsi = st.checkbox("DEBUG: show RSI values", value=False)

    # Diagnostics moved to sidebar
    with st.expander("Diagnostics / Controls", expanded=False):
        st.write("Last refresh:", st.session_state.get("last_refresh"))
        st.write("Diagnostics:", st.session_state.get("diagnostics"))
        if st.button("Retry failed downloads (sidebar)"):
            failed = st.session_state.get("diagnostics", {}).get("failed_downloads", [])
            if failed:
                rows, diag = retry_failed_downloads(st.session_state.get("results_rows", []), failed, sma_fast, sma_med, ema_slow, lookback_dip, window_52w, stop_loss_pct, entry_prices_map, view_mode, debug_show_rsi=debug_show_rsi)
                st.session_state.results_rows = rows
                st.session_state.diagnostics = diag
                st.experimental_rerun()

# Auto-refresh controls (replace streamlit_autorefresh)
refresh_minutes = st.sidebar.slider("Auto refresh interval (minutes)", 1, 30, 5)
manual_refresh = st.sidebar.button("🔄 Refresh now")

if "last_auto_refresh_ts" not in st.session_state:
    st.session_state.last_auto_refresh_ts = datetime.now().timestamp()

now_ts = datetime.now().timestamp()
elapsed_seconds = now_ts - st.session_state.last_auto_refresh_ts
auto_interval_seconds = int(refresh_minutes) * 60

auto_refresh_triggered = False
if elapsed_seconds >= auto_interval_seconds:
    auto_refresh_triggered = True
    st.session_state.last_auto_refresh_ts = now_ts

should_run = debug_force or run_btn_manual or manual_refresh or auto_refresh_triggered

# Run processing when requested
if should_run:
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
        recent_price_override=None,
        recent_rsi_override=None,
        debug_show_rsi=debug_show_rsi
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

# -------------------------
# Render table with styling
# -------------------------
display_rows = st.session_state.get("results_rows", [])
display_df = pd.DataFrame(display_rows)

# Ensure numeric conversions
for col in ['R_RSI','Prev_RSI','Buy Price','Recent Price','Buy RSI','Sell Price','Sell RSI']:
    if col in display_df.columns:
        display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

def _style_all(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    # RSI vs Prev_RSI coloring
    if 'R_RSI' in df.columns and 'Prev_RSI' in df.columns:
        css_series = pd.Series("", index=df.index, dtype="object")
        for i in df.index:
            curr = df.at[i, 'R_RSI']
            prev = df.at[i, 'Prev_RSI']
            try:
                if pd.notna(curr) and pd.notna(prev):
                    if curr > prev:
                        css_series.at[i] = "background-color: #5cb85c; color: white;"
                    elif curr < prev:
                        css_series.at[i] = "background-color: #d9534f; color: white;"
            except Exception:
                css_series.at[i] = ""
        styles['R_RSI'] = css_series

    # Symbol coloring based on Breakout column
    if 'Symbol' in df.columns and 'Breakout' in df.columns:
        symbol_css = pd.Series("", index=df.index, dtype="object")
        mask = df['Breakout'].astype(bool)
        symbol_css.loc[mask] = "background-color: #28a745; color: white; font-weight: 600;"
        styles['Symbol'] = symbol_css

    return styles

styled = display_df.style.apply(_style_all, axis=None).format({
    'Buy Price': '{:,.2f}',
    'Recent Price': '{:,.2f}',
    'R_RSI': '{:.1f}',
    'Prev_RSI': '{:.1f}'
}, na_rep="")

st.dataframe(styled, width="stretch")
st.write("Styling applied at:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# Optional legend below the table
st.markdown(
    "<div style='display:flex;gap:12px;align-items:center'>"
    "<div style='background:#5cb85c;color:white;padding:4px 8px;border-radius:4px'>RSI <= 30</div>"
    "<div style='background:#d9534f;color:white;padding:4px 8px;border-radius:4px'>RSI >= 70</div>"
    "<div style='background:#28a745;color:white;padding:4px 8px;border-radius:4px'>Symbol: Breakout</div>"
    "</div>",
    unsafe_allow_html=True
)

styled = display_df.style.apply(_style_all, axis=None)
st.dataframe(styled, width="stretch")


# -------------------------
# RSI-only chart (Streamlit line chart)
# -------------------------
if not display_df.empty and "Symbol" in display_df.columns:
    selected_symbol_rsi = st.selectbox(
        "Select ticker for RSI-only chart",
        display_df["Symbol"].unique(),
        key="rsi_only_select"
    )

    if selected_symbol_rsi:
        df_selected = fetch_ticker(selected_symbol_rsi, period="2y", interval="1d")
        df_selected = compute_indicators(df_selected)

        if df_selected is None or df_selected.empty:
            st.warning(f"No price data available for {selected_symbol_rsi}")
        elif "RSI" not in df_selected.columns:
            st.warning(f"RSI not computed for {selected_symbol_rsi}")
        else:
            st.subheader(f"RSI (line) for {selected_symbol_rsi}")
            st.line_chart(df_selected["RSI"], width="stretch")

# -------------------------
# Price + RSI chart (matplotlib)
# -------------------------
if not display_df.empty and "Symbol" in display_df.columns:
    selected_symbol_price = st.selectbox(
        "Select ticker for Price + RSI chart",
        display_df["Symbol"].unique(),
        key="price_rsi_select"
    )

    if selected_symbol_price:
        df_selected = fetch_ticker(selected_symbol_price, period="2y", interval="1d")
        df_selected = compute_indicators(df_selected)

        if df_selected is None or df_selected.empty:
            st.warning(f"No price data available for {selected_symbol_price}")
        elif "RSI" not in df_selected.columns:
            st.warning(f"RSI not computed for {selected_symbol_price}")
        else:
            st.subheader(f"Price and RSI for {selected_symbol_price}")

            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(df_selected.index, df_selected["Close"], color="blue", label="Close Price")
            ax1.set_ylabel("Price", color="blue")
            ax1.tick_params(axis="y", labelcolor="blue")

            ax2 = ax1.twinx()
            ax2.plot(df_selected.index, df_selected["RSI"], color="red", label="RSI")
            ax2.axhline(70, color="gray", linestyle="--")
            ax2.axhline(30, color="gray", linestyle="--")
            ax2.set_ylabel("RSI", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

            # Safe Buy/Sell markers from display_df
            row = display_df.loc[display_df["Symbol"] == selected_symbol_price]
            if not row.empty:
                buy_date = row.iloc[0].get("Buy Date")
                sell_date = row.iloc[0].get("Sell Date")
            else:
                buy_date = None
                sell_date = None

            buy_dt = pd.to_datetime(buy_date) if pd.notna(buy_date) else None
            sell_dt = pd.to_datetime(sell_date) if pd.notna(sell_date) else None

            if buy_dt is not None:
                if buy_dt in df_selected.index:
                    rsi_val = df_selected.loc[buy_dt, "RSI"]
                    ax2.scatter(buy_dt, rsi_val, color="green", marker="^", s=120, label="Buy")
                else:
                    idx = df_selected.index.get_indexer([buy_dt], method="nearest")
                    if len(idx) > 0 and idx[0] >= 0:
                        nearest = df_selected.index[idx[0]]
                        rsi_val = df_selected.loc[nearest, "RSI"]
                        ax2.scatter(nearest, rsi_val, color="green", marker="^", s=120, label="Buy")

            if sell_dt is not None:
                if sell_dt in df_selected.index:
                    rsi_val = df_selected.loc[sell_dt, "RSI"]
                    ax2.scatter(sell_dt, rsi_val, color="red", marker="v", s=120, label="Sell")
                else:
                    idx = df_selected.index.get_indexer([sell_dt], method="nearest")
                    if len(idx) > 0 and idx[0] >= 0:
                        nearest = df_selected.index[idx[0]]
                        rsi_val = df_selected.loc[nearest, "RSI"]
                        ax2.scatter(nearest, rsi_val, color="red", marker="v", s=120, label="Sell")

            ax1.legend(loc="upper left")
            ax2.legend(loc="upper right")
            st.pyplot(fig)
