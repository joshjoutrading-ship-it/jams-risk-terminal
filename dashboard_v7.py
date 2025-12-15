import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import requests

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="JAMS Capital | Risk Management Terminal",
    page_icon="■",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# Bloomberg-like CSS
# =========================
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');

.main, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"],
.block-container {
    background-color: #000000 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    padding: 0.5rem !important;
}

h1, h2, h3 {
    color: #FF9500 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    margin: 15px 0 10px 0 !important;
}

h1 {
    text-align: center;
    font-size: 2.2rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px;
}

p, div, span, label, td, th,
.stMarkdown, .stMarkdown p, .stMarkdown div, .stMarkdown span {
    color: #FFFFFF !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 400 !important;
}

.dataframe {
    background-color: #000000 !important;
    border: 1px solid #333333 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
}

.dataframe th {
    background-color: #1a1a1a !important;
    color: #FF9500 !important;
    font-weight: 600 !important;
    text-align: center !important;
    padding: 6px !important;
    border: 1px solid #333333 !important;
}

.dataframe td {
    color: #FFFFFF !important;
    background-color: #000000 !important;
    text-align: center !important;
    padding: 6px !important;
    border: 1px solid #333333 !important;
    font-weight: 400 !important;
}

.stButton button {
    background-color: #FF9500 !important;
    color: #000000 !important;
    font-weight: 600 !important;
    border: none !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
}

.risk-score-large {
    font-size: 6rem !important;
    font-weight: 700 !important;
    text-align: center;
    margin: 20px 0 !important;
    color: #FFFFFF !important;
}

.terminal-line {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #FFFFFF !important;
    font-size: 0.95rem !important;
    line-height: 1.45 !important;
    margin: 3px 0 !important;
}

.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid #FF9500;
    background: #111;
    font-size: 0.85rem;
}

.hr {
    height: 1px;
    background: #333;
    margin: 10px 0 18px 0;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {visibility: hidden;}

/* Selectbox / dropdown: black text on white background */
div[data-baseweb="select"] * { color: #000000 !important; }
div[data-baseweb="select"] > div { background-color: #FFFFFF !important; border: 2px solid #FF9500 !important; }
div[data-baseweb="select"] svg { fill: #000000 !important; }

/* Date input text */
div[data-testid="stDateInput"] input { color: #000000 !important; background-color: #FFFFFF !important; border: 2px solid #FF9500 !important; }

/* Number input text */
div[data-testid="stNumberInput"] input { color: #000000 !important; }

</style>
""", unsafe_allow_html=True)

# =========================
# Utilities
# =========================
def _to_date(x) -> date:
    if isinstance(x, date):
        return x
    return pd.to_datetime(x).date()

def zscore(s: pd.Series, window: int = 252) -> pd.Series:
    s = s.astype(float)
    mu = s.rolling(window, min_periods=max(20, window//5)).mean()
    sd = s.rolling(window, min_periods=max(20, window//5)).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)

def regime_from_z(z: float) -> str:
    if pd.isna(z):
        return "INSUFFICIENT"
    if z >= 1.0:
        return "ELEVATED"
    if z <= -1.0:
        return "DEPRESSED"
    return "NORMAL"

def signal_from_z(z: float, direction: str) -> float:
    if pd.isna(z):
        return np.nan
    zc = float(np.clip(z, -3.0, 3.0))
    if direction == "higher_worse":
        return (zc + 3.0) / 6.0 * 100.0
    if direction == "lower_worse":
        return (3.0 - zc) / 6.0 * 100.0
    return (abs(zc) / 3.0) * 100.0

def plot_timeseries(df: pd.DataFrame, title: str, y_cols, y2_cols=None, y_title="Value", y2_title="", height=420) -> go.Figure:
    y2_cols = y2_cols or []
    fig = go.Figure()
    for c in y_cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c))
    for c in y2_cols:
        fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=c, yaxis="y2"))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(color="#FF9500", family="IBM Plex Mono", size=14)),
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#FFFFFF", family="IBM Plex Mono", size=10),
        height=height,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(
            x=0.01, y=0.99,
            bgcolor="rgba(0,0,0,0.6)",
            bordercolor="#FF9500",
            borderwidth=1,
            font=dict(color="#FFFFFF", family="IBM Plex Mono", size=10)
        ),
        xaxis=dict(
            showgrid=True, gridwidth=1, gridcolor="#333333",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                bgcolor="#FF9500",
                activecolor="#FFFFFF",
                font=dict(color="#000000", family="IBM Plex Mono", size=10),
                buttons=list([
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(count=3, label="3Y", step="year", stepmode="backward"),
                    dict(step="all", label="ALL")
                ])
            )
        ),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="#333333", title=y_title),
    )
    if y2_cols:
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False, title=y2_title))
    return fig

# =========================
# Data Providers
# =========================
@st.cache_data(ttl=3600)
def fetch_fred_series(series_id: str, start: date, end: date) -> pd.Series:
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}&cosd={start_s}&coed={end_s}"
    df = pd.read_csv(url)
    df.columns = [c.strip() for c in df.columns]
    date_col = "DATE" if "DATE" in df.columns else df.columns[0]
    val_col = series_id if series_id in df.columns else df.columns[1]
    df[date_col] = pd.to_datetime(df[date_col])
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    return df.set_index(date_col)[val_col].sort_index()

@st.cache_data(ttl=900)
def fetch_yf_prices(tickers, start: date, end: date) -> pd.DataFrame:
    df = yf.download(
        tickers,
        start=start.strftime("%Y-%m-%d"),
        end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=True
    )
    if isinstance(df.columns, pd.MultiIndex):
        px = df["Close"].copy()
    else:
        px = df[["Close"]].rename(columns={"Close": tickers[0]})
    px = px.dropna(how="all").ffill()
    px.index = pd.to_datetime(px.index)
    return px

@st.cache_data(ttl=3600)
def fetch_cot_sp500_legacy_net_spec(start: date, end: date, market_code: str = "13874+") -> pd.Series:
    """
    Robust COT fetch:
      - Try PRE API
      - Fall back to CFTC historical compressed ZIP (deacotYYYY.zip)
    Returns weekly series indexed by report date.
    """
    # 1) Try PRE API
    try:
        base = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        where = (
            f"cftc_contract_market_code='{market_code}' "
            f"AND report_date_as_yyyy_mm_dd >= '{start.strftime('%Y-%m-%d')}' "
            f"AND report_date_as_yyyy_mm_dd <= '{end.strftime('%Y-%m-%d')}'"
        )
        params = {"$where": where, "$order": "report_date_as_yyyy_mm_dd asc", "$limit": 5000}
        r = requests.get(base, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json()
        if rows:
            df = pd.DataFrame(rows)
            df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
            long_col = "noncomm_positions_long_all" if "noncomm_positions_long_all" in df.columns else "noncomm_positions_long"
            short_col = "noncomm_positions_short_all" if "noncomm_positions_short_all" in df.columns else "noncomm_positions_short"
            df[long_col] = pd.to_numeric(df[long_col], errors="coerce")
            df[short_col] = pd.to_numeric(df[short_col], errors="coerce")
            s = (df[long_col] - df[short_col]).rename("COT_NET_NONCOMM")
            return pd.Series(s.values, index=df["report_date_as_yyyy_mm_dd"]).dropna().sort_index()
    except Exception:
        pass

    # 2) Fallback ZIP
    import io, zipfile

    def _load_year(y: int) -> pd.DataFrame:
        url = f"https://www.cftc.gov/files/dea/history/deacot{y}.zip"
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        members = [n for n in z.namelist() if n.lower().endswith((".txt", ".csv"))] or z.namelist()
        name = members[0]
        raw = z.open(name).read().decode("utf-8", errors="replace")
        dfy = pd.read_csv(io.StringIO(raw))
        dfy.columns = [c.strip() for c in dfy.columns]
        return dfy

    frames = []
    for y in range(start.year, end.year + 1):
        try:
            frames.append(_load_year(y))
        except Exception:
            continue
    if not frames:
        return pd.Series(dtype=float, name="COT_NET_NONCOMM")

    df = pd.concat(frames, ignore_index=True)
    cols = {c.lower(): c for c in df.columns}

    # date col
    date_col = None
    for k in ["report_date_as_yyyy-mm-dd", "report_date_as_yyyy_mm_dd"]:
        if k in cols:
            date_col = cols[k]
            break
    if date_col is None:
        for k, v in cols.items():
            if "report_date" in k:
                date_col = v
                break
    if date_col is None:
        return pd.Series(dtype=float, name="COT_NET_NONCOMM")

    mkt_col = None
    for k, v in cols.items():
        if "cftc_contract_market_code" in k:
            mkt_col = v
            break
    if mkt_col is None:
        return pd.Series(dtype=float, name="COT_NET_NONCOMM")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df[(df[date_col].dt.date >= start) & (df[date_col].dt.date <= end)]
    df = df[df[mkt_col].astype(str).str.strip().isin([market_code, market_code.replace("+", "")])]

    long_col = None
    short_col = None
    for k, v in cols.items():
        if long_col is None and "noncommercial" in k and "long" in k:
            long_col = v
        if short_col is None and "noncommercial" in k and "short" in k:
            short_col = v
    if long_col is None or short_col is None:
        return pd.Series(dtype=float, name="COT_NET_NONCOMM")

    df[long_col] = pd.to_numeric(df[long_col], errors="coerce")
    df[short_col] = pd.to_numeric(df[short_col], errors="coerce")
    net = (df[long_col] - df[short_col]).rename("COT_NET_NONCOMM")
    s = pd.Series(net.values, index=df[date_col]).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    return s

# =========================
# Core Modules
# =========================
def build_modules(start: date, end: date, z_window: int) -> dict:
    out = {}

    # HY OAS
    hy = fetch_fred_series("BAMLH0A0HYM2", start, end).ffill()
    df = pd.DataFrame({"HY_OAS": hy})
    df["Z"] = zscore(df["HY_OAS"], z_window)
    df["SIGNAL"] = df["Z"].apply(lambda x: signal_from_z(x, "higher_worse"))
    out["HY_OAS"] = df

    # HYG/SPY
    px = fetch_yf_prices(["HYG", "SPY"], start, end)
    ratio = (px["HYG"] / px["SPY"]).rename("HYG_SPY")
    df = pd.DataFrame({"HYG_SPY": ratio}).ffill()
    df["Z"] = zscore(df["HYG_SPY"], z_window)
    df["SIGNAL"] = df["Z"].apply(lambda x: signal_from_z(x, "lower_worse"))
    out["HYG_SPY"] = df

    # SOFR - 3M TBill
    sofr = fetch_fred_series("SOFR", start, end).ffill()
    tb3 = fetch_fred_series("DTB3", start, end).ffill()
    spread = (sofr - tb3).rename("SOFR_MINUS_TB3M")
    df = pd.DataFrame({"SOFR_MINUS_TB3M": spread}).ffill()
    df["Z"] = zscore(df["SOFR_MINUS_TB3M"], z_window)
    df["SIGNAL"] = df["Z"].apply(lambda x: signal_from_z(x, "higher_worse"))
    out["SOFR_TBILL"] = df

    # 10Y Breakeven
    be = fetch_fred_series("T10YIE", start, end).ffill()
    df = pd.DataFrame({"T10YIE": be})
    df["Z"] = zscore(df["T10YIE"], z_window)
    df["SIGNAL"] = df["Z"].apply(lambda x: signal_from_z(x, "higher_worse"))
    out["BREAKEVEN_10Y"] = df

    # COT (weekly -> daily ffill)
    cot = fetch_cot_sp500_legacy_net_spec(start, end).sort_index()
    cot_daily = cot.reindex(pd.date_range(pd.to_datetime(start), pd.to_datetime(end), freq="D")).ffill()
    df = pd.DataFrame({"COT_NET_NONCOMM": cot_daily})
    # Use ~2 years of weekly equivalent after ffill
    df["Z"] = zscore(df["COT_NET_NONCOMM"], max(60, z_window // 3))
    df["SIGNAL"] = df["Z"].apply(lambda x: signal_from_z(x, "both_worse"))
    out["COT_SP500"] = df

    return out

def composite_snapshot(modules: dict, as_of: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for name, df in modules.items():
        dfx = df[df.index <= as_of]
        if dfx.empty:
            continue
        last = dfx.iloc[-1]
        level_col = [c for c in df.columns if c not in ["Z", "SIGNAL"]][0]
        rows.append({
            "MODULE": name,
            "REGIME": regime_from_z(last["Z"]),
            "Z": np.round(float(last["Z"]), 2) if pd.notna(last["Z"]) else np.nan,
            "SIGNAL(0-100)": np.round(float(last["SIGNAL"]), 1) if pd.notna(last["SIGNAL"]) else np.nan,
            "LEVEL": np.round(float(last[level_col]), 4) if pd.notna(last[level_col]) else np.nan
        })
    if not rows:
        return pd.DataFrame(columns=["MODULE", "REGIME", "Z", "SIGNAL(0-100)", "LEVEL"])
    return pd.DataFrame(rows).sort_values("SIGNAL(0-100)", ascending=False)

# =========================
# App
# =========================
def main():
    st.markdown("# JAMS CAPITAL RISK MANAGEMENT TERMINAL")
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1.2, 1.2, 1.4])
    with c1:
        if st.button("REFRESH DATA"):
            st.cache_data.clear()
            st.rerun()
    with c2:
        auto_refresh = st.checkbox("AUTO REFRESH", value=False)
    with c3:
        lookback = st.selectbox("DEFAULT LOOKBACK", ["6M", "1Y", "3Y", "5Y", "10Y", "MAX"], index=2)
    with c4:
        z_window = st.selectbox("Z-SCORE WINDOW", [63, 126, 252, 504], index=2, help="Trading days. 252 = ~1Y.")
    with c5:
        st.write(f"LAST UPDATE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    today = datetime.utcnow().date()
    if lookback == "6M":
        default_start = today - timedelta(days=183)
    elif lookback == "1Y":
        default_start = today - timedelta(days=365)
    elif lookback == "3Y":
        default_start = today - timedelta(days=365*3)
    elif lookback == "5Y":
        default_start = today - timedelta(days=365*5)
    elif lookback == "10Y":
        default_start = today - timedelta(days=365*10)
    else:
        default_start = date(1990, 1, 1)

    d1, d2, d3 = st.columns([1.2, 1.2, 2.6])
    with d1:
        start_date = st.date_input("START DATE", value=default_start)
    with d2:
        end_date = st.date_input("END DATE", value=today)
    with d3:
        st.markdown(
            "<div class='terminal-line'>CHARTS SUPPORT PAN/ZOOM + RANGE SLIDER + QUICK RANGE BUTTONS (1M/3M/6M/1Y/3Y/ALL).</div>",
            unsafe_allow_html=True
        )

    start_date = _to_date(start_date)
    end_date = _to_date(end_date)
    if end_date <= start_date:
        st.error("END DATE must be after START DATE.")
        st.stop()

    # Build core modules
    modules = build_modules(start_date, end_date, int(z_window))

    st.markdown("## DATA INPUTS")
    st.markdown("<div class='terminal-line'>All modules are sourced programmatically (FRED, Yahoo Finance, CFTC). No uploads required.</div>", unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Snapshot table
    as_of = pd.to_datetime(end_date)
    snap_df = composite_snapshot(modules, as_of)

    left, right = st.columns([1.15, 2.85])
    with left:
        st.markdown("## SIGNAL SNAPSHOT")
        if snap_df.empty:
            st.markdown("<div class='terminal-line'>DATA INSUFFICIENT.</div>", unsafe_allow_html=True)
        else:
            # themed HTML table
            header = "<tr><th>MODULE</th><th>REGIME</th><th>Z</th><th>SIGNAL(0-100)</th></tr>"
            body = ""
            for _, r in snap_df[["MODULE", "REGIME", "Z", "SIGNAL(0-100)"]].iterrows():
                body += f"<tr><td>{r['MODULE']}</td><td>{r['REGIME']}</td><td>{r['Z']}</td><td>{r['SIGNAL(0-100)']}</td></tr>"
            st.markdown(f"<table class='dataframe' style='width:100%; border-collapse:collapse;'>{header}{body}</table>", unsafe_allow_html=True)

    with right:
        st.markdown("## EXECUTIVE SUMMARY (SIGNALIZED)")
        if snap_df.empty:
            msg = "DATA INSUFFICIENT: Unable to compute signals for the selected range."
        else:
            worst = snap_df.iloc[0]["MODULE"]
            comp = float(np.nanmean(snap_df["SIGNAL(0-100)"]))
            comp10 = float(np.clip(round(comp / 10.0, 1), 0.0, 10.0))
            msg = (
                f"RISK STATE: composite signal is {comp10:.1f}/10 (avg across modules). "
                f"Highest-pressure module: {worst}. Validate directionality in the tabs below."
            )
        st.markdown(f"<div class='terminal-line'>{msg}</div>", unsafe_allow_html=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # -------------------------
    # V1 proxy tabs (excluding VIX and forward risk score, per your request)
    # -------------------------
    v1_px = fetch_yf_prices(["HYG", "TLT", "UUP", "FXY", "RSP", "SPY", "XLU", "XLK"], start_date, end_date).ffill()
    v1_hyg_tlt = (v1_px["HYG"] / v1_px["TLT"]).rename("HYG_TLT")
    v1_usd_jpy = (v1_px["UUP"] / v1_px["FXY"]).rename("USD_JPY_PROXY")
    v1_rsp_spy = (v1_px["RSP"] / v1_px["SPY"]).rename("RSP_SPY")
    v1_def = (v1_px["XLU"] / v1_px["XLK"]).rename("XLU_XLK")

    v1_df = pd.concat([v1_hyg_tlt, v1_usd_jpy, v1_rsp_spy, v1_def], axis=1).dropna()

    tabs = st.tabs([
        "CREDIT STRESS HYG/TLT (V1)",
        "CURRENCY STRESS USD/JPY (V1)",
        "MARKET BREADTH RSP/SPY (V1)",
        "DEFENSIVE ROTATION XLU/XLK (V1)",
        "HY OAS (FRED)",
        "HYG/SPY",
        "SOFR - 3M T-Bill",
        "10Y Breakeven",
        "COT S&P 500"
    ])

    with tabs[0]:
        df = pd.DataFrame({"HYG_TLT": v1_df["HYG_TLT"]})
        df["Z"] = zscore(df["HYG_TLT"], int(z_window))
        st.plotly_chart(plot_timeseries(df, "CREDIT STRESS (V1): HYG / TLT | LEVEL + Z-SCORE", ["HYG_TLT"], ["Z"], "Ratio", "Z"), use_container_width=True)

    with tabs[1]:
        df = pd.DataFrame({"USD_JPY_PROXY": v1_df["USD_JPY_PROXY"]})
        df["Z"] = zscore(df["USD_JPY_PROXY"], int(z_window))
        st.plotly_chart(plot_timeseries(df, "CURRENCY STRESS (V1): USD/JPY PROXY (UUP/FXY) | LEVEL + Z-SCORE", ["USD_JPY_PROXY"], ["Z"], "Proxy Ratio", "Z"), use_container_width=True)

    with tabs[2]:
        df = pd.DataFrame({"RSP_SPY": v1_df["RSP_SPY"]})
        df["Z"] = zscore(df["RSP_SPY"], int(z_window))
        st.plotly_chart(plot_timeseries(df, "MARKET BREADTH (V1): RSP / SPY | LEVEL + Z-SCORE", ["RSP_SPY"], ["Z"], "Ratio", "Z"), use_container_width=True)

    with tabs[3]:
        df = pd.DataFrame({"XLU_XLK": v1_df["XLU_XLK"]})
        df["Z"] = zscore(df["XLU_XLK"], int(z_window))
        st.plotly_chart(plot_timeseries(df, "DEFENSIVE ROTATION (V1): XLU / XLK | LEVEL + Z-SCORE", ["XLU_XLK"], ["Z"], "Ratio", "Z"), use_container_width=True)

    with tabs[4]:
        df = modules["HY_OAS"].copy()
        st.plotly_chart(plot_timeseries(df, "HIGH YIELD OAS (BAMLH0A0HYM2) | LEVEL + Z-SCORE", ["HY_OAS"], ["Z"], "Spread (%)", "Z"), use_container_width=True)

    with tabs[5]:
        df = modules["HYG_SPY"].copy()
        st.plotly_chart(plot_timeseries(df, "RISK APPETITE: HYG / SPY RATIO | LEVEL + Z-SCORE", ["HYG_SPY"], ["Z"], "Ratio", "Z"), use_container_width=True)

    with tabs[6]:
        df = modules["SOFR_TBILL"].copy()
        st.plotly_chart(plot_timeseries(df, "FUNDING STRESS: SOFR - 3M T-BILL | LEVEL + Z-SCORE", ["SOFR_MINUS_TB3M"], ["Z"], "Spread (%)", "Z"), use_container_width=True)

    with tabs[7]:
        df = modules["BREAKEVEN_10Y"].copy()
        st.plotly_chart(plot_timeseries(df, "10Y BREAKEVEN INFLATION (T10YIE) | LEVEL + Z-SCORE", ["T10YIE"], ["Z"], "Breakeven (%)", "Z"), use_container_width=True)

        reg = df["Z"].apply(regime_from_z)
        reg_counts = reg.value_counts(dropna=False).rename_axis("REGIME").reset_index(name="DAYS")
        rows = "".join([f"<tr><td>{r['REGIME']}</td><td>{int(r['DAYS'])}</td></tr>" for _, r in reg_counts.iterrows()])
        st.markdown(f"""
        <table class="dataframe" style="width:320px; border-collapse:collapse; margin-top:10px;">
            <tr><th>REGIME</th><th>DAYS</th></tr>
            {rows}
        </table>
        """, unsafe_allow_html=True)

    with tabs[8]:
        df = modules["COT_SP500"].copy()
        st.plotly_chart(plot_timeseries(df, "COT S&P 500 | NET NON-COMMERCIAL POSITIONING", ["COT_NET_NONCOMM"], ["Z"], "Contracts (Net)", "Z"), use_container_width=True)
        st.markdown("<div class='terminal-line'>COT is weekly (report date). The series is forward-filled daily for visualization and scoring.</div>", unsafe_allow_html=True)

    if auto_refresh:
        st.caption("Auto refresh enabled (refreshes every 60 seconds).")
        st.sleep(60)
        st.rerun()

if __name__ == "__main__":
    main()
