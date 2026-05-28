# -*- coding: utf-8 -*-
"""
Delhi Climate AI Forecasting Dashboard
Dark glassmorphism theme, animated UI, interactive Plotly charts.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller, kpss, acf as sm_acf, pacf as sm_pacf
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error, r2_score
import torch
import warnings

warnings.filterwarnings("ignore")


# ─── METRICS HELPERS (match final.py exactly) ─────────────────────────────────

def calc_smape(actual, predicted):
    denom = (np.abs(actual) + np.abs(predicted)) / 2.0 + 1e-8
    return float(np.mean(np.abs(actual - predicted) / denom) * 100)

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Delhi Climate AI Forecast",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── App background: deep space gradient ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    background-attachment: fixed;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10, 8, 30, 0.96) !important;
    border-right: 1px solid rgba(0, 212, 255, 0.15) !important;
}
[data-testid="stSidebar"] * { color: #c8c8e8 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    gap: 2px;
    border: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 8px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.4) !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    padding: 8px 18px !important;
    transition: all 0.25s ease !important;
    border: 1px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,rgba(0,212,255,0.18),rgba(123,47,247,0.18)) !important;
    color: #00d4ff !important;
    border: 1px solid rgba(0,212,255,0.25) !important;
    box-shadow: 0 2px 14px rgba(0,212,255,0.12) !important;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 14px !important;
    padding: 16px 14px !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 28px rgba(0,212,255,0.1) !important;
}
[data-testid="stMetricLabel"] p {
    color: rgba(180,180,220,0.55) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1.3px !important;
}
[data-testid="stMetricValue"] {
    color: #00d4ff !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
}

/* ── Select box ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(0,212,255,0.22) !important;
    border-radius: 8px !important;
}

/* ── Slider track ── */
[data-testid="stSlider"] [role="slider"] {
    background: #00d4ff !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.93rem !important;
    padding: 10px 0 !important;
    width: 100%;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 18px rgba(0,212,255,0.22) !important;
    letter-spacing: 0.4px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(123,47,247,0.4) !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.22); border-radius: 3px; }

/* ── Animations ── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes pulse-border {
    0%, 100% { box-shadow: 0 0 8px rgba(0,212,255,0.25); }
    50%       { box-shadow: 0 0 22px rgba(123,47,247,0.5); }
}

/* ── Reusable utility classes ── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 22px;
    animation: fadeInUp 0.45s ease both;
    box-shadow: 0 4px 24px rgba(0,0,0,0.22);
    margin-bottom: 12px;
}
.gradient-text {
    background: linear-gradient(90deg, #00d4ff 0%, #7b2ff7 50%, #ff6b35 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 3.5s linear infinite;
}
.verdict-badge {
    display: inline-block;
    padding: 3px 13px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 1.6px;
    text-transform: uppercase;
}
.badge-green { background: rgba(0,255,136,0.12); color: #00ff88; border: 1px solid rgba(0,255,136,0.38); }
.badge-red   { background: rgba(255,71,87,0.12);  color: #ff4757; border: 1px solid rgba(255,71,87,0.38); }
.badge-yellow{ background: rgba(255,215,0,0.12);  color: #ffd700; border: 1px solid rgba(255,215,0,0.38); }
.stat-row    { display:flex; gap:8px; flex-wrap:wrap; margin:6px 0; }
.stat-item   { background:rgba(255,255,255,0.04); border-radius:8px; padding:5px 11px; font-size:0.8rem; color:rgba(200,200,230,0.7); }
.stat-value  { color:#00d4ff; font-weight:700; }
.interp-box  { border-radius:8px; padding:13px 18px; margin:14px 0; font-size:0.87rem; }
</style>
""", unsafe_allow_html=True)


# ─── DATA LOADING & PREPROCESSING ────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_and_preprocess():
    df = pd.read_csv("DailyDelhiClimateTrain.csv")
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df.asfreq("D")

    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx].copy()
    test  = df.iloc[split_idx:].copy()

    for split_df in [train, test]:
        mask = (split_df["meanpressure"] < 900) | (split_df["meanpressure"] > 1100)
        split_df.loc[mask, "meanpressure"] = np.nan
        split_df["meanpressure"] = split_df["meanpressure"].interpolate().ffill().bfill()

    for col in ["meantemp", "humidity", "wind_speed"]:
        Q1, Q3 = train[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lo_b, hi_b = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        for split_df in [train, test]:
            split_df.loc[(split_df[col] < lo_b) | (split_df[col] > hi_b), col] = np.nan
            split_df[col] = split_df[col].interpolate().ffill().bfill()

    return df, train, test


@st.cache_data(show_spinner=False)
def run_stationarity(series_tuple):
    s = pd.Series(list(series_tuple)).dropna()
    adf_stat, adf_p, adf_lags, _, adf_crit, _ = adfuller(s, autolag="AIC")
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(s, regression="c", nlags="auto")
    except Exception:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = None, None, None, {}
    return {
        "adf":  {"stat": adf_stat,  "p": adf_p,  "lags": adf_lags,  "crit": adf_crit},
        "kpss": {"stat": kpss_stat, "p": kpss_p, "lags": kpss_lags, "crit": kpss_crit},
    }


@st.cache_data(show_spinner=False)
def get_acf_pacf(series_tuple, nlags):
    s = pd.Series(list(series_tuple)).dropna()
    acf_vals,  acf_ci  = sm_acf(s,  nlags=nlags, alpha=0.05)
    pacf_vals, pacf_ci = sm_pacf(s, nlags=nlags, alpha=0.05, method="ywm")
    return acf_vals, acf_ci, pacf_vals, pacf_ci


@st.cache_data(show_spinner=False)
def get_stl(series_tuple):
    s = pd.Series(list(series_tuple))
    res = STL(s, period=365, robust=True).fit()
    return res.trend.values, res.seasonal.values, res.resid.values


# ─── MODEL ────────────────────────────────────────────────────────────────────

BOLT_MODEL_TYPE = "bolt"
BOLT_MODEL_ID   = "amazon/chronos-bolt-small"
BOLT_CTX_CAP    = 512
BOLT_LABEL      = "Chronos Bolt Small"


@st.cache_resource(show_spinner=False)
def load_chronos(model_id: str, model_type: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16
    if model_type == "standard":
        from chronos import ChronosPipeline
        return ChronosPipeline.from_pretrained(model_id, device_map=device, torch_dtype=dtype)
    from chronos import BaseChronosPipeline
    return BaseChronosPipeline.from_pretrained(model_id, device_map=device, torch_dtype=dtype)


# ─── ROLLING FORECAST WITH CONFIDENCE INTERVALS ───────────────────────────────

def rolling_forecast_ci(model_type, pipeline, ctx_vals, test_values,
                        step=14, ctx_cap=512, num_samples=100):
    medians, lowers, uppers = [], [], []
    ctx = list(ctx_vals)

    for i in range(0, len(test_values), step):
        horizon    = min(step, len(test_values) - i)
        ctx_window = np.array(ctx[-ctx_cap:], dtype=np.float32)
        mu, sigma  = ctx_window.mean(), ctx_window.std() + 1e-8
        ctx_norm   = torch.tensor((ctx_window - mu) / sigma, dtype=torch.float32)

        if model_type == "standard":
            # Returns (batch, num_samples, horizon)
            fc      = pipeline.predict(ctx_norm, prediction_length=horizon, num_samples=num_samples)
            samples = fc.squeeze(0).numpy()          # (num_samples, horizon)
            if samples.ndim == 1:
                samples = samples.reshape(1, -1)
            med = np.quantile(samples, 0.50, axis=0) * sigma + mu
            lo  = np.quantile(samples, 0.10, axis=0) * sigma + mu
            hi  = np.quantile(samples, 0.90, axis=0) * sigma + mu
        else:
            # Bolt: Returns (batch, num_quantile_levels, horizon)
            fc    = pipeline.predict(ctx_norm, prediction_length=horizon)
            fc_np = fc.squeeze(0).numpy()            # (num_quantiles, horizon)
            if fc_np.ndim == 1:
                med = fc_np * sigma + mu
                lo  = med - 2.0
                hi  = med + 2.0
            else:
                nq  = fc_np.shape[0]
                lo  = fc_np[0]       * sigma + mu
                med = fc_np[nq // 2] * sigma + mu
                hi  = fc_np[-1]      * sigma + mu

        medians.extend(med[:horizon])
        lowers.extend(lo[:horizon])
        uppers.extend(hi[:horizon])
        ctx.extend(test_values[i : i + horizon])

    return np.array(medians), np.array(lowers), np.array(uppers)


# ─── SHARED PLOTLY LAYOUT BASE ────────────────────────────────────────────────

def base_layout(**overrides):
    layout = dict(
        template      = "plotly_dark",
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font          = dict(family="Inter, system-ui, sans-serif", color="#c8c8e8", size=12),
        legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.08)",
                             borderwidth=1, font_size=11),
        margin        = dict(l=48, r=20, t=52, b=44),
        xaxis         = dict(gridcolor="rgba(255,255,255,0.04)", showgrid=True,
                             zeroline=False, linecolor="rgba(255,255,255,0.08)"),
        yaxis         = dict(gridcolor="rgba(255,255,255,0.04)", showgrid=True,
                             zeroline=False, linecolor="rgba(255,255,255,0.08)"),
        hoverlabel    = dict(bgcolor="rgba(12,10,34,0.92)", font_color="#e8e8ff",
                             bordercolor="rgba(0,212,255,0.3)", font_size=12),
        hovermode     = "x unified",
    )
    layout.update(overrides)
    return layout


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:14px 0 20px;">
        <div style="font-size:2rem;margin-bottom:4px;">🌡️</div>
        <div class="gradient-text" style="font-size:1.05rem;font-weight:900;letter-spacing:1px;">
            DELHI CLIMATE
        </div>
        <div style="color:rgba(180,180,220,0.42);font-size:0.66rem;letter-spacing:2.5px;
                    text-transform:uppercase;margin-top:2px;">
            AI Forecasting Dashboard
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    df, train, test = load_and_preprocess()

    st.markdown("""
    <div style="color:rgba(180,180,220,0.45);font-size:0.65rem;letter-spacing:1.8px;
                text-transform:uppercase;margin-bottom:8px;">Dataset Info</div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class="glass-card" style="padding:14px 16px;">
        <div class="stat-row">
            <div class="stat-item">Total <span class="stat-value">{len(df):,}</span></div>
            <div class="stat-item">Train <span class="stat-value">{len(train):,}</span></div>
            <div class="stat-item">Test  <span class="stat-value">{len(test):,}</span></div>
        </div>
        <div style="margin-top:8px;color:rgba(180,180,220,0.4);font-size:0.73rem;">
            📅 {df.index[0].strftime('%b %Y')} &rarr; {df.index[-1].strftime('%b %Y')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    run_btn = st.button("⚡  Run Forecast", use_container_width=True)

    if run_btn:
        ctx_vals    = train["meantemp"].values.astype(np.float32)
        test_values = test["meantemp"].values.astype(np.float32)
        actual_arr  = test["meantemp"].values

        with st.spinner(f"🤖 Loading {BOLT_LABEL}…"):
            pipeline = load_chronos(BOLT_MODEL_ID, BOLT_MODEL_TYPE)
        with st.spinner("⚡ Computing rolling forecast…"):
            med_r, lo_r, hi_r = rolling_forecast_ci(
                BOLT_MODEL_TYPE, pipeline, ctx_vals, test_values,
                step=14, ctx_cap=BOLT_CTX_CAP
            )

        mse_r  = float(mean_squared_error(actual_arr, med_r))
        rmse_r = float(np.sqrt(mse_r))

        st.session_state["forecast_results"] = (med_r, lo_r, hi_r)
        st.session_state["forecast_done"]    = True
        st.success(f"✅ {BOLT_LABEL}  |  RMSE: {rmse_r:.2f} °C")

    elif "forecast_done" in st.session_state:
        med_r, _, _ = st.session_state["forecast_results"]
        actual_arr  = test["meantemp"].values
        rmse_r = float(np.sqrt(mean_squared_error(actual_arr, med_r)))
        st.success(f"✅ {BOLT_LABEL}  |  RMSE: {rmse_r:.2f} °C")

    st.markdown("---")
    st.markdown("""
    <div style="color:rgba(180,180,220,0.28);font-size:0.65rem;text-align:center;line-height:1.7;">
        Delhi Daily Climate Dataset<br>2013 – 2017 · 1,462 days<br>
        Powered by Chronos &amp; Streamlit
    </div>
    """, unsafe_allow_html=True)


# ─── PAGE HEADER ──────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding:4px 0 22px;">
    <h1 class="gradient-text"
        style="font-size:2.55rem;font-weight:900;margin:0;line-height:1.1;letter-spacing:-0.5px;">
        Delhi Climate Forecasting
    </h1>
    <p style="color:rgba(180,180,220,0.42);font-size:0.78rem;letter-spacing:2.2px;
              text-transform:uppercase;margin-top:5px;">
        Foundation Model &nbsp;·&nbsp; Time Series Analysis &nbsp;·&nbsp; Interactive Dashboard
    </p>
</div>
""", unsafe_allow_html=True)


# ─── TABS ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌡️  Raw Data",
    "🔬  Stationarity",
    "📊  ACF & PACF",
    "🔮  Forecast",
    "🌀  Decomposition",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ACTUAL VS PREDICTED
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if "forecast_done" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:90px 20px 80px;">
            <div style="font-size:3.2rem;margin-bottom:18px;">📈</div>
            <div style="font-size:1.05rem;color:rgba(180,180,220,0.55);font-weight:600;
                        margin-bottom:8px;">No forecast yet</div>
            <div style="font-size:0.82rem;color:rgba(180,180,220,0.32);">
                Click <b style="color:#00d4ff;">⚡ Run Forecast</b> in the sidebar
                to generate predictions with Chronos Bolt Small
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        med, lo, hi = st.session_state["forecast_results"]
        actual      = test["meantemp"].values

        fig = go.Figure()

        # 80% CI shaded band
        fig.add_trace(go.Scatter(
            x=list(test.index) + list(test.index[::-1]),
            y=list(hi) + list(lo[::-1]),
            fill="toself", fillcolor="rgba(123,47,247,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% CI", hoverinfo="skip",
        ))
        # Full historical context (muted)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["meantemp"],
            name="Historical", mode="lines",
            line=dict(color="rgba(200,200,230,0.18)", width=1.2),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Temp: %{y:.1f} °C<extra></extra>",
        ))
        # Forecast line
        fig.add_trace(go.Scatter(
            x=test.index, y=med,
            name=f"Forecast · {BOLT_LABEL}",
            mode="lines", line=dict(color="#ff6b35", width=2.2, dash="dash"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: %{y:.1f} °C<extra></extra>",
        ))
        # Actual test period
        fig.add_trace(go.Scatter(
            x=test.index, y=actual,
            name="Actual", mode="lines",
            line=dict(color="#00d4ff", width=2.4),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:.1f} °C<extra></extra>",
        ))

        fig.update_layout(
            **base_layout(
                title=dict(
                    text=f"Actual vs Best Model Forecast · Delhi Temperature",
                    font_size=15, x=0.02,
                ),
                yaxis_title="Temperature (°C)",
                height=520,
                xaxis=dict(
                    gridcolor="rgba(255,255,255,0.04)", showgrid=True,
                    zeroline=False, linecolor="rgba(255,255,255,0.08)",
                    rangeselector=dict(
                        buttons=[
                            dict(count=1, label="1Y", step="year", stepmode="backward"),
                            dict(count=2, label="2Y", step="year", stepmode="backward"),
                            dict(step="all", label="All"),
                        ],
                        bgcolor="rgba(255,255,255,0.04)",
                        activecolor="rgba(0,212,255,0.18)",
                        font=dict(color="#c8c8e8", size=11),
                        bordercolor="rgba(255,255,255,0.08)",
                        x=0.0, y=1.04,
                    ),
                    rangeslider=dict(visible=True, bgcolor="rgba(255,255,255,0.02)", thickness=0.055),
                    type="date",
                ),
            )
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — STATIONARITY TESTS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    raw_vals  = df["meantemp"].dropna().values
    stat_res  = run_stationarity(tuple(raw_vals))
    adf_r     = stat_res["adf"]
    kpss_r    = stat_res["kpss"]

    adf_stat  = adf_r["p"] < 0.05
    kpss_stat = (kpss_r["p"] > 0.05) if kpss_r["p"] is not None else None

    # ── Two test cards side-by-side ───────────────────────────────────────────
    col_adf, col_kpss = st.columns(2)

    with col_adf:
        clr   = "#00ff88" if adf_stat  else "#ff4757"
        txt   = "STATIONARY"    if adf_stat  else "NOT STATIONARY"
        bcls  = "badge-green"   if adf_stat  else "badge-red"
        rows  = "".join(
            f'<tr><td style="padding:5px 0;color:rgba(180,180,220,0.48);">Critical {k}</td>'
            f'<td style="text-align:right;color:rgba(200,200,230,0.7);">{v:.4f}</td></tr>'
            for k, v in adf_r["crit"].items()
        )
        st.markdown(f"""
        <div class="glass-card" style="border-top:3px solid {clr};">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;">
                <span style="font-size:1.05rem;font-weight:700;color:#e8e8ff;">ADF Test</span>
                <span class="verdict-badge {bcls}">{txt}</span>
            </div>
            <table style="width:100%;border-collapse:collapse;font-size:0.83rem;">
                <tr><td style="padding:6px 0;color:rgba(180,180,220,0.48);">ADF Statistic</td>
                    <td style="text-align:right;color:#00d4ff;font-weight:700;">{adf_r['stat']:.5f}</td></tr>
                <tr><td style="padding:6px 0;color:rgba(180,180,220,0.48);">p-value</td>
                    <td style="text-align:right;color:#00d4ff;font-weight:700;">{adf_r['p']:.5f}</td></tr>
                <tr><td style="padding:6px 0;color:rgba(180,180,220,0.48);">Lags Used</td>
                    <td style="text-align:right;color:#00d4ff;font-weight:700;">{adf_r['lags']}</td></tr>
                {rows}
            </table>
            <div style="margin-top:14px;font-size:0.75rem;color:rgba(180,180,220,0.38);line-height:1.6;">
                H₀: Series has a unit root (non-stationary)<br>
                Reject H₀ when p &lt; 0.05 → series is stationary
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_kpss:
        if kpss_r["stat"] is not None:
            kclr  = "#00ff88" if kpss_stat  else "#ff4757"
            ktxt  = "STATIONARY"    if kpss_stat  else "NOT STATIONARY"
            kbcls = "badge-green"   if kpss_stat  else "badge-red"
            krows = "".join(
                f'<tr><td style="padding:5px 0;color:rgba(180,180,220,0.48);">Critical {k}</td>'
                f'<td style="text-align:right;color:rgba(200,200,230,0.7);">{v:.4f}</td></tr>'
                for k, v in kpss_r["crit"].items()
            )
            st.markdown(f"""
            <div class="glass-card" style="border-top:3px solid {kclr};">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;">
                    <span style="font-size:1.05rem;font-weight:700;color:#e8e8ff;">KPSS Test</span>
                    <span class="verdict-badge {kbcls}">{ktxt}</span>
                </div>
                <table style="width:100%;border-collapse:collapse;font-size:0.83rem;">
                    <tr><td style="padding:6px 0;color:rgba(180,180,220,0.48);">KPSS Statistic</td>
                        <td style="text-align:right;color:#7b2ff7;font-weight:700;">{kpss_r['stat']:.5f}</td></tr>
                    <tr><td style="padding:6px 0;color:rgba(180,180,220,0.48);">p-value</td>
                        <td style="text-align:right;color:#7b2ff7;font-weight:700;">{kpss_r['p']:.5f}</td></tr>
                    <tr><td style="padding:6px 0;color:rgba(180,180,220,0.48);">Lags Used</td>
                        <td style="text-align:right;color:#7b2ff7;font-weight:700;">{kpss_r['lags']}</td></tr>
                    {krows}
                </table>
                <div style="margin-top:14px;font-size:0.75rem;color:rgba(180,180,220,0.38);line-height:1.6;">
                    H₀: Series is stationary around a constant<br>
                    Reject H₀ when p &lt; 0.05 → series is NOT stationary
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("KPSS test could not be computed.")

    # ── Interpretation banner ─────────────────────────────────────────────────
    if adf_stat and kpss_stat:
        msg, brd = "✅ Both tests agree: the series is <b>stationary</b>.", "#00ff88"
    elif not adf_stat and kpss_stat is False:
        msg, brd = "❌ Both tests agree: the series is <b>non-stationary</b>. Differencing is recommended.", "#ff4757"
    else:
        msg, brd = "⚠️ Tests disagree — the series may be <b>trend-stationary</b> or <b>difference-stationary</b>.", "#ffd700"

    st.markdown(f"""
    <div class="interp-box"
         style="background:rgba(255,255,255,0.03);border-left:3px solid {brd};color:rgba(200,200,230,0.8);">
        {msg}
    </div>
    """, unsafe_allow_html=True)

    # ── Original vs differenced Plotly chart ──────────────────────────────────
    raw_s  = df["meantemp"].dropna()
    diff_s = raw_s.diff().dropna()

    fig_stat = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=["Original Series", "1st-Order Differenced Series"],
        vertical_spacing=0.1,
    )
    fig_stat.add_trace(go.Scatter(
        x=raw_s.index, y=raw_s.values, name="Original",
        mode="lines", line=dict(color="#00d4ff", width=1.4),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y:.1f} °C<extra></extra>",
    ), row=1, col=1)
    fig_stat.add_trace(go.Scatter(
        x=diff_s.index, y=diff_s.values, name="Differenced",
        mode="lines", line=dict(color="#ff6b35", width=1.3),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>Δ%{y:.2f} °C<extra></extra>",
    ), row=2, col=1)
    fig_stat.update_layout(
        **base_layout(height=440,
                      title=dict(text="Stationarity Visualization", font_size=14, x=0.02),
                      hovermode="x unified"),
    )
    for r in (1, 2):
        fig_stat.update_yaxes(gridcolor="rgba(255,255,255,0.04)", row=r, col=1)
    st.plotly_chart(fig_stat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — ACF & PACF
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style="color:rgba(180,180,220,0.5);font-size:0.82rem;margin-bottom:16px;line-height:1.6;">
        <b style="color:#00d4ff;">ACF</b> measures correlation of the series with its own past lags.
        &nbsp;<b style="color:#7b2ff7;">PACF</b> isolates the direct lag effect, removing intermediate
        correlations. Significant spikes outside the dashed bounds indicate useful lag structure.
    </div>
    """, unsafe_allow_html=True)

    nlags = st.slider("Number of lags", min_value=20, max_value=100, value=60, step=5)

    series_tuple = tuple(df["meantemp"].dropna().values)
    acf_vals, _, pacf_vals, _ = get_acf_pacf(series_tuple, nlags)

    n    = len(df["meantemp"].dropna())
    sig  = 1.96 / np.sqrt(n)
    lags = np.arange(len(acf_vals))

    def make_cf_fig(vals, sig_bound, title, bar_color, sig_color):
        colors = [bar_color if abs(v) > sig_bound else bar_color.replace("ff", "44")
                  for v in vals]
        fig = go.Figure()
        # stems
        for i, (lag, val) in enumerate(zip(lags, vals)):
            fig.add_shape(type="line", x0=lag, x1=lag, y0=0, y1=val,
                          line=dict(color=colors[i], width=1.8))
        # dots
        fig.add_trace(go.Scatter(
            x=lags, y=vals, mode="markers",
            marker=dict(color=colors, size=6, line=dict(color="#fff", width=0.5)),
            name="Value",
            hovertemplate="Lag %{x}: %{y:.4f}<extra></extra>",
        ))
        # significance bounds
        fig.add_hline(y= sig_bound, line_dash="dash", line_color=sig_color,
                      line_width=1.2, annotation_text="+95% CI",
                      annotation_font=dict(color=sig_color, size=9))
        fig.add_hline(y=-sig_bound, line_dash="dash", line_color=sig_color, line_width=1.2)
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.12)", line_width=1)
        fig.update_layout(
            **base_layout(height=360, showlegend=False,
                          title=dict(text=title, font_size=14, x=0.02),
                          xaxis_title="Lag", yaxis_title="Correlation"),
        )
        return fig

    ca, cp = st.columns(2)
    with ca:
        st.plotly_chart(make_cf_fig(acf_vals,  sig, "Autocorrelation Function (ACF)",
                                    "#00d4ff", "rgba(255,71,87,0.7)"),
                        use_container_width=True)
    with cp:
        st.plotly_chart(make_cf_fig(pacf_vals, sig, "Partial Autocorrelation (PACF)",
                                    "#7b2ff7", "rgba(255,71,87,0.7)"),
                        use_container_width=True)

    sig_acf_count  = int(np.sum(np.abs(acf_vals[1:])  > sig))
    sig_pacf_count = int(np.sum(np.abs(pacf_vals[1:]) > sig))
    st.markdown(f"""
    <div class="interp-box"
         style="background:rgba(255,255,255,0.03);border-left:3px solid #00d4ff;color:rgba(200,200,230,0.75);">
        <b style="color:#00d4ff;">ACF:</b> {sig_acf_count} significant lags out of {nlags} &nbsp;|&nbsp;
        <b style="color:#7b2ff7;">PACF:</b> {sig_pacf_count} significant lags out of {nlags}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — FORECAST & METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    if "forecast_done" not in st.session_state:
        st.markdown("""
        <div style="text-align:center;padding:70px 20px 60px;">
            <div style="font-size:3.5rem;margin-bottom:18px;animation:fadeInUp 0.5s ease;">🔮</div>
            <div style="font-size:1.05rem;color:rgba(180,180,220,0.55);margin-bottom:8px;
                        font-weight:600;">No forecast computed yet</div>
            <div style="font-size:0.82rem;color:rgba(180,180,220,0.32);">
                Click <b style="color:#00d4ff;">⚡ Run Forecast</b> in the sidebar<br>
                to generate predictions with Chronos Bolt Small
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        med, lo, hi = st.session_state["forecast_results"]
        actual      = test["meantemp"].values

        # ── Metrics (match final.py store_metrics exactly) ───────────────────
        mse   = float(mean_squared_error(actual, med))
        rmse  = float(np.sqrt(mse))
        smape = calc_smape(actual, med)
        r2    = float(r2_score(actual, med))

        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("MSE",   f"{mse:.4f}",    help="Mean Squared Error")
        with c2: st.metric("RMSE",  f"{rmse:.2f} °C",  help="Root Mean Squared Error")
        with c3: st.metric("sMAPE", f"{smape:.2f}%", help="Symmetric Mean Absolute Percentage Error")
        with c4: st.metric("R²",    f"{r2:.4f}",     help="Coefficient of determination")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Forecast chart ────────────────────────────────────────────────────
        model_short = BOLT_LABEL
        fig_fc = go.Figure()

        # CI shaded band
        fig_fc.add_trace(go.Scatter(
            x=list(test.index) + list(test.index[::-1]),
            y=list(hi) + list(lo[::-1]),
            fill="toself",
            fillcolor="rgba(123,47,247,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="80% CI",
            hoverinfo="skip",
        ))
        # CI boundary lines (thin)
        fig_fc.add_trace(go.Scatter(
            x=test.index, y=hi, name="Upper CI",
            mode="lines", line=dict(color="rgba(123,47,247,0.35)", width=0.8, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))
        fig_fc.add_trace(go.Scatter(
            x=test.index, y=lo, name="Lower CI",
            mode="lines", line=dict(color="rgba(123,47,247,0.35)", width=0.8, dash="dot"),
            hoverinfo="skip", showlegend=False,
        ))
        # Forecast median line
        fig_fc.add_trace(go.Scatter(
            x=test.index, y=med,
            name=f"Forecast · {model_short}",
            mode="lines",
            line=dict(color="#ff6b35", width=2.2, dash="dash"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: %{y:.1f} °C<extra></extra>",
        ))
        # Actual line
        fig_fc.add_trace(go.Scatter(
            x=test.index, y=actual,
            name="Actual",
            mode="lines",
            line=dict(color="#00d4ff", width=2.4),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:.1f} °C<extra></extra>",
        ))

        fig_fc.update_layout(
            **base_layout(
                title=dict(text="Foundation Model Forecast vs Actual Temperature · Test Period",
                           font_size=15, x=0.02),
                yaxis_title="Temperature (°C)",
                height=470,
                xaxis=dict(
                    gridcolor="rgba(255,255,255,0.04)", showgrid=True,
                    zeroline=False, linecolor="rgba(255,255,255,0.08)",
                    rangeslider=dict(visible=True, bgcolor="rgba(255,255,255,0.02)",
                                     thickness=0.055),
                    type="date",
                ),
            )
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # ── Error distribution ────────────────────────────────────────────────
        errors = actual - med
        col_hist, col_scatter = st.columns([1, 1])
        with col_hist:
            fig_err = go.Figure()
            fig_err.add_trace(go.Histogram(
                x=errors, nbinsx=35,
                marker=dict(color="#7b2ff7", opacity=0.75,
                            line=dict(color="rgba(0,212,255,0.3)", width=0.5)),
                name="Error",
                hovertemplate="Error: %{x:.1f} °C  Count: %{y}<extra></extra>",
            ))
            fig_err.add_vline(x=0, line_color="rgba(255,255,255,0.38)",
                              line_dash="dash", line_width=1.5)
            fig_err.update_layout(
                **base_layout(
                    title=dict(text="Forecast Error Distribution", font_size=13, x=0.02),
                    xaxis_title="Error (Actual − Forecast) °C",
                    yaxis_title="Count",
                    height=300, showlegend=False,
                )
            )
            st.plotly_chart(fig_err, use_container_width=True)

        with col_scatter:
            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scatter(
                x=actual, y=med,
                mode="markers",
                marker=dict(
                    color=np.abs(errors), colorscale="Plasma",
                    size=5, opacity=0.65,
                    colorbar=dict(title="Error", thickness=10, len=0.7),
                ),
                name="Actual vs Forecast",
                hovertemplate="Actual: %{x:.1f}°C<br>Forecast: %{y:.1f}°C<extra></extra>",
            ))
            mn, mx = float(min(actual.min(), med.min())), float(max(actual.max(), med.max()))
            fig_scat.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.25)", width=1.5, dash="dot"),
                name="Perfect Forecast",
            ))
            fig_scat.update_layout(
                **base_layout(
                    title=dict(text="Actual vs Forecast Scatter", font_size=13, x=0.02),
                    xaxis_title="Actual (°C)",
                    yaxis_title="Forecast (°C)",
                    height=300,
                )
            )
            st.plotly_chart(fig_scat, use_container_width=True)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — STL DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    series_clean = df["meantemp"].dropna()
    trend, seasonal, resid = get_stl(tuple(series_clean.values))
    idx = series_clean.index

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Trend Min",       f"{trend.min():.1f} °C")
    with c2: st.metric("Trend Max",       f"{trend.max():.1f} °C")
    with c3: st.metric("Seasonal Range",  f"{seasonal.max() - seasonal.min():.1f} °C")
    with c4: st.metric("Residual Std",    f"{resid.std():.3f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 4-panel Plotly chart ──────────────────────────────────────────────────
    components = [series_clean.values, trend, seasonal, resid]
    comp_names = ["Observed", "Trend", "Seasonal", "Residual"]
    comp_cols  = ["#00d4ff", "#7b2ff7", "#00ff88", "#ff6b35"]

    fig_stl = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=comp_names,
        vertical_spacing=0.05,
        row_heights=[0.3, 0.25, 0.25, 0.2],
    )
    for row, (comp, name, color) in enumerate(zip(components, comp_names, comp_cols), start=1):
        fig_stl.add_trace(go.Scatter(
            x=idx, y=comp, name=name,
            mode="lines", line=dict(color=color, width=1.4),
            hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{name}: %{{y:.2f}} °C<extra></extra>",
        ), row=row, col=1)
        if name == "Residual":
            fig_stl.add_hline(y=0, line_color="rgba(255,255,255,0.15)",
                              line_dash="dash", line_width=1, row=row, col=1)

    fig_stl.update_layout(
        **base_layout(
            height=720,
            title=dict(text="STL Decomposition — Delhi Mean Temperature", font_size=15, x=0.02),
            showlegend=False,
            hovermode="x unified",
        ),
    )
    for r in range(1, 5):
        fig_stl.update_yaxes(gridcolor="rgba(255,255,255,0.04)", row=r, col=1)

    st.plotly_chart(fig_stl, use_container_width=True)

    # ── Interpretation card ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="glass-card" style="margin-top:4px;">
        <div style="font-size:0.95rem;font-weight:700;color:#e8e8ff;margin-bottom:14px;">
            Decomposition Insights
        </div>
        <div style="font-size:0.84rem;color:rgba(200,200,230,0.72);line-height:1.8;">
            <span style="color:#7b2ff7;font-weight:600;">Trend:</span>
            Delhi temperatures span <b style="color:#00d4ff;">{trend.min():.1f} – {trend.max():.1f} °C</b>,
            revealing the underlying multi-year temperature signal after removing seasonality.<br>
            <span style="color:#00ff88;font-weight:600;">Seasonal:</span>
            Annual cycle amplitude of
            <b style="color:#00d4ff;">{seasonal.max() - seasonal.min():.1f} °C</b>
            — reflecting Delhi's strong summer–winter contrast with a 365-day period.<br>
            <span style="color:#ff6b35;font-weight:600;">Residual:</span>
            Standard deviation of <b style="color:#00d4ff;">{resid.std():.3f}</b>
            — small residuals confirm the STL fit captures most variance.
        </div>
    </div>
    """, unsafe_allow_html=True)
