"""
Monte Carlo Options Pricer — Enhanced Interactive Streamlit App
===============================================================
Run locally:   streamlit run app.py
Run via Docker: docker compose up --build  → http://localhost:8501
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from monte_carlo_options import bs_price, monte_carlo

# ── Page config (must be first Streamlit call) ─────────────────────────────────
st.set_page_config(
    page_title="MC Options Pricer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Dark gradient background ───────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0b0f1a 0%, #111827 60%, #0f172a 100%);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #0b0f1a 100%);
    border-right: 1px solid #1e293b;
}

/* ── Page title ─────────────────────────────────────────────────────────── */
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0;
    line-height: 1.2;
}
.hero-sub {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 4px;
    margin-bottom: 24px;
}

/* ── KPI cards ───────────────────────────────────────────────────────────── */
.kpi-row {
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin: 20px 0 28px 0;
}
.kpi-card {
    flex: 1;
    min-width: 130px;
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 18px 16px 14px 16px;
    text-align: center;
    cursor: default;
    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(96, 165, 250, 0.2);
    border-color: #3b82f6;
}
.kpi-icon  { font-size: 1.3rem; margin-bottom: 6px; }
.kpi-label {
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
    margin-bottom: 6px;
}
.kpi-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.kpi-delta {
    font-size: 10.5px;
    margin-top: 5px;
    font-weight: 500;
}
.c-green  { color: #4ade80; }
.c-red    { color: #f87171; }
.c-blue   { color: #60a5fa; }
.c-purple { color: #c084fc; }
.c-amber  { color: #fbbf24; }
.c-teal   { color: #2dd4bf; }
.c-pink   { color: #f472b6; }

/* ── Section divider ─────────────────────────────────────────────────────── */
.section-label {
    font-size: 11px;
    font-weight: 700;
    color: #334155;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin: 24px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #1e293b;
}

/* ── Sidebar labels ──────────────────────────────────────────────────────── */
[data-testid="stSidebar"] label { color: #94a3b8 !important; }
[data-testid="stSidebar"] .stSlider > label { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Parameters")

    S     = st.slider("Spot price S ($)",          50.0, 200.0, 100.0, step=1.0)
    K     = st.slider("Strike price K ($)",         50.0, 200.0, 105.0, step=1.0)
    sigma = st.slider("Volatility σ (%)",            5.0,  80.0,  20.0, step=0.5) / 100
    r     = st.slider("Risk-free rate r (%)",        0.0,  15.0,   4.5, step=0.25) / 100
    T     = st.slider("Time to expiry T (years)",   0.05,   2.0,   0.5, step=0.05)

    st.divider()
    st.markdown("### 🎛️ Simulation")

    n_sims      = st.selectbox("Paths", [1_000, 10_000, 50_000], index=1, format_func=lambda x: f"{x:,}")
    n_show      = st.slider("Paths shown in chart", 5, 50, 20)
    opt_type    = st.radio("Greeks / table view", ["Call", "Put", "Both"], index=2, horizontal=True)
    antithetic  = st.checkbox("Antithetic variates", value=True,
                               help="Pairs each random draw with its negative mirror — halves variance at zero extra cost.")
    seed_val    = st.number_input("Random seed (0 = random)", value=42, min_value=0)
    seed        = int(seed_val) if seed_val > 0 else None

    run = st.button("▶  Run Simulation", type="primary", use_container_width=True)

# ── Simulation ─────────────────────────────────────────────────────────────────
if "mc_res" not in st.session_state:
    mc_res = monte_carlo(100, 105, 0.5, 0.045, 0.20,
                         n_sims=10_000, n_paths_plot=50, antithetic=True, seed=42)
    bs_res = bs_price(100, 105, 0.5, 0.045, 0.20)
    st.session_state.update(mc_res=mc_res, bs_res=bs_res,
                             sim_S=100.0, sim_K=105.0, sim_T=0.5,
                             sim_r=0.045, sim_sigma=0.20)

if run:
    mc_res = monte_carlo(S, K, T, r, sigma,
                         n_sims=n_sims, n_paths_plot=50, antithetic=antithetic, seed=seed)
    bs_res = bs_price(S, K, T, r, sigma)
    st.session_state.update(mc_res=mc_res, bs_res=bs_res,
                             sim_S=S, sim_K=K, sim_T=T,
                             sim_r=r, sim_sigma=sigma)

mc_res    = st.session_state["mc_res"]
bs_res    = st.session_state["bs_res"]
sim_K     = st.session_state["sim_K"]
sim_S     = st.session_state["sim_S"]
sim_T     = st.session_state["sim_T"]
sim_r     = st.session_state["sim_r"]
sim_sigma = st.session_state["sim_sigma"]

mc_call      = mc_res["mc_call"]
mc_put       = mc_res["mc_put"]
call_ci      = mc_res["call_ci"]
put_ci       = mc_res["put_ci"]
terminal     = mc_res["terminal"]
sample_paths = mc_res["sample_paths"][:n_show]
actual_sims  = mc_res["n_sims"]
time_axis    = np.linspace(0, sim_T, sample_paths.shape[1])

itm_mask = terminal >= sim_K
pct_itm  = 100.0 * itm_mask.mean()
moneyness = sim_S / sim_K
parity_err = abs((mc_call - mc_put) - (sim_S - sim_K * np.exp(-sim_r * sim_T)))
call_diff_pct = (mc_call - bs_res["call"]) / bs_res["call"] * 100
put_diff_pct  = (mc_put  - bs_res["put"])  / bs_res["put"]  * 100

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_BG = "rgba(15,23,42,0)"
PLOTLY_PAPER = "rgba(15,23,42,0)"
GRID_COLOR = "#1e293b"

# ── Hero title ─────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">📈 Monte Carlo Options Pricer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">European Call & Put · Geometric Brownian Motion simulation · '
    'Black-Scholes cross-validation</p>',
    unsafe_allow_html=True,
)

# ── KPI cards ──────────────────────────────────────────────────────────────────
def sign_color(v):
    return "c-green" if v >= 0 else "c-red"

moneyness_label = "ITM" if moneyness > 1.005 else ("OTM" if moneyness < 0.995 else "ATM")
moneyness_color = "c-green" if moneyness > 1.005 else ("c-red" if moneyness < 0.995 else "c-amber")
parity_color    = "c-green" if parity_err < 0.005 else ("c-amber" if parity_err < 0.02 else "c-red")
parity_icon     = "✅" if parity_err < 0.005 else ("⚠️" if parity_err < 0.02 else "❌")

kpi_html = f"""
<div class="kpi-row">

  <div class="kpi-card">
    <div class="kpi-icon">📞</div>
    <div class="kpi-label">MC Call Price</div>
    <div class="kpi-value c-blue">${mc_call:.4f}</div>
    <div class="kpi-delta {sign_color(-call_diff_pct)}">
      BS ${bs_res['call']:.4f} &nbsp;·&nbsp; {call_diff_pct:+.2f}%
    </div>
  </div>

  <div class="kpi-card">
    <div class="kpi-icon">🤝</div>
    <div class="kpi-label">MC Put Price</div>
    <div class="kpi-value c-pink">${mc_put:.4f}</div>
    <div class="kpi-delta {sign_color(-put_diff_pct)}">
      BS ${bs_res['put']:.4f} &nbsp;·&nbsp; {put_diff_pct:+.2f}%
    </div>
  </div>

  <div class="kpi-card">
    <div class="kpi-icon">🎯</div>
    <div class="kpi-label">Moneyness (S/K)</div>
    <div class="kpi-value {moneyness_color}">{moneyness:.3f}</div>
    <div class="kpi-delta {moneyness_color}">{moneyness_label}</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-icon">📊</div>
    <div class="kpi-label">% ITM (for Call)</div>
    <div class="kpi-value c-teal">{pct_itm:.1f}%</div>
    <div class="kpi-delta c-teal">{actual_sims:,} paths simulated</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-icon">🎲</div>
    <div class="kpi-label">95% CI · Call / Put</div>
    <div class="kpi-value c-purple">±${call_ci:.4f}</div>
    <div class="kpi-delta c-purple">Put ±${put_ci:.4f}</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-icon">{parity_icon}</div>
    <div class="kpi-label">Put-Call Parity Err</div>
    <div class="kpi-value {parity_color}">${parity_err:.5f}</div>
    <div class="kpi-delta {parity_color}">{"Good" if parity_err < 0.005 else "Elevated"}</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-icon">Δ</div>
    <div class="kpi-label">Delta (Call / Put)</div>
    <div class="kpi-value c-amber">{bs_res['delta_call']:.3f}</div>
    <div class="kpi-delta c-red">Put {bs_res['delta_put']:.3f}</div>
  </div>

  <div class="kpi-card">
    <div class="kpi-icon">ν</div>
    <div class="kpi-label">Vega (per 1% σ)</div>
    <div class="kpi-value c-green">${bs_res['vega']:.4f}</div>
    <div class="kpi-delta c-amber">σ = {sim_sigma*100:.1f}% &nbsp; T = {sim_T:.2f}yr</div>
  </div>

</div>
"""
st.markdown(kpi_html, unsafe_allow_html=True)

# ── Chart helpers ──────────────────────────────────────────────────────────────
def apply_dark_layout(fig, title=""):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor=PLOTLY_PAPER,
        plot_bgcolor=PLOTLY_BG,
        title=dict(text=title, font=dict(size=13, color="#94a3b8"), x=0.01),
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        legend=dict(
            bgcolor="rgba(15,23,42,0.7)",
            bordercolor="#1e293b",
            borderwidth=1,
            font=dict(size=11),
        ),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )
    return fig

# ── Chart 1: Coloured price paths ─────────────────────────────────────────────
terminal_shown = sample_paths[:, -1]
# Rank each path by its terminal value → map to a 0-1 scale for the colorscale
ranks = terminal_shown.argsort().argsort() / max(len(terminal_shown) - 1, 1)

# Sample colours from the Plasma colorscale: cool (purple/blue) → warm (yellow/orange)
path_colors = px.colors.sample_colorscale("plasma", ranks.tolist())

fig_paths = go.Figure()

for i, (path, color) in enumerate(zip(sample_paths, path_colors)):
    is_lead = (i == int(np.argmax(ranks)))    # highest-terminal path gets thicker line
    fig_paths.add_trace(go.Scatter(
        x=time_axis, y=path,
        mode="lines",
        line=dict(color=color, width=2.2 if is_lead else 0.9),
        opacity=0.85 if is_lead else 0.55,
        name=f"Path {i+1}  →  S_T=${terminal_shown[i]:.1f}",
        hovertemplate="t=%{x:.2f}yr  S=$%{y:.2f}<extra>Path %{fullData.name}</extra>",
        showlegend=False,
    ))

# Strike line
fig_paths.add_hline(
    y=sim_K, line_dash="dash", line_color="#f87171", line_width=1.5,
    annotation_text=f"  Strike K=${sim_K:.0f}",
    annotation_font_color="#f87171",
    annotation_position="top left",
)
# Colourbar legend (fake scatter trace)
fig_paths.add_trace(go.Scatter(
    x=[None], y=[None], mode="markers",
    marker=dict(
        colorscale="plasma", showscale=True,
        color=[0], cmin=0, cmax=1,
        colorbar=dict(
            title=dict(text="S_T rank", font=dict(size=10, color="#64748b")),
            thickness=10, len=0.6, x=1.01,
            tickvals=[0, 0.5, 1],
            ticktext=["Low", "Mid", "High"],
            tickfont=dict(size=9, color="#64748b"),
            bgcolor="rgba(0,0,0,0)",
        ),
    ),
    showlegend=False,
))

apply_dark_layout(
    fig_paths,
    f"GBM Price Paths — {n_show} shown  ({actual_sims:,} simulated)  |  colour = terminal value",
)
fig_paths.update_layout(height=380)

# ── Chart 2: Terminal distribution ────────────────────────────────────────────
fig_dist = go.Figure()

fig_dist.add_trace(go.Histogram(
    x=terminal[~itm_mask], nbinsx=55,
    name="OTM (call)", marker_color="#f472b6",
    opacity=0.80,
    hovertemplate="S_T = $%{x:.1f}<br>Count = %{y}<extra>OTM</extra>",
))
fig_dist.add_trace(go.Histogram(
    x=terminal[itm_mask], nbinsx=55,
    name="ITM (call)", marker_color="#60a5fa",
    opacity=0.80,
    hovertemplate="S_T = $%{x:.1f}<br>Count = %{y}<extra>ITM</extra>",
))
fig_dist.update_layout(barmode="overlay")

fig_dist.add_vline(x=sim_K, line_dash="dash", line_color="#f87171", line_width=1.5,
                   annotation_text=f"  K=${sim_K:.0f}", annotation_font_color="#f87171",
                   annotation_position="top left")
fig_dist.add_vline(x=terminal.mean(), line_dash="dot", line_color="#fbbf24", line_width=1.2,
                   annotation_text=f"  Mean ${terminal.mean():.1f}", annotation_font_color="#fbbf24",
                   annotation_position="top right")

apply_dark_layout(fig_dist, f"Terminal Price Distribution  ·  {pct_itm:.1f}% ITM for Call")
fig_dist.update_layout(
    height=380,
    xaxis_title="S_T ($)",
    yaxis_title="Count",
    legend=dict(orientation="h", y=0.98, x=0.01),
)

# ── Row 1 ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Simulation</div>', unsafe_allow_html=True)
col_l, col_r = st.columns(2, gap="medium")
col_l.plotly_chart(fig_paths, use_container_width=True, config={"displayModeBar": True})
col_r.plotly_chart(fig_dist,  use_container_width=True, config={"displayModeBar": True})

# ── Chart 3: Pricing table (Plotly table) ─────────────────────────────────────
def pct_diff_str(mc, bsv):
    if abs(bsv) < 1e-10:
        return "N/A"
    return f"{(mc - bsv) / bsv * 100:+.2f}%"

show_call = opt_type in ("Call", "Both")
show_put  = opt_type in ("Put",  "Both")

rows_hdr   = ["", "MC Price", "BS Price", "% Diff vs BS", "95% CI (half-width)"]
rows_call  = ["📞 Call", f"${mc_call:.4f}", f"${bs_res['call']:.4f}",
              pct_diff_str(mc_call, bs_res["call"]), f"±${call_ci:.4f}"]
rows_put   = ["🤝 Put",  f"${mc_put:.4f}",  f"${bs_res['put']:.4f}",
              pct_diff_str(mc_put,  bs_res["put"]),  f"±${put_ci:.4f}"]

data_rows = []
if show_call:
    data_rows.append(rows_call)
if show_put:
    data_rows.append(rows_put)

cell_text   = [list(col) for col in zip(*data_rows)] if data_rows else [[] for _ in rows_hdr]
fill_colors = []
for col_data in cell_text:
    col_colors = []
    for j, _ in enumerate(col_data):
        col_colors.append("#1a2744" if (j == 0 and show_call) else "#1a1535")
    fill_colors.append(col_colors)

fig_table = go.Figure(go.Table(
    header=dict(
        values=[f"<b>{h}</b>" for h in rows_hdr],
        fill_color="#0f172a",
        align="center",
        font=dict(color="#94a3b8", size=12),
        line_color="#1e293b",
        height=36,
    ),
    cells=dict(
        values=cell_text,
        fill_color=fill_colors if fill_colors else "#0f172a",
        align="center",
        font=dict(color="#f1f5f9", size=13),
        line_color="#1e293b",
        height=34,
    ),
))
fig_table.update_layout(
    paper_bgcolor=PLOTLY_PAPER,
    margin=dict(l=0, r=0, t=36, b=0),
    height=160 + 40 * len(data_rows),
    title=dict(
        text=f"Pricing Summary  ·  Put-Call Parity error = ${parity_err:.5f} {parity_icon}",
        font=dict(size=12, color="#64748b"), x=0.01,
    ),
)

# ── Chart 4: Greeks (Plotly bar) ───────────────────────────────────────────────
if opt_type == "Put":
    greek_values = [bs_res["delta_put"], bs_res["gamma"],
                    bs_res["theta_put"], bs_res["vega"], bs_res["rho_put"]]
    greeks_title = "Black-Scholes Greeks — Put"
else:
    greek_values = [bs_res["delta_call"], bs_res["gamma"],
                    bs_res["theta_call"], bs_res["vega"], bs_res["rho_call"]]
    greeks_title = "Black-Scholes Greeks — Call"

greek_names   = ["Delta (Δ)", "Gamma (Γ)", "Theta (Θ) $/day", "Vega (ν) /1%σ", "Rho (ρ) /1%r"]
greek_desc    = [
    "Sensitivity to spot price S",
    "Rate of change of Delta",
    "Time decay per calendar day",
    "Sensitivity to 1% volatility move",
    "Sensitivity to 1% rate move",
]
bar_colors    = ["#60a5fa" if v >= 0 else "#f87171" for v in greek_values]

fig_greeks = go.Figure(go.Bar(
    x=greek_names,
    y=greek_values,
    marker_color=bar_colors,
    marker_line_color="rgba(0,0,0,0)",
    text=[f"{v:.4f}" for v in greek_values],
    textposition="outside",
    textfont=dict(size=11, color="#cbd5e1"),
    customdata=greek_desc,
    hovertemplate="<b>%{x}</b><br>Value: %{y:.5f}<br>%{customdata}<extra></extra>",
))

fig_greeks.add_hline(y=0, line_color="#334155", line_width=1)
apply_dark_layout(fig_greeks, greeks_title)
fig_greeks.update_layout(
    height=340,
    yaxis_title="Value",
    showlegend=False,
)
# Extend y-axis so text labels aren't clipped
y_range = max(greek_values) - min(greek_values) if greek_values else 1
pad = y_range * 0.25 or 0.05
fig_greeks.update_yaxes(
    range=[min(greek_values) - pad, max(greek_values) + pad]
)

# ── Row 2 ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Pricing &amp; Greeks</div>', unsafe_allow_html=True)
col_tbl, col_grk = st.columns([1, 1.6], gap="medium")
col_tbl.plotly_chart(fig_table,  use_container_width=True, config={"displayModeBar": False})
col_grk.plotly_chart(fig_greeks, use_container_width=True, config={"displayModeBar": True})

# ── Chart 5: Payoff profile ────────────────────────────────────────────────────
s_range = np.linspace(sim_S * 0.5, sim_S * 1.5, 300)
call_payoff = np.maximum(s_range - sim_K, 0)
put_payoff  = np.maximum(sim_K - s_range, 0)
bs_calls = np.array([bs_price(s, sim_K, max(sim_T, 0.001), sim_r, sim_sigma)["call"] for s in s_range])
bs_puts  = np.array([bs_price(s, sim_K, max(sim_T, 0.001), sim_r, sim_sigma)["put"]  for s in s_range])

fig_payoff = go.Figure()
if show_call:
    fig_payoff.add_trace(go.Scatter(
        x=s_range, y=bs_calls, mode="lines", name="Call (BS)",
        line=dict(color="#60a5fa", width=2.5),
        hovertemplate="S=$%{x:.1f}<br>Call=$%{y:.4f}<extra></extra>",
    ))
    fig_payoff.add_trace(go.Scatter(
        x=s_range, y=call_payoff, mode="lines", name="Call payoff at expiry",
        line=dict(color="#60a5fa", width=1.2, dash="dash"),
        opacity=0.5,
        hovertemplate="S=$%{x:.1f}<br>Payoff=$%{y:.4f}<extra></extra>",
    ))
if show_put:
    fig_payoff.add_trace(go.Scatter(
        x=s_range, y=bs_puts, mode="lines", name="Put (BS)",
        line=dict(color="#f472b6", width=2.5),
        hovertemplate="S=$%{x:.1f}<br>Put=$%{y:.4f}<extra></extra>",
    ))
    fig_payoff.add_trace(go.Scatter(
        x=s_range, y=put_payoff, mode="lines", name="Put payoff at expiry",
        line=dict(color="#f472b6", width=1.2, dash="dash"),
        opacity=0.5,
        hovertemplate="S=$%{x:.1f}<br>Payoff=$%{y:.4f}<extra></extra>",
    ))

fig_payoff.add_vline(x=sim_K, line_dash="dash", line_color="#fbbf24", line_width=1.5,
                     annotation_text=f"  K=${sim_K:.0f}", annotation_font_color="#fbbf24",
                     annotation_position="top left")
fig_payoff.add_vline(x=sim_S, line_dash="dot", line_color="#a3e635", line_width=1.2,
                     annotation_text=f"  S=${sim_S:.0f}", annotation_font_color="#a3e635",
                     annotation_position="top right")

apply_dark_layout(fig_payoff, f"Option Value Profile  ·  T={sim_T:.2f}yr  σ={sim_sigma*100:.0f}%  r={sim_r*100:.1f}%")
fig_payoff.update_layout(
    height=320,
    xaxis_title="Spot Price S ($)",
    yaxis_title="Option Value ($)",
    legend=dict(orientation="h", y=-0.18),
)

# ── Chart 6: Volatility sensitivity (Vega surface) ────────────────────────────
sigma_range = np.linspace(0.05, 0.60, 60)
T_range     = np.linspace(0.1,  2.0,  50)
SIG, TT = np.meshgrid(sigma_range, T_range)

call_surface = np.vectorize(
    lambda sig, t: bs_price(sim_S, sim_K, t, sim_r, sig)["call"]
)(SIG, TT)

fig_surface = go.Figure(go.Surface(
    x=sigma_range * 100, y=T_range, z=call_surface,
    colorscale="plasma",
    contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)),
    hovertemplate="σ=%{x:.0f}%<br>T=%{y:.2f}yr<br>Call=$%{z:.3f}<extra></extra>",
    showscale=True,
    colorbar=dict(
        title=dict(text="Call ($)", font=dict(size=10, color="#64748b")),
        thickness=12, len=0.7, x=1.01,
        tickfont=dict(size=9, color="#64748b"),
        bgcolor="rgba(0,0,0,0)",
    ),
))
# Mark current point
current_call = bs_price(sim_S, sim_K, sim_T, sim_r, sim_sigma)["call"]
fig_surface.add_trace(go.Scatter3d(
    x=[sim_sigma * 100], y=[sim_T], z=[current_call],
    mode="markers+text",
    marker=dict(size=7, color="#a3e635", symbol="circle"),
    text=["◀ Current"],
    textfont=dict(color="#a3e635", size=11),
    name="Current params",
    hovertemplate=f"σ={sim_sigma*100:.0f}%<br>T={sim_T:.2f}yr<br>Call=${current_call:.3f}<extra>Current</extra>",
))
fig_surface.update_layout(
    template=PLOTLY_TEMPLATE,
    paper_bgcolor=PLOTLY_PAPER,
    title=dict(
        text="Call Price Surface  (σ vs T)  —  drag to rotate",
        font=dict(size=13, color="#94a3b8"), x=0.01,
    ),
    scene=dict(
        xaxis=dict(title="Volatility σ (%)", gridcolor=GRID_COLOR, backgroundcolor="rgba(0,0,0,0)"),
        yaxis=dict(title="Time T (yr)",       gridcolor=GRID_COLOR, backgroundcolor="rgba(0,0,0,0)"),
        zaxis=dict(title="Call Price ($)",    gridcolor=GRID_COLOR, backgroundcolor="rgba(0,0,0,0)"),
        bgcolor="rgba(0,0,0,0)",
    ),
    margin=dict(l=0, r=0, t=50, b=0),
    height=440,
    legend=dict(bgcolor="rgba(15,23,42,0.8)", bordercolor="#1e293b", borderwidth=1),
)

# ── Row 3 ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Option Analytics</div>', unsafe_allow_html=True)
col_pay, col_surf = st.columns([1, 1.2], gap="medium")
col_pay.plotly_chart(fig_payoff,  use_container_width=True, config={"displayModeBar": True})
col_surf.plotly_chart(fig_surface, use_container_width=True, config={"displayModeBar": True})

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<div style="text-align:center; color:#334155; font-size:11px; margin-top:32px; '
    'padding-top:16px; border-top:1px solid #1e293b;">'
    'Monte Carlo Options Pricer &nbsp;·&nbsp; GBM simulation &nbsp;·&nbsp; '
    'Black-Scholes analytics &nbsp;·&nbsp; Plotly interactive charts'
    '</div>',
    unsafe_allow_html=True,
)
