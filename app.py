import streamlit as st
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from datetime import date, datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import json
import os

# ════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE PÁGINA
# ════════════════════════════════════════════════════════════════
st.set_page_config(layout="wide", page_title="Modelo de Opciones", page_icon="📊")

# ════════════════════════════════════════════════════════════════
#  CSS PERSONALIZADO — Diseño refinado estilo terminal financiero
# ════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Header principal */
h1 { 
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.6rem !important;
    letter-spacing: -0.02em;
    color: #0f172a !important;
    border-bottom: 3px solid #0ea5e9;
    padding-bottom: 0.5rem;
}
h2, h3 {
    font-family: 'IBM Plex Mono', monospace !important;
    color: #1e293b !important;
}

/* Métricas */
[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #0ea5e9;
    border-radius: 6px;
    padding: 12px 16px !important;
    transition: box-shadow 0.15s;
}
[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 12px rgba(14,165,233,0.12);
}
[data-testid="metric-container"] label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 1.35rem !important;
    font-weight: 600;
    color: #0f172a !important;
}

/* Positive/negative metric deltas */
[data-testid="stMetricDelta"] svg { display: none; }

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #f1f5f9;
    border-radius: 8px;
    padding: 4px;
    gap: 2px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px;
    font-weight: 500;
    border-radius: 6px;
    color: #64748b;
    padding: 6px 16px;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: white !important;
    color: #0ea5e9 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Expanders */
[data-testid="stExpander"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    background: #fafafa;
}

/* Buttons */
.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    font-weight: 500;
    border-radius: 6px !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
}

/* ── Dataframes ─────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
/* Forzar fuente monospace en celdas numéricas */
[data-testid="stDataFrame"] td {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
}
[data-testid="stDataFrame"] th {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    background: #0f172a !important;
    color: #e2e8f0 !important;
}
/* Scrollbar discreta */
[data-testid="stDataFrame"] ::-webkit-scrollbar { height: 5px; width: 5px; }
[data-testid="stDataFrame"] ::-webkit-scrollbar-track { background: #f1f5f9; }
[data-testid="stDataFrame"] ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

/* ── Info/Warning/Success ────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
    font-size: 13px;
    border-left-width: 4px !important;
}

/* ── Separadores ─────────────────────────────── */
hr { border-color: #e2e8f0 !important; margin: 1.2rem 0 !important; }

/* ── Caption ─────────────────────────────────── */
.stCaption {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 10px !important;
    color: #94a3b8 !important;
    letter-spacing: 0.04em;
}

/* ── Subheaders con línea lateral ────────────── */
h3::before {
    content: "";
    display: inline-block;
    width: 3px;
    height: 0.85em;
    background: #0ea5e9;
    border-radius: 2px;
    margin-right: 8px;
    vertical-align: middle;
}

/* ── Número inputs y selects ─────────────────── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-baseweb="select"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    border-radius: 6px !important;
}

/* ── Sliders ─────────────────────────────────── */
[data-testid="stSlider"] [data-baseweb="slider"] {
    padding: 4px 0;
}

/* ── Cards de resumen ────────────────────────── */
.stat-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    border-radius: 10px;
    padding: 16px 20px;
    color: white;
    font-family: 'IBM Plex Mono', monospace;
    box-shadow: 0 4px 12px rgba(14,165,233,0.15);
}
.stat-card .label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #7dd3fc;
    margin-bottom: 4px;
}
.stat-card .value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #f0f9ff;
}
.stat-card .sub {
    font-size: 10px;
    color: #64748b;
    margin-top: 4px;
}

/* ── Badges ──────────────────────────────────── */
.strategy-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}
.pnl-positive { color: #16a34a; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.pnl-negative { color: #dc2626; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }

/* ── What-if panel ───────────────────────────── */
.whatif-panel {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 1px solid #bae6fd;
    border-radius: 10px;
    padding: 16px;
    margin: 8px 0;
}

/* ── Cadena de opciones: highlight ATM ───────── */
.atm-row {
    background: #fef3c7 !important;
    font-weight: 600;
}

/* ── Scrollable section label ────────────────── */
.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #64748b;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 4px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  INICIALIZACIÓN DE SESSION STATE
# ════════════════════════════════════════════════════════════════
for key, default in [
    ("cartera", []),
    ("hidden_charts", {}),
    ("saved_portfolios", {}),
    ("closed_positions", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ════════════════════════════════════════════════════════════════
#  CACHÉ DE DATOS DE MERCADO  (TTL = 60 s)
# ════════════════════════════════════════════════════════════════
@st.cache_data(ttl=60)
def get_stock_info(ticker: str) -> dict:
    """Precio actual + cambio % de una acción."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="2d")
        price = float(hist["Close"].iloc[-1]) if not hist.empty else None
        prev  = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
        change_pct = ((price - prev) / prev * 100) if (price and prev) else None
        last_ts = hist.index[-1].to_pydatetime() if not hist.empty else None
        info = t.info
        return {
            "price":       round(price, 4) if price else None,
            "change_pct":  round(change_pct, 2) if change_pct else None,
            "name":        info.get("shortName", ticker),
            "last_update": last_ts,
        }
    except Exception as e:
        return {"price": None, "change_pct": None, "name": ticker, "last_update": None, "error": str(e)}


@st.cache_data(ttl=60)
def get_option_chain(ticker: str, expiry: str):
    """Devuelve (calls_df, puts_df) para un vencimiento concreto."""
    try:
        chain = yf.Ticker(ticker).option_chain(expiry)
        return chain.calls, chain.puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


@st.cache_data(ttl=60)
def get_available_expiries(ticker: str):
    try:
        return list(yf.Ticker(ticker).options)
    except Exception:
        return []


@st.cache_data(ttl=300)
def get_historical_close(ticker: str, target_date: date):
    """Precio de cierre más cercano a target_date (hacia atrás hasta 7 días)."""
    try:
        hist = yf.Ticker(ticker).history(
            start=target_date - timedelta(days=7),
            end=target_date + timedelta(days=1),
        )
        if hist.empty:
            return None
        hist = hist[hist.index.date <= target_date]
        return float(hist["Close"].iloc[-1]) if not hist.empty else None
    except Exception:
        return None


@st.cache_data(ttl=3600)
def get_historical_volatility(ticker: str, window: int = 30) -> float:
    """Volatilidad histórica realizada a N días."""
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if len(hist) < window + 1:
            return 0.25
        log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        hv = float(log_returns.rolling(window).std().iloc[-1]) * np.sqrt(252)
        return round(hv, 4) if hv > 0 else 0.25
    except Exception:
        return 0.25


@st.cache_data(ttl=3600)
def get_iv_rank(ticker: str, current_iv: float) -> dict:
    """
    IV Rank y IV Percentile basados en IV histórica del último año.
    IV Rank = (IV_actual - IV_min) / (IV_max - IV_min) * 100
    """
    try:
        t = yf.Ticker(ticker)
        # Obtener opciones ATM para múltiples fechas históricas
        opts = t.options
        if not opts:
            return {"iv_rank": None, "iv_pct": None, "iv_min": None, "iv_max": None}
        
        # Aproximar IV histórica usando HV
        hist = t.history(period="1y")
        if len(hist) < 30:
            return {"iv_rank": None, "iv_pct": None, "iv_min": None, "iv_max": None}
        
        # Rolling 30-day HV como proxy de IV histórica
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        rolling_hv = log_ret.rolling(30).std() * np.sqrt(252)
        rolling_hv = rolling_hv.dropna()
        
        iv_min = float(rolling_hv.min())
        iv_max = float(rolling_hv.max())
        
        if iv_max == iv_min:
            return {"iv_rank": 50.0, "iv_pct": 50.0, "iv_min": iv_min, "iv_max": iv_max}
        
        iv_rank = (current_iv - iv_min) / (iv_max - iv_min) * 100
        iv_pct  = float(np.mean(rolling_hv <= current_iv) * 100)
        
        return {
            "iv_rank": round(iv_rank, 1),
            "iv_pct":  round(iv_pct, 1),
            "iv_min":  round(iv_min, 4),
            "iv_max":  round(iv_max, 4),
        }
    except Exception:
        return {"iv_rank": None, "iv_pct": None, "iv_min": None, "iv_max": None}


def get_implied_vol(ticker: str, expiry: str, strike: float, option_type: str, default: float = 0.25) -> float:
    """Obtiene la IV del mercado; si falla devuelve el default."""
    try:
        calls, puts = get_option_chain(ticker, expiry)
        df = calls if option_type == "call" else puts
        if df.empty:
            return default
        idx = (df["strike"] - strike).abs().idxmin()
        iv = df.loc[idx, "impliedVolatility"]
        return float(iv) if iv and iv > 0.01 else default
    except Exception:
        return default


def get_market_option_price(ticker: str, expiry: str, strike: float, option_type: str):
    """Mid-price de una opción. Devuelve None si no está disponible."""
    try:
        calls, puts = get_option_chain(ticker, expiry)
        df = calls if option_type == "call" else puts
        if df.empty:
            return None
        idx = (df["strike"] - strike).abs().idxmin()
        row = df.loc[idx]
        bid = float(row.get("bid", 0) or 0)
        ask = float(row.get("ask", 0) or 0)
        if bid > 0 and ask > 0:
            return round((bid + ask) / 2, 4)
        last = float(row.get("lastPrice", 0) or 0)
        return round(last, 4) if last > 0 else None
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
#  BLACK-SCHOLES: PRECIO Y GRIEGAS
# ════════════════════════════════════════════════════════════════
def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float:
    """Precio Black-Scholes. T en años."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == "call" else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    else:
        return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> dict:
    """
    Griegas Black-Scholes correctas y consistentes.
    Todas las griegas son POR OPCIÓN (no por contrato de 100 acc).
    Delta  ∈ [-1, 1]
    Gamma  por $1 de movimiento
    Vega   por +1% de IV
    Theta  decaimiento diario en $
    Rho    por +1% de tipo de interés
    """
    zero = {"Delta": 0.0, "Gamma": 0.0, "Vega": 0.0, "Theta": 0.0, "Rho": 0.0}
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        if option_type == "call":
            zero["Delta"] = 1.0 if S > K else 0.0
        else:
            zero["Delta"] = -1.0 if S < K else 0.0
        return zero

    try:
        d1     = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2     = d1 - sigma * np.sqrt(T)
        pdf_d1 = norm.pdf(d1)
        sqrtT  = np.sqrt(T)

        # Delta ∈ [-1, 1] — sin escalar
        delta = norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1.0

        # Gamma por $1 — sin escalar
        gamma = pdf_d1 / (S * sigma * sqrtT)

        # Vega por +1% de IV
        vega  = S * pdf_d1 * sqrtT * 0.01

        # Theta diario
        theta_common = -(S * pdf_d1 * sigma) / (2.0 * sqrtT)
        if option_type == "call":
            theta = (theta_common - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
        else:
            theta = (theta_common + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0

        # Rho por +1%
        if option_type == "call":
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01
        else:
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01

        return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

    except Exception:
        return zero


# ════════════════════════════════════════════════════════════════
#  MARGEN ESTILO INTERACTIVE BROKERS
# ════════════════════════════════════════════════════════════════
def strategy_margin_ib(ops: list, S: float = None, method: str = "reg_t") -> dict:
    """
    Calcula margen de estrategia combinada siguiendo metodología IB Reg-T.

    Devuelve:
      margin_combined  — margen neto de la estrategia
      margin_individual — suma de márgenes individuales
      saving           — ahorro vs. márgenes individuales
      breakdown        — descripción del cálculo
    """
    if not ops:
        return {"margin_combined": 0.0, "margin_individual": 0.0, "saving": 0.0, "breakdown": "Sin posiciones"}

    # Tickers únicos y precio de referencia
    tickers = list({op["Ticker"] for op in ops})
    if S is None:
        info = get_stock_info(ops[0]["Ticker"])
        S = info["price"] or float(ops[0]["Strike"])

    ventas = [op for op in ops if op["Dirección"] == "Venta"]
    compras = [op for op in ops if op["Dirección"] == "Compra"]

    # Margen individual (suma pata a pata)
    margin_ind = sum(op_margin(op, S) for op in ops)

    # Identificar tipo de estrategia para aplicar regla de margen
    strikes   = [float(op["Strike"]) for op in ops]
    dirs      = [op["Dirección"] for op in ops]
    tipos     = [op["Tipo"] for op in ops]
    cantidades = [int(op["Cantidad"]) for op in ops]

    strat = strategy_type(strikes, dirs, tipos)
    n     = len(ops)

    # ── Spreads verticales (Bull/Bear Call/Put Spread) ────────────
    if n == 2 and len(ventas) == 1 and len(compras) == 1:
        # Mismo tipo (call-call o put-put): margen = diferencia de strikes
        if tipos[0] == tipos[1]:
            k_sorted = sorted(zip(strikes, dirs, cantidades), key=lambda x: x[0])
            k_low, d_low, qty_low = k_sorted[0]
            k_high, d_high, qty_high = k_sorted[1]
            qty = max(qty_low, qty_high)

            if tipos[0] == "call":
                # Bear call spread (venta call bajo, compra call alto)
                # Margen = max(diferencia strikes, 0) * 100 * qty
                margin_comb = (k_high - k_low) * 100 * qty
                breakdown = (f"Bear Call Spread: (K_alto - K_bajo) × 100 × contratos\n"
                             f"= ({k_high:.2f} - {k_low:.2f}) × 100 × {qty} = {margin_comb:,.2f}$")
            else:
                # Bull put spread (venta put alto, compra put bajo)
                margin_comb = (k_high - k_low) * 100 * qty
                breakdown = (f"Bull Put Spread: (K_alto - K_bajo) × 100 × contratos\n"
                             f"= ({k_high:.2f} - {k_low:.2f}) × 100 × {qty} = {margin_comb:,.2f}$")
        else:
            # Call + Put diferentes tipos
            margin_comb = margin_ind
            breakdown = "Combinación call+put: suma de márgenes individuales"

    # ── Iron Condor ───────────────────────────────────────────────
    elif n == 4 and sum(1 for d in dirs if d == "Venta") == 2 and sum(1 for t in tipos if t == "call") == 2:
        call_ops = [(float(op["Strike"]), op["Dirección"], int(op["Cantidad"]))
                    for op in ops if op["Tipo"] == "call"]
        put_ops  = [(float(op["Strike"]), op["Dirección"], int(op["Cantidad"]))
                    for op in ops if op["Tipo"] == "put"]

        call_sorted = sorted(call_ops, key=lambda x: x[0])
        put_sorted  = sorted(put_ops,  key=lambda x: x[0])

        # Ancho de los spreads
        call_width = call_sorted[1][0] - call_sorted[0][0]
        put_width  = put_sorted[1][0]  - put_sorted[0][0]
        qty = max(c[2] for c in call_ops + put_ops)

        # IB: margen = max(call_width, put_width) * 100 * qty
        max_width  = max(call_width, put_width)
        margin_comb = max_width * 100 * qty
        breakdown = (f"Iron Condor: max(call_width, put_width) × 100 × contratos\n"
                     f"Call spread: {call_width:.2f} | Put spread: {put_width:.2f}\n"
                     f"= {max_width:.2f} × 100 × {qty} = {margin_comb:,.2f}$")

    # ── Iron Butterfly ────────────────────────────────────────────
    elif n == 4 and "Iron Butterfly" in strat:
        strike_vals = sorted(set(strikes))
        if len(strike_vals) >= 2:
            wing_width = max(strike_vals) - min(strike_vals)
            qty = max(cantidades)
            margin_comb = wing_width * 100 * qty
            breakdown = (f"Iron Butterfly: wing_width × 100 × contratos\n"
                         f"= {wing_width:.2f} × 100 × {qty} = {margin_comb:,.2f}$")
        else:
            margin_comb = margin_ind
            breakdown = "Iron Butterfly: suma de márgenes individuales"

    # ── Straddle / Strangle vendido ────────────────────────────────
    elif n == 2 and len(ventas) == 2:
        # IB: margen del lado mayor + prima del lado menor
        margins_per_leg = []
        for op in ventas:
            K_op  = float(op["Strike"])
            p_op  = float(op.get("PrecioEntrada", op.get("Prima", 0)))
            qty_op = int(op["Cantidad"])
            if op["Tipo"] == "call":
                otm = max(K_op - S, 0)
                m   = max(0.20 * S - otm, 0.10 * S) + p_op
            else:
                otm = max(S - K_op, 0)
                m   = max(0.20 * S - otm, 0.10 * K_op) + p_op
            margins_per_leg.append((m * qty_op * 100, p_op * qty_op * 100, op["Tipo"]))

        margins_per_leg.sort(key=lambda x: x[0], reverse=True)
        major_margin = margins_per_leg[0][0]
        minor_credit = margins_per_leg[1][1]
        margin_comb  = major_margin + minor_credit
        breakdown = (f"Straddle/Strangle vendido: margen_lado_mayor + prima_lado_menor\n"
                     f"Lado mayor ({margins_per_leg[0][2]}): {major_margin:,.2f}$\n"
                     f"Prima lado menor: {minor_credit:,.2f}$\n"
                     f"= {margin_comb:,.2f}$")

    # ── Mariposa vendida ───────────────────────────────────────────
    elif "Mariposa" in strat:
        k_sorted = sorted(zip(strikes, dirs, tipos, cantidades), key=lambda x: x[0])
        wings = [x for x in k_sorted if x[1] == "Venta"]
        if len(wings) >= 1:
            wing_strike = wings[0][0]
            body_strikes = [x[0] for x in k_sorted if x[1] == "Compra"]
            if body_strikes:
                max_loss_per = abs(wing_strike - body_strikes[0])
                qty = max(cantidades)
                margin_comb = max_loss_per * 100 * qty
                breakdown = f"Mariposa: pérdida máxima × 100 × contratos = {margin_comb:,.2f}$"
            else:
                margin_comb = margin_ind
                breakdown = "Mariposa: suma márgenes individuales"
        else:
            margin_comb = margin_ind
            breakdown = "Mariposa: suma márgenes individuales"

    # ── Posición simple o sin identificar ─────────────────────────
    else:
        margin_comb = margin_ind
        breakdown = f"{strat}: suma de márgenes por pata"

    saving = max(margin_ind - margin_comb, 0)

    return {
        "margin_combined":    round(margin_comb, 2),
        "margin_individual":  round(margin_ind, 2),
        "saving":             round(saving, 2),
        "breakdown":          breakdown,
        "strategy_name":      strat,
    }


# ════════════════════════════════════════════════════════════════
#  UTILIDADES DE FECHAS
# ════════════════════════════════════════════════════════════════
def days_to_expiry(expiry) -> int:
    hoy = datetime.now().date()
    if isinstance(expiry, str):
        expiry = datetime.strptime(expiry, "%Y-%m-%d").date()
    return max((expiry - hoy).days, 0)


# ════════════════════════════════════════════════════════════════
#  MÉTRICAS DE UNA OPERACIÓN INDIVIDUAL
# ════════════════════════════════════════════════════════════════
def op_breakeven(op: dict) -> float:
    K     = float(op["Strike"])
    prima = float(op.get("PrecioEntrada", op.get("Prima", 0)))
    return round(K + prima if op["Tipo"] == "call" else K - prima, 4)


def op_prob_profit(op: dict, S: float, sigma: float) -> float:
    """Probabilidad de profit a vencimiento. Retorna float [0,1]."""
    try:
        be  = op_breakeven(op)
        dte = days_to_expiry(op["Vencimiento"])
        T   = max(dte / 365.0, 1e-6)
        if be <= 0:
            return 0.0
        d2 = (np.log(S / be) - 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        tipo      = op["Tipo"]
        direccion = op["Dirección"]
        if tipo == "call":
            return float(norm.cdf(d2) if direccion == "Compra" else norm.cdf(-d2))
        else:
            return float(norm.cdf(-d2) if direccion == "Compra" else norm.cdf(d2))
    except Exception:
        return 0.0


def op_max_return(op: dict):
    """
    Máximo retorno a vencimiento por posición completa (qty × 100).
    Compra call : ilimitado al alza
    Compra put  : (K - prima) × mult  — subyacente cae a 0
    Venta call/put : prima cobrada × mult
    """
    K         = float(op["Strike"])
    prima     = float(op.get("PrecioEntrada", op.get("Prima", 0)))
    cantidad  = int(op["Cantidad"])
    mult      = cantidad * 100
    tipo      = op["Tipo"]
    direccion = op["Dirección"]
    if direccion == "Compra":
        return "Ilimitado" if tipo == "call" else round(max(K - prima, 0) * mult, 2)
    else:
        # Vendedor: gana si expira OTM → se queda toda la prima
        return round(prima * mult, 2)


def op_max_risk(op: dict):
    """
    Máximo riesgo a vencimiento por posición completa (qty × 100).
    Compra call/put : prima pagada (pérdida total si expira OTM)
    Venta call      : ilimitado al alza
    Venta put       : (K - prima_cobrada) × mult  — subyacente cae a 0
    """
    K        = float(op["Strike"])
    prima    = float(op.get("PrecioEntrada", op.get("Prima", 0)))
    cantidad = int(op["Cantidad"])
    mult     = cantidad * 100
    tipo     = op["Tipo"]
    direccion = op["Dirección"]
    if direccion == "Compra":
        # Pérdida máxima = prima pagada
        return round(prima * mult, 2)
    else:
        if tipo == "call":
            return "Ilimitado"          # Call vendida: riesgo ilimitado al alza
        else:
            # Put vendida: subyacente → 0, pero ya cobré la prima
            return round(max(K - prima, 0) * mult, 2)


def op_entry_cost_credit(op: dict) -> float:
    prima    = float(op.get("PrecioEntrada", op.get("Prima", 0)))
    cantidad = int(op["Cantidad"])
    mult     = cantidad * 100
    return round(-prima * mult if op["Dirección"] == "Compra" else prima * mult, 2)


def op_margin(op: dict, S=None) -> float:
    """Margen CBOE/MEFF simplificado para opciones vendidas."""
    if op["Dirección"] != "Venta":
        return 0.0
    try:
        K        = float(op["Strike"])
        prima    = float(op.get("PrecioEntrada", op.get("Prima", 0)))
        cantidad = int(op["Cantidad"])
        if S is None:
            info = get_stock_info(op["Ticker"])
            S = info["price"] or K
        if op["Tipo"] == "call":
            otm_amount = max(K - S, 0)
            margin_per = max(0.20 * S - otm_amount, 0.10 * S) + prima
        else:
            otm_amount = max(S - K, 0)
            margin_per = max(0.20 * S - otm_amount, 0.10 * K) + prima
        return round(margin_per * cantidad * 100, 2)
    except Exception:
        return 0.0


def op_pnl_market(op: dict, market_price):
    """PnL no realizado usando precio de mercado actual de la opción."""
    if market_price is None:
        return "N/D"
    prima    = float(op.get("PrecioEntrada", op.get("Prima", 0)))
    cantidad = int(op["Cantidad"])
    mult     = cantidad * 100
    diff     = market_price - prima
    sign     = 1 if op["Dirección"] == "Compra" else -1
    return round(diff * sign * mult, 2)


def op_expired_pnl(op: dict):
    """PnL realizado para opciones ya vencidas."""
    try:
        expiry = datetime.strptime(op["Vencimiento"], "%Y-%m-%d").date()
        S_exp  = get_historical_close(op["Ticker"], expiry)
        if S_exp is None:
            return "N/D"
        K        = float(op["Strike"])
        prima    = float(op.get("PrecioEntrada", op.get("Prima", 0)))
        cantidad = int(op["Cantidad"])
        tipo     = op["Tipo"]
        sign     = 1 if op["Dirección"] == "Compra" else -1
        intrinsic = max(S_exp - K, 0) if tipo == "call" else max(K - S_exp, 0)
        return round((intrinsic - prima) * sign * cantidad * 100, 2)
    except Exception as e:
        return f"Error: {e}"


def op_otm_pct(op: dict, S: float) -> str:
    """Porcentaje OTM/ITM del strike respecto al precio actual."""
    try:
        K = float(op["Strike"])
        if S and S > 0:
            pct = (K - S) / S * 100
            label = "ITM" if (
                (op["Tipo"] == "call" and S > K) or
                (op["Tipo"] == "put"  and S < K)
            ) else "OTM"
            return f"{label} {abs(pct):.1f}%"
        return "—"
    except Exception:
        return "—"


# ════════════════════════════════════════════════════════════════
#  MÉTRICAS DE ESTRATEGIA (VECTORIZADAS)
# ════════════════════════════════════════════════════════════════
def _strat_pnl(strikes, primas, direcciones, cantidades, tipos, S_arr):
    S_arr = np.asarray(S_arr, dtype=float)
    total = np.zeros_like(S_arr)
    for K, p, d, n, t in zip(strikes, primas, direcciones, cantidades, tipos):
        intr = np.maximum(S_arr - K, 0) if t == "call" else np.maximum(K - S_arr, 0)
        sign = 1 if d == "Compra" else -1
        total += (intr - p) * sign * n * 100
    return total


def strategy_breakeven(strikes, primas, direcciones, cantidades, tipos):
    S_range = np.linspace(max(min(strikes) * 0.2, 0.01), max(strikes) * 2.5, 5000)
    pnl     = _strat_pnl(strikes, primas, direcciones, cantidades, tipos, S_range)
    bes = []
    for i in range(1, len(pnl)):
        if pnl[i-1] * pnl[i] < 0:
            be = S_range[i-1] - pnl[i-1] * (S_range[i] - S_range[i-1]) / (pnl[i] - pnl[i-1])
            bes.append(round(be, 2))
        elif pnl[i] == 0:
            bes.append(round(float(S_range[i]), 2))
    unique_bes = sorted(set(bes))
    if not unique_bes:
        return "N/D"
    return unique_bes[0] if len(unique_bes) == 1 else ", ".join(str(b) for b in unique_bes)


def strategy_prob_profit(strikes, primas, direcciones, cantidades, tipos, S0, vencimiento, sigma=0.25):
    """Monte Carlo con semilla fija."""
    try:
        dte = days_to_expiry(vencimiento)
        if dte <= 0:
            return 0.0   # Retornar float, no string
        T   = dte / 365.0
        rng = np.random.default_rng(seed=42)
        z   = rng.standard_normal(10_000)
        S_T = S0 * np.exp((-0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
        pnl = _strat_pnl(strikes, primas, direcciones, cantidades, tipos, S_T)
        return round(float(np.mean(pnl > 0) * 100), 1)
    except Exception:
        return 0.0


def strategy_risk_reward(strikes, primas, direcciones, cantidades, tipos):
    S_range = np.linspace(max(min(strikes) * 0.1, 0.01), max(strikes) * 3.5, 5000)
    pnl     = _strat_pnl(strikes, primas, direcciones, cantidades, tipos, S_range)
    max_ret = float(np.max(pnl))
    max_rsk = float(np.min(pnl))
    has_naked_call = any(d == "Venta" and t == "call" for d, t in zip(direcciones, tipos))
    max_rsk_str    = "Ilimitado" if (has_naked_call and pnl[-1] < pnl[-5]) else round(max_rsk, 2)
    has_long_call  = any(d == "Compra" and t == "call" for d, t in zip(direcciones, tipos))
    max_ret_str    = "Ilimitado" if (has_long_call and pnl[-1] > pnl[-5]) else round(max_ret, 2)
    return max_ret_str, max_rsk_str


def strategy_entry_credit(primas, direcciones, cantidades) -> float:
    total = sum(
        (-p * n * 100) if d == "Compra" else (p * n * 100)
        for p, d, n in zip(primas, direcciones, cantidades)
    )
    return round(total, 2)


def strategy_type(strikes, direcciones, tipos) -> str:
    compras = sum(1 for d in direcciones if d == "Compra")
    ventas  = len(direcciones) - compras
    calls   = sum(1 for t in tipos if t == "call")
    puts    = len(tipos) - calls
    n       = len(strikes)

    if n == 1:
        return f"{'Compra' if compras else 'Venta'} {tipos[0].capitalize()}"
    if n == 2:
        if compras == 1 and ventas == 1:
            if calls == 2:
                sorted_k = sorted(zip(strikes, direcciones), key=lambda x: x[0])
                return "Bull Call Spread" if sorted_k[0][1] == "Compra" else "Bear Call Spread"
            if puts  == 2:
                sorted_k = sorted(zip(strikes, direcciones), key=lambda x: x[0])
                return "Bull Put Spread" if sorted_k[1][1] == "Compra" else "Bear Put Spread"
            return "Combinación Call+Put"
        if compras == 2: return "Backspread"
        if ventas  == 2:
            if calls == 1 and puts == 1:
                k_sorted = sorted(zip(strikes, tipos), key=lambda x: x[0])
                return "Strangle vendido" if k_sorted[0][0] != k_sorted[1][0] else "Straddle vendido"
            return "Ratio Spread"
    if n == 3:
        if compras == 1 and ventas == 2: return "Mariposa vendida"
        if compras == 2 and ventas == 1: return "Mariposa comprada"
    if n == 4:
        if compras == 2 and ventas == 2:
            return "Iron Condor" if calls == 2 and puts == 2 else "Iron Butterfly"
    return f"Estrategia {compras}C/{ventas}V"


# ════════════════════════════════════════════════════════════════
#  PERSISTENCIA DE CARTERAS
# ════════════════════════════════════════════════════════════════
PORTFOLIO_FILE = "saved_portfolios.json"

def _normalize_op(op: dict) -> dict:
    op = dict(op)
    op.setdefault("PrecioEntrada", op.get("Prima", 0))
    op.setdefault("FechaEntrada",  op.get("Fecha", str(date.today())))
    op.setdefault("IV", 0.25)
    return op

def save_portfolio(name: str, portfolio: list):
    st.session_state["saved_portfolios"][name] = [_normalize_op(op) for op in portfolio]
    with open(PORTFOLIO_FILE, "w") as f:
        json.dump(st.session_state["saved_portfolios"], f, indent=2)

def load_portfolios():
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE) as f:
                data = json.load(f)
            st.session_state["saved_portfolios"] = {
                name: [_normalize_op(op) for op in ops]
                for name, ops in data.items()
            }
        except Exception:
            st.session_state["saved_portfolios"] = {}

load_portfolios()

# ════════════════════════════════════════════════════════════════
#  VALIDACIÓN DE INPUTS
# ════════════════════════════════════════════════════════════════
def validate_operation(ticker, strike, prima, cantidad, vencimiento) -> list:
    errors = []
    if not ticker:
        errors.append("El ticker no puede estar vacío.")
    if strike <= 0:
        errors.append("El strike debe ser mayor que 0.")
    if prima < 0:
        errors.append("La prima no puede ser negativa.")
    if cantidad < 1:
        errors.append("La cantidad debe ser al menos 1 contrato.")
    if isinstance(vencimiento, date) and vencimiento <= date.today():
        errors.append("El vencimiento debe ser una fecha futura.")
    return errors


# ════════════════════════════════════════════════════════════════
#  HELPERS DE FORMATO Y COLOR
# ════════════════════════════════════════════════════════════════
def fmt(val, dec=2) -> str:
    if isinstance(val, (int, float)):
        return f"{val:,.{dec}f}"
    return str(val)

def color_pnl(val) -> str:
    if isinstance(val, (int, float)):
        base = "font-family: 'IBM Plex Mono', monospace; font-weight: 600; font-size: 12px"
        if val > 0:  return f"color: #16a34a; {base}"
        if val < 0:  return f"color: #dc2626; {base}"
    return "font-family: 'IBM Plex Mono', monospace"

def color_prob(val) -> str:
    if isinstance(val, (int, float)):
        base = "font-family: 'IBM Plex Mono', monospace; font-weight: 700; border-radius: 4px; padding: 2px 6px; font-size: 11px"
        if val >= 65: return f"background-color: #dcfce7; color: #14532d; {base}"
        if val >= 50: return f"background-color: #fef9c3; color: #713f12; {base}"
        return f"background-color: #fee2e2; color: #7f1d1d; {base}"
    return ""

def color_dir(val) -> str:
    base = "font-weight: 700; font-size: 11px; border-radius: 4px; padding: 2px 8px"
    if val == "Compra": return f"background-color: #dbeafe; color: #1d4ed8; {base}"
    if val == "Venta":  return f"background-color: #fef3c7; color: #b45309; {base}"
    return ""

def color_tipo(val) -> str:
    base = "font-weight: 700; font-size: 11px; font-family: 'IBM Plex Mono', monospace; border-radius: 4px; padding: 2px 7px"
    if val == "call": return f"background-color: #e0f2fe; color: #0369a1; {base}"
    if val == "put":  return f"background-color: #ede9fe; color: #7c3aed; {base}"
    return ""

def color_otm(val) -> str:
    if not isinstance(val, str):
        return ""
    base = "font-family: 'IBM Plex Mono', monospace; font-size: 11px; font-weight: 600"
    if val.startswith("ITM"): return f"color: #0369a1; {base}"
    if val.startswith("OTM"): return f"color: #64748b; {base}"
    return ""

def style_table(df, pnl_cols=None, prob_cols=None, dir_cols=None, tipo_cols=None, otm_cols=None):
    """Aplica estilos premium a todas las tablas."""
    s = df.style.set_table_styles([
        # Header
        {"selector": "thead tr th", "props": [
            ("background", "linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%)"),
            ("color", "#e2e8f0"),
            ("font-family", "'IBM Plex Mono', monospace"),
            ("font-size", "10px"),
            ("font-weight", "600"),
            ("text-transform", "uppercase"),
            ("letter-spacing", "0.09em"),
            ("padding", "10px 14px"),
            ("border-bottom", "2px solid #0ea5e9"),
            ("white-space", "nowrap"),
        ]},
        # Rows alternos
        {"selector": "tbody tr:nth-child(odd)",  "props": [("background", "#ffffff")]},
        {"selector": "tbody tr:nth-child(even)", "props": [("background", "#f8fafc")]},
        # Hover
        {"selector": "tbody tr:hover td", "props": [
            ("background", "#e0f2fe !important"),
            ("transition", "background 0.12s ease"),
        ]},
        # Celdas
        {"selector": "tbody td", "props": [
            ("padding", "8px 14px"),
            ("font-family", "'IBM Plex Sans', sans-serif"),
            ("font-size", "12px"),
            ("color", "#1e293b"),
            ("border-bottom", "1px solid #f1f5f9"),
            ("vertical-align", "middle"),
        ]},
        # Última fila (totales si la hay)
        {"selector": "tbody tr:last-child td", "props": [
            ("border-bottom", "2px solid #0ea5e9"),
            ("font-weight", "600"),
        ]},
        # Bordes exteriores
        {"selector": "table", "props": [
            ("border-collapse", "collapse"),
            ("border-radius", "8px"),
            ("overflow", "hidden"),
            ("border", "1px solid #e2e8f0"),
            ("width", "100%"),
        ]},
    ])
    if pnl_cols:
        for c in pnl_cols:
            if c in df.columns:
                s = s.map(color_pnl, subset=[c])
    if prob_cols:
        for c in prob_cols:
            if c in df.columns:
                s = s.map(color_prob, subset=[c])
    if dir_cols:
        for c in dir_cols:
            if c in df.columns:
                s = s.map(color_dir, subset=[c])
    if tipo_cols:
        for c in tipo_cols:
            if c in df.columns:
                s = s.map(color_tipo, subset=[c])
    if otm_cols:
        for c in otm_cols:
            if c in df.columns:
                s = s.map(color_otm, subset=[c])
    return s


# ════════════════════════════════════════════════════════════════
#  HELPERS DE GRÁFICOS
# ════════════════════════════════════════════════════════════════
def add_chart_references(fig, strikes, ref_price=None, color_strike="#94a3b8", color_price="#f59e0b"):
    seen = set()
    for k in strikes:
        if k not in seen:
            fig.add_vline(x=k, line_dash="dot", line_color=color_strike, line_width=1,
                          annotation_text=f"K={k:.0f}", annotation_position="top",
                          annotation_font_size=10, annotation_font_color=color_strike)
            seen.add(k)
    if ref_price:
        fig.add_vline(x=ref_price, line_dash="dash", line_color=color_price, line_width=2,
                      annotation_text=f"S={ref_price:.2f}", annotation_position="top right",
                      annotation_font_size=10, annotation_font_color=color_price)

PLOTLY_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="IBM Plex Sans, system-ui, sans-serif", size=12, color="#1e293b"),
    xaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False, showline=True,
               linecolor="#e2e8f0", tickfont=dict(family="IBM Plex Mono", size=11)),
    yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False, showline=True,
               linecolor="#e2e8f0", tickfont=dict(family="IBM Plex Mono", size=11)),
    legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#e2e8f0", borderwidth=1,
                font=dict(size=11, family="IBM Plex Sans")),
    margin=dict(l=60, r=20, t=50, b=60),
    hovermode="x unified",
)


# ════════════════════════════════════════════════════════════════
#  GRÁFICO RADAR DE GRIEGAS
# ════════════════════════════════════════════════════════════════
def plot_greeks_radar(tot_greeks: dict, n_contracts: int) -> go.Figure:
    """Radar chart normalizado de las 5 griegas de la cartera."""
    delta_norm = tot_greeks["Delta"] / (n_contracts * 100) if n_contracts > 0 else 0

    # Normalizar a escala [-1, 1] para visualización
    values_raw = {
        "Δ Delta":  delta_norm,
        "Γ Gamma":  tot_greeks["Gamma"],
        "ν Vega":   tot_greeks["Vega"],
        "Θ Theta":  tot_greeks["Theta"],
        "ρ Rho":    tot_greeks["Rho"],
    }

    # Normalización min-max para el radar (solo forma, no magnitud)
    abs_max = max(abs(v) for v in values_raw.values()) or 1
    categories = list(values_raw.keys())
    values_norm = [abs(v) / abs_max for v in values_raw.values()]
    colors_sign = ["#16a34a" if v >= 0 else "#dc2626" for v in values_raw.values()]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_norm + [values_norm[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(14, 165, 233, 0.12)",
        line=dict(color="#0ea5e9", width=2),
        name="Perfil de Griegas",
        hovertemplate="<b>%{theta}</b><br>Valor: %{r:.3f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(family="IBM Plex Mono", size=11)),
            bgcolor="white",
        ),
        showlegend=False,
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=40, b=40),
        height=280,
    )
    return fig


# ════════════════════════════════════════════════════════════════
#  HEATMAP PnL POR TICKER × DTE
# ════════════════════════════════════════════════════════════════
def plot_pnl_by_ticker(rows: list) -> go.Figure:
    """Barras de PnL agrupadas por operación individual."""
    pnl_data = [
        {"label": f"{r['Ticker']} K={r['Strike']:.0f} {r['Dir.']}", "pnl": r["PnL $"]}
        for r in rows if isinstance(r["PnL $"], (int, float))
    ]
    if not pnl_data:
        return None
    labels = [d["label"] for d in pnl_data]
    pnls   = [d["pnl"]   for d in pnl_data]
    colors = ["#16a34a" if p >= 0 else "#dc2626" for p in pnls]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=pnls,
        marker_color=colors, marker_line_width=0,
        text=[f"${p:+,.0f}" for p in pnls],
        textposition="outside",
        textfont=dict(family="IBM Plex Mono", size=11),
        hovertemplate="<b>%{x}</b><br>PnL: $%{y:,.2f}<extra></extra>",
        name="PnL por posición",
    ))
    fig.add_hline(y=0, line_color="#94a3b8", line_width=1, line_dash="dash")
    fig.update_layout(
        title=dict(text="PnL por Posición", font=dict(size=13, family="IBM Plex Mono")),
        showlegend=False, height=260,
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="IBM Plex Sans", size=12, color="#1e293b"),
        xaxis=dict(showgrid=False, tickangle=-30, tickfont=dict(family="IBM Plex Mono", size=10)),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9", zeroline=False,
                   tickfont=dict(family="IBM Plex Mono", size=10)),
        margin=dict(l=50, r=20, t=45, b=70),
    )
    return fig


def plot_pnl_heatmap(rows: list) -> go.Figure:
    """Heatmap de PnL por ticker."""
    pnl_data = [(r["Ticker"], r["DTE"], r["PnL $"]) for r in rows if isinstance(r["PnL $"], (int, float))]
    if not pnl_data:
        return None

    df = pd.DataFrame(pnl_data, columns=["Ticker", "DTE", "PnL"])
    df["DTE_bucket"] = pd.cut(df["DTE"], bins=[0, 7, 30, 60, 90, 365],
                               labels=["0-7d", "8-30d", "31-60d", "61-90d", ">90d"])
    pivot = df.groupby(["Ticker", "DTE_bucket"])["PnL"].sum().unstack(fill_value=0)

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0, "#dc2626"], [0.5, "#f1f5f9"], [1, "#16a34a"]],
        zmid=0,
        text=[[f"${v:,.0f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=11, family="IBM Plex Mono"),
        hovertemplate="Ticker: %{y}<br>DTE: %{x}<br>PnL: %{text}<extra></extra>",
        showscale=True,
        colorbar=dict(title="PnL $", tickfont=dict(size=10, family="IBM Plex Mono")),
    ))
    safe_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "hovermode")}
    fig.update_layout(
        title=dict(text="Mapa de PnL por Ticker × DTE", font=dict(size=13, family="IBM Plex Mono")),
        xaxis=dict(title="Bucket DTE", tickfont=dict(family="IBM Plex Mono", size=10)),
        yaxis=dict(title="Ticker",     tickfont=dict(family="IBM Plex Mono", size=10)),
        height=max(220, len(pivot) * 55 + 80),
        **safe_layout,
    )
    return fig


# ════════════════════════════════════════════════════════════════
#  GAUGE DELTA NETO
# ════════════════════════════════════════════════════════════════
def plot_delta_gauge(delta_norm: float) -> go.Figure:
    """Gauge de exposición delta neta [-1, 1]."""
    color = "#16a34a" if delta_norm > 0.1 else ("#dc2626" if delta_norm < -0.1 else "#f59e0b")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(delta_norm, 3),
        number=dict(font=dict(family="IBM Plex Mono", size=22, color=color), suffix=""),
        gauge=dict(
            axis=dict(range=[-1, 1], tickwidth=1, tickcolor="#94a3b8",
                      tickfont=dict(family="IBM Plex Mono", size=10)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="white",
            borderwidth=1,
            bordercolor="#e2e8f0",
            steps=[
                dict(range=[-1, -0.3], color="#fee2e2"),
                dict(range=[-0.3, 0.3], color="#fef9c3"),
                dict(range=[0.3, 1],   color="#dcfce7"),
            ],
            threshold=dict(line=dict(color="#0f172a", width=2), thickness=0.75, value=delta_norm),
        ),
        title=dict(text="Δ Delta Neto", font=dict(family="IBM Plex Mono", size=12, color="#64748b")),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(
        height=200,
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=30, b=10),
    )
    return fig


# ════════════════════════════════════════════════════════════════
#  VOLATILITY CONE
# ════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def compute_vol_cone(ticker: str) -> dict:
    """Calcula el cono de volatilidad histórica para 10/20/30/60/90 días."""
    try:
        hist = yf.Ticker(ticker).history(period="2y")
        if len(hist) < 90:
            return {}
        log_ret = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        cone = {}
        for w in [10, 20, 30, 60, 90]:
            rv = log_ret.rolling(w).std() * np.sqrt(252)
            rv = rv.dropna()
            cone[w] = {
                "min":    float(rv.min()),
                "p25":    float(rv.quantile(0.25)),
                "median": float(rv.median()),
                "p75":    float(rv.quantile(0.75)),
                "max":    float(rv.max()),
                "current": float(rv.iloc[-1]),
            }
        return cone
    except Exception:
        return {}


def plot_vol_cone(ticker: str, current_iv: float = None) -> go.Figure:
    """Gráfico del cono de volatilidad."""
    cone = compute_vol_cone(ticker)
    if not cone:
        return None

    windows = list(cone.keys())
    fig = go.Figure()

    # Rango (min-max)
    fig.add_trace(go.Scatter(
        x=windows + windows[::-1],
        y=[cone[w]["max"] * 100 for w in windows] + [cone[w]["min"] * 100 for w in windows[::-1]],
        fill="toself", fillcolor="rgba(148,163,184,0.15)",
        line=dict(color="rgba(148,163,184,0)"),
        name="Rango Min-Max", showlegend=True,
    ))
    # P25-P75
    fig.add_trace(go.Scatter(
        x=windows + windows[::-1],
        y=[cone[w]["p75"] * 100 for w in windows] + [cone[w]["p25"] * 100 for w in windows[::-1]],
        fill="toself", fillcolor="rgba(14,165,233,0.15)",
        line=dict(color="rgba(14,165,233,0)"),
        name="Rango P25-P75",
    ))
    # Mediana
    fig.add_trace(go.Scatter(
        x=windows, y=[cone[w]["median"] * 100 for w in windows],
        mode="lines+markers", name="Mediana HV",
        line=dict(color="#0ea5e9", width=2, dash="dash"),
        marker=dict(size=6),
    ))
    # Actual HV
    fig.add_trace(go.Scatter(
        x=windows, y=[cone[w]["current"] * 100 for w in windows],
        mode="lines+markers", name="HV Actual",
        line=dict(color="#0f172a", width=2),
        marker=dict(size=7, symbol="circle"),
    ))
    # IV actual (línea horizontal)
    if current_iv:
        fig.add_hline(y=current_iv * 100, line_dash="dot", line_color="#f59e0b", line_width=2,
                      annotation_text=f"IV={current_iv*100:.1f}%", annotation_position="right",
                      annotation_font_color="#f59e0b")

    layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("xaxis", "yaxis", "xaxis_title", "yaxis_title")}
    fig.update_layout(
        title=dict(text=f"Cono de Volatilidad — {ticker}", font=dict(size=13, family="IBM Plex Mono")),
        xaxis=dict(
            tickvals=windows, ticktext=[f"{w}d" for w in windows],
            title="Ventana (días)",
            showgrid=True, gridcolor="#f1f5f9", zeroline=False, showline=True,
            linecolor="#e2e8f0", tickfont=dict(family="IBM Plex Mono", size=11),
        ),
        yaxis=dict(
            title="Volatilidad %",
            showgrid=True, gridcolor="#f1f5f9", zeroline=False, showline=True,
            linecolor="#e2e8f0", tickfont=dict(family="IBM Plex Mono", size=11),
        ),
        **layout,
    )
    return fig


# ════════════════════════════════════════════════════════════════
#  WHAT-IF INTERACTIVO
# ════════════════════════════════════════════════════════════════
def compute_whatif_pnl(cartera: list, price_shock_pct: float, iv_shock_pct: float,
                        days_forward: int, precios_sub: dict) -> float:
    """PnL total de la cartera bajo escenarios de precio, IV y tiempo."""
    total = 0.0
    for op in cartera:
        S_base = precios_sub.get(op["Ticker"])
        if S_base is None:
            continue
        K     = float(op["Strike"])
        prima = float(op.get("PrecioEntrada", op.get("Prima", 0)))
        qty   = int(op["Cantidad"])
        sigma = float(op.get("IV") or 0.25)
        dte   = days_to_expiry(op["Vencimiento"])
        sign  = 1 if op["Dirección"] == "Compra" else -1

        S_new     = S_base * (1 + price_shock_pct / 100)
        sigma_new = max(sigma + iv_shock_pct / 100, 0.01)
        T_new     = max((dte - days_forward) / 365.0, 1e-6)

        new_price = bs_price(S_new, K, T_new, 0.05, sigma_new, op["Tipo"])
        total    += (new_price - prima) * sign * qty * 100
    return round(total, 2)


# ════════════════════════════════════════════════════════════════
#  INTERFAZ PRINCIPAL
# ════════════════════════════════════════════════════════════════
st.title("📊 Modelo de Opciones y Seguimiento de Cartera")

tab1, tab2, tab3, tab4 = st.tabs([
    "🔗 Cadena de Opciones",
    "💼 Cartera",
    "📈 Gráficos de Sensibilidad",
    "📋 Historial de Posiciones",
])

# ════════════════════════════════════════════════════════════════
#  TAB 1 — CADENA DE OPCIONES
# ════════════════════════════════════════════════════════════════
with tab1:
    st.header("Cadena de Opciones")

    col_t, col_b = st.columns([3, 1])
    with col_t:
        ticker = st.text_input("Ticker", value="AAPL", key="ticker_input").upper().strip()
    with col_b:
        st.write("")
        consultar = st.button("Consultar", key="btn_consultar")

    if ticker:
        info = get_stock_info(ticker)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nombre",    info["name"])
        c2.metric("Precio",    fmt(info["price"]) if info["price"] else "—")
        c3.metric("Cambio %",  f"{fmt(info['change_pct'])}%" if info["change_pct"] else "—")
        if info.get("last_update"):
            c4.metric("Últ. actualización", info["last_update"].strftime("%d/%m %H:%M"))
        if not info["price"]:
            st.warning(f"No se encontraron datos para «{ticker}». Verifica el ticker.")

    st.markdown("---")
    expiries = get_available_expiries(ticker)

    if not expiries:
        st.info("Introduce un ticker válido para ver opciones.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_expiry = st.selectbox("Vencimiento", expiries, key="sel_expiry")
        with col2:
            tipo_chain = st.radio("Tipo", ["Calls", "Puts"], key="tipo_chain", horizontal=True)

        calls_df, puts_df = get_option_chain(ticker, selected_expiry)
        chain_df = calls_df if tipo_chain == "Calls" else puts_df

        if chain_df.empty:
            st.warning("No hay datos disponibles para este vencimiento.")
        else:
            # ── Enriquecer con BS price y distancia ATM ──────────
            chain_enriched = chain_df.copy()
            spot = info.get("price")
            dte_chain = days_to_expiry(selected_expiry)
            T_chain   = max(dte_chain / 365.0, 1e-6)

            if spot and "impliedVolatility" in chain_enriched.columns:
                def _bs_row(row):
                    iv_ = float(row.get("impliedVolatility") or 0.25)
                    k_  = float(row["strike"])
                    return round(bs_price(spot, k_, T_chain, 0.05, iv_,
                                          "call" if tipo_chain == "Calls" else "put"), 3)
                chain_enriched["BS Price"] = chain_enriched.apply(_bs_row, axis=1)
                chain_enriched["Dist ATM%"] = ((chain_enriched["strike"] - spot) / spot * 100).round(2)

            cols_show = [c for c in [
                "strike", "bid", "ask", "lastPrice", "BS Price",
                "impliedVolatility"
            ] if c in chain_enriched.columns]

            fmt_map = {}
            for col, fmt_str in [
                ("lastPrice", "{:.2f}"), ("bid", "{:.2f}"), ("ask", "{:.2f}"),
                ("BS Price", "{:.3f}"),
                ("impliedVolatility", "{:.1%}"),
            ]:
                if col in cols_show:
                    fmt_map[col] = fmt_str

            # Estilo: ATM highlight (amarillo) + gradiente IV
            def highlight_atm(row):
                if spot and spot > 0:
                    dist_pct = abs(float(row.get("strike", 0)) - spot) / spot * 100
                    if dist_pct < 1.5:
                        return ["background-color: #fef9c3; font-weight: 700; color: #92400e"] * len(row)
                return [""] * len(row)

            styled_chain = (
                chain_enriched[cols_show]
                .style
                .format(fmt_map)
                .apply(highlight_atm, axis=1)
                .background_gradient(
                    subset=["impliedVolatility"] if "impliedVolatility" in cols_show else [],
                    cmap="YlOrRd", vmin=0.05, vmax=1.0,
                )
                .set_table_styles([
                    {"selector": "thead tr th", "props": [
                        ("background", "linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%)"),
                        ("color", "#e2e8f0"),
                        ("font-family", "'IBM Plex Mono', monospace"),
                        ("font-size", "10px"),
                        ("font-weight", "600"),
                        ("text-transform", "uppercase"),
                        ("letter-spacing", "0.08em"),
                        ("padding", "9px 13px"),
                        ("border-bottom", "2px solid #0ea5e9"),
                        ("white-space", "nowrap"),
                    ]},
                    {"selector": "tbody tr:nth-child(even)", "props": [("background", "#f8fafc")]},
                    {"selector": "tbody tr:hover td", "props": [("background", "#e0f2fe !important")]},
                    {"selector": "tbody td", "props": [
                        ("padding", "8px 14px"),
                        ("font-family", "'IBM Plex Mono', monospace"),
                        ("font-size", "12px"),
                        ("border-bottom", "1px solid #f1f5f9"),
                        ("color", "#1e293b"),
                    ]},
                    {"selector": "table", "props": [
                        ("border-collapse", "collapse"),
                        ("border", "1px solid #e2e8f0"),
                        ("width", "100%"),
                    ]},
                ])
            )

            if spot:
                st.caption(f"Filas amarillas = ATM (S={spot:.2f})  |  DTE: {dte_chain}d")
            st.dataframe(styled_chain, use_container_width=True, height=400)


            with st.expander("➕ Añadir a cartera desde la cadena", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    dir_op   = st.radio("Dirección", ["Compra", "Venta"], key="dir_chain", horizontal=True)
                    cantidad = st.number_input("Contratos", min_value=1, value=1, step=1, key="qty_chain")
                with c2:
                    row_idx  = st.number_input("Fila", min_value=0, max_value=len(chain_enriched) - 1, value=0, key="row_chain")
                with c3:
                    iv_override = st.number_input(
                        "IV override (0 = usar mercado)", min_value=0.0, max_value=5.0,
                        value=0.0, step=0.01, key="iv_override_chain",
                    )

                if st.button("Añadir a cartera", key="btn_add_chain"):
                    row  = chain_enriched.iloc[row_idx]
                    bid_p = float(row.get("bid", 0) or 0)
                    ask_p = float(row.get("ask", 0) or 0)
                    mid   = round((bid_p + ask_p) / 2, 4) if bid_p > 0 and ask_p > 0 else float(row.get("lastPrice", 0) or 0)
                    iv_mkt = float(row.get("impliedVolatility", 0.25) or 0.25)
                    iv_use = iv_override if iv_override > 0 else iv_mkt

                    nueva_op = {
                        "Ticker":        ticker,
                        "Tipo":          "call" if tipo_chain == "Calls" else "put",
                        "Vencimiento":   selected_expiry,
                        "Strike":        float(row["strike"]),
                        "Prima":         mid,
                        "Cantidad":      int(cantidad),
                        "Dirección":     dir_op,
                        "Fecha":         str(date.today()),
                        "Desembolso":    round(mid * 100 * cantidad * (1 if dir_op == "Compra" else -1), 2),
                        "PrecioEntrada": mid,
                        "FechaEntrada":  str(date.today()),
                        "IV":            round(iv_use, 4),
                    }
                    if dir_op == "Venta":
                        s_info = get_stock_info(ticker)
                        nueva_op["MargenRequerido"] = op_margin(nueva_op, s_info["price"])

                    st.session_state["cartera"].append(nueva_op)
                    st.success(f"✅ {dir_op} de {cantidad} contratos añadida ({tipo_chain[:-1]} K={row['strike']}).")


# ════════════════════════════════════════════════════════════════
#  TAB 2 — CARTERA
# ════════════════════════════════════════════════════════════════
with tab2:
    st.header("Cartera de Opciones")
    cartera = st.session_state["cartera"]

    # ── Añadir manualmente ──────────────────────────────────────
    with st.expander("➕ Añadir operación manualmente", expanded=not bool(cartera)):
        with st.form("form_manual"):
            c1, c2, c3 = st.columns(3)
            with c1:
                m_ticker = st.text_input("Ticker", value="AAPL").upper().strip()
                m_tipo   = st.selectbox("Tipo", ["call", "put"])
                m_dir    = st.selectbox("Dirección", ["Compra", "Venta"])
            with c2:
                m_strike = st.number_input("Strike",    value=150.0, min_value=0.01)
                m_prima  = st.number_input("Prima",     value=2.0,   min_value=0.0)
                m_qty    = st.number_input("Contratos", value=1, min_value=1, step=1)
            with c3:
                m_venc   = st.date_input("Vencimiento", value=date.today() + timedelta(days=30))
                m_iv     = st.number_input("IV estimada (ej: 0.25 = 25%)", min_value=0.01, max_value=5.0, value=0.25, step=0.01)

            if st.form_submit_button("Añadir operación"):
                errors = validate_operation(m_ticker, m_strike, m_prima, m_qty, m_venc)
                if errors:
                    for e in errors:
                        st.error(e)
                else:
                    nueva_op = {
                        "Ticker":        m_ticker,
                        "Tipo":          m_tipo,
                        "Vencimiento":   str(m_venc),
                        "Strike":        m_strike,
                        "Prima":         m_prima,
                        "Cantidad":      m_qty,
                        "Dirección":     m_dir,
                        "Fecha":         str(date.today()),
                        "Desembolso":    round(m_prima * 100 * m_qty * (1 if m_dir == "Compra" else -1), 2),
                        "PrecioEntrada": m_prima,
                        "FechaEntrada":  str(date.today()),
                        "IV":            m_iv,
                    }
                    if m_dir == "Venta":
                        s_info = get_stock_info(m_ticker)
                        nueva_op["MargenRequerido"] = op_margin(nueva_op, s_info["price"])
                    st.session_state["cartera"].append(nueva_op)
                    st.success("✅ Operación añadida.")
                    st.rerun()

    if not cartera:
        st.info("La cartera está vacía. Añade operaciones desde la cadena o manualmente.")
    else:
        tickers_u    = list({op["Ticker"] for op in cartera})
        precios_sub  = {t: get_stock_info(t)["price"] for t in tickers_u}

        rows         = []
        tot_desemb   = 0.0
        tot_margen   = 0.0
        # Griegas en unidades absolutas (por opción × qty × 100)
        tot_greeks   = {g: 0.0 for g in ["Delta", "Gamma", "Vega", "Theta", "Rho"]}
        n_contracts  = sum(int(op["Cantidad"]) for op in cartera)

        for i, op in enumerate(cartera):
            t_op  = op["Ticker"]
            S     = precios_sub.get(t_op)
            K     = float(op["Strike"])
            dte   = days_to_expiry(op["Vencimiento"])
            T     = max(dte / 365.0, 1e-6)
            r     = 0.05
            sigma = float(op.get("IV") or 0) or get_implied_vol(t_op, op["Vencimiento"], K, op["Tipo"])

            mkt_price = get_market_option_price(t_op, op["Vencimiento"], K, op["Tipo"])
            pnl       = op_pnl_market(op, mkt_price)

            try:
                fe    = datetime.strptime(op["FechaEntrada"], "%Y-%m-%d").date()
                dias  = max((date.today() - fe).days, 1)
                pnl_d = round(pnl / dias, 2) if isinstance(pnl, (int, float)) else "N/D"
            except Exception:
                pnl_d = "N/D"

            sign  = 1 if op["Dirección"] == "Compra" else -1
            g     = bs_greeks(S or K, K, T, r, sigma, op["Tipo"])
            qty   = int(op["Cantidad"])
            # Acumular griegas escaladas: por opción × 100 acciones × qty × signo dirección
            for gk in tot_greeks:
                tot_greeks[gk] += g[gk] * 100 * qty * sign

            margen     = op_margin(op, S)
            tot_margen += margen
            desemb     = float(op.get("Desembolso", 0))
            tot_desemb += desemb

            # Delta unitario ∈ [-1,1]
            delta_unit  = round(g["Delta"] * sign, 4)
            theta_day   = round(g["Theta"] * 100 * sign, 4)
            otm_str     = op_otm_pct(op, S) if S else "—"
            margen_unit = round(margen / qty, 2) if margen > 0 and qty > 0 else "—"

            rows.append({
                "#":             i,
                "Ticker":        t_op,
                "Tipo":          op["Tipo"],
                "Dir.":          op["Dirección"],
                "Strike":        K,
                "OTM/ITM":       otm_str,
                "Prima entrada": float(op.get("PrecioEntrada", op.get("Prima", 0))),
                "Prima actual":  round(mkt_price, 4) if mkt_price else "N/D",
                "Qty":           qty,
                "Venc.":         op["Vencimiento"],
                "DTE":           dte,
                "PnL $":         pnl,
                "PnL $/día":     pnl_d,
                "Δ/cto":         delta_unit,
                "Θ/día $":       theta_day,
                "Margen/cto $":  margen_unit,
            })

        df_cart = pd.DataFrame(rows)
        styled = (
            style_table(df_cart, pnl_cols=["PnL $", "PnL $/día"], dir_cols=["Dir."], tipo_cols=["Tipo"], otm_cols=["OTM/ITM"])
            .format({
                "Strike":        "{:,.2f}",
                "Prima entrada": "{:.4f}",
                "Prima actual":  lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else str(x),
                "Qty":           "{:d}",
                "DTE":           "{:d}",
                "PnL $":         lambda x: f"${x:+,.2f}" if isinstance(x, (int, float)) else str(x),
                "PnL $/día":     lambda x: f"${x:+,.2f}" if isinstance(x, (int, float)) else str(x),
                "Δ/cto":         "{:+.4f}",
                "Θ/día $":       lambda x: f"${x:+.2f}" if isinstance(x, (int, float)) else str(x),
                "Margen/cto $":  lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else str(x),
            })
        )
        st.dataframe(styled, use_container_width=True)

        # ── Alertas DTE < 7 días ─────────────────────────────────
        alertas_dte = [r for r in rows if r["DTE"] <= 7]
        if alertas_dte:
            st.warning(
                f"⚠️ **{len(alertas_dte)} posición(es) vencen en ≤7 días:** " +
                ", ".join(f"{r['Ticker']} K={r['Strike']} (DTE={r['DTE']})" for r in alertas_dte) +
                " — Considera roll, cierre o dejar expirar."
            )

        # ═══════════════════════════════════════════════════════
        #  RESUMEN VISUAL COMPLETO
        # ═══════════════════════════════════════════════════════
        st.subheader("📊 Resumen de Cartera")

        pnl_vals  = [r["PnL $"] for r in rows if isinstance(r["PnL $"], (int, float))]
        pnl_total = sum(pnl_vals)
        pnl_pct   = pnl_total / abs(tot_desemb) * 100 if tot_desemb else 0
        theta_total = tot_greeks["Theta"]

        # KPIs principales
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Operaciones",       len(cartera))
        c2.metric("Desembolso neto",   f"${tot_desemb:,.0f}")
        pnl_color = "normal" if pnl_total >= 0 else "inverse"
        c3.metric("PnL Total",         f"${pnl_total:,.2f}", delta=f"{pnl_pct:.1f}%")
        c4.metric("Theta/día total",   f"${theta_total:,.2f}")
        c5.metric("Margen total",      f"${tot_margen:,.0f}")
        c6.metric("DTE mínimo",        f"{min(r['DTE'] for r in rows)}d")

        # (gráficos de resumen eliminados)

        # ── Griegas totales ──────────────────────────────────────
        st.subheader("🔢 Griegas Totales de la Cartera")
        st.caption("Valores escalados: por opción × 100 acciones × cantidad × ±1 (compra/venta)")

        # KPI métricas totales
        gc = st.columns(5)
        greek_defs = [
            ("Delta", "Δ Delta",  4, "Exposición direccional total"),
            ("Gamma", "Γ Gamma",  6, "Curvatura total del delta por $1 de movimiento"),
            ("Vega",  "ν Vega",   4, "Cambio total en valor por +1% de IV"),
            ("Theta", "Θ Theta",  4, "Decaimiento temporal total diario ($)"),
            ("Rho",   "ρ Rho",    4, "Cambio total en valor por +1% de tipos"),
        ]
        for col_g, (gk, lbl, dec, help_txt) in zip(gc, greek_defs):
            col_g.metric(lbl, fmt(tot_greeks[gk], dec), help=help_txt)

        # Tabla individual por operación — ancho completo, sin radar
        st.markdown("**Griegas por operación** (escaladas × 100 × contratos × ±1 dirección)")
        greek_rows = []
        for i, op in enumerate(cartera):
            S_g   = precios_sub.get(op["Ticker"])
            K_g   = float(op["Strike"])
            T_g   = max(days_to_expiry(op["Vencimiento"]) / 365.0, 1e-6)
            sig_g = float(op.get("IV") or 0.25)
            qty_g = int(op["Cantidad"])
            sgn_g = 1 if op["Dirección"] == "Compra" else -1
            gg    = bs_greeks(S_g or K_g, K_g, T_g, 0.05, sig_g, op["Tipo"])
            scale = 100 * qty_g * sgn_g
            greek_rows.append({
                "Op": f"{i}: {op['Ticker']} {op['Tipo'].upper()} K={K_g:.0f} {op['Dirección']}",
                "Δ Delta":  round(gg["Delta"] * scale, 2),
                "Γ Gamma":  round(gg["Gamma"] * scale, 4),
                "ν Vega":   round(gg["Vega"]  * scale, 2),
                "Θ Theta":  round(gg["Theta"] * scale, 2),
                "ρ Rho":    round(gg["Rho"]   * scale, 2),
            })
        df_gg = pd.DataFrame(greek_rows)
        total_row = {"Op": "▶ TOTAL"}
        for gk_col in ["Δ Delta", "Γ Gamma", "ν Vega", "Θ Theta", "ρ Rho"]:
            total_row[gk_col] = round(df_gg[gk_col].sum(), 4)
        df_gg = pd.concat([df_gg, pd.DataFrame([total_row])], ignore_index=True)

        def _color_greek_val(val):
            if isinstance(val, (int, float)):
                if val > 0: return "color:#16a34a;font-weight:600;font-family:'IBM Plex Mono',monospace"
                if val < 0: return "color:#dc2626;font-weight:600;font-family:'IBM Plex Mono',monospace"
                return "font-family:'IBM Plex Mono',monospace"
            return "font-family:'IBM Plex Mono',monospace"

        greek_num_cols = ["Δ Delta", "Γ Gamma", "ν Vega", "Θ Theta", "ρ Rho"]
        st.dataframe(
            style_table(df_gg)
                .map(_color_greek_val, subset=greek_num_cols)
                .format({c: (lambda x: fmt(x, 4) if isinstance(x, (int, float)) else str(x))
                         for c in greek_num_cols}),
            use_container_width=True,
            height=min(420, (len(df_gg) + 1) * 40),
        )

        # ═══════════════════════════════════════════════════════
        #  WHAT-IF INTERACTIVO
        # ═══════════════════════════════════════════════════════
        st.subheader("🔮 Análisis What-If")
        with st.expander("Simular escenarios de mercado", expanded=False):
            st.markdown("Ajusta los sliders para ver el impacto en el PnL de la cartera completa en tiempo real.")
            wi_c1, wi_c2, wi_c3 = st.columns(3)
            with wi_c1:
                price_shock = st.slider("Variación precio (%)", -30, 30, 0, 1, key="wi_price",
                                         help="Shock de precio sobre el subyacente")
            with wi_c2:
                iv_shock = st.slider("Variación IV (pp)", -15, 15, 0, 1, key="wi_iv",
                                      help="Cambio en volatilidad implícita en puntos porcentuales")
            with wi_c3:
                days_fwd = st.slider("Días hacia adelante", 0, 60, 0, 1, key="wi_days",
                                      help="Simular passage del tiempo")

            wi_pnl = compute_whatif_pnl(cartera, price_shock, iv_shock, days_fwd, precios_sub)
            wi_delta_pnl = wi_pnl - (pnl_total if pnl_total != 0 else 0)

            wc1, wc2, wc3, wc4 = st.columns(4)
            wc1.metric("Precio shock",    f"{price_shock:+.0f}%")
            wc2.metric("IV shock",        f"{iv_shock:+.0f}pp")
            wc3.metric("Días",            f"+{days_fwd}d")

            # Color del PnL what-if
            pnl_color_html = "#16a34a" if wi_pnl >= 0 else "#dc2626"
            wc4.markdown(
                f"""<div style='background:#f0f9ff;border:1px solid #bae6fd;border-radius:8px;padding:12px;text-align:center'>
                <div style='font-size:10px;color:#64748b;font-family:IBM Plex Mono;text-transform:uppercase;letter-spacing:0.08em'>PnL Simulado</div>
                <div style='font-size:1.4rem;font-weight:700;color:{pnl_color_html};font-family:IBM Plex Mono'>${wi_pnl:+,.2f}</div>
                </div>""",
                unsafe_allow_html=True
            )

            # Mini matriz de escenarios
            st.markdown("**Matriz de escenarios** (filas: precio, columnas: IV)")
            price_scenarios = [-10, -5, 0, 5, 10]
            iv_scenarios    = [-5, 0, 5]
            matrix_data = {}
            for iv_s in iv_scenarios:
                col_data = []
                for p_s in price_scenarios:
                    pnl_s = compute_whatif_pnl(cartera, p_s, iv_s, days_fwd, precios_sub)
                    col_data.append(pnl_s)
                matrix_data[f"IV{iv_s:+d}pp"] = col_data

            df_matrix = pd.DataFrame(matrix_data, index=[f"Precio{p:+d}%" for p in price_scenarios])
            st.dataframe(
                df_matrix.style
                    .map(color_pnl)
                    .format("${:,.2f}"),
                use_container_width=True,
            )

        # ═══════════════════════════════════════════════════════
        #  ESTIMATED RETURNS — OPERACIONES INDIVIDUALES
        # ═══════════════════════════════════════════════════════
        st.subheader("📊 Estimated Returns — Operaciones Individuales")
        er_rows = []
        for i, op in enumerate(cartera):
            S_er     = precios_sub.get(op["Ticker"]) or float(op["Strike"])
            sigma_er = float(op.get("IV") or 0.25)
            # op_prob_profit retorna [0,1] — ya multiplica por 100 aquí
            pp       = round(op_prob_profit(op, S_er, sigma_er) * 100, 1)
            mr       = op_max_return(op)
            mk       = op_max_risk(op)
            ec       = op_entry_cost_credit(op)
            be       = op_breakeven(op)
            iv_rank_data = get_iv_rank(op["Ticker"], sigma_er)

            # Ratio R/R: sólo calculable cuando ambos son numéricos
            if isinstance(mr, (int, float)) and isinstance(mk, (int, float)) and mk != 0:
                rr = round(mr / abs(mk), 2)
            else:
                rr = "—"

            er_rows.append({
                "Operación":       f"{i}: {op['Ticker']} {op['Tipo'].upper()} K={op['Strike']} {op['Dirección']}",
                "Breakeven":       be,
                "Prob. Profit %":  pp,
                "Máx. Retorno $":  mr,
                "Máx. Riesgo $":   mk,
                "Ratio R/R":       rr,
                "Coste/Crédito $": ec,
                "IV %":            f"{sigma_er*100:.1f}%",
                "IV Rank":         f"{iv_rank_data['iv_rank']:.0f}" if iv_rank_data["iv_rank"] is not None else "—",
            })

        df_er = pd.DataFrame(er_rows)
        st.dataframe(
            style_table(df_er, pnl_cols=["Coste/Crédito $", "Máx. Retorno $"], prob_cols=["Prob. Profit %"])
            .format({
                "Breakeven":       lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else str(x),
                "Máx. Retorno $":  lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else str(x),
                "Máx. Riesgo $":   lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else str(x),
                "Coste/Crédito $": lambda x: f"${x:+,.2f}" if isinstance(x, (int, float)) else str(x),
                "Ratio R/R":       lambda x: f"{x:.2f}x" if isinstance(x, (int, float)) else str(x),
            }),
            use_container_width=True,
        )

        # ═══════════════════════════════════════════════════════
        #  ESTIMATED RETURNS — ESTRATEGIAS AGRUPADAS + MARGEN IB
        # ═══════════════════════════════════════════════════════
        st.subheader("🎯 Estimated Returns — Estrategias Agrupadas")
        grupos = {}
        for i, op in enumerate(cartera):
            grupos.setdefault(f"{op['Ticker']}_{op['Vencimiento']}", []).append(i)

        strat_rows = []
        margin_details = []

        for key, idxs in grupos.items():
            ops_g      = [cartera[i] for i in idxs]
            t_g        = ops_g[0]["Ticker"]
            v_g        = ops_g[0]["Vencimiento"]
            S_g        = precios_sub.get(t_g) or float(ops_g[0]["Strike"])
            sigma_g    = float(ops_g[0].get("IV") or 0.25)

            strikes_g  = [float(o["Strike"])  for o in ops_g]
            primas_g   = [float(o.get("PrecioEntrada", o.get("Prima", 0))) for o in ops_g]
            dirs_g     = [o["Dirección"] for o in ops_g]
            qtys_g     = [int(o["Cantidad"]) for o in ops_g]
            tipos_g    = [o["Tipo"] for o in ops_g]

            be     = strategy_breakeven(strikes_g, primas_g, dirs_g, qtys_g, tipos_g)
            pp     = strategy_prob_profit(strikes_g, primas_g, dirs_g, qtys_g, tipos_g, S_g, v_g, sigma_g)
            mr, mk = strategy_risk_reward(strikes_g, primas_g, dirs_g, qtys_g, tipos_g)
            ec     = strategy_entry_credit(primas_g, dirs_g, qtys_g)
            tipo_str = strategy_type(strikes_g, dirs_g, tipos_g)

            # Margen IB
            margin_info = strategy_margin_ib(ops_g, S_g)
            margin_details.append({"key": key, "tipo": tipo_str, **margin_info})

            strat_rows.append({
                "Estrategia":          f"{t_g} — {tipo_str}",
                "Vencimiento":         v_g,
                "Legs":                len(idxs),
                "Breakeven":           be,
                "Prob. Profit %":      pp,
                "Máx. Retorno $":      mr,
                "Máx. Riesgo $":       mk,
                "Crédito/Coste $":     ec,
                "Margen IB $":         margin_info["margin_combined"],
                "Margen Ind. $":       margin_info["margin_individual"],
                "Ahorro Margen $":     margin_info["saving"],
            })

        if strat_rows:
            df_st = pd.DataFrame(strat_rows)
            st.dataframe(
                style_table(df_st, prob_cols=["Prob. Profit %"], pnl_cols=["Crédito/Coste $", "Ahorro Margen $"])
                .format({
                    "Breakeven":       lambda x: f"{x:,.2f}" if isinstance(x, (int, float)) else str(x),
                    "Prob. Profit %":  lambda x: f"{x:.1f}%" if isinstance(x, (int, float)) else str(x),
                    "Máx. Retorno $":  lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else str(x),
                    "Máx. Riesgo $":   lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else str(x),
                    "Crédito/Coste $": lambda x: f"${x:+,.2f}" if isinstance(x, (int, float)) else str(x),
                    "Margen IB $":     lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else str(x),
                    "Margen Ind. $":   lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else str(x),
                    "Ahorro Margen $": lambda x: f"${x:,.0f}" if isinstance(x, (int, float)) else str(x),
                }),
                use_container_width=True,
            )

            # Detalle del cálculo de márgenes
            with st.expander("🔍 Detalle cálculo de márgenes (metodología IB Reg-T)"):
                for md in margin_details:
                    if md["margin_individual"] > 0:
                        saving_pct = md["saving"] / md["margin_individual"] * 100 if md["margin_individual"] > 0 else 0
                        st.markdown(f"""
**{md['key'].replace('_', ' ')} — {md['tipo']}**
```
{md['breakdown']}
```
💡 Ahorro vs. suma individual: **${md['saving']:,.2f}** ({saving_pct:.1f}% de reducción)
""")

        # ── Gestión de operaciones ───────────────────────────────
        st.subheader("🔧 Gestión de Operaciones")
        c1, c2, c3 = st.columns(3)
        with c1:
            idx_close = st.number_input("Índice a cerrar", 0, len(cartera)-1, 0, key="idx_close")
            if st.button("Cerrar posición", key="btn_close"):
                op = cartera[idx_close]
                if days_to_expiry(op["Vencimiento"]) <= 0:
                    pnl_r = op_expired_pnl(op)
                else:
                    mkt = get_market_option_price(op["Ticker"], op["Vencimiento"], float(op["Strike"]), op["Tipo"])
                    pnl_r = op_pnl_market(op, mkt)
                closed = dict(op)
                closed["FechaCierre"] = str(date.today())
                closed["PnLReal"]     = pnl_r
                st.session_state["closed_positions"].append(closed)
                st.session_state["cartera"].pop(idx_close)
                pnl_str = fmt(pnl_r) if isinstance(pnl_r, (int, float)) else str(pnl_r)
                st.success(f"Posición cerrada. PnL: {pnl_str} $")
                st.rerun()
        with c2:
            idx_del = st.number_input("Índice a eliminar", 0, len(cartera)-1, 0, key="idx_del")
            if st.button("Eliminar operación", key="btn_del"):
                st.session_state["cartera"].pop(idx_del)
                st.rerun()
        with c3:
            if st.button("🗑️ Limpiar toda la cartera", key="btn_clear"):
                st.session_state["cartera"] = []
                st.rerun()

    # ── Guardar / Cargar carteras ────────────────────────────────
    with st.expander("💾 Guardar / Cargar Cartera"):
        port_name = st.text_input("Nombre de la cartera", key="port_name")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Guardar", key="btn_save") and port_name:
                save_portfolio(port_name, st.session_state["cartera"])
                st.success(f"Cartera «{port_name}» guardada.")
        saved = st.session_state["saved_portfolios"]
        if saved:
            with c2:
                sel = st.selectbox("Carteras guardadas", list(saved.keys()), key="sel_port")
                if st.button(f"Cargar", key="btn_load"):
                    st.session_state["cartera"] = [dict(op) for op in saved[sel]]
                    st.success(f"Cartera «{sel}» cargada.")
                    st.rerun()
            with c3:
                if st.button("Eliminar seleccionada", key="btn_del_port"):
                    del st.session_state["saved_portfolios"][sel]
                    with open(PORTFOLIO_FILE, "w") as f:
                        json.dump(st.session_state["saved_portfolios"], f)
                    st.rerun()


# ════════════════════════════════════════════════════════════════
#  TAB 3 — GRÁFICOS DE SENSIBILIDAD
# ════════════════════════════════════════════════════════════════
with tab3:
    st.header("Gráficos de Sensibilidad")
    cartera = st.session_state["cartera"]

    if not cartera:
        st.info("Añade operaciones a la cartera para ver los gráficos.")
    else:
        tickers_u = list({op["Ticker"] for op in cartera})
        precios_s = {t: get_stock_info(t)["price"] for t in tickers_u}

        ref_ticker  = cartera[0]["Ticker"]
        ref_price   = precios_s.get(ref_ticker) or float(cartera[0]["Strike"])
        all_strikes = [float(op["Strike"]) for op in cartera]
        strike_min  = min(all_strikes)
        strike_max  = max(all_strikes)
        spread      = max(strike_max - strike_min, ref_price * 0.10)
        auto_min    = round(max(min(strike_min, ref_price) - spread * 1.5, 0.01), 2)
        auto_max    = round(max(strike_max, ref_price) + spread * 1.5, 2)

        st.subheader("Configuración")
        c1, c2 = st.columns(2)
        with c1:
            col_pmin, col_pmax = st.columns(2)
            with col_pmin:
                p_min = st.number_input("Precio mínimo", value=auto_min, key="g_pmin")
            with col_pmax:
                p_max = st.number_input("Precio máximo", value=auto_max, key="g_pmax")
            show_ind = st.checkbox("Mostrar operaciones individuales", value=True, key="g_ind")
            st.caption(f"Rango auto: {auto_min:.2f} – {auto_max:.2f}  |  S={ref_price:.2f}  |  K: {strike_min:.2f}–{strike_max:.2f}")
        with c2:
            step_opts  = {"1 día": 1, "3 días": 3, "1 semana": 7, "2 semanas": 14}
            step_sel   = st.selectbox("Paso temporal", list(step_opts.keys()), index=2, key="g_step")
            step_days  = step_opts[step_sel]
            show_all_t = st.checkbox("Todos los periodos en un gráfico", value=True, key="g_all")

        precios = np.linspace(p_min, p_max, 300)

        st.subheader("Seleccionar operaciones")
        sel_ops = []
        for i, op in enumerate(cartera):
            lbl = f"{i}: {op['Ticker']} {op['Tipo'].upper()} K={op['Strike']} {op['Dirección']} x{op['Cantidad']}"
            if st.checkbox(lbl, value=True, key=f"chk_{i}"):
                sel_ops.append(i)

        if not sel_ops:
            st.warning("Selecciona al menos una operación.")
        else:
            g1, g2, g3, g4, g5 = st.tabs([
                "📈 PnL Actual",
                "📊 Griegas",
                "⌛ PnL a Vencimiento",
                "🔄 Evolución Temporal",
                "📉 Cono de Volatilidad",
            ])

            # ── PnL actual ───────────────────────────────────────
            with g1:
                fig = go.Figure()
                total_pnl = np.zeros(len(precios))
                for i in sel_ops:
                    op    = cartera[i]
                    K     = float(op["Strike"])
                    prima = float(op.get("PrecioEntrada", op.get("Prima", 0)))
                    qty   = int(op["Cantidad"])
                    sigma = float(op.get("IV") or 0.25)
                    T     = max(days_to_expiry(op["Vencimiento"]) / 365.0, 1e-6)
                    sign  = 1 if op["Dirección"] == "Compra" else -1
                    pnl_arr = np.array([
                        (bs_price(S, K, T, 0.05, sigma, op["Tipo"]) - prima) * sign * qty * 100
                        for S in precios
                    ])
                    total_pnl += pnl_arr
                    if show_ind:
                        fig.add_trace(go.Scatter(x=precios, y=pnl_arr, mode="lines",
                                                  name=f"Op{i} {op['Ticker']} {op['Tipo']} {K}",
                                                  line=dict(width=1, dash="dot")))
                fig.add_trace(go.Scatter(x=precios, y=total_pnl, mode="lines",
                                          name="PnL Total", line=dict(color="#dc2626", width=3)))
                fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)
                add_chart_references(fig, [float(cartera[i]["Strike"]) for i in sel_ops],
                                      precios_s.get(cartera[sel_ops[0]]["Ticker"]))
                fig.update_layout(title="PnL Actual (Black-Scholes)",
                                   xaxis_title="Precio subyacente", yaxis_title="P&L ($)", **PLOTLY_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            # ── Griegas ──────────────────────────────────────────
            with g2:
                for greek in ["Delta", "Gamma", "Vega", "Theta"]:
                    fig_g = go.Figure()
                    total_g = np.zeros(len(precios))
                    for i in sel_ops:
                        op    = cartera[i]
                        K     = float(op["Strike"])
                        qty   = int(op["Cantidad"])
                        sigma = float(op.get("IV") or 0.25)
                        T     = max(days_to_expiry(op["Vencimiento"]) / 365.0, 1e-6)
                        sign  = 1 if op["Dirección"] == "Compra" else -1
                        g_arr = np.array([
                            bs_greeks(S, K, T, 0.05, sigma, op["Tipo"])[greek] * qty * 100 * sign
                            for S in precios
                        ])
                        total_g += g_arr
                        if show_ind:
                            fig_g.add_trace(go.Scatter(x=precios, y=g_arr, mode="lines",
                                                        name=f"Op{i}", line=dict(width=1, dash="dot")))
                    fig_g.add_trace(go.Scatter(x=precios, y=total_g, mode="lines",
                                                name=f"{greek} Total", line=dict(color="#2563eb", width=3)))
                    fig_g.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)
                    add_chart_references(fig_g, [float(cartera[i]["Strike"]) for i in sel_ops],
                                          precios_s.get(cartera[sel_ops[0]]["Ticker"]))
                    fig_g.update_layout(title=f"Perfil de {greek}",
                                         xaxis_title="Precio subyacente", yaxis_title=greek, **PLOTLY_LAYOUT)
                    st.plotly_chart(fig_g, use_container_width=True)

            # ── PnL a vencimiento ────────────────────────────────
            with g3:
                fig_exp   = go.Figure()
                total_exp = np.zeros(len(precios))
                for i in sel_ops:
                    op    = cartera[i]
                    K     = float(op["Strike"])
                    prima = float(op.get("PrecioEntrada", op.get("Prima", 0)))
                    qty   = int(op["Cantidad"])
                    sign  = 1 if op["Dirección"] == "Compra" else -1
                    intr  = np.maximum(precios - K, 0) if op["Tipo"] == "call" else np.maximum(K - precios, 0)
                    pnl_v = (intr - prima) * sign * qty * 100
                    total_exp += pnl_v
                    if show_ind:
                        fig_exp.add_trace(go.Scatter(x=precios, y=pnl_v, mode="lines",
                                                      name=f"Op{i} {op['Ticker']} {op['Tipo']} {K}",
                                                      line=dict(width=1, dash="dot")))
                fig_exp.add_trace(go.Scatter(x=precios, y=total_exp, mode="lines",
                                              name="PnL Total", line=dict(color="#ea580c", width=3),
                                              fill="tozeroy", fillcolor="rgba(234,88,12,0.08)"))
                fig_exp.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)
                add_chart_references(fig_exp, [float(cartera[i]["Strike"]) for i in sel_ops],
                                      precios_s.get(cartera[sel_ops[0]]["Ticker"]))
                fig_exp.update_layout(title="PnL a Vencimiento",
                                       xaxis_title="Precio subyacente", yaxis_title="P&L ($)", **PLOTLY_LAYOUT)
                st.plotly_chart(fig_exp, use_container_width=True)

            # ── Evolución temporal ───────────────────────────────
            with g4:
                min_dte = min(days_to_expiry(cartera[i]["Vencimiento"]) for i in sel_ops)
                if min_dte <= 0:
                    st.warning("Las operaciones seleccionadas ya han vencido.")
                else:
                    colors  = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                                "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
                    fig_evo = go.Figure() if show_all_t else None

                    for idx_d, day in enumerate(range(min_dte, -1, -step_days)):
                        T_evo    = max(day / 365.0, 1e-6)
                        total_ev = np.zeros(len(precios))
                        for i in sel_ops:
                            op    = cartera[i]
                            K     = float(op["Strike"])
                            prima = float(op.get("PrecioEntrada", op.get("Prima", 0)))
                            qty   = int(op["Cantidad"])
                            sigma = float(op.get("IV") or 0.25)
                            sign  = 1 if op["Dirección"] == "Compra" else -1
                            pnl_e = np.array([
                                (bs_price(S, K, T_evo, 0.05, sigma, op["Tipo"]) - prima) * sign * qty * 100
                                for S in precios
                            ])
                            total_ev += pnl_e

                        fecha_lbl = (date.today() + timedelta(days=min_dte - day)).strftime("%d/%m/%Y")
                        label     = f"{fecha_lbl} (DTE {day})"
                        color     = colors[idx_d % len(colors)]

                        if show_all_t:
                            fig_evo.add_trace(go.Scatter(x=precios, y=total_ev, mode="lines",
                                                          name=label, line=dict(width=2, color=color)))
                        else:
                            f_d = go.Figure()
                            f_d.add_trace(go.Scatter(x=precios, y=total_ev, mode="lines",
                                                      name=label, line=dict(color=color, width=2)))
                            f_d.add_hline(y=0, line_dash="dash", line_color="gray")
                            f_d.update_layout(title=f"Evolución — {label}",
                                               xaxis_title="Precio subyacente", yaxis_title="P&L ($)", **PLOTLY_LAYOUT)
                            st.plotly_chart(f_d, use_container_width=True)

                    if show_all_t and fig_evo:
                        fig_evo.add_hline(y=0, line_dash="dash", line_color="#94a3b8", line_width=1)
                        add_chart_references(fig_evo, [float(cartera[i]["Strike"]) for i in sel_ops],
                                              precios_s.get(cartera[sel_ops[0]]["Ticker"]))
                        fig_evo.update_layout(title="Evolución Temporal del PnL",
                                               xaxis_title="Precio subyacente", yaxis_title="P&L ($)",
                                               **PLOTLY_LAYOUT)
                        st.plotly_chart(fig_evo, use_container_width=True)

            # ── Cono de Volatilidad ──────────────────────────────
            with g5:
                st.subheader("Cono de Volatilidad Histórica")
                tickers_sel = list({cartera[i]["Ticker"] for i in sel_ops})
                for tk in tickers_sel:
                    iv_ref = float(cartera[next(i for i in sel_ops if cartera[i]["Ticker"] == tk)].get("IV") or 0.25)
                    fig_vc = plot_vol_cone(tk, iv_ref)
                    if fig_vc:
                        st.plotly_chart(fig_vc, use_container_width=True)
                        cone_data = compute_vol_cone(tk)
                        if cone_data:
                            iv_rank_d = get_iv_rank(tk, iv_ref)
                            cc1, cc2, cc3, cc4 = st.columns(4)
                            cc1.metric(f"{tk} IV actual",    f"{iv_ref*100:.1f}%")
                            cc2.metric("HV30 actual",        f"{cone_data.get(30, {}).get('current', 0)*100:.1f}%")
                            cc3.metric("IV Rank (52s)",      f"{iv_rank_d['iv_rank']:.0f}" if iv_rank_d["iv_rank"] else "—")
                            cc4.metric("IV Percentil",       f"{iv_rank_d['iv_pct']:.0f}%" if iv_rank_d["iv_pct"] else "—")
                    else:
                        st.info(f"No hay suficientes datos históricos para {tk}.")


# ════════════════════════════════════════════════════════════════
#  TAB 4 — HISTORIAL DE POSICIONES CERRADAS
# ════════════════════════════════════════════════════════════════
with tab4:
    st.header("📋 Historial de Posiciones Cerradas")
    closed = st.session_state["closed_positions"]

    if not closed:
        st.info("Aún no has cerrado ninguna posición.")
    else:
        hist_rows = [{
            "Ticker":          op.get("Ticker"),
            "Tipo":            op.get("Tipo"),
            "Dirección":       op.get("Dirección"),
            "Strike":          float(op.get("Strike", 0)),
            "Prima entrada":   float(op.get("PrecioEntrada", op.get("Prima", 0))),
            "Precio cierre":   op.get("PrecioCierre", "N/D"),
            "Qty":             int(op.get("Cantidad", 1)),
            "Vencimiento":     op.get("Vencimiento"),
            "Fecha entrada":   op.get("FechaEntrada", op.get("Fecha", "—")),
            "Fecha cierre":    op.get("FechaCierre", "—"),
            "PnL Realizado $": op.get("PnLReal", "N/D"),
        } for op in closed]

        df_hist = pd.DataFrame(hist_rows)
        pnl_r   = [r["PnL Realizado $"] for r in hist_rows if isinstance(r["PnL Realizado $"], (int, float))]

        # KPIs historial
        wins     = sum(1 for p in pnl_r if p > 0)
        losses   = sum(1 for p in pnl_r if p < 0)
        avg_win  = np.mean([p for p in pnl_r if p > 0]) if wins  > 0 else 0
        avg_loss = np.mean([p for p in pnl_r if p < 0]) if losses > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Posiciones cerradas", len(closed))
        c2.metric("PnL realizado total", f"${sum(pnl_r):,.2f}")
        c3.metric("Win rate",            f"{wins/len(pnl_r)*100:.1f}%" if pnl_r else "—")
        c4.metric("Media wins",          f"${avg_win:,.2f}" if avg_win else "—")
        c5.metric("Media losses",        f"${avg_loss:,.2f}" if avg_loss else "—")

        # Gráfico de PnL acumulado
        if pnl_r:
            cumulative = np.cumsum(pnl_r)
            fig_cum = go.Figure()
            colors_bar = ["#16a34a" if p >= 0 else "#dc2626" for p in pnl_r]
            fig_cum.add_trace(go.Bar(
                x=list(range(len(pnl_r))), y=pnl_r,
                marker_color=colors_bar, name="PnL por operación",
                opacity=0.7,
            ))
            fig_cum.add_trace(go.Scatter(
                x=list(range(len(cumulative))), y=cumulative,
                mode="lines+markers", name="PnL Acumulado",
                line=dict(color="#0ea5e9", width=2),
                yaxis="y2",
            ))
            fig_cum.update_layout(
                title="Historial de PnL",
                xaxis_title="Operación #",
                yaxis_title="PnL ($)",
                yaxis2=dict(title="PnL Acumulado ($)", overlaying="y", side="right",
                            showgrid=False, tickfont=dict(family="IBM Plex Mono", size=10)),
                **PLOTLY_LAYOUT,
            )
            st.plotly_chart(fig_cum, use_container_width=True)

        st.dataframe(
            style_table(df_hist, pnl_cols=["PnL Realizado $"])
            .format({
                "Strike":          "{:.2f}",
                "Prima entrada":   "{:.4f}",
                "PnL Realizado $": lambda x: fmt(x) if isinstance(x, (int, float)) else str(x),
            }),
            use_container_width=True,
        )

        if st.button("🗑️ Limpiar historial", key="btn_clear_hist"):
            st.session_state["closed_positions"] = []
            st.rerun()
