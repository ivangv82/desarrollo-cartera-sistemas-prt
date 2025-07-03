import streamlit as st
import pandas as pd
import numpy as np
import io
from dateutil import parser
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Configuración de página ---
st.set_page_config(
    page_title="📊 Backtest PRT Completo",
    layout="wide"
)

st.title("🚀 Análisis Completo de Backtest PRT")
st.markdown("Carga tu reporte de operaciones de ProRealTime y obtén métricas clásicas, curvas y ratios de riesgo adaptados al timeframe.")

# --- Función de carga con fallback de fechas ---
@st.cache_data
def load_prt_trades(file):
    try:
        file.seek(0)
        df = pd.read_csv(file, sep='\t', decimal=',', thousands='.')
        if df.shape[1] < 2:
            file.seek(0)
            df = pd.read_csv(file, sep=',', decimal='.', thousands=',')
    except Exception:
        file.seek(0)
        df = pd.read_csv(file, sep=None, engine='python')

    column_map = {
        'Fecha entrada': 'Entry Date', 'entry date': 'Entry Date',
        'Fecha salida': 'Exit Date', 'exit date': 'Exit Date',
        'Tipo': 'Side', 'side': 'Side',
        'Rdto Abs': 'Profit', 'profit': 'Profit',
        'Ganancia unitaria': 'Profit %', 'profit %': 'Profit %',
        'MFE': 'MFE', 'mfe': 'MFE',
        'MAE': 'MAE', 'mae': 'MAE',
        'Nº barras': 'Nº barras'
    }
    
    df = df.rename(columns=lambda c: column_map.get(str(c).strip(), str(c).strip()))

    if 'Entry Date' not in df.columns or 'Exit Date' not in df.columns or 'Profit' not in df.columns:
        st.error("El archivo no contiene las columnas necesarias ('Fecha entrada', 'Fecha salida', 'Rdto Abs'). Por favor, verifica el formato.")
        st.stop()
        
    df['__raw_entry'] = df['Entry Date'].astype(str)
    df['__raw_exit']  = df['Exit Date'].astype(str)

    month_map = {
      'ene':'Jan','feb':'Feb','mar':'Mar','abr':'Apr','may':'May','jun':'Jun',
      'jul':'Jul','ago':'Aug','sep':'Sep','sept':'Sep','oct':'Oct','nov':'Nov','dic':'Dec'
    }
    def clean_and_parse(s_raw):
        s = str(s_raw).lower()
        for es, en in month_map.items():
            s = s.replace(es, en)
        try:
            dt = pd.to_datetime(s, dayfirst=True, errors='coerce')
            if pd.isna(dt):
                dt = parser.parse(s, dayfirst=True, fuzzy=True)
            return dt
        except:
            return pd.NaT

    df['Entry Date'] = df['__raw_entry'].apply(clean_and_parse)
    df['Exit Date']  = df['__raw_exit'].apply(clean_and_parse)

    mask_bad = df['Entry Date'].isna() | df['Exit Date'].isna()
    if mask_bad.any():
        st.warning(f"{mask_bad.sum()} fila(s) con fecha irreconocible y serán descartadas.")
        df = df.loc[~mask_bad].copy()
    
    df = df.drop(columns=['__raw_entry','__raw_exit'])

    if 'Profit %' in df.columns:
        profit_pct_str = df['Profit %'].astype(str).str.replace('%','', regex=False).str.replace(',','.', regex=False)
        df['Profit %'] = pd.to_numeric(profit_pct_str, errors='coerce').fillna(0.0) / 100
    else:
        df['Profit %'] = 0.0

    profit_str = df['Profit'].astype(str)
    profit_str = profit_str.str.replace(r'\.(?=.*\d)', '', regex=True)
    profit_str = profit_str.str.replace(',', '.', regex=False)
    profit_str = profit_str.str.replace(r'[^\d.-]', '', regex=True)
    df['Profit'] = pd.to_numeric(profit_str, errors='coerce')
    df['Profit'] = df['Profit'].fillna(0.0)

    return df

# --- Equity y métricas ---
def compute_equity(trades, init_cap):
    df = trades.sort_values('Exit Date').copy()
    # Usamos los retornos porcentuales sobre el capital para un Monte Carlo más realista
    df['Return'] = df['Profit'] / (init_cap + df['Profit'].shift(1).cumsum().fillna(0))
    df['Equity'] = init_cap * (1 + df['Return']).cumprod()
    
    # Aseguramos que el primer valor de equity sea el capital inicial
    df.loc[df.index[0], 'Equity'] = init_cap + df.loc[df.index[0], 'Profit']

    equity_curve = pd.DataFrame({
        'Date':   [trades['Entry Date'].min()] + df['Exit Date'].tolist(),
        'Equity': [init_cap] + df['Equity'].tolist()
    }).set_index('Date')

    return equity_curve, df['Return']

def calculate_metrics(trades, equity_df, ppy, timeframe):
    if equity_df.empty or len(equity_df) < 2:
        return {k: 0 for k in ["Beneficio Total", "Crecimiento Capital", "CAGR", "Sharpe Ratio", "Max Drawdown %", "Max Drawdown $", "Recovery Factor", "Calmar Ratio", "Total Operaciones", "Duración Media (velas)", "% Ganadoras", "Retorno Medio/Op. (%)", "Factor de Beneficio", "Ratio Payoff"]}

    equity = equity_df['Equity']
    ini, fin = equity.iloc[0], equity.iloc[-1]
    total_profit = fin - ini
    growth = fin/ini - 1 if ini != 0 else 0
    days = (equity.index[-1] - equity.index[0]).days or 1
    cagr = (fin/ini)**(365.0/days) - 1 if ini != 0 else 0

    cummax = equity.cummax()
    dd_rel = (equity - cummax)/cummax
    mdd_pct = dd_rel.min() if not dd_rel.empty else 0
    mdd_abs = (equity - cummax).min() if not (equity - cummax).empty else 0

    resample_freq = {"1mn":"T", "5mn":"5T", "15mn":"15T", "30mn":"30T", "1h":"H", "4h":"4H", "1d":"D", "1w":"W", "1mes":"MS"}[timeframe]
    equity_resampled = equity.resample(resample_freq).ffill().dropna()
    ret = equity_resampled.pct_change().dropna()

    if len(ret) < 2: sharpe = 0.0
    else: std_dev = ret.std(); sharpe = (ret.mean()/std_dev * np.sqrt(ppy)) if std_dev>0 else 0.0

    n = len(trades)
    wins = trades[trades['Profit']>0]; losses = trades[trades['Profit']<0]
    win_rate = len(wins)/n if n>0 else 0.0

    if 'Nº barras' in trades.columns and pd.to_numeric(trades['Nº barras'], errors='coerce').notna().all():
        avg_dur = pd.to_numeric(trades['Nº barras']).mean()
    else:
        avg_dur_days = (trades['Exit Date']-trades['Entry Date']).dt.total_seconds().mean()/(24*3600); avg_dur = avg_dur_days*(ppy/252)

    avg_ret_trade = trades['Profit %'].mean() if 'Profit %' in trades.columns else 0.0
    gross_profit = wins['Profit'].sum(); gross_loss = abs(losses['Profit'].sum())
    pf = gross_profit/gross_loss if gross_loss>0 else np.inf

    avg_win = wins['Profit %'].mean() if not wins.empty and 'Profit %' in wins.columns else 0.0
    avg_loss = abs(losses['Profit %'].mean()) if not losses.empty and 'Profit %' in losses.columns else 0.0
    payoff = avg_win/avg_loss if avg_loss>0 else np.inf

    rec_factor = total_profit/abs(mdd_abs) if mdd_abs!=0 else np.inf
    calmar = cagr/abs(mdd_pct) if mdd_pct!=0 else np.inf

    return {
        "Beneficio Total":total_profit, "Crecimiento Capital":growth, "CAGR":cagr, "Sharpe Ratio":sharpe,
        "Max Drawdown %":mdd_pct, "Max Drawdown $":mdd_abs, "Recovery Factor":rec_factor, "Calmar Ratio":calmar,
        "Total Operaciones":n, "Duración Media (velas)":avg_dur, "% Ganadoras":win_rate,
        "Retorno Medio/Op. (%)":avg_ret_trade, "Factor de Beneficio":pf, "Ratio Payoff":payoff
    }

# --- Sidebar ---
st.sidebar.header("📁 Carga de Datos")
trades_file = st.sidebar.file_uploader("Reporte PRT (CSV o TXT)", type=["csv","txt"])
initial_cap = st.sidebar.number_input("Capital Inicial", value=10000.0, min_value=0.0, step=100.0, format="%.2f")
timeframe = st.sidebar.selectbox("Timeframe de Velas", ["1mn","5mn","15mn","30mn","1h","4h","1d","1w","1mes"], index=4)

if timeframe in ["1mn","5mn","15mn","30mn","1h","4h"]:
    trading_hours_per_day = st.sidebar.number_input("Horas de trading por día", 1.0, 24.0, 6.5, 0.5, help="Ej: Bolsa USA=6.5h, Forex/Cripto=24h")
    minutes_in_tf = {"1mn":1, "5mn":5, "15mn":15, "30mn":30, "1h":60, "4h":240}[timeframe]
    ppy = (trading_hours_per_day * 60 / minutes_in_tf) * 252
else: ppy = {"1d":252, "1w":52, "1mes":12}[timeframe]
st.sidebar.caption(f"Periodos por año calculados: **{int(ppy)}**")

if not trades_file: st.info("Sube el archivo de operaciones para comenzar."); st.stop()

# --- Procesamiento ---
trades = load_prt_trades(trades_file)
if trades.empty: st.stop()
equity, trade_returns = compute_equity(trades, initial_cap) # Obtenemos retornos por operación
metrics = calculate_metrics(trades, equity, ppy, timeframe)

# --- Funciones de Monte Carlo ---
@st.cache_data
def run_monte_carlo(returns, n_sims, horizon):
    arr = np.asarray(returns, dtype=float); arr = arr[np.isfinite(arr)]
    if arr.size == 0 or horizon <= 0: return np.ones((horizon, n_sims))
    sims = np.zeros((horizon, n_sims))
    for i in range(n_sims):
        sample = np.random.choice(arr, size=horizon, replace=True)
        sims[:, i] = np.cumprod(1 + sample)
    return sims

@st.cache_data
def run_block_bootstrap_monte_carlo(returns, n_sims, block_size, horizon):
    """Simulación Monte Carlo usando Block Bootstrap."""
    arr = np.asarray(returns, dtype=float); arr = arr[np.isfinite(arr)]
    n_returns = len(arr)
    if n_returns < block_size or block_size <= 0: return None
    
    n_blocks = (horizon // block_size) + 1
    possible_starts = np.arange(n_returns - block_size + 1)
    sims = np.zeros((horizon, n_sims))

    for i in range(n_sims):
        random_starts = np.random.choice(possible_starts, size=n_blocks, replace=True)
        sim_returns = np.concatenate([arr[start:start+block_size] for start in random_starts])
        sim_returns = sim_returns[:horizon]
        sims[:, i] = np.cumprod(1 + sim_returns)
    return sims

def max_dd(equity_path):
    cm = np.maximum.accumulate(equity_path); return ((equity_path - cm) / cm).min()

# --- Pestañas ---
tabs = st.tabs(["📊 Resumen", "📈 Equity & DD", "📝 Operaciones", "🎲 MC Simple", "🎲 MC Bloques"])

with tabs[0]: # Resumen
    st.header("📋 Resumen de Métricas")
    # ... (código de métricas sin cambios)

with tabs[1]: # Equity & Drawdown
    st.header("📈 Curva de Equity y Drawdown")
    # ... (código de gráficos de equity sin cambios)

with tabs[2]: # Operaciones
    st.header("📝 Detalle de Operaciones")
    # ... (código de tabla de operaciones sin cambios)

with tabs[3]: # Monte Carlo Simple
    st.header("🎲 Simulación Monte Carlo (Bootstrap Simple)")
    # ... (código de MC simple sin cambios)

with tabs[4]: # Monte Carlo Block Bootstrap
    st.header("🎲 Simulación Monte Carlo (Block Bootstrap)")
    st.markdown("Muestrea bloques de retornos para intentar preservar la autocorrelación.")
    
    mc_block_cols = st.columns(2)
    block_size = mc_block_cols[0].number_input("Tamaño de bloque (operaciones)", 1, 100, 5, 1)
    n_sims_bb = mc_block_cols[1].number_input("Número de simulaciones (BB)", 100, 10000, 1000, 100)

    horizon_bb = len(trade_returns)
    if horizon_bb < 1: st.warning("No hay operaciones suficientes para ejecutar."); st.stop()

    if st.button("▶️ Ejecutar Simulación Block Bootstrap"):
        with st.spinner("Corriendo simulaciones con bloques..."):
            sims_rel_bb = run_block_bootstrap_monte_carlo(trade_returns.values, n_sims_bb, block_size, horizon_bb)

            if sims_rel_bb is None:
                st.error(f"Error: El tamaño de bloque ({block_size}) debe ser menor que el número de operaciones ({horizon_bb}).")
            else:
                initial_value = float(equity.iloc[0, 0])
                sims_eq_bb = sims_rel_bb * initial_value
                final_vals_bb = sims_eq_bb[-1, :]

                # Calcular estadísticas
                stats_bb = {
                    "Media": final_vals_bb.mean(), "Mediana": np.median(final_vals_bb),
                    "P10": np.percentile(final_vals_bb, 10), "P90": np.percentile(final_vals_bb, 90),
                    "VaR 95%": np.percentile(final_vals_bb, 5), "CVaR 95%": final_vals_bb[final_vals_bb <= np.percentile(final_vals_bb, 5)].mean()
                }
                mdds_bb = np.apply_along_axis(max_dd, 0, sims_eq_bb) * 100

                st.subheader("📈 Estadísticas del Capital Final (Block Bootstrap)")
                stat_cols_bb = st.columns(len(stats_bb))
                for idx, (label, value) in enumerate(stats_bb.items()):
                    stat_cols_bb[idx].metric(label, f"${value:,.2f}")

                # Gráficos (reutilizamos la lógica anterior)
                sim_dates = trade_returns.index
                fig_env_bb = go.Figure()
                fig_env_bb.add_trace(go.Scatter(x=sim_dates, y=np.percentile(sims_eq_bb, 95, axis=1), fill=None, mode='lines', line_color='lightgrey', showlegend=False))
                fig_env_bb.add_trace(go.Scatter(x=sim_dates, y=np.percentile(sims_eq_bb, 5, axis=1), fill='tonexty', mode='lines', line_color='lightgrey', name='5%-95%'))
                fig_env_bb.add_trace(go.Scatter(x=sim_dates, y=np.percentile(sims_eq_bb, 50, axis=1), mode='lines', name='Mediana', line=dict(color='orange', dash='dash')))
                fig_env_bb.add_trace(go.Scatter(x=equity.index, y=equity['Equity'], mode='lines', name='Histórico', line=dict(color='blue', width=3)))
                fig_env_bb.update_layout(title="Simulaciones Monte Carlo (Block Bootstrap) vs. Curva Histórica", xaxis_title='Operación #', yaxis_title='Capital')
                st.plotly_chart(fig_env_bb, use_container_width=True)
                
                # ... (Puedes añadir aquí los histogramas de la misma forma que en el MC simple, usando las variables `final_vals_bb` y `mdds_bb`)
