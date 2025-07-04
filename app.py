import streamlit as st
import pandas as pd
import numpy as np
import io
from dateutil import parser
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Configuración de página ---
st.set_page_config(page_title="📊 Analizador de Backtests", layout="wide")
st.title("🚀 Analizador de Estrategias PRT")

# --- Funciones de Lógica ---
@st.cache_data
def load_prt_trades(file):
    if file is None: return None
    encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    df = None
    for encoding in encodings_to_try:
        try:
            file.seek(0)
            temp_df = pd.read_csv(file, sep='\t', decimal=',', thousands='.', encoding=encoding, skipinitialspace=True)
            if temp_df.shape[1] > 1: df = temp_df; break
            file.seek(0)
            temp_df = pd.read_csv(file, sep=',', decimal='.', thousands=',', encoding=encoding, skipinitialspace=True)
            if temp_df.shape[1] > 1: df = temp_df; break
        except Exception: continue
    if df is None:
        st.error(f"Error al leer el archivo '{file.name}'. Su formato o codificación no es compatible."); return None
    column_map = {'Fecha entrada': 'Entry Date','Fecha salida': 'Exit Date','Tipo': 'Side','Rdto Abs': 'Profit','Ganancia unitaria': 'Profit %','MFE': 'MFE','MAE': 'MAE','Nº barras': 'Nº barras'}
    df = df.rename(columns=lambda c: column_map.get(str(c).strip(), str(c).strip()))
    if not all(col in df.columns for col in ['Entry Date', 'Exit Date', 'Profit']):
        st.error(f"El archivo '{file.name}' no contiene las columnas necesarias."); return None
    df['__raw_entry'] = df['Entry Date'].astype(str); df['__raw_exit']  = df['Exit Date'].astype(str)
    month_map = {'ene':'Jan','feb':'Feb','mar':'Mar','abr':'Apr','may':'May','jun':'Jun','jul':'Jul','ago':'Aug','sep':'Sep','sept':'Sep','oct':'Oct','nov':'Nov','dic':'Dec'}
    def clean_and_parse(s_raw):
        s = str(s_raw).lower()
        for es, en in month_map.items(): s = s.replace(es, en)
        try:
            dt = pd.to_datetime(s, dayfirst=True, errors='coerce')
            if pd.isna(dt): dt = parser.parse(s, dayfirst=True, fuzzy=True)
            return dt
        except: return pd.NaT
    df['Entry Date'] = df['__raw_entry'].apply(clean_and_parse); df['Exit Date']  = df['__raw_exit'].apply(clean_and_parse)
    mask_bad = df['Entry Date'].isna() | df['Exit Date'].isna()
    if mask_bad.any(): st.warning(f"{mask_bad.sum()} filas con fecha irreconocible en '{file.name}' serán descartadas."); df = df.loc[~mask_bad].copy()
    df = df.drop(columns=['__raw_entry','__raw_exit'])
    if 'Profit %' in df.columns:
        profit_pct_str = df['Profit %'].astype(str).str.replace('%','', regex=False).str.replace(',','.', regex=False)
        df['Profit %'] = pd.to_numeric(profit_pct_str, errors='coerce').fillna(0.0) / 100
    else: df['Profit %'] = 0.0
    profit_str = df['Profit'].astype(str).str.replace(r'\.(?=.*\d)', '', regex=True).str.replace(',', '.', regex=False).str.replace(r'[^\d.-]', '', regex=True)
    df['Profit'] = pd.to_numeric(profit_str, errors='coerce').fillna(0.0)
    return df

def compute_equity(trades, init_cap):
    if trades is None or trades.empty: return None, None
    df = trades.sort_values('Exit Date').copy().reset_index(drop=True)
    equity_before_trade = init_cap + df['Profit'].shift(1).cumsum().fillna(0)
    df['Return'] = df['Profit'].divide(equity_before_trade.replace(0, np.nan)).fillna(0)
    df['Equity'] = init_cap + df['Profit'].cumsum()
    equity_curve = pd.DataFrame({'Date': pd.to_datetime([trades['Entry Date'].min()] + df['Exit Date'].tolist()), 'Equity': [init_cap] + df['Equity'].tolist()}).set_index('Date')
    return equity_curve, df['Return']

def calculate_metrics(trades, equity_df, ppy, timeframe):
    if equity_df is None or equity_df.empty or len(equity_df) < 2: return {}
    equity = equity_df['Equity']
    ini, fin = equity.iloc[0], equity.iloc[-1]
    total_profit = fin - ini; growth = fin/ini - 1 if ini != 0 else 0
    days = (equity.index[-1] - equity.index[0]).days or 1
    cagr = (fin/ini)**(365.0/days) - 1 if ini != 0 else 0
    cummax = equity.cummax(); dd_rel = (equity - cummax)/cummax
    mdd_pct = dd_rel.min() if not dd_rel.empty else 0
    mdd_abs = (equity - cummax).min() if not (equity - cummax).empty else 0
    resample_freq = {"1mn":"T", "5mn":"5T", "15mn":"15T", "30mn":"30T", "1h":"H", "4h":"4H", "1d":"D", "1w":"W", "1mes":"MS"}[timeframe]
    equity_resampled = equity.resample(resample_freq).ffill().dropna()
    ret = equity_resampled.pct_change().dropna()
    if len(ret) < 2: sharpe = 0.0
    else: std_dev = ret.std(); sharpe = (ret.mean()/std_dev * np.sqrt(ppy)) if std_dev > 0 else 0.0
    n = len(trades); wins = trades[trades['Profit']>0]; losses = trades[trades['Profit']<0]
    win_rate = len(wins)/n if n > 0 else 0.0
    if 'Nº barras' in trades.columns and pd.to_numeric(trades['Nº barras'], errors='coerce').notna().all():
        avg_dur = pd.to_numeric(trades['Nº barras']).mean()
    else:
        avg_dur_days = (trades['Exit Date'] - trades['Entry Date']).dt.total_seconds().mean() / (24*3600) if n > 0 else 0; avg_dur = avg_dur_days * (ppy/252)
    avg_ret_trade = trades['Profit %'].mean() if 'Profit %' in trades.columns else 0.0
    gross_profit = wins['Profit'].sum(); gross_loss = abs(losses['Profit'].sum())
    pf = gross_profit/gross_loss if gross_loss > 0 else np.inf
    avg_win = wins['Profit %'].mean() if not wins.empty and 'Profit %' in wins.columns else 0.0
    avg_loss = abs(losses['Profit %'].mean()) if not losses.empty and 'Profit %' in losses.columns else 0.0
    payoff = avg_win/avg_loss if avg_loss>0 else np.inf
    rec_factor = total_profit/abs(mdd_abs) if mdd_abs!=0 else np.inf
    calmar = cagr/abs(mdd_pct) if mdd_pct!=0 else np.inf
    return {"Beneficio Total":total_profit, "Crecimiento Capital":growth, "CAGR":cagr, "Sharpe Ratio":sharpe, "Max Drawdown %":mdd_pct, "Max Drawdown $":mdd_abs, "Recovery Factor":rec_factor, "Calmar Ratio":calmar, "Total Operaciones":n, "Duración Media (velas)":avg_dur, "% Ganadoras":win_rate, "Retorno Medio/Op. (%)":avg_ret_trade, "Factor de Beneficio":pf, "Ratio Payoff":payoff}

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

# --- INICIALIZACIÓN DE ESTADO ---
if 'compare_mode' not in st.session_state:
    st.session_state.compare_mode = False

def activate_compare_mode():
    st.session_state.compare_mode = True

# --- SIDEBAR DINÁMICO PARA N ESTRATEGIAS ---
st.sidebar.header("📁 Carga de Estrategias")
num_strategies = st.sidebar.number_input(
    "Número de estrategias a comparar", 
    min_value=1, 
    max_value=10,
    value=2,  # Lo ponemos en 2 por defecto para que sea más intuitivo
    step=1
)

uploaded_files = []
initial_caps = []

st.sidebar.markdown("---")

for i in range(num_strategies):
    # Usamos un expander para que el sidebar no se alargue demasiado
    # CORRECCIÓN: Usamos `expanded=True` para que todos aparezcan abiertos por defecto.
    with st.sidebar.expander(f"Estrategia {i+1}", expanded=True): 
        
        file = st.file_uploader(
            f"Reporte Estrategia {i+1}", 
            type=["csv", "txt"], 
            key=f"file_{i}"
        )
        
        cap = st.number_input(
            f"Capital Inicial {i+1}", 
            value=10000.0, 
            min_value=0.0, 
            step=1000.0, 
            format="%.2f", 
            key=f"cap_{i}"
        )
        
        if file:
            uploaded_files.append(file)
            initial_caps.append(cap)

# --- LÓGICA DE PROCESAMIENTO Y RENDERIZADO ---
if not uploaded_files:
    st.info("Por favor, carga al menos un archivo de estrategia para comenzar.")
    st.stop()

strategies_data = []
for i, file in enumerate(uploaded_files):
    trades = load_prt_trades(file)
    equity, returns = compute_equity(trades, initial_caps[i])
    metrics = calculate_metrics(trades, equity, ppy, timeframe)
    strategies_data.append({
        "name": file.name,
        "trades": trades,
        "equity": equity,
        "returns": returns,
        "metrics": metrics,
        "initial_cap": initial_caps[i]
    })

# --- RENDERIZADO DE PESTAÑAS ---
st.header("Resultados del Análisis")
tabs = st.tabs(["📊 Resumen", "📈 Curvas", "📝 Operaciones", "🎲 Análisis Avanzado"])

with tabs[0]:
    st.subheader("Resumen Comparativo de Métricas")
    metric_series_list = [pd.Series(s["metrics"], name=s["name"].split('.')[0]) for s in strategies_data if s["metrics"]]
    if metric_series_list:
        df_comp = pd.concat(metric_series_list, axis=1)
        def format_value(val, key):
            if pd.isna(val): return "-"
            if key in ["Beneficio Total", "Max Drawdown $"]: return f"${val:,.2f}"
            if key in ["Crecimiento Capital", "CAGR", "Max Drawdown %", "% Ganadoras", "Retorno Medio/Op. (%)"]: return f"{val:.2%}"
            if key in ["Total Operaciones"]: return f"{int(val)}"
            return f"{val:.2f}"
        df_display = df_comp.copy()
        for col in df_display.columns: df_display[col] = [format_value(val, idx) for idx, val in df_comp[col].items()]
        st.dataframe(df_display, use_container_width=True)
    else:
        st.warning("No se pudieron calcular métricas para ninguna estrategia.")

with tabs[1]:
    st.subheader("Curvas de Capital (Normalizadas)")
    fig_eq = go.Figure()
    colors = ['royalblue', 'darkorange', 'green', 'indianred', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    for i, s_data in enumerate(strategies_data):
        if s_data["equity"] is not None:
            norm_equity = (s_data["equity"]['Equity'] / s_data["equity"]['Equity'].iloc[0]) * 100
            fig_eq.add_trace(go.Scatter(x=norm_equity.index, y=norm_equity, mode='lines', name=s_data["name"].split('.')[0], line=dict(color=colors[i % len(colors)])))
    fig_eq.update_layout(title="Comparativa de Curvas de Capital", yaxis_title="Capital Normalizado (Base 100)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_eq, use_container_width=True)

    st.markdown("---")
    st.subheader("Curvas de Drawdown por Fecha")
    fig_dd = go.Figure()
    for i, s_data in enumerate(strategies_data):
        if s_data["equity"] is not None:
            equity_curve = s_data["equity"]['Equity']
            dd_pct = (equity_curve - equity_curve.cummax()) / equity_curve.cummax() * 100
            fig_dd.add_trace(go.Scatter(x=dd_pct.index, y=dd_pct.values, mode='lines', name=s_data["name"].split('.')[0], line=dict(color=colors[i % len(colors)]), fill='tozeroy', opacity=0.6))
    fig_dd.update_layout(title="Comparativa de Curvas de Drawdown", yaxis_title="Drawdown (%)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_dd, use_container_width=True)

with tabs[2]:
    st.subheader("Detalle de Operaciones")
    for s_data in strategies_data:
        with st.expander(f"Ver operaciones de: {s_data['name']}"):
            st.dataframe(s_data["trades"])

with tabs[3]:
    st.header("🎲 Análisis Avanzado Individual")
    strategy_names = [s["name"] for s in strategies_data]
    if not strategy_names:
        st.warning("No hay estrategias cargadas para analizar.")
        st.stop()
        
    choice = st.selectbox("Elige la estrategia para analizar:", strategy_names, key="adv_choice")
    
    selected_data = next((s for s in strategies_data if s["name"] == choice), None)

    if selected_data:
        active_returns = selected_data["returns"]
        active_equity = selected_data["equity"]
        active_cap = selected_data["initial_cap"]
        active_name = selected_data["name"]

        if active_returns is None or active_returns.empty:
            st.warning(f"No hay datos de operaciones suficientes para realizar el análisis en '{active_name}'.")
        else:
            with st.expander("1. Simulación Monte Carlo (Bootstrap Simple)", expanded=True):
                n_sims = st.number_input("Número de simulaciones", 100, 10000, 1000, 100, key=f"n_sims_simple_{active_name}")
                if st.button("▶️ Ejecutar MC Simple", key=f"btn_simple_{active_name}"):
                    with st.spinner(f"Corriendo MC Simple para {active_name}..."):
                        sims_rel = run_monte_carlo(active_returns.values, n_sims, len(active_returns))
                        sims_eq = sims_rel * active_cap
                        final_vals = sims_eq[-1, :]; stats = {"Media": final_vals.mean(), "Mediana": np.median(final_vals), "VaR 95%": np.percentile(final_vals, 5)}
                        mdds = np.apply_along_axis(max_dd, 0, sims_eq) * 100
                        st.subheader("Resultados MC Simple"); st_cols = st.columns(len(stats))
                        for idx, (lbl, val) in enumerate(stats.items()): st_cols[idx].metric(lbl, f"${val:,.2f}")
                        fig1 = go.Figure(go.Histogram(x=final_vals, nbinsx=50)); fig1.update_layout(title="Histograma de Capital Final", showlegend=False); st.plotly_chart(fig1, use_container_width=True)
                        fig2 = go.Figure(go.Histogram(x=mdds, nbinsx=50)); fig2.update_layout(title="Histograma Max Drawdown (%)", showlegend=False); st.plotly_chart(fig2, use_container_width=True)

            with st.expander("2. Simulación Monte Carlo (Block Bootstrap)"):
                cols_bb = st.columns(2)
                block_size = cols_bb[0].number_input("Tamaño de bloque", 1, 100, 5, 1, key=f"block_size_{active_name}")
                n_sims_bb = cols_bb[1].number_input("Nº Simulaciones", 100, 10000, 1000, 100, key=f"n_sims_bb_{active_name}")
                if st.button("▶️ Ejecutar MC por Bloques", key=f"btn_bb_{active_name}"):
                    with st.spinner(f"Corriendo MC por Bloques para {active_name}..."):
                        sims_rel_bb = run_block_bootstrap_monte_carlo(active_returns.values, n_sims_bb, block_size, len(active_returns))
                        if sims_rel_bb is None: st.error("Error: Tamaño de bloque inválido.")
                        else:
                            sims_eq_bb = sims_rel_bb * active_cap; final_vals_bb = sims_eq_bb[-1, :]; stats_bb = {"Media": final_vals_bb.mean(), "Mediana": np.median(final_vals_bb), "VaR 95%": np.percentile(final_vals_bb, 5)}
                            st.subheader("Resultados MC por Bloques"); st_cols_bb = st.columns(len(stats_bb))
                            for idx, (lbl, val) in enumerate(stats_bb.items()): st_cols_bb[idx].metric(lbl, f"${val:,.2f}")
                            fig_hist_bb = go.Figure(go.Histogram(x=final_vals_bb, nbinsx=50)); fig_hist_bb.update_layout(title="Histograma de Capital Final (Bloques)", showlegend=False); st.plotly_chart(fig_hist_bb, use_container_width=True)
            
            with st.expander("3. Stress Test de Recuperación"):
                cols_st = st.columns(3)
                shock_pct = cols_st[0].number_input("Shock inicial (%)", -99.0, -1.0, -20.0, 1.0, format="%.1f", key=f"shock_{active_name}") / 100.0
                horizon_ops = cols_st[1].number_input("Horizonte recuperación (ops)", 1, 10000, 252, 1, key=f"horizon_{active_name}")
                n_sims_st = cols_st[2].number_input("Nº Simulaciones", 100, 10000, 500, 100, key=f"n_sims_st_{active_name}")
                if st.button("▶️ Ejecutar Stress Test", key=f"btn_st_{active_name}"):
                    with st.spinner(f"Corriendo Stress Test para {active_name}..."):
                        shocked_cap = active_cap * (1 + shock_pct); ret_arr = active_returns.values[np.isfinite(active_returns.values)]
                        sims_post_shock_rel = run_monte_carlo(ret_arr, n_sims_st, horizon_ops); sims_post_shock_abs = sims_post_shock_rel * shocked_cap
                        ttrs = [next((i+1 for i, v in enumerate(path) if v >= active_cap), np.nan) for path in sims_post_shock_abs.T]
                        recovered_count = np.count_nonzero(~np.isnan(ttrs)); pct_recov = 100 * recovered_count / n_sims_st if n_sims_st > 0 else 0
                        med_ttr = np.nanmedian(ttrs) if recovered_count > 0 else 'N/A'
                        st.subheader("Resultados Stress Test"); c1, c2 = st.columns(2)
                        c1.metric("% Recuperadas", f"{pct_recov:.1f}%"); c2.metric("Mediana Recuperación (ops)", f"{med_ttr:.0f}" if isinstance(med_ttr, (int, float)) else med_ttr)
                        fig_ttr = go.Figure()
                        if recovered_count > 0: fig_ttr.add_trace(go.Histogram(x=[t for t in ttrs if pd.notna(t)], nbinsx=50))
                        fig_ttr.update_layout(title="Histograma de Operaciones hasta Recuperación", showlegend=False); st.plotly_chart(fig_ttr, use_container_width=True)
    else:
        st.warning("No se encontraron datos para la estrategia seleccionada.")
