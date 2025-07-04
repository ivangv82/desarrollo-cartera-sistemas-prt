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

# --- carga datos ---
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

# --- curva de capital ---
def compute_equity(trades, init_cap):
    if trades is None or trades.empty: return None, None
    df = trades.sort_values('Exit Date').copy().reset_index(drop=True)
    equity_before_trade = init_cap + df['Profit'].shift(1).cumsum().fillna(0)
    df['Return'] = df['Profit'].divide(equity_before_trade.replace(0, np.nan)).fillna(0)
    df['Equity'] = init_cap + df['Profit'].cumsum()
    equity_curve = pd.DataFrame({'Date': pd.to_datetime([trades['Entry Date'].min()] + df['Exit Date'].tolist()), 'Equity': [init_cap] + df['Equity'].tolist()}).set_index('Date')
    return equity_curve, df['Return']

# --- metricas sistemas ---
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

# --- montecarlo ---
@st.cache_data
def run_monte_carlo(returns, n_sims, horizon):
    arr = np.asarray(returns, dtype=float); arr = arr[np.isfinite(arr)]
    if arr.size == 0 or horizon <= 0: return np.ones((horizon, n_sims))
    sims = np.zeros((horizon, n_sims))
    for i in range(n_sims):
        sample = np.random.choice(arr, size=horizon, replace=True)
        sims[:, i] = np.cumprod(1 + sample)
    return sims

# --- montecarlo bloques---
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

# --- mdd ---
def max_dd(equity_path):
    cm = np.maximum.accumulate(equity_path)
    dd = (equity_path - cm) / cm
    return dd.min()

# --- INICIALIZACIÓN DE ESTADO ---
if 'compare_mode' not in st.session_state:
    st.session_state.compare_mode = False

def activate_compare_mode():
    st.session_state.compare_mode = True

# --- SIDEBAR DINÁMICO ---
st.sidebar.header("📁 Carga de Datos")
trades_file_a = st.sidebar.file_uploader("Reporte de Estrategia", type=["csv","txt"], key="file_a")
initial_cap_a = st.sidebar.number_input("Capital Inicial", value=10000.0, min_value=0.0, step=1000.0, format="%.2f", key="cap_a")

trades_file_b = None
if st.session_state.compare_mode:
    trades_file_b = st.sidebar.file_uploader("Reporte de Estrategia B", type=["csv","txt"], key="file_b")
    initial_cap_b = st.sidebar.number_input("Capital Inicial B", value=10000.0, min_value=0.0, step=1000.0, format="%.2f", key="cap_b")
else:
    st.sidebar.button("➕ Comparar con otra Estrategia", on_click=activate_compare_mode, use_container_width=True)

# --- PARAMETROS GLOBALES ---
st.sidebar.header("⚙️ Parámetros Globales")
timeframe = st.sidebar.selectbox("Timeframe de Velas", ["1mn","5mn","15mn","30mn","1h","4h","1d","1w","1mes"], index=6)
if timeframe in ["1mn","5mn","15mn","30mn","1h","4h"]:
    trading_hours_per_day = st.sidebar.number_input("Horas de trading/día", 1.0, 24.0, 6.5, 0.5)
    minutes_in_tf = {"1mn":1, "5mn":5, "15mn":15, "30mn":30, "1h":60, "4h":240}[timeframe]
    ppy = (trading_hours_per_day * 60 / minutes_in_tf) * 252
else: ppy = {"1d":252, "1w":52, "1mes":12}[timeframe]
st.sidebar.caption(f"Periodos por año calculados: {int(ppy)}")

# --- LÓGICA DE PROCESAMIENTO Y RENDERIZADO ---
if not trades_file_a:
    st.info("Por favor, carga un archivo de estrategia para comenzar.")
    st.stop()

trades_a = load_prt_trades(trades_file_a)
equity_a, returns_a = compute_equity(trades_a, initial_cap_a)
metrics_a = calculate_metrics(trades_a, equity_a, ppy, timeframe)

############################################################################################
# --- MODO COMPARATIVO ---
############################################################################################
if st.session_state.compare_mode and trades_file_b:
    st.markdown(f"### Comparativa: `{trades_file_a.name}` vs `{trades_file_b.name}`")
    trades_b = load_prt_trades(trades_file_b)
    equity_b, returns_b = compute_equity(trades_b, initial_cap_b)
    metrics_b = calculate_metrics(trades_b, equity_b, ppy, timeframe)
    
    tabs = st.tabs(["📊 Resumen", "📈 Curvas", "📝 Operaciones", "🎲 Análisis Avanzado"])
    
    with tabs[0]:
        st.header("Resumen Comparativo de Métricas")
        if metrics_a and metrics_b:
            df_a = pd.Series(metrics_a, name="Estrategia A"); df_b = pd.Series(metrics_b, name="Estrategia B")
            df_comp = pd.concat([df_a, df_b], axis=1)
            def format_value(val, key):
                if pd.isna(val): return "-"
                if key in ["Beneficio Total", "Max Drawdown $"]: return f"${val:,.2f}"
                if key in ["Crecimiento Capital", "CAGR", "Max Drawdown %", "% Ganadoras", "Retorno Medio/Op. (%)"]: return f"{val:.2%}"
                if key in ["Total Operaciones"]: return f"{int(val)}"
                return f"{val:.2f}"
            df_display = df_comp.copy()
            for col in df_display.columns: df_display[col] = [format_value(val, idx) for idx, val in df_comp[col].items()]
            st.dataframe(df_display, use_container_width=True)
        else: st.warning("No se pudieron calcular las métricas para una o ambas estrategias.")
    # Draw down
    with tabs[1]:
        st.header("Curvas de Capital y Drawdowns")
        if equity_a is not None and equity_b is not None:
            st.subheader("Curvas de Capital (Normalizadas)")
            norm_equity_a = (equity_a['Equity'] / equity_a['Equity'].iloc[0]) * 100
            norm_equity_b = (equity_b['Equity'] / equity_b['Equity'].iloc[0]) * 100
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=norm_equity_a.index, y=norm_equity_a, mode='lines', name='Estrategia A', line=dict(color='royalblue')))
            fig_eq.add_trace(go.Scatter(x=norm_equity_b.index, y=norm_equity_b, mode='lines', name='Estrategia B', line=dict(color='darkorange')))
            fig_eq.update_layout(yaxis_title="Capital Normalizado (Base 100)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_eq, use_container_width=True)
            st.markdown("---")
            
            # --- GRÁFICO DE DRAWDOWN COMBINADO POR FECHA (CORREGIDO) ---
            st.subheader("Curvas de Drawdown por Fecha")
            dd_pct_a = (equity_a['Equity'] - equity_a['Equity'].cummax()) / equity_a['Equity'].cummax() * 100
            dd_pct_b = (equity_b['Equity'] - equity_b['Equity'].cummax()) / equity_b['Equity'].cummax() * 100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_pct_a.index, y=dd_pct_a.values, mode='lines', name='Drawdown A', line=dict(color='indianred'), fill='tozeroy', opacity=0.7))
            fig_dd.add_trace(go.Scatter(x=dd_pct_b.index, y=dd_pct_b.values, mode='lines', name='Drawdown B', line=dict(color='orange'), fill='tozeroy', opacity=0.7))
            fig_dd.update_layout(yaxis_title="Drawdown (%)", xaxis_title="Fecha", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_dd, use_container_width=True)    
    

    with tabs[2]:
        st.header("Detalle de Operaciones")
        with st.expander("Ver operaciones de Estrategia A"): st.dataframe(trades_a)
        with st.expander("Ver operaciones de Estrategia B"): st.dataframe(trades_b)
    # --- ANALISIS AVANZADO --- #
    with tabs[3]:
        st.header("🎲 Análisis Avanzado")
        
        # --- Selector de Estrategia ---
        strategy_choice = st.selectbox(
            "Elige la estrategia para analizar en detalle:",
            ("Estrategia A", "Estrategia B"),
            key="adv_choice_comp"
        )
    
        # Asignar los datos de la estrategia elegida a variables "activas"
        if strategy_choice == "Estrategia A":
            active_returns, active_equity, active_cap = returns_a, equity_a, initial_cap_a
        else:
            active_returns, active_equity, active_cap = returns_b, equity_b, initial_cap_b
    
        st.info(f"Mostrando análisis avanzado para **{strategy_choice}**.")
    
        if active_returns is None or active_returns.empty:
            st.warning("No hay datos de operaciones suficientes para realizar el análisis en la estrategia seleccionada.")
        else:
            # --- Expander para cada tipo de análisis ---
            with st.expander("1. Simulación Monte Carlo (Bootstrap Simple)"):
                n_sims = st.number_input("Número de simulaciones", 100, 10000, 1000, 100, key="c_mc_simple_sims")
                horizon = len(active_returns)
                
                if st.button("▶️ Ejecutar MC Simple", key="c_btn_mc_simple"):
                    with st.spinner("Corriendo simulaciones..."):
                        sims_rel = run_monte_carlo(active_returns.values, n_sims, horizon)
                        sims_eq = sims_rel * active_cap
                        final_vals = sims_eq[-1, :]
                        stats = {"Media": final_vals.mean(), "Mediana": np.median(final_vals), "VaR 95%": np.percentile(final_vals, 5)}
                        mdds = np.apply_along_axis(max_dd, 0, sims_eq) * 100
    
                    st.subheader("📈 Estadísticas Clave (MC Simple)")
                    stat_cols = st.columns(len(stats))
                    for idx, (label, value) in enumerate(stats.items()):
                        stat_cols[idx].metric(label, f"${value:,.2f}")
                    
                    # Gráficos...
                    fig_hist_simple = go.Figure(go.Histogram(x=final_vals, nbinsx=50, name="Frecuencia"))
                    fig_hist_simple.update_layout(title="Histograma de Capital Final")
                    st.plotly_chart(fig_hist_simple, use_container_width=True)
    
    
            with st.expander("2. Simulación Monte Carlo (Block Bootstrap)"):
                cols_bb = st.columns(2)
                block_size = cols_bb[0].number_input("Tamaño de bloque", 1, 100, 5, 1, key="c_mc_block_size")
                n_sims_bb = cols_bb[1].number_input("Nº Simulaciones", 100, 10000, 1000, 100, key="c_mc_block_sims")
                horizon_bb = len(active_returns)
                
                if st.button("▶️ Ejecutar MC por Bloques", key="c_btn_mc_block"):
                    with st.spinner("Corriendo simulaciones con bloques..."):
                        sims_rel_bb = run_block_bootstrap_monte_carlo(active_returns.values, n_sims_bb, block_size, horizon_bb)
                        if sims_rel_bb is None:
                            st.error(f"Error: El tamaño de bloque ({block_size}) es inválido.")
                        else:
                            sims_eq_bb = sims_rel_bb * active_cap
                            final_vals_bb = sims_eq_bb[-1, :]
                            stats_bb = {"Media": final_vals_bb.mean(), "Mediana": np.median(final_vals_bb), "VaR 95%": np.percentile(final_vals_bb, 5)}
                            mdds_bb = np.apply_along_axis(max_dd, 0, sims_eq_bb) * 100
    
                            st.subheader("📈 Estadísticas Clave (Block Bootstrap)")
                            stat_cols_bb = st.columns(len(stats_bb))
                            for idx, (label, value) in enumerate(stats_bb.items()):
                                stat_cols_bb[idx].metric(label, f"${value:,.2f}")
    
                            # Gráficos...
                            fig_hist_bb = go.Figure(go.Histogram(x=final_vals_bb, nbinsx=50, name="Frecuencia"))
                            fig_hist_bb.update_layout(title="Histograma de Capital Final (Block Bootstrap)")
                            st.plotly_chart(fig_hist_bb, use_container_width=True)
    
    
            with st.expander("3. Stress Test de Recuperación"):
                cols_st = st.columns(3)
                shock_pct = cols_st[0].number_input("Shock inicial (%)", -99.0, -1.0, -20.0, 1.0, format="%.1f", key="c_st_shock") / 100.0
                horizon_ops = cols_st[1].number_input("Horizonte recuperación (ops)", 1, 10000, 252, 1, key="c_st_horizon")
                n_sims_st = cols_st[2].number_input("Nº Simulaciones", 100, 10000, 500, 100, key="c_st_sims")
                
                if st.button("▶️ Ejecutar Stress Test", key="c_btn_st"):
                    with st.spinner("Corriendo stress tests..."):
                        shocked_cap = active_cap * (1 + shock_pct)
                        ret_arr = active_returns.values[np.isfinite(active_returns.values)]
                        sims_post_shock_rel = run_monte_carlo(ret_arr, n_sims_st, horizon_ops)
                        sims_post_shock_abs = sims_post_shock_rel * shocked_cap
    
                        ttrs = []
                        for i in range(n_sims_st):
                            path = sims_post_shock_abs[:, i]
                            rec_indices = np.where(path >= active_cap)[0]
                            ttrs.append(rec_indices[0] + 1 if rec_indices.size > 0 else np.nan)
                        ttrs = np.array(ttrs)
                        
                        recovered_count = np.count_nonzero(~np.isnan(ttrs))
                        pct_recov = 100 * recovered_count / n_sims_st if n_sims_st > 0 else 0
                        med_ttr = np.nanmedian(ttrs) if recovered_count > 0 else 'N/A'
                        
                        st.subheader("📊 Estadísticas de Recuperación")
                        c1, c2 = st.columns(2)
                        c1.metric("% Simulaciones Recuperadas", f"{pct_recov:.1f}%")
                        c2.metric("Mediana Tiempo Recuperación (ops)", f"{med_ttr:.0f}" if isinstance(med_ttr, (int, float)) else med_ttr)
    
                        # Gráfico...
                        fig_ttr = go.Figure()
                        if recovered_count > 0:
                            fig_ttr.add_trace(go.Histogram(x=ttrs[~np.isnan(ttrs)], nbinsx=50))
                        fig_ttr.update_layout(title="Histograma de Operaciones hasta Recuperación", xaxis_title="Operaciones", yaxis_title="Frecuencia")
                        st.plotly_chart(fig_ttr, use_container_width=True)
############################################################################################
# --- MODO ANÁLISIS INDIVIDUAL ---
############################################################################################
else:
    st.markdown(f"### Análisis Individual: `{trades_file_a.name}`")
    tabs = st.tabs(["📊 Resumen", "📈 Equity & DD", "📝 Operaciones", "🎲 MC Simple", "🎲 MC Bloques", "⚠️ Stress Test"])

    with tabs[0]:
        st.header("Resumen de Métricas")
        cols = st.columns(4)
        metric_order = ["Beneficio Total", "Crecimiento Capital", "CAGR", "Sharpe Ratio", "Max Drawdown $", "Max Drawdown %", "Recovery Factor", "Calmar Ratio", "Total Operaciones", "% Ganadoras", "Factor de Beneficio", "Ratio Payoff", "Retorno Medio/Op. (%)", "Duración Media (velas)"]
        for i, key in enumerate(metric_order):
            if key not in metrics_a: continue
            val = metrics_a[key]
            if key in ["Beneficio Total", "Max Drawdown $"]: disp = f"${val:,.2f}"
            elif key in ["Crecimiento Capital", "CAGR", "Max Drawdown %", "% Ganadoras", "Retorno Medio/Op. (%)"]: disp = f"{val*100:.2f}%"
            elif np.isinf(val): disp = "∞"
            elif isinstance(val, (int, float)): disp = f"{val:.2f}"
            else: disp = str(val)
            cols[i%4].metric(label=key, value=disp)
    
    with tabs[1]:
        st.header("Curva de Equity y Drawdown")
        if equity_a is not None:
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7,0.3], subplot_titles=("Curva de Equity","Drawdown (%)"))
            fig.add_trace(go.Scatter(x=equity_a.index, y=equity_a['Equity'], mode='lines', name='Equity', line=dict(color='royalblue')), row=1, col=1)
            dd_pct = (equity_a['Equity'] - equity_a['Equity'].cummax()) / equity_a['Equity'].cummax() * 100
            fig.add_trace(go.Scatter(x=dd_pct.index, y=dd_pct.values, fill='tozeroy', mode='lines', name='Drawdown', line=dict(color='indianred')), row=2, col=1)
            fig.update_layout(height=500, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.header("Detalle de Operaciones")
        st.dataframe(trades_a)

    # Lógica completa para las pestañas avanzadas en modo individual
    
    # --- MONTECARLO SIMPLE --- #
    with tabs[3]:
        st.header("🎲 Simulación Monte Carlo (Bootstrap Simple)")
        st.markdown("Muestreo aleatorio de los retornos por operación para simular miles de posibles futuros de la curva de capital.")
    
        n_sims = st.number_input("Número de simulaciones", 100, 10000, 1000, 100, key="s_mc_simple_sims")
        
        if returns_a is None or returns_a.empty:
            st.warning("No hay operaciones suficientes para ejecutar la simulación.")
        else:
            horizon = len(returns_a)
            if st.button("▶️ Ejecutar MC Simple", key="s_btn_mc_simple"):
                with st.spinner("Corriendo simulaciones..."):
                    # Simulación
                    sims_rel = run_monte_carlo(returns_a.values, n_sims, horizon)
                    sims_eq = sims_rel * initial_cap_a
    
                    # Estadísticas finales
                    final_vals = sims_eq[-1, :]
                    stats = {
                        "Media": final_vals.mean(), "Mediana": np.median(final_vals),
                        "P10": np.percentile(final_vals, 10), "P90": np.percentile(final_vals, 90),
                        "VaR 95%": np.percentile(final_vals, 5), "CVaR 95%": final_vals[final_vals <= np.percentile(final_vals, 5)].mean()
                    }
                    
                    # MDD por simulación
                    mdds = np.apply_along_axis(max_dd, 0, sims_eq) * 100
    
                st.subheader("📈 Estadísticas del Capital Final")
                stat_cols = st.columns(len(stats))
                for idx, (label, value) in enumerate(stats.items()):
                    stat_cols[idx].metric(label, f"${value:,.2f}")
    
                # Envelope plot
                sim_plot_dates = equity_a.index[1:]
                fig_env = go.Figure()
                fig_env.add_trace(go.Scatter(x=sim_plot_dates, y=np.percentile(sims_eq, 95, axis=1), fill=None, mode='lines', line_color='lightgrey', showlegend=False))
                fig_env.add_trace(go.Scatter(x=sim_plot_dates, y=np.percentile(sims_eq, 5, axis=1), fill='tonexty', mode='lines', line_color='lightgrey', name='5%-95%'))
                fig_env.add_trace(go.Scatter(x=sim_plot_dates, y=np.percentile(sims_eq, 50, axis=1), mode='lines', name='Mediana', line=dict(color='orange', dash='dash')))
                fig_env.add_trace(go.Scatter(x=equity_a.index, y=equity_a['Equity'], mode='lines', name='Histórico', line=dict(color='blue', width=3)))
                fig_env.update_layout(title="Simulaciones vs. Curva Histórica", xaxis_title='Fecha', yaxis_title='Capital')
                st.plotly_chart(fig_env, use_container_width=True)
    
                # Histogramas
                st.subheader("📊 Histograma Capital Final")
                hist1 = go.Figure()
                hist1.add_trace(go.Histogram(x=final_vals, nbinsx=50, name="Frecuencia"))
                hist1.add_vline(x=stats["Media"], line_dash="dash", annotation_text="Media", line_color="black")
                hist1.add_vline(x=stats["Mediana"], line_dash="dash", annotation_text="Mediana", line_color="orange")
                hist1.add_vline(x=stats["VaR 95%"], line_dash="dot", annotation_text="VaR 95%", line_color="red")
                hist1.update_layout(xaxis_title="Capital Final", yaxis_title="Frecuencia", showlegend=False)
                st.plotly_chart(hist1, use_container_width=True)
    
                st.subheader("📉 Histograma de Max Drawdown (%)")
                hist2 = go.Figure()
                hist2.add_trace(go.Histogram(x=mdds, nbinsx=50, name="Frecuencia"))
                hist2.add_vline(x=np.median(mdds), line_dash="dash", annotation_text="Mediana", line_color="orange")
                hist2.add_vline(x=np.percentile(mdds, 5), line_dash="dot", annotation_text="P95 (peor 5%)", line_color="red")
                hist2.update_layout(xaxis_title="Max Drawdown (%)", yaxis_title="Frecuencia", showlegend=False)
                st.plotly_chart(hist2, use_container_width=True)

    # --- MONTECARLO POR BLOQUES --- #
    with tabs[4]:
        st.header("🎲 Simulación Monte Carlo (Block Bootstrap)")
        st.markdown("Muestrea bloques de retornos para intentar preservar la autocorrelación de la estrategia.")
        
        cols_bb = st.columns(2)
        block_size = cols_bb[0].number_input("Tamaño de bloque (operaciones)", 1, 100, 5, 1, key="s_mc_block_size")
        n_sims_bb = cols_bb[1].number_input("Nº Simulaciones", 100, 10000, 1000, 100, key="s_mc_block_sims")
    
        if returns_a is None or returns_a.empty:
            st.warning("No hay operaciones suficientes para ejecutar la simulación.")
        else:
            horizon_bb = len(returns_a)
            if st.button("▶️ Ejecutar MC por Bloques", key="s_btn_mc_block"):
                with st.spinner("Corriendo simulaciones con bloques..."):
                    sims_rel_bb = run_block_bootstrap_monte_carlo(returns_a.values, n_sims_bb, block_size, horizon_bb)
    
                    if sims_rel_bb is None:
                        st.error(f"Error: El tamaño de bloque ({block_size}) debe ser menor que el número de operaciones ({horizon_bb}).")
                    else:
                        sims_eq_bb = sims_rel_bb * initial_cap_a
                        final_vals_bb = sims_eq_bb[-1, :]
    
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
                        
                        sim_plot_dates = equity_a.index[1:]
                        fig_env_bb = go.Figure()
                        fig_env_bb.add_trace(go.Scatter(x=sim_plot_dates, y=np.percentile(sims_eq_bb, 95, axis=1), fill=None, mode='lines', line_color='lightgrey', showlegend=False))
                        fig_env_bb.add_trace(go.Scatter(x=sim_plot_dates, y=np.percentile(sims_eq_bb, 5, axis=1), fill='tonexty', mode='lines', line_color='lightgrey', name='5%-95%'))
                        fig_env_bb.add_trace(go.Scatter(x=sim_plot_dates, y=np.percentile(sims_eq_bb, 50, axis=1), mode='lines', name='Mediana', line=dict(color='orange', dash='dash')))
                        fig_env_bb.add_trace(go.Scatter(x=equity_a.index, y=equity_a['Equity'], mode='lines', name='Histórico', line=dict(color='blue', width=3)))
                        fig_env_bb.update_layout(title="Simulaciones (Block Bootstrap) vs. Curva Histórica", xaxis_title='Fecha', yaxis_title='Capital')
                        st.plotly_chart(fig_env_bb, use_container_width=True)
                        
                        st.subheader("📊 Histograma Capital Final (Block Bootstrap)")
                        hist1_bb = go.Figure()
                        hist1_bb.add_trace(go.Histogram(x=final_vals_bb, nbinsx=50, name="Frecuencia"))
                        hist1_bb.add_vline(x=stats_bb["Media"], line_dash="dash", annotation_text="Media", line_color="black")
                        hist1_bb.add_vline(x=stats_bb["Mediana"], line_dash="dash", annotation_text="Mediana", line_color="orange")
                        hist1_bb.add_vline(x=stats_bb["VaR 95%"], line_dash="dot", annotation_text="VaR 95%", line_color="red")
                        hist1_bb.update_layout(xaxis_title="Capital Final", yaxis_title="Frecuencia", showlegend=False)
                        st.plotly_chart(hist1_bb, use_container_width=True)
    
                        st.subheader("📉 Histograma de Max Drawdown (%) (Block Bootstrap)")
                        hist2_bb = go.Figure()
                        hist2_bb.add_trace(go.Histogram(x=mdds_bb, nbinsx=50, name="Frecuencia"))
                        hist2_bb.add_vline(x=np.median(mdds_bb), line_dash="dash", annotation_text="Mediana", line_color="orange")
                        hist2_bb.add_vline(x=np.percentile(mdds_bb, 5), line_dash="dot", annotation_text="P95 (peor 5%)", line_color="red")
                        hist2_bb.update_layout(xaxis_title="Max Drawdown (%)", yaxis_title="Frecuencia", showlegend=False)
                        st.plotly_chart(hist2_bb, use_container_width=True)

    # --- TEST DE STRESS --- #
    with tabs[5]:
        st.header("⚠️ Stress Test (Recuperación tras un Shock)")
        st.markdown("Simula la recuperación del capital tras una pérdida inicial súbita, medido en número de operaciones.")
    
        cols_st = st.columns(3)
        shock_pct = cols_st[0].number_input("Shock inicial (%)", -99.0, -1.0, -20.0, 1.0, format="%.1f", key="s_st_shock") / 100.0
        horizon_ops = cols_st[1].number_input("Horizonte recuperación (ops)", 1, 10000, 252, 1, key="s_st_horizon")
        n_sims_st = cols_st[2].number_input("Nº Simulaciones", 100, 10000, 500, 100, key="s_st_sims")
    
        if returns_a is None or returns_a.empty:
            st.warning("No hay operaciones suficientes para ejecutar el stress test.")
        else:
            if st.button("▶️ Ejecutar Stress Test", key="s_btn_st"):
                with st.spinner("Corriendo stress tests..."):
                    shocked_cap = initial_cap_a * (1 + shock_pct)
                    ret_arr = returns_a.values[np.isfinite(returns_a.values)]
                    
                    # Simulación
                    sims_post_shock_rel = run_monte_carlo(ret_arr, n_sims_st, horizon_ops)
                    sims_post_shock_abs = sims_post_shock_rel * shocked_cap
    
                    # Tiempo hasta recuperación (Time to Recovery - TTR)
                    ttrs = []
                    for i in range(n_sims_st):
                        path = sims_post_shock_abs[:, i]
                        rec_indices = np.where(path >= initial_cap_a)[0]
                        ttrs.append(rec_indices[0] + 1 if rec_indices.size > 0 else np.nan)
                    ttrs = np.array(ttrs)
    
                    # Métricas de recuperación
                    recovered_count = np.count_nonzero(~np.isnan(ttrs))
                    pct_recov = 100 * recovered_count / n_sims_st if n_sims_st > 0 else 0
                    med_ttr = np.nanmedian(ttrs) if recovered_count > 0 else 'N/A'
                    p90_ttr = np.nanpercentile(ttrs, 90) if recovered_count > 0 else 'N/A'
    
                st.subheader("📊 Estadísticas de Recuperación")
                c1, c2, c3 = st.columns(3)
                c1.metric("% Simulaciones Recuperadas", f"{pct_recov:.1f}%")
                c2.metric("Mediana Tiempo Recuperación (ops)", f"{med_ttr:.0f}" if isinstance(med_ttr, (int, float)) else med_ttr)
                c3.metric("P90 Tiempo Recuperación (ops)", f"{p90_ttr:.0f}" if isinstance(p90_ttr, (int, float)) else p90_ttr)
    
                # Histograma de TTR
                st.subheader("📈 Histograma de Operaciones hasta Recuperación")
                fig_ttr = go.Figure()
                if recovered_count > 0:
                    fig_ttr.add_trace(go.Histogram(x=ttrs[~np.isnan(ttrs)], nbinsx=50))
                fig_ttr.update_layout(xaxis_title="Operaciones hasta recuperar capital inicial", yaxis_title="Frecuencia")
                st.plotly_chart(fig_ttr, use_container_width=True)
