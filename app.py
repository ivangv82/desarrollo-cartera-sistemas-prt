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

# --- MODO COMPARATIVO ---
if st.session_state.compare_mode and trades_file_b:
    st.markdown(f"### Comparativa: `{trades_file_a.name}` vs `{trades_file_b.name}`")
    trades_b = load_prt_trades(trades_file_b)
    equity_b, returns_b = compute_equity(trades_b, initial_cap_b)
    metrics_b = calculate_metrics(trades_b, equity_b, ppy, timeframe)
    
    tabs = st.tabs(["📊 Resumen", "📈 Curvas", "📝 Operaciones", "🎲 Análisis Avanzado"])
    
    with tabs[0]:
        st.header("Resumen Comparativo de Métricas")
        if metrics_a and metrics_b:
            df_a = pd.Series(metrics_a, name=f"A: {trades_file_a.name}"); df_b = pd.Series(metrics_b, name=f"B: {trades_file_b.name}")
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

    with tabs[1]:
        st.header("Curvas de Capital y Drawdowns")
        if equity_a is not None and equity_b is not None:
            st.subheader("Curvas de Capital (Normalizadas)")
            norm_equity_a = (equity_a['Equity'] / equity_a['Equity'].iloc[0]) * 100
            norm_equity_b = (equity_b['Equity'] / equity_b['Equity'].iloc[0]) * 100
            fig_eq = go.Figure()
            fig_eq.add_trace(go.Scatter(x=norm_equity_a.index, y=norm_equity_a, mode='lines', name=f"A: {trades_file_a.name}", line=dict(color='royalblue')))
            fig_eq.add_trace(go.Scatter(x=norm_equity_b.index, y=norm_equity_b, mode='lines', name=f"B: {trades_file_b.name}", line=dict(color='darkorange')))
            fig_eq.update_layout(yaxis_title="Capital Normalizado (Base 100)", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_eq, use_container_width=True)
            st.markdown("---")
            
            st.subheader("Curvas de Drawdown por Fecha")
            dd_pct_a = (equity_a['Equity'] - equity_a['Equity'].cummax()) / equity_a['Equity'].cummax() * 100
            dd_pct_b = (equity_b['Equity'] - equity_b['Equity'].cummax()) / equity_b['Equity'].cummax() * 100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_pct_a.index, y=dd_pct_a.values, mode='lines', name=f"Drawdown A", line=dict(color='indianred'), fill='tozeroy', opacity=0.7))
            fig_dd.add_trace(go.Scatter(x=dd_pct_b.index, y=dd_pct_b.values, mode='lines', name=f"Drawdown B", line=dict(color='orange'), fill='tozeroy', opacity=0.7))
            fig_dd.update_layout(yaxis_title="Drawdown (%)", xaxis_title="Fecha", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_dd, use_container_width=True)

    with tabs[2]:
        st.header("Detalle de Operaciones")
        with st.expander(f"Ver operaciones de Estrategia A: {trades_file_a.name}"): st.dataframe(trades_a)
        with st.expander(f"Ver operaciones de Estrategia B: {trades_file_b.name}"): st.dataframe(trades_b)
    
    with tabs[3]:
        st.header("🎲 Análisis Avanzado")
        strategy_choice = st.selectbox("Elige la estrategia para analizar en detalle:", (f"Estrategia A: {trades_file_a.name}", f"Estrategia B: {trades_file_b.name}"), key="adv_choice_comp")

        if "Estrategia A" in strategy_choice:
            active_returns, active_equity, active_cap, active_name = returns_a, equity_a, initial_cap_a, "Estrategia A"
        else:
            active_returns, active_equity, active_cap, active_name = returns_b, equity_b, initial_cap_b, "Estrategia B"

        st.info(f"Mostrando análisis avanzado para **{active_name}**.")

        if active_returns is None or active_returns.empty:
            st.warning(f"No hay datos de operaciones suficientes para realizar el análisis en {active_name}.")
        else:
            with st.expander("1. Simulación Monte Carlo (Bootstrap Simple)", expanded=True):
                n_sims_simple = st.number_input("Número de simulaciones", 100, 10000, 1000, 100, key="c_mc_simple_sims")
                if st.button("▶️ Ejecutar MC Simple", key="c_btn_mc_simple"):
                    with st.spinner(f"Corriendo MC Simple para {active_name}..."):
                        sims_rel = run_monte_carlo(active_returns.values, n_sims_simple, len(active_returns))
                        sims_eq = sims_rel * active_cap
                        final_vals = sims_eq[-1, :]
                        stats = {"Media": final_vals.mean(), "Mediana": np.median(final_vals), "VaR 95%": np.percentile(final_vals, 5)}
                        mdds = np.apply_along_axis(max_dd, 0, sims_eq) * 100
                        st.subheader("Resultados MC Simple"); st_cols = st.columns(len(stats));
                        for idx, (lbl, val) in enumerate(stats.items()): st_cols[idx].metric(lbl, f"${val:,.2f}")
                        fig1 = go.Figure(go.Histogram(x=final_vals, nbinsx=50)); fig1.update_layout(title="Histograma Capital Final"); st.plotly_chart(fig1, use_container_width=True)
                        fig2 = go.Figure(go.Histogram(x=mdds, nbinsx=50)); fig2.update_layout(title="Histograma Max Drawdown (%)"); st.plotly_chart(fig2, use_container_width=True)

            with st.expander("2. Simulación Monte Carlo (Block Bootstrap)"):
                cols_bb = st.columns(2)
                block_size = cols_bb[0].number_input("Tamaño de bloque", 1, 100, 5, 1, key="c_mc_block_size")
                n_sims_bb = cols_bb[1].number_input("Nº Simulaciones", 100, 10000, 1000, 100, key="c_mc_block_sims")
                if st.button("▶️ Ejecutar MC por Bloques", key="c_btn_mc_block"):
                    with st.spinner(f"Corriendo MC por Bloques para {active_name}..."):
                        sims_rel_bb = run_block_bootstrap_monte_carlo(active_returns.values, n_sims_bb, block_size, len(active_returns))
                        if sims_rel_bb is None: st.error("Error: Tamaño de bloque inválido.")
                        else:
                            sims_eq_bb = sims_rel_bb * active_cap; final_vals_bb = sims_eq_bb[-1, :]
                            stats_bb = {"Media": final_vals_bb.mean(), "Mediana": np.median(final_vals_bb), "VaR 95%": np.percentile(final_vals_bb, 5)}
                            st.subheader("Resultados MC por Bloques"); st_cols_bb = st.columns(len(stats_bb))
                            for idx, (lbl, val) in enumerate(stats_bb.items()): st_cols_bb[idx].metric(lbl, f"${val:,.2f}")
                            fig_bb = go.Figure(go.Histogram(x=final_vals_bb, nbinsx=50)); fig_bb.update_layout(title="Histograma Capital Final (Bloques)"); st.plotly_chart(fig_bb, use_container_width=True)

            with st.expander("3. Stress Test de Recuperación"):
                cols_st = st.columns(3)
                shock_pct = cols_st[0].number_input("Shock inicial (%)", -99.0, -1.0, -20.0, 1.0, format="%.1f", key="c_st_shock") / 100.0
                horizon_ops = cols_st[1].number_input("Horizonte recuperación (ops)", 1, 10000, 252, 1, key="c_st_horizon")
                n_sims_st = cols_st[2].number_input("Nº Simulaciones", 100, 10000, 500, 100, key="c_st_sims")
                if st.button("▶️ Ejecutar Stress Test", key="c_btn_st"):
                    with st.spinner(f"Corriendo Stress Test para {active_name}..."):
                        shocked_cap = active_cap * (1 + shock_pct); ret_arr = active_returns.values[np.isfinite(active_returns.values)]
                        sims_post_shock_rel = run_monte_carlo(ret_arr, n_sims_st, horizon_ops); sims_post_shock_abs = sims_post_shock_rel * shocked_cap
                        ttrs = [next((i+1 for i, v in enumerate(path) if v >= active_cap), np.nan) for path in sims_post_shock_abs.T]
                        recovered_count = np.count_nonzero(~np.isnan(ttrs)); pct_recov = 100 * recovered_count / n_sims_st if n_sims_st > 0 else 0
                        med_ttr = np.nanmedian(ttrs) if recovered_count > 0 else 'N/A'
                        st.subheader("Resultados Stress Test"); c1, c2 = st.columns(2)
                        c1.metric("% Recuperadas", f"{pct_recov:.1f}%"); c2.metric("Mediana Recuperación (ops)", f"{med_ttr:.0f}" if isinstance(med_ttr, (int, float)) else med_ttr)
                        fig_ttr = go.Figure();
                        if recovered_count > 0: fig_ttr.add_trace(go.Histogram(x=[t for t in ttrs if pd.notna(t)], nbinsx=50))
                        fig_ttr.update_layout(title="Histograma de Operaciones hasta Recuperación"); st.plotly_chart(fig_ttr, use_container_width=True)

# --- MODO ANÁLISIS INDIVIDUAL ---
else:
    st.markdown(f"### Análisis Individual: `{trades_file_a.name}`")
    tabs = st.tabs(["📊 Resumen", "📈 Equity & DD", "📝 Operaciones", "🎲 MC Simple", "🎲 MC Bloques", "⚠️ Stress Test"])

    with tabs[0]:
        st.header("Resumen de Métricas")
        if not metrics_a: st.warning("No se pudieron calcular las métricas.")
        else:
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

    with tabs[3]:
        st.header("🎲 Simulación Monte Carlo (Bootstrap Simple)")
        n_sims_s = st.number_input("Número de simulaciones", 100, 10000, 1000, 100, key="s_mc_simple_sims")
        if returns_a is None or returns_a.empty: st.warning("No hay operaciones suficientes.")
        else:
            horizon_s = len(returns_a)
            if st.button("▶️ Ejecutar MC Simple", key="s_btn_mc_simple"):
                with st.spinner("Corriendo simulaciones..."):
                    sims_rel_s = run_monte_carlo(returns_a.values, n_sims_s, horizon_s); sims_eq_s = sims_rel_s * initial_cap_a
                    final_vals_s = sims_eq_s[-1, :]; stats_s = {"Media": final_vals_s.mean(), "Mediana": np.median(final_vals_s), "P10": np.percentile(final_vals_s, 10), "P90": np.percentile(final_vals_s, 90), "VaR 95%": np.percentile(final_vals_s, 5), "CVaR 95%": final_vals_s[final_vals_s <= np.percentile(final_vals_s, 5)].mean()}; mdds_s = np.apply_along_axis(max_dd, 0, sims_eq_s) * 100
                    st.subheader("📈 Estadísticas del Capital Final"); stat_cols_s = st.columns(len(stats_s))
                    for idx, (label, value) in enumerate(stats_s.items()): stat_cols_s[idx].metric(label, f"${value:,.2f}")
                    sim_plot_dates_s = equity_a.index[1:]; fig_env_s = go.Figure()
                    fig_env_s.add_trace(go.Scatter(x=sim_plot_dates_s, y=np.percentile(sims_eq_s, 95, axis=1), fill=None, mode='lines', line_color='lightgrey', showlegend=False)); fig_env_s.add_trace(go.Scatter(x=sim_plot_dates_s, y=np.percentile(sims_eq_s, 5, axis=1), fill='tonexty', mode='lines', line_color='lightgrey', name='5%-95%'))
                    fig_env_s.add_trace(go.Scatter(x=sim_plot_dates_s, y=np.percentile(sims_eq_s, 50, axis=1), mode='lines', name='Mediana', line=dict(color='orange', dash='dash'))); fig_env_s.add_trace(go.Scatter(x=equity_a.index, y=equity_a['Equity'], mode='lines', name='Histórico', line=dict(color='blue', width=3)))
                    fig_env_s.update_layout(title="Simulaciones vs. Curva Histórica", xaxis_title='Fecha', yaxis_title='Capital'); st.plotly_chart(fig_env_s, use_container_width=True)
                    st.subheader("📊 Histograma Capital Final"); hist1_s = go.Figure(go.Histogram(x=final_vals_s, nbinsx=50)); hist1_s.add_vline(x=stats_s["Media"], line_dash="dash", annotation_text="Media", line_color="black"); hist1_s.add_vline(x=stats_s["Mediana"], line_dash="dash", annotation_text="Mediana", line_color="orange"); hist1_s.add_vline(x=stats_s["VaR 95%"], line_dash="dot", annotation_text="VaR 95%", line_color="red"); hist1_s.update_layout(showlegend=False); st.plotly_chart(hist1_s, use_container_width=True)
                    st.subheader("📉 Histograma de Max Drawdown (%)"); hist2_s = go.Figure(go.Histogram(x=mdds_s, nbinsx=50)); hist2_s.add_vline(x=np.median(mdds_s), line_dash="dash", annotation_text="Mediana", line_color="orange"); hist2_s.add_vline(x=np.percentile(mdds_s, 5), line_dash="dot", annotation_text="P95 (peor 5%)", line_color="red"); hist2_s.update_layout(showlegend=False); st.plotly_chart(hist2_s, use_container_width=True)

    with tabs[4]:
        st.header("🎲 Simulación Monte Carlo (Block Bootstrap)")
        cols_bb_s = st.columns(2)
        block_size_s = cols_bb_s[0].number_input("Tamaño de bloque", 1, 100, 5, 1, key="s_mc_block_size")
        n_sims_bb_s = cols_bb_s[1].number_input("Nº Simulaciones", 100, 10000, 1000, 100, key="s_mc_block_sims")
        if returns_a is None or returns_a.empty: st.warning("No hay operaciones suficientes.")
        else:
            horizon_bb_s = len(returns_a)
            if st.button("▶️ Ejecutar MC por Bloques", key="s_btn_mc_block"):
                with st.spinner("Corriendo simulaciones con bloques..."):
                    sims_rel_bb_s = run_block_bootstrap_monte_carlo(returns_a.values, n_sims_bb_s, block_size_s, horizon_bb_s)
                    if sims_rel_bb_s is None: st.error("Error: Tamaño de bloque inválido.")
                    else:
                        sims_eq_bb_s = sims_rel_bb_s * initial_cap_a; final_vals_bb_s = sims_eq_bb_s[-1, :]; stats_bb_s = {"Media": final_vals_bb_s.mean(), "Mediana": np.median(final_vals_bb_s), "P10": np.percentile(final_vals_bb_s, 10), "P90": np.percentile(final_vals_bb_s, 90), "VaR 95%": np.percentile(final_vals_bb_s, 5), "CVaR 95%": final_vals_bb_s[final_vals_bb_s <= np.percentile(final_vals_bb_s, 5)].mean()}; mdds_bb_s = np.apply_along_axis(max_dd, 0, sims_eq_bb_s) * 100
                        st.subheader("📈 Estadísticas del Capital Final (Block Bootstrap)"); stat_cols_bb_s = st.columns(len(stats_bb_s)); 
                        for idx, (lbl, val) in enumerate(stats_bb_s.items()): stat_cols_bb_s[idx].metric(lbl, f"${val:,.2f}")
                        # ... (resto de gráficos para MC Bloques)

    with tabs[5]:
        st.header("⚠️ Stress Test")
        cols_st_s = st.columns(3)
        shock_pct_s = cols_st_s[0].number_input("Shock inicial (%)", -99.0, -1.0, -20.0, 1.0, format="%.1f", key="s_st_shock") / 100.0
        horizon_ops_s = cols_st_s[1].number_input("Horizonte recuperación (ops)", 1, 10000, 252, 1, key="s_st_horizon")
        n_sims_st_s = cols_st_s[2].number_input("Nº Simulaciones", 100, 10000, 500, 100, key="s_st_sims")
        if returns_a is None or returns_a.empty: st.warning("No hay operaciones suficientes.")
        else:
            if st.button("▶️ Ejecutar Stress Test", key="s_btn_st"):
                with st.spinner("Corriendo stress tests..."):
                    shocked_cap_s = initial_cap_a * (1 + shock_pct_s); ret_arr_s = returns_a.values[np.isfinite(returns_a.values)]
                    sims_rel_st = run_monte_carlo(ret_arr_s, n_sims_st_s, horizon_ops_s); sims_abs_st = sims_rel_st * shocked_cap_s
                    ttrs_s = [next((i+1 for i, v in enumerate(path) if v >= initial_cap_a), np.nan) for path in sims_abs_st.T]
                    recovered_count_s = np.count_nonzero(~np.isnan(ttrs_s)); pct_recov_s = 100 * recovered_count_s / n_sims_st_s if n_sims_st_s > 0 else 0
                    med_ttr_s = np.nanmedian(ttrs_s) if recovered_count_s > 0 else 'N/A'; p90_ttr_s = np.nanpercentile(ttrs_s, 90) if recovered_count_s > 0 else 'N/A'
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

