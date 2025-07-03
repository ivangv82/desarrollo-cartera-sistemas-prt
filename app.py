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
tabs = st.tabs([
    "📊 Resumen", "📈 Equity & DD", "📝 Operaciones",
    "🎲 MC Simple", "🎲 MC Bloques", "⚠️ Stress Test"
])

with tabs[0]:
    st.header("📋 Resumen de Métricas")
    cols = st.columns(4)
    metric_order = [
        "Beneficio Total", "Crecimiento Capital", "CAGR", "Sharpe Ratio",
        "Max Drawdown $", "Max Drawdown %", "Recovery Factor", "Calmar Ratio",
        "Total Operaciones", "% Ganadoras", "Factor de Beneficio", "Ratio Payoff",
        "Retorno Medio/Op. (%)", "Duración Media (velas)"
    ]
    
    for i, key in enumerate(metric_order):
        if key not in metrics: continue
        val = metrics[key]
        
        if key in ["Beneficio Total", "Max Drawdown $"]:
            disp = f"${val:,.2f}"
        elif key in ["Crecimiento Capital", "CAGR", "Max Drawdown %", "% Ganadoras", "Retorno Medio/Op. (%)"]:
            disp = f"{val*100:.2f}%"
        elif np.isinf(val):
            disp = "∞"
        elif isinstance(val, (int, float)):
            disp = f"{val:.2f}"
        else:
            disp = str(val)
        
        cols[i%4].metric(label=key, value=disp)

with tabs[1]:
    st.header("📈 Curva de Equity y Drawdown")
    dates = equity.index; eq = equity['Equity']
    cummax = eq.cummax(); dd_pct = (eq-cummax)/cummax*100

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08, row_heights=[0.7,0.3],
                        subplot_titles=("Curva de Equity","Drawdown (%)"))

    fig.add_trace(go.Scatter(
        x=dates, y=eq, mode='lines', name='Equity',
        line=dict(width=2,color='royalblue'),
        hovertemplate='%{x|%d %b %Y %H:%M}<br>Equity: %{y:$,.2f}<extra></extra>'
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=dates, y=dd_pct, mode='lines', name='Drawdown',
        fill='tozeroy', line=dict(color='indianred'),
        hovertemplate='%{x|%d %b %Y %H:%M}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ), row=2, col=1)

    fig.update_layout(height=600, margin=dict(l=50,r=20,t=50,b=50),
                      showlegend=False, hovermode='x unified')
    fig.update_yaxes(title_text="Capital ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("📝 Detalle de Operaciones")
    display_cols = ['Entry Date','Exit Date','Side','Profit','Profit %']
    if 'MFE' in trades.columns: display_cols.append('MFE')
    if 'MAE' in trades.columns: display_cols.append('MAE')
    if 'Nº barras' in trades.columns: display_cols.append('Nº barras')
    
    existing_cols = [col for col in display_cols if col in trades.columns]
    st.dataframe(
        trades[existing_cols].reset_index(drop=True),
        use_container_width=True
    )
# Pestaña 3: Monte Carlo
with tabs[3]:
    st.header("🎲 Simulación Monte Carlo (Bootstrap de Retornos)")

    n_sims = st.number_input(
        "Número de simulaciones",
        min_value=100, max_value=10000,
        value=1000, step=100
    )

    rets = equity['Equity'].pct_change().dropna()
    horizon = len(rets)
    if horizon < 1:
        st.warning("No hay retornos suficientes para ejecutar Monte Carlo.")
        st.stop()

    if st.button("▶️ Ejecutar Monte Carlo"):
        with st.spinner("Corriendo simulaciones..."):
            # --- LÍNEA CORREGIDA ---
            # Usamos .iloc[0, 0] para obtener el valor de la 1ª fila y 1ª columna
            initial_value = float(equity.iloc[0, 0])
            
            # Simulaciones
            sims_rel = run_monte_carlo(rets.values, n_sims, horizon)
            sims_eq  = sims_rel * initial_value

            # Estadísticas finales
            final_vals = sims_eq[-1, :]
            mean_f   = final_vals.mean()
            med_f    = np.median(final_vals)
            p10_f    = np.percentile(final_vals, 10)
            p90_f    = np.percentile(final_vals, 90)
            var95    = np.percentile(final_vals, 5)
            cvar95   = final_vals[final_vals <= var95].mean()

            def max_dd(arr):
                # Añadimos el valor inicial al principio para un cálculo correcto
                path = np.insert(arr, 0, initial_value)
                cm = np.maximum.accumulate(path)
                return ((path - cm) / cm).min()
            
            mdds = np.array([max_dd(sims_eq[:, i]) for i in range(n_sims)]) * 100

        st.subheader("📈 Estadísticas del Capital Final")
        stats = {
            "Media":   mean_f, "Mediana": med_f,
            "P10":     p10_f, "P90":     p90_f,
            "VaR 95%": var95, "CVaR 95%": cvar95
        }
        cols = st.columns(len(stats))
        for idx, (label, value) in enumerate(stats.items()):
            cols[idx].metric(label, f"${value:,.2f}")

        # Envelope plot
        sim_dates = rets.index
        fig_env = go.Figure()
        
        # Percentile bands
        fig_env.add_trace(go.Scatter(
            x=sim_dates, y=np.percentile(sims_eq, 95, axis=1),
            fill=None, mode='lines', line_color='lightgrey', showlegend=False
        ))
        fig_env.add_trace(go.Scatter(
            x=sim_dates, y=np.percentile(sims_eq, 5, axis=1),
            fill='tonexty', mode='lines', line_color='lightgrey', name='5%-95% Percentil'
        ))
        
        # Median
        fig_env.add_trace(go.Scatter(
            x=sim_dates, y=np.percentile(sims_eq, 50, axis=1), mode='lines',
            name='Mediana (P50)', line=dict(color='orange', dash='dash')
        ))

        fig_env.update_layout(
            title="Simulaciones Monte Carlo vs. Curva Histórica",
            xaxis_title='Fecha', yaxis_title='Capital',
            hovermode='x unified', template='plotly_white'
        )
        # Original equity curve
        fig_env.add_trace(go.Scatter(
            x=equity.index, y=equity['Equity'], mode='lines',
            name='Histórico', line=dict(color='blue', width=2)
        ))
        st.plotly_chart(fig_env, use_container_width=True)

        # Histograma de Capital Final
        st.subheader("📊 Histograma Capital Final")
        hist1 = go.Figure()
        hist1.add_trace(go.Histogram(x=final_vals, nbinsx=50, name="Frecuencia"))
        hist1.add_vline(x=mean_f,  line_dash="dash", annotation_text="Media",  line_color="black")
        hist1.add_vline(x=med_f,   line_dash="dash", annotation_text="Mediana", line_color="orange")
        hist1.add_vline(x=var95,   line_dash="dot",  annotation_text="VaR 95%", line_color="red")
        hist1.update_layout(
            xaxis_title="Capital Final", yaxis_title="Frecuencia",
            template='plotly_white', showlegend=False
        )
        st.plotly_chart(hist1, use_container_width=True)

        # Histograma de Max Drawdown
        st.subheader("📉 Histograma de Max Drawdown (%)")
        hist2 = go.Figure()
        hist2.add_trace(go.Histogram(x=mdds, nbinsx=50, name="Frecuencia"))
        hist2.add_vline(x=np.median(mdds), line_dash="dash", annotation_text="Mediana", line_color="orange")
        hist2.add_vline(x=np.percentile(mdds, 95),   line_dash="dot",  annotation_text="P95",   line_color="red")
        hist2.update_layout(
            xaxis_title="Max Drawdown (%)", yaxis_title="Frecuencia",
            template='plotly_white', showlegend=False
        )
        st.plotly_chart(hist2, use_container_width=True)

with tabs[4]: 
    st.header("🎲 Simulación Monte Carlo (Block Bootstrap)")
    st.markdown("Muestrea bloques de retornos para intentar preservar la autocorrelación de la estrategia.")
    
    mc_block_cols = st.columns(2)
    block_size = mc_block_cols[0].number_input("Tamaño de bloque (operaciones)", 1, 100, 5, 1, key="bb_block_size")
    n_sims_bb = mc_block_cols[1].number_input("Número de simulaciones (BB)", 100, 10000, 1000, 100, key="bb_n_sims")

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

                # Calcular MDD para cada simulación
                mdds_bb = np.apply_along_axis(max_dd, 0, sims_eq_bb) * 100

                st.subheader("📈 Estadísticas del Capital Final (Block Bootstrap)")
                stats_bb = {
                    "Media": final_vals_bb.mean(), "Mediana": np.median(final_vals_bb),
                    "P10": np.percentile(final_vals_bb, 10), "P90": np.percentile(final_vals_bb, 90),
                    "VaR 95%": np.percentile(final_vals_bb, 5), "CVaR 95%": final_vals_bb[final_vals_bb <= np.percentile(final_vals_bb, 5)].mean()
                }
                stat_cols_bb = st.columns(len(stats_bb))
                for idx, (label, value) in enumerate(stats_bb.items()):
                    stat_cols_bb[idx].metric(label, f"${value:,.2f}")

                # --- GRÁFICO ENVELOPE (CORREGIDO) ---
                # Usamos el índice de la curva de equity omitiendo el primer punto para que coincida con la longitud de la simulación
                sim_plot_dates = equity.index[1:]

                fig_env_bb = go.Figure()
                fig_env_bb.add_trace(go.Scatter(x=sim_plot_dates, y=np.percentile(sims_eq_bb, 95, axis=1), fill=None, mode='lines', line_color='lightgrey', showlegend=False))
                fig_env_bb.add_trace(go.Scatter(x=sim_plot_dates, y=np.percentile(sims_eq_bb, 5, axis=1), fill='tonexty', mode='lines', line_color='lightgrey', name='5%-95%'))
                fig_env_bb.add_trace(go.Scatter(x=sim_plot_dates, y=np.percentile(sims_eq_bb, 50, axis=1), mode='lines', name='Mediana', line=dict(color='orange', dash='dash')))
                # Dibujamos la curva histórica por encima para que se vea bien
                fig_env_bb.add_trace(go.Scatter(x=equity.index, y=equity['Equity'], mode='lines', name='Histórico', line=dict(color='blue', width=3)))
                
                fig_env_bb.update_layout(title="Simulaciones Monte Carlo (Block Bootstrap) vs. Curva Histórica", xaxis_title='Fecha', yaxis_title='Capital', hovermode='x unified')
                st.plotly_chart(fig_env_bb, use_container_width=True)
                
                # --- HISTOGRAMAS (AÑADIDOS) ---
                # Histograma de Capital Final
                st.subheader("📊 Histograma Capital Final (Block Bootstrap)")
                hist1_bb = go.Figure()
                hist1_bb.add_trace(go.Histogram(x=final_vals_bb, nbinsx=50, name="Frecuencia"))
                hist1_bb.add_vline(x=stats_bb["Media"],   line_dash="dash", annotation_text="Media",   line_color="black")
                hist1_bb.add_vline(x=stats_bb["Mediana"], line_dash="dash", annotation_text="Mediana", line_color="orange")
                hist1_bb.add_vline(x=stats_bb["VaR 95%"], line_dash="dot",  annotation_text="VaR 95%", line_color="red")
                hist1_bb.update_layout(xaxis_title="Capital Final", yaxis_title="Frecuencia", showlegend=False)
                st.plotly_chart(hist1_bb, use_container_width=True)

                # Histograma de Max Drawdown
                st.subheader("📉 Histograma de Max Drawdown (%) (Block Bootstrap)")
                hist2_bb = go.Figure()
                hist2_bb.add_trace(go.Histogram(x=mdds_bb, nbinsx=50, name="Frecuencia"))
                hist2_bb.add_vline(x=np.median(mdds_bb),      line_dash="dash", annotation_text="Mediana", line_color="orange")
                hist2_bb.add_vline(x=np.percentile(mdds_bb, 95), line_dash="dot",  annotation_text="P95",     line_color="red")
                hist2_bb.update_layout(xaxis_title="Max Drawdown (%)", yaxis_title="Frecuencia", showlegend=False)
                st.plotly_chart(hist2_bb, use_container_width=True)

    # --- Comparativa automática para varios tamaños de bloque ---
    st.markdown("---")
    st.subheader("🔍 Comparativa Block Size vs Métricas")

    # Definimos los tamaños de bloque a comparar
    block_sizes = [5, 10, 20]
    comparativa = []

    for bs in block_sizes:
        sims_rel_bb = run_block_bootstrap_monte_carlo(
            trade_returns.values, n_sims_bb, bs, horizon_bb
        )
        if sims_rel_bb is None:
            continue

        initial_value = float(equity.iloc[0, 0])
        sims_eq_bb = sims_rel_bb * initial_value
        final_vals_bb = sims_eq_bb[-1, :]

        # Métricas
        mean_v   = final_vals_bb.mean()
        med_v    = np.median(final_vals_bb)
        var95_v  = np.percentile(final_vals_bb, 5)
        cvar95_v = final_vals_bb[final_vals_bb <= var95_v].mean()

        # Drawdowns
        mdds_bb = np.apply_along_axis(max_dd, 0, sims_eq_bb) * 100
        med_dd   = np.median(mdds_bb)
        p95_dd   = np.percentile(mdds_bb, 95)

        comparativa.append({
            "Block Size": bs,
            "Media Final": mean_v,
            "Mediana Final": med_v,
            "VaR 95%": var95_v,
            "CVaR 95%": cvar95_v,
            "Mediana MDD %": med_dd,
            "P95 MDD %": p95_dd
        })

    if comparativa:
        df_comp = pd.DataFrame(comparativa)
        # Formateamos para mostrar
        df_comp_display = df_comp.copy()
        df_comp_display[["Media Final","Mediana Final","VaR 95%","CVaR 95%"]] = \
            df_comp_display[["Media Final","Mediana Final","VaR 95%","CVaR 95%"]].applymap(lambda x: f"${x:,.0f}")
        df_comp_display[["Mediana MDD %","P95 MDD %"]] = \
            df_comp_display[["Mediana MDD %","P95 MDD %"]].applymap(lambda x: f"{x:.2f}%")
        st.dataframe(df_comp_display, use_container_width=True)
    else:
        st.warning("No se pudo generar la comparativa para los tamaños de bloque seleccionados.")

# --- TEST DE STRESS
with tabs[5]: 
    st.header("⚠️ Stress Test (Recuperación por Velas)")
    st.markdown("Simula la recuperación tras un shock inicial, medido en número de velas.")

    # Inputs
    shock_pct = st.number_input(
        "Choque inicial sobre el capital (%)",
        min_value=-99.0, max_value=-1.0,
        value=-20.0, step=1.0,
        format="%.1f"
    ) / 100.0

    horizon_bars = st.number_input(
        "Horizonte máximo de recuperación (velas)",
        min_value=1, max_value=10000,
        value=252, step=1
    )

    sims_stress = st.number_input(
        "Número de simulaciones post-shock",
        min_value=100, max_value=10000,
        value=500, step=100
    )

    if st.button("▶️ Ejecutar Stress Test"):
        with st.spinner("Corriendo stress tests..."):
            # 1) Shock en el capital
            init = float(equity['Equity'].iat[0])
            shocked = init * (1 + shock_pct)

            # 2) Retornos históricos por vela
            ret_arr = trade_returns.values
            ret_arr = ret_arr[np.isfinite(ret_arr)]

            # 3) Simulaciones post-shock
            sims = np.zeros((horizon_bars, sims_stress))
            for i in range(sims_stress):
                seq = np.random.choice(ret_arr, size=horizon_bars, replace=True)
                sims[:, i] = shocked * np.cumprod(1 + seq)


            # 4) Tiempo hasta recuperación: la primera vela donde equity >= init
            ttrs = []
            for path in sims:
                rec_indices = np.where(path >= init)[0]
                ttrs.append(rec_indices[0] if rec_indices.size>0 else np.nan)
            ttrs = np.array(ttrs)

            # 5) Métricas
            pct_recov = 100 * np.count_nonzero(~np.isnan(ttrs)) / sims_stress
            med_ttr   = np.nanmedian(ttrs)
            p90_ttr   = np.nanpercentile(ttrs[~np.isnan(ttrs)], 90) if np.count_nonzero(~np.isnan(ttrs))>0 else np.nan

        # Resultados
        st.subheader("📊 Estadísticas de Recuperación")
        c1, c2, c3 = st.columns(3)
        c1.metric("% Recuperadas", f"{pct_recov:.1f}%")
        c2.metric("Mediana Velas TTR", f"{med_ttr:.0f}")
        c3.metric("P90 Velas TTR", f"{p90_ttr:.0f}")

        # Histograma de TTR
        st.subheader("📈 Histograma de Velas hasta Recuperación")
        fig_ttr = go.Figure()
        fig_ttr.add_trace(go.Histogram(
            x=ttrs[~np.isnan(ttrs)],
            nbinsx=50
        ))
        fig_ttr.update_layout(
            xaxis_title="Velas hasta recuperar capital inicial",
            yaxis_title="Frecuencia",
            template="plotly_white"
        )
        st.plotly_chart(fig_ttr, use_container_width=True)

        # Curva de supervivencia
        st.subheader("📉 Curva de Supervivencia por Velas")
        surv = [(path >= init).cumprod().sum() / sims_stress * 100 for path in sims.T]
        # mejor: calculamos % vivo en cada vela
        alive = np.array([(sims[:,j] >= init) for j in range(sims_stress)]).T
        pct_alive = alive.sum(axis=1) / sims_stress * 100
        fig_surv = go.Figure(go.Scatter(
            x=list(range(1, horizon_bars+1)),
            y=pct_alive,
            mode="lines"
        ))
        fig_surv.update_layout(
            xaxis_title="Velas tras shock",
            yaxis_title="% Simulaciones con capital ≥ inicial",
            yaxis_range=[0,100],
            template="plotly_white"
        )
        st.plotly_chart(fig_surv, use_container_width=True)
