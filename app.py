import streamlit as st
import pandas as pd
import numpy as np
from dateutil import parser
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# --- Configuración de página ---
st.set_page_config(page_title="📊 Analizador de Backtests", layout="wide")
st.title("🚀 Analizador de Estrategias PRT")
st.markdown("Carga múltiples reportes de ProRealTime, compáralos y profundiza en cada uno.")

# --- Funciones auxiliares ---
@st.cache_data
def load_prt_trades(file):
    encs = ['utf-8','latin-1','iso-8859-1','cp1252']
    df = None
    for enc in encs:
        try:
            file.seek(0)
            tmp = pd.read_csv(file, sep='\t', decimal=',', thousands='.', encoding=enc)
            if tmp.shape[1] > 1:
                df = tmp; break
            file.seek(0)
            tmp = pd.read_csv(file, sep=',', decimal='.', thousands=',', encoding=enc)
            if tmp.shape[1] > 1:
                df = tmp; break
        except:
            continue
    if df is None:
        return None

    colmap = {
        'Fecha entrada':'Entry Date','Fecha salida':'Exit Date',
        'Tipo':'Side','Rdto Abs':'Profit','Ganancia unitaria':'Profit %',
        'MFE':'MFE','MAE':'MAE','Nº barras':'Nº barras'
    }
    df = df.rename(columns=lambda c: colmap.get(c.strip(), c.strip()))
    if not all(c in df.columns for c in ['Entry Date','Exit Date','Profit']):
        return None

    df['__e'] = df['Entry Date'].astype(str)
    df['__x'] = df['Exit Date'].astype(str)
    month_map = {
      'ene':'Jan','feb':'Feb','mar':'Mar','abr':'Apr','may':'May','jun':'Jun',
      'jul':'Jul','ago':'Aug','sep':'Sep','sept':'Sep','oct':'Oct','nov':'Nov','dic':'Dec'
    }
    def parse_dt(s):
        s2 = s.lower()
        for es,en in month_map.items():
            s2 = s2.replace(es,en)
        dt = pd.to_datetime(s2, dayfirst=True, errors='coerce')
        if pd.isna(dt):
            try: dt = parser.parse(s2, dayfirst=True, fuzzy=True)
            except: return pd.NaT
        return dt

    df['Entry Date'] = df['__e'].map(parse_dt)
    df['Exit Date']  = df['__x'].map(parse_dt)
    df = df.drop(columns=['__e','__x']).dropna(subset=['Entry Date','Exit Date'])

    if 'Profit %' in df:
        pct = df['Profit %'].astype(str).str.replace('%','').str.replace(',','.')
        df['Profit %'] = pd.to_numeric(pct, errors='coerce').fillna(0)/100
    else:
        df['Profit %'] = 0.0

    p = ( df['Profit']
          .astype(str)
          .str.replace(r'\.(?=\d{3}(?:[.,]|$))','',regex=True)
          .str.replace(',','.',regex=False)
          .str.replace(r'[^\d\.-]','',regex=True)
    )
    df['Profit'] = pd.to_numeric(p,errors='coerce').fillna(0)
    return df

def compute_equity(trades, init_cap):
    df = trades.sort_values('Exit Date').reset_index(drop=True)
    df['Equity'] = init_cap + df['Profit'].cumsum()
    prev = init_cap + df['Profit'].shift(1).cumsum().fillna(0)
    df['Return'] = df['Profit'] / prev.replace(0,np.nan).fillna(0)
    dates = [trades['Entry Date'].min()] + df['Exit Date'].tolist()
    eqs   = [init_cap] + df['Equity'].tolist()
    eq = pd.DataFrame({'Date':dates,'Equity':eqs}).set_index('Date')
    return eq, df['Return']

def calculate_metrics(trades, equity, ppy, tf):
    if equity is None or equity.empty:
        return {}
    e = equity['Equity']
    ini,fin = e.iloc[0],e.iloc[-1]
    tot = fin-ini
    growth = fin/ini -1 if ini!=0 else 0
    days=(e.index[-1]-e.index[0]).days or 1
    cagr=(fin/ini)**(365/days)-1 if ini!=0 else 0
    cm=e.cummax()
    dd_rel=(e-cm)/cm
    mdd_pct=dd_rel.min()
    mdd_abs=(e-cm).min()
    freq_map={"1mn":"T","5mn":"5T","15mn":"15T","30mn":"30T","1h":"H","4h":"4H","1d":"D","1w":"W","1mes":"MS"}
    r = e.resample(freq_map[tf]).ffill().pct_change().dropna()
    sharpe = (r.mean()/r.std()*np.sqrt(ppy)) if len(r)>1 and r.std()>0 else 0
    n=len(trades)
    wins=trades[trades['Profit']>0]
    losses=trades[trades['Profit']<0]
    wr=len(wins)/n if n>0 else 0
    pf = wins['Profit'].sum()/abs(losses['Profit'].sum()) if losses['Profit'].sum()!=0 else np.inf
    aw = wins['Profit %'].mean() if not wins.empty else 0
    al = abs(losses['Profit %'].mean()) if not losses.empty else 0
    payoff = aw/al if al>0 else np.inf
    rec = tot/abs(mdd_abs) if mdd_abs!=0 else np.inf
    calmar = cagr/abs(mdd_pct) if mdd_pct!=0 else np.inf

    return {
        "Beneficio Total":     tot,
        "Crecimiento Capital": growth,
        "CAGR":                cagr,
        "Sharpe Ratio":        sharpe,
        "Max Drawdown %":      mdd_pct,
        "Max Drawdown $":      mdd_abs,
        "Recovery Factor":     rec,
        "Calmar Ratio":        calmar,
        "Total Operaciones":   n,
        "% Ganadoras":         wr,
        "Factor de Beneficio": pf,
        "Ratio Payoff":        payoff
    }

def max_dd(path):
    cm = np.maximum.accumulate(path)
    return ((path-cm)/cm).min()

@st.cache_data
def run_monte_carlo(returns, n_sims, horizon):
    arr = np.asarray(returns, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size==0 or horizon<=0:
        return np.ones((horizon,n_sims))
    sims = np.zeros((horizon,n_sims))
    for i in range(n_sims):
        sample = np.random.choice(arr,size=horizon,replace=True)
        sims[:,i] = np.cumprod(1+sample)
    return sims

@st.cache_data
def run_block_bootstrap_monte_carlo(returns, n_sims, block_size, horizon):
    arr = np.asarray(returns,dtype=float); arr=arr[np.isfinite(arr)]
    n_ret=len(arr)
    if block_size<=0 or n_ret<block_size:
        return None
    n_blocks=(horizon//block_size)+1
    starts=np.arange(n_ret-block_size+1)
    sims=np.zeros((horizon,n_sims))
    for i in range(n_sims):
        rs = np.random.choice(starts, size=n_blocks, replace=True)
        seq = np.concatenate([arr[s:s+block_size] for s in rs])[:horizon]
        sims[:,i]=np.cumprod(1+seq)
    return sims

# --- Sidebar: carga múltiple ---
st.sidebar.header("📁 Carga de Estrategias")
strategy_files = st.sidebar.file_uploader(
    "Sube reportes (múltiples):", type=["csv","txt"],
    accept_multiple_files=True
)
initial_caps = []
for i,f in enumerate(strategy_files):
    cap = st.sidebar.number_input(
        f"Capital Estrat. {i+1}", value=10000.0,
        min_value=0.0, step=100.0, format="%.2f", key=f"cap{i}"
    )
    initial_caps.append(cap)

# Parámetros globales
st.sidebar.header("⚙️ Parámetros Globales")
timeframe = st.sidebar.selectbox(
    "Timeframe de Velas",
    ["1mn","5mn","15mn","30mn","1h","4h","1d","1w","1mes"], index=4
)
if timeframe in ["1mn","5mn","15mn","30mn","1h","4h"]:
    th = st.sidebar.number_input("Horas trading/día",1.0,24.0,6.5,0.5)
    mins={"1mn":1,"5mn":5,"15mn":15,"30mn":30,"1h":60,"4h":240}[timeframe]
    ppy=(th*60/mins)*252
else:
    ppy={"1d":252,"1w":52,"1mes":12}[timeframe]
st.sidebar.caption(f"Periodos/año: {int(ppy)}")

if not strategy_files:
    st.info("Sube al menos un reporte para continuar.")
    st.stop()

# --- Procesar estrategias ---
names, equities, returns, metrics_list = [], [], [], []
for idx, f in enumerate(strategy_files):
    df_tr = load_prt_trades(f)
    if df_tr is None or df_tr.empty:
        continue
    eq, ret = compute_equity(df_tr, initial_caps[idx])
    met = calculate_metrics(df_tr, eq, ppy, timeframe)
    names.append(f.name)
    equities.append(eq)
    returns.append(ret)
    metrics_list.append(met)

# --- Pestañas principales ---
tabs = st.tabs(["Estadísticas","Curvas y MDD","Análisis Avanzado"])

# 1) Estadísticas comparativas
with tabs[0]:
    dfm = pd.DataFrame(metrics_list, index=names)
    def fmt(v,k):
        if k in ["Beneficio Total","Max Drawdown $"]:
            return f"${v:,.2f}"
        if k in ["Crecimiento Capital","CAGR","Max Drawdown %","% Ganadoras"]:
            return f"{v:.2%}"
        return f"{v:.2f}"
    df_disp = dfm.copy()
    for col in df_disp.columns:
        df_disp[col] = df_disp[col].map(lambda x: fmt(x,col))

    st.header("📋 Estadísticas Comparativas")
    st.dataframe(df_disp, use_container_width=True)

# 2) Curvas de capital + MDD
with tabs[1]:
    st.header("📈 Curvas de Capital Normalizadas")
    fig = go.Figure()
    for name, eq in zip(names, equities):
        norm = eq['Equity']/eq['Equity'].iloc[0]*100
        fig.add_trace(go.Scatter(x=norm.index, y=norm, mode='lines', name=name))
    fig.update_layout(xaxis_title="Fecha", yaxis_title="Capital Normalizado", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.header("📉 Drawdown (%)")
    fig2 = go.Figure()
    for name, eq in zip(names, equities):
        dd = (eq['Equity'] - eq['Equity'].cummax())/eq['Equity'].cummax()*100
        fig2.add_trace(go.Scatter(x=dd.index, y=dd, mode='lines', fill='tozeroy', name=name))
    fig2.update_layout(xaxis_title="Fecha", yaxis_title="Drawdown (%)", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# 3) Análisis avanzado de una estrategia
with tabs[2]:
    choice = st.selectbox("Elige estrategia para análisis detallado:", names)
    i = names.index(choice)
    eq_a, ret_a, cap_a = equities[i], returns[i], initial_caps[i]

    st.header(f"🔍 Análisis Avanzado: {choice}")

    sub = st.tabs(["📊 Métricas","📈 Equity & DD","📝 Operaciones","🎲 MC Simple","🎲 MC Bloques","⚠️ Stress Test"])

    # Métricas detalladas
    with sub[0]:
        met = metrics_list[i]
        cols = st.columns(4)
        order = [
            "Beneficio Total","Crecimiento Capital","CAGR","Sharpe Ratio",
            "Max Drawdown $","Max Drawdown %","Recovery Factor","Calmar Ratio",
            "Total Operaciones","% Ganadoras","Factor de Beneficio","Ratio Payoff"
        ]
        for idx,key in enumerate(order):
            if key in met:
                val=met[key]
                disp=fmt(val,key)
                cols[idx%4].metric(key, disp)

    # Curva & DD
    with sub[1]:
        dates, eqv = eq_a.index, eq_a['Equity']
        cm = eqv.cummax(); dd = (eqv-cm)/cm*100
        fig3 = make_subplots(rows=2,cols=1,shared_xaxes=True,
                             row_heights=[0.7,0.3],vertical_spacing=0.08)
        fig3.add_trace(go.Scatter(x=dates,y=eqv,mode='lines',name='Equity'),row=1,col=1)
        fig3.add_trace(go.Scatter(x=dates,y=dd,mode='lines',fill='tozeroy',name='Drawdown'),row=2,col=1)
        fig3.update_layout(height=600,showlegend=False,template="plotly_white")
        st.plotly_chart(fig3,use_container_width=True)

    # Detalle de operaciones
    with sub[2]:
        df_tr = load_prt_trades(strategy_files[i])
        st.dataframe(df_tr, use_container_width=True)
        
    # --- Simulación Monte Carlo Simple ---
    with sub[3]:
        st.header("🎲 Monte Carlo Simple")
        n_sims = st.number_input("Número de simulaciones", 100, 20000, 1000, 100, key="mc_simple_n")
        if st.button("▶️ Ejecutar MC Simple", key="mc_simple_run"):
            # 1) Spinner: cálculos
            with st.spinner("Corriendo simulaciones..."):
                arr      = returns_a.values[np.isfinite(returns_a.values)]
                horizon  = len(arr)
                sims_rel = run_monte_carlo(arr, n_sims, horizon)
                init     = float(eq_a['Equity'].iloc[0])
                sims_eq  = sims_rel * init

                # percentiles
                p10 = np.percentile(sims_eq, 10, axis=1)
                p50 = np.percentile(sims_eq, 50, axis=1)
                p90 = np.percentile(sims_eq, 90, axis=1)
                dates_mc = eq_a.index[1:]

                # estadísticas finales
                final_vals = sims_eq[-1, :]
                var95   = np.percentile(final_vals, 5)
                cvar95  = final_vals[final_vals <= var95].mean()
                stats = {
                    "Media": final_vals.mean(),
                    "Mediana": np.median(final_vals),
                    "P10": np.percentile(final_vals, 10),
                    "P90": np.percentile(final_vals, 90),
                    "VaR 95%": var95,
                    "CVaR 95%": cvar95
                }

                # drawdowns simulados
                def compute_mdd(path):
                    full = np.insert(path, 0, init)
                    cm = np.maximum.accumulate(full)
                    dd = (full - cm) / cm
                    return dd.min() * 100

                mdds = np.array([compute_mdd(sims_eq[:, i]) for i in range(n_sims)])

            # 2) Envelope plot
            fig_env = go.Figure()
            fig_env.add_trace(go.Scatter(x=dates_mc, y=p90, mode='lines', line_color='lightgrey', showlegend=False))
            fig_env.add_trace(go.Scatter(x=dates_mc, y=p10, fill='tonexty', mode='lines', line_color='lightgrey', name='10–90%'))
            fig_env.add_trace(go.Scatter(x=dates_mc, y=p50, mode='lines', line=dict(color='orange', dash='dash'), name='Mediana P50'))
            fig_env.add_trace(go.Scatter(x=eq_a.index, y=eq_a['Equity'], mode='lines', line=dict(color='blue', width=2), name='Histórico'))
            fig_env.update_layout(
                title="Envelope Monte Carlo Simple",
                xaxis_title="Fecha", yaxis_title="Capital",
                template="plotly_white", hovermode="x unified"
            )
            st.plotly_chart(fig_env, use_container_width=True)

            # 3) Métricas finales
            st.subheader("📈 Estadísticas del Capital Final")
            cols = st.columns(len(stats))
            for j, (label, val) in enumerate(stats.items()):
                cols[j].metric(label, f"${val:,.2f}")

            # 4) Histograma Capital Final
            st.subheader("📊 Histograma Capital Final")
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(x=final_vals, nbinsx=50))
            fig_hist.add_vline(x=stats["Mediana"],  line_dash="dash", annotation_text="Mediana", line_color="orange")
            fig_hist.add_vline(x=stats["CVaR 95%"], line_dash="dot", annotation_text="CVaR 95%", line_color="red")
            fig_hist.update_layout(xaxis_title="Capital Final", yaxis_title="Frecuencia", template="plotly_white", showlegend=False)
            st.plotly_chart(fig_hist, use_container_width=True)

            # 5) Histograma Máx Drawdown
            st.subheader("📉 Histograma Máx Drawdown (%)")
            fig_ddh = go.Figure()
            fig_ddh.add_trace(go.Histogram(x=mdds, nbinsx=50))
            fig_ddh.add_vline(x=np.median(mdds),      line_dash="dash", annotation_text="Mediana", line_color="orange")
            fig_ddh.add_vline(x=np.percentile(mdds,95), line_dash="dot",  annotation_text="P95",    line_color="red")
            fig_ddh.update_layout(xaxis_title="Max Drawdown (%)", yaxis_title="Frecuencia", template="plotly_white", showlegend=False)
            st.plotly_chart(fig_ddh, use_container_width=True)


        

      
    
    # --- Monte Carlo Block Bootstrap ---
    with sub[4]:
        st.header("🎲 Monte Carlo Block Bootstrap")
        block_size = st.number_input("Tamaño de bloque (ops)", 1, len(ret_a), 5, 1, key="mc_block_bs")
        n_sims_bb  = st.number_input("Número de simulaciones", 100, 20000, 1000, 100, key="mc_block_n")
        if st.button("▶️ Ejecutar MC Bloques", key="mc_block_run"):
            with st.spinner("Corriendo MC por bloques..."):
                bb = run_block_bootstrap_monte_carlo(ret_a.values, n_sims_bb, block_size, len(ret_a))
                if bb is None:
                    st.error("Block size inválido (debe ser < número de operaciones).")
                else:
                    init = float(eq_a['Equity'].iloc[0])
                    sims_eq_bb = bb * init
    
                    # percentiles
                    p10_bb = np.percentile(sims_eq_bb, 10, axis=1)
                    p50_bb = np.percentile(sims_eq_bb, 50, axis=1)
                    p90_bb = np.percentile(sims_eq_bb, 90, axis=1)
                    dates_bb = eq_a.index[1:]
    
                    # 1) Envelope plot bloques
                    fig_env_bb = go.Figure()
                    fig_env_bb.add_trace(go.Scatter(x=dates_bb, y=p90_bb,
                                                   mode='lines', line_color='lightgrey', showlegend=False))
                    fig_env_bb.add_trace(go.Scatter(x=dates_bb, y=p10_bb,
                                                   fill='tonexty', mode='lines', line_color='lightgrey', name='10–90%'))
                    fig_env_bb.add_trace(go.Scatter(x=dates_bb, y=p50_bb,
                                                   mode='lines', line=dict(color='orange', dash='dash'), name='Mediana'))
                    fig_env_bb.add_trace(go.Scatter(x=eq_a.index, y=eq_a['Equity'],
                                                   mode='lines', line=dict(color='blue', width=2), name='Histórico'))
                    fig_env_bb.update_layout(
                        title="Envelope MC Block Bootstrap",
                        xaxis_title="Fecha", yaxis_title="Capital",
                        template="plotly_white", hovermode="x unified"
                    )
                    st.plotly_chart(fig_env_bb, use_container_width=True)
    
                    # 2) Estadísticas
                    final_bb = sims_eq_bb[-1, :]
                    var95_bb = np.percentile(final_bb, 5)
                    cvar95_bb = final_bb[final_bb <= var95_bb].mean()
                    stats_bb = {
                        "Media": final_bb.mean(),
                        "Mediana": np.median(final_bb),
                        "P10": np.percentile(final_bb, 10),
                        "P90": np.percentile(final_bb, 90),
                        "VaR 95%": var95_bb,
                        "CVaR 95%": cvar95_bb
                    }
                    st.subheader("📈 Estadísticas Capital Final (BB)")
                    cols_bb = st.columns(len(stats_bb))
                    for idx, (lbl, val) in enumerate(stats_bb.items()):
                        cols_bb[idx].metric(lbl, f"${val:,.2f}")
    
                    # 3) Histograma Capital Final BB
                    fig_hist_bb = go.Figure()
                    fig_hist_bb.add_trace(go.Histogram(x=final_bb, nbinsx=50))
                    fig_hist_bb.add_vline(x=stats_bb["Mediana"],  line_dash="dash", annotation_text="Mediana", line_color="orange")
                    fig_hist_bb.add_vline(x=stats_bb["CVaR 95%"], line_dash="dot",  annotation_text="CVaR 95%", line_color="red")
                    fig_hist_bb.update_layout(
                        title="Histograma Capital Final (BB)",
                        xaxis_title="Capital Final", yaxis_title="Frecuencia",
                        template="plotly_white", showlegend=False
                    )
                    st.plotly_chart(fig_hist_bb, use_container_width=True)
    
                    # 4) Histograma Máx Drawdown BB
                    mdds_bb = np.array([max_dd(sims_eq_bb[:, j]) * 100 for j in range(n_sims_bb)])
                    fig_dd_bb = go.Figure()
                    fig_dd_bb.add_trace(go.Histogram(x=mdds_bb, nbinsx=50))
                    fig_dd_bb.add_vline(x=np.median(mdds_bb),      line_dash="dash", annotation_text="Mediana", line_color="orange")
                    fig_dd_bb.add_vline(x=np.percentile(mdds_bb,95), line_dash="dot",  annotation_text="P95",    line_color="red")
                    fig_dd_bb.update_layout(
                        title="Histograma Máx Drawdown (%) (BB)",
                        xaxis_title="Max Drawdown (%)", yaxis_title="Frecuencia",
                        template="plotly_white", showlegend=False
                    )
                    st.plotly_chart(fig_dd_bb, use_container_width=True)


    # Stress Test
    with sub[5]:
        shock = st.number_input("Shock inicial (%)",-99.0,-1.0,-20.0,1.0,key="st_s")
        ops_h = st.number_input("Horizonte (ops)",1,10000,252,1,key="st_h")
        n_st = st.number_input("Simulaciones ST",100,10000,500,100,key="st_n")
        if st.button("▶️ Ejecutar Stress Test", key="btn_st"):
            init=float(eq_a['Equity'].iloc[0])
            shocked=init*(1+shock/100)
            arr = ret_a.values[np.isfinite(ret_a.values)]
            sims_st=np.zeros((ops_h,n_st))
            for j in range(n_st):
                seq=np.random.choice(arr,ops_h,replace=True)
                sims_st[:,j]=shocked*np.cumprod(1+seq)
            ttrs=[(np.where(sims_st[:,j]>=init)[0][0]+1 if np.any(sims_st[:,j]>=init) else np.nan)
                  for j in range(n_st)]
            ttrs=np.array(ttrs)
            pct=100*np.count_nonzero(~np.isnan(ttrs))/n_st
            med=np.nanmedian(ttrs)
            c1,c2=st.columns(2)
            c1.metric("% Recup.",f"{pct:.1f}%")
            c2.metric("Mediana ops",f"{med:.0f}")
            fig4=go.Figure(go.Histogram(x=ttrs[~np.isnan(ttrs)],nbinsx=50))
            fig4.update_layout(title="Ops hasta recuperación",template="plotly_white")
            st.plotly_chart(fig4,use_container_width=True)
