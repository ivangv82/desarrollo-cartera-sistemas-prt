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

    # Renombrado
    colmap = {
        'Fecha entrada':'Entry Date','Fecha salida':'Exit Date',
        'Tipo':'Side','Rdto Abs':'Profit','Ganancia unitaria':'Profit %',
        'MFE':'MFE','MAE':'MAE','Nº barras':'Nº barras'
    }
    df = df.rename(columns=lambda c: colmap.get(c.strip(), c.strip()))

    if not all(c in df.columns for c in ['Entry Date','Exit Date','Profit']):
        return None

    # Parseo fechas
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

    # Profit %
    if 'Profit %' in df:
        pct = df['Profit %'].astype(str).str.replace('%','').str.replace(',','.')
        df['Profit %'] = pd.to_numeric(pct, errors='coerce').fillna(0)/100
    else:
        df['Profit %'] = 0.0

    # Profit absoluto
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

# --- Comparativa de métricas ---
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

st.header("📊 Comparativa de Métricas")
st.dataframe(df_disp, use_container_width=True)

# --- Curvas normalizadas ---
st.header("📈 Curvas Normalizadas (Base 100)")
fig = go.Figure()
for name, eq in zip(names, equities):
    norm = eq['Equity']/eq['Equity'].iloc[0]*100
    fig.add_trace(go.Scatter(x=norm.index, y=norm, mode='lines', name=name))
fig.update_layout(xaxis_title="Fecha", yaxis_title="Capital Normalizado", template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

# --- Selección de estrategia para análisis profundo ---
choice = st.selectbox("Elige estrategia para análisis detallado:", names)
i = names.index(choice)
eq_a = equities[i]; ret_a = returns[i]; cap_a = initial_caps[i]

tabs = st.tabs([
    "📊 Métricas", "📈 Equity & DD", "📝 Operaciones",
    "🎲 MC Simple", "🎲 MC Bloques", "⚠️ Stress Test"
])

with tabs[0]:
    st.header(f"Métricas Detalladas: {choice}")
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

with tabs[1]:
    st.header("Curva de Equity y Drawdown")
    dates, eqv = eq_a.index, eq_a['Equity']
    cm = eqv.cummax(); dd = (eqv-cm)/cm*100
    fig2 = make_subplots(rows=2,cols=1,shared_xaxes=True,
                         row_heights=[0.7,0.3],vertical_spacing=0.08)
    fig2.add_trace(go.Scatter(x=dates,y=eqv,mode='lines',name='Equity'),row=1,col=1)
    fig2.add_trace(go.Scatter(x=dates,y=dd,mode='lines',fill='tozeroy',name='Drawdown (%)'),row=2,col=1)
    fig2.update_layout(height=600,showlegend=False,template="plotly_white")
    st.plotly_chart(fig2,use_container_width=True)

with tabs[2]:
    st.header("Detalle de Operaciones")
    df_tr = load_prt_trades(strategy_files[i])
    st.dataframe(df_tr, use_container_width=True)

with tabs[3]:
    st.header("🎲 Monte Carlo Simple")
    n_sims = st.number_input("Simulaciones",100,10000,1000,100,key="mc_sim")
    horizon=len(ret_a)
    if horizon<1:
        st.warning("No hay retornos.")
    elif st.button("▶️ Ejecutar MC Simple"):
        sims = run_monte_carlo(ret_a.values, n_sims, horizon)
        init = float(eq_a['Equity'].iloc[0])
        sims_eq = sims * init
        final = sims_eq[-1,:]
        # stats
        mean,med = final.mean(),np.median(final)
        var95=np.percentile(final,5)
        cvar95=final[final<=var95].mean()
        cols=st.columns(4)
        cols[0].metric("Media",f"${mean:,.2f}")
        cols[1].metric("Mediana",f"${med:,.2f}")
        cols[2].metric("VaR 95%",f"${var95:,.2f}")
        cols[3].metric("CVaR 95%",f"${cvar95:,.2f}")
        # envelope
        dates_mc=eq_a.index[1:]
        p10,p50,p90 = np.percentile(sims_eq,10,axis=1),np.percentile(sims_eq,50,axis=1),np.percentile(sims_eq,90,axis=1)
        fmc=go.Figure()
        fmc.add_trace(go.Scatter(x=dates_mc,y=p90,fill=None,mode='lines',line_color='lightgrey'))
        fmc.add_trace(go.Scatter(x=dates_mc,y=p10,fill='tonexty',mode='lines',line_color='lightgrey',name='10–90%'))
        fmc.add_trace(go.Scatter(x=dates_mc,y=p50,mode='lines',line=dict(color='orange',dash='dash'),name='P50'))
        fmc.add_trace(go.Scatter(x=eq_a.index,y=eq_a['Equity'],mode='lines',line=dict(color='blue',width=2),name='Histórico'))
        fmc.update_layout(title="Envelope MC Simple",template="plotly_white")
        st.plotly_chart(fmc,use_container_width=True)

with tabs[4]:
    st.header("🎲 Monte Carlo Block Bootstrap")
    bs = st.number_input("Block size",1,100,5,1,key="bb_size")
    n_bb = st.number_input("Simulaciones",100,10000,1000,100,key="bb_sim")
    if st.button("▶️ Ejecutar MC Bloques"):
        sims_bb = run_block_bootstrap_monte_carlo(ret_a.values, n_bb, bs, len(ret_a))
        if sims_bb is None:
            st.error("Block size inválido.")
        else:
            init=float(eq_a['Equity'].iloc[0])
            sims_eq_bb = sims_bb*init
            final_bb = sims_eq_bb[-1,:]
            mean,med = final_bb.mean(),np.median(final_bb)
            var95=np.percentile(final_bb,5)
            cols=st.columns(4)
            cols[0].metric("Media",f"${mean:,.2f}")
            cols[1].metric("Mediana",f"${med:,.2f}")
            cols[2].metric("VaR 95%",f"${var95:,.2f}")
            # histogram
            fig_bb = go.Figure(go.Histogram(x=final_bb,nbinsx=50))
            fig_bb.update_layout(title="Histograma Capital Final (BB)",template="plotly_white")
            st.plotly_chart(fig_bb,use_container_width=True)

with tabs[5]:
    st.header("⚠️ Stress Test")
    shock = st.number_input("Shock inicial (%)",-99.0,-1.0,-20.0,1.0,key="st_shock")/100
    horizon_ops=st.number_input("Horizonte (ops)",1,10000,252,1,key="st_hor")
    n_st=st.number_input("Simulaciones",100,10000,500,100,key="st_sim")
    if st.button("▶️ Ejecutar Stress Test"):
        init=float(eq_a['Equity'].iloc[0])
        shocked=init*(1+shock)
        arr=ret_a.values[np.isfinite(ret_a.values)]
        sims = np.zeros((horizon_ops,n_st))
        for j in range(n_st):
            seq=np.random.choice(arr,horizon_ops,replace=True)
            sims[:,j]=shocked*np.cumprod(1+seq)
        # TTR ops
        ttrs=[]
        for j in range(n_st):
            idxs=np.where(sims[:,j]>=init)[0]
            ttrs.append(idxs[0]+1 if len(idxs)>0 else np.nan)
        ttrs=np.array(ttrs)
        pct = 100*np.count_nonzero(~np.isnan(ttrs))/n_st
        med = np.nanmedian(ttrs)
        c1,c2=st.columns(2)
        c1.metric("% Recup.",f"{pct:.1f}%")
        c2.metric("Mediana ops TTR",f"{med:.0f}")
        fig_st=go.Figure(go.Histogram(x=ttrs[~np.isnan(ttrs)],nbinsx=50))
        fig_st.update_layout(title="Histograma Ops hasta Recup.",template="plotly_white")
        st.plotly_chart(fig_st,use_container_width=True)
