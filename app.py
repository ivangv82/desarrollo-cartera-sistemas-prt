import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
st.markdown("Carga tu reporte de operaciones de ProRealTime y obtén todas las métricas clásicas más la curva de equity.")

# --- Función de carga con fallback de fechas ---
@st.cache_data
def load_prt_trades(file):
    df = pd.read_csv(file, sep='\t', decimal=',')
    if df.shape[1] == 1:
        df = pd.read_csv(file, sep=',', decimal=',')
        if "" in df.columns:
            df = df.drop(columns=[""])
    df = df.rename(columns={
        'Fecha entrada': 'Entry Date',
        'Fecha salida':  'Exit Date',
        'Tipo':          'Side',
        'Rdto Abs':      'Profit',
        'Ganancia unitaria': 'Profit %',
        'MFE':           'MFE',
        'MAE':           'MAE'
    })
    df['__raw_entry'] = df['Entry Date'].astype(str)
    df['__raw_exit']  = df['Exit Date'].astype(str)

    month_map = {
      'ene':'Jan','feb':'Feb','mar':'Mar','abr':'Apr','may':'May','jun':'Jun',
      'jul':'Jul','ago':'Aug','sep':'Sep','sept':'Sep','oct':'Oct','nov':'Nov','dic':'Dec'
    }
    def clean_and_parse(s_raw):
        s = s_raw.lower()
        for es, en in month_map.items():
            s = s.replace(es, en)
        dt = pd.to_datetime(s, dayfirst=True, infer_datetime_format=True, errors='coerce')
        if pd.isna(dt):
            try:
                dt = parser.parse(s, dayfirst=True, fuzzy=True)
            except:
                dt = pd.NaT
        return dt

    df['Entry Date'] = df['__raw_entry'].map(clean_and_parse)
    df['Exit Date']  = df['__raw_exit'].map(clean_and_parse)

    mask_bad = df['Entry Date'].isna() | df['Exit Date'].isna()
    if mask_bad.any():
        st.warning(f"{mask_bad.sum()} fila(s) con fecha irreconocible y serán descartadas.")
        df = df.loc[~mask_bad].copy()
    df = df.drop(columns=['__raw_entry','__raw_exit'])

    df['Profit %'] = (
        df.get('Profit %', 0).astype(str)
          .str.replace('%','', regex=False)
          .str.replace(',','.', regex=False)
          .astype(float, errors='ignore')
          .fillna(0.0) / 100
    )
    df['Profit'] = (
        df.get('Profit', 0).astype(str)
          .str.replace('[^0-9,.-]','', regex=True)
          .str.replace('\.','', regex=True)
          .str.replace(',','.', regex=False)
          .astype(float, errors='ignore')
          .fillna(0.0)
    )
    return df

# --- Métricas y equity ---
def compute_equity(trades, init_cap):
    df = trades.sort_values('Exit Date').copy()
    df['Equity'] = init_cap + df['Profit'].cumsum()
    equity = pd.DataFrame({
        'Date':   [df['Entry Date'].min()] + df['Exit Date'].tolist(),
        'Equity': [init_cap] + df['Equity'].tolist()
    }).set_index('Date')
    return equity

def calculate_metrics(trades, equity_df, periods_per_year):
    equity = equity_df['Equity']
    ini, fin = equity.iloc[0], equity.iloc[-1]
    total_profit = fin - ini
    growth = fin/ini - 1
    days = (equity.index[-1] - equity.index[0]).days or 1
    cagr = (fin/ini)**(365.0/days) - 1

    cummax = equity.cummax()
    dd_rel = (equity - cummax)/cummax
    mdd_pct = dd_rel.min()
    mdd_abs = (equity - cummax).min()

    daily_ret = equity.pct_change().dropna()
    std_dev = daily_ret.std()
    sharpe = (daily_ret.mean()/std_dev * np.sqrt(periods_per_year)) if std_dev>0 else 0.0

    n = len(trades)
    wins, losses = trades[trades['Profit']>0], trades[trades['Profit']<0]
    win_rate = len(wins)/n if n>0 else 0.0
    
    if 'Nº barras' in trades.columns:
        avg_dur = trades['Nº barras'].astype(float).mean()
    else:
    avg_dur = (trades['Exit Date'] - trades['Entry Date']).dt.days.mean() * (252 if timeframe=='1d' else 1)
    avg_ret_trade = trades['Profit %'].mean()

    gross_profit, gross_loss = wins['Profit'].sum(), abs(losses['Profit'].sum())
    pf = gross_profit/gross_loss if gross_loss>0 else np.nan
    avg_win = wins['Profit %'].mean() if not wins.empty else 0.0
    avg_loss = abs(losses['Profit %'].mean()) if not losses.empty else 0.0
    payoff = avg_win/avg_loss if avg_loss>0 else np.nan

    rec_factor = total_profit/abs(mdd_abs) if mdd_abs!=0 else np.nan
    calmar = cagr/abs(mdd_pct) if mdd_pct!=0 else np.nan

    return {
        "Beneficio Total":       total_profit,
        "Crecimiento Capital":   growth,
        "CAGR":                  cagr,
        "Sharpe Ratio":          sharpe,
        "Max Drawdown %":        mdd_pct,
        "Max Drawdown $":        mdd_abs,
        "Recovery Factor":       rec_factor,
        "Calmar Ratio":          calmar,
        "Total Operaciones":     n,
        "% Ganadoras":           win_rate,
        "Duración Media (velas)": avg_dur,
        "Retorno Medio/Op. (%)": avg_ret_trade,
        "Factor de Beneficio":   pf,
        "Ratio Payoff":          payoff
    }

# --- Sidebar: carga y timeframe ---
st.sidebar.header("📁 Carga de Datos")
trades_file   = st.sidebar.file_uploader("Reporte PRT (CSV)", type=["csv","txt"])
initial_cap   = st.sidebar.number_input("Capital Inicial", value=10000.0, min_value=0.0, step=100.0, format="%.2f")
timeframe     = st.sidebar.selectbox("Timeframe de Velas", ["1mn","5mn","15mn","30mn","1h","4h","1d","1w","1mes"])
if not trades_file:
    st.sidebar.warning("Sube el CSV de operaciones primero.")
    st.stop()

# mapping timeframe → periodos por año
ppy_map = {
    "1mn": 252*6.5*60,   # 6.5h trading dia
    "5mn": 252*6.5*12,
    "15mn":252*6.5*4,
    "30mn":252*6.5*2,
    "1h":  252*6.5,
    "4h":  252*1.625,
    "1d":  252,
    "1w":  52,
    "1mes":12
}
ppy = ppy_map.get(timeframe, 252)

# --- Procesamiento ---
trades = load_prt_trades(trades_file)
equity = compute_equity(trades, initial_cap)
metrics = calculate_metrics(trades, equity, ppy)

# --- Pestañas ---
tabs = st.tabs(["📊 Resumen Métricas","📈 Equity & Drawdown","📝 Operaciones (Trades)"])

with tabs[0]:
    st.header("📋 Resumen de Métricas")
    cols = st.columns(4)
    keys = list(metrics.keys())
    money_keys   = {"Beneficio Total","Max Drawdown $"}
    percent_keys = {"Crecimiento Capital","CAGR","Max Drawdown %","% Ganadoras","Retorno Medio/Op. (%)"}
    ratio_keys   = {"Sharpe Ratio","Factor de Beneficio","Ratio Payoff","Recovery Factor","Calmar Ratio"}
    int_keys     = {"Total Operaciones"}
    float_keys   = {"Duración Media (velas)"}
    for i, key in enumerate(keys):
        val = metrics[key]
        if key in money_keys:
            disp = f"${val:,.2f}"
        elif key in percent_keys:
            disp = f"{val*100:.2f}%"
        elif key in ratio_keys:
            disp = f"{val:.2f}"
        elif key in int_keys:
            disp = f"{int(val)}"
        elif key in float_keys:
            disp = f"{val:.1f}"
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
    fig.add_trace(go.Scatter(x=dates,y=eq,mode='lines',name='Equity',
        line=dict(width=2,color='royalblue'),
        hovertemplate='%{x|%d %b %Y}<br>Equity: %{y:$,.2f}<extra></extra>'),
        row=1, col=1)
    fig.add_trace(go.Scatter(x=dates,y=dd_pct,mode='lines',name='Drawdown',
        fill='tozeroy',line=dict(color='indianred'),
        hovertemplate='%{x|%d %b %Y}<br>Drawdown: %{y:.2f}%<extra></extra>'),
        row=2, col=1)
    fig.update_layout(height=600, margin=dict(l=50,r=20,t=50,b=50),
                      showlegend=False, hovermode='x unified')
    fig.update_xaxes(tickformat="%d %b %Y", tickangle=45, nticks=10, row=2, col=1)
    fig.update_yaxes(title_text="Equity",   row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True)

with tabs[2]:
    st.header("📝 Detalle de Operaciones")
    st.dataframe(trades[['Entry Date','Exit Date','Side','Profit','Profit %','MFE','MAE']]
                 .reset_index(drop=True), use_container_width=True)
