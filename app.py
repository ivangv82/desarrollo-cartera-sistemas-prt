import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- Configuración de página ---
st.set_page_config(
    page_title="📊 Backtest PRT Completo",
    layout="wide"
)

st.title("🚀 Análisis Completo de Backtest PRT")
st.markdown("Carga tu reporte de operaciones de ProRealTime y obtén todas las métricas clásicas más la curva de equity.")

# --- Función de carga ---
@st.cache_data
def load_prt_trades(file):
    """Carga y limpia el CSV de operaciones exportado por ProRealTime."""
    # 1) Intento tab separado (export estándar PRT)
    df = pd.read_csv(file, sep='\t', decimal=',')
    # 2) Si solo hay UNA columna, fue con coma => recargamos con sep=','
    if df.shape[1] == 1:
        df = pd.read_csv(file, sep=',', decimal=',')
        # A veces la primera columna queda vacía, la eliminamos
        if "" in df.columns:
            df = df.drop(columns=[""])
    
    # Renombramos a nuestro estándar
    df = df.rename(columns={
        'Fecha entrada': 'Entry Date',
        'Fecha salida':  'Exit Date',
        'Tipo':          'Side',
        'Rdto Abs':      'Profit',
        'Ganancia unitaria': 'Profit %',
        'MFE':           'MFE',
        'MAE':           'MAE'
    })

    # Mapear meses ES→EN para parseo fiable
   # Mapear meses ES→EN
    month_map = {
        'ene':'Jan','feb':'Feb','mar':'Mar','abr':'Apr','may':'May','jun':'Jun',
        'jul':'Jul','ago':'Aug','sep':'Sep','sept':'Sep',
        'oct':'Oct','nov':'Nov','dic':'Dec'
    }
    
    for col in ['Entry Date','Exit Date']:
        s = df[col].astype(str).str.lower()
        for es, en in month_map.items():
            s = s.str.replace(es, en, regex=True)
        # Inferimos el formato, acepta yy o yyyy, distintos separadores, comas, espacios…
        df[col] = pd.to_datetime(
            s,
            dayfirst=True,
            infer_datetime_format=True,
            errors='coerce'
        )

    # Eliminamos filas sin fecha válida
    df = df.dropna(subset=['Entry Date','Exit Date'])

    # Profit % → decimal
    df['Profit %'] = (
        df['Profit %'].astype(str)
          .str.replace('%','', regex=False)
          .str.replace(',','.', regex=False)
          .astype(float, errors='ignore')
          .fillna(0.0) / 100
    )
    # Profit absoluto → float
    df['Profit'] = (
        df['Profit'].astype(str)
          .str.replace('[^0-9,.-]','', regex=True)
          .str.replace('\.','', regex=True)   # quita miles
          .str.replace(',','.', regex=False)  # pasa coma a punto
          .astype(float, errors='ignore')
          .fillna(0.0)
    )

    return df


# --- Funciones métricas ---
def compute_equity(trades, init_cap):
    df = trades.sort_values('Exit Date').copy()
    df['Equity'] = init_cap + df['Profit'].cumsum()
    equity = pd.DataFrame({
        'Date':   [df['Entry Date'].min()] + df['Exit Date'].tolist(),
        'Equity': [init_cap] + df['Equity'].tolist()
    })
    equity = equity.set_index('Date')
    return equity

def calculate_metrics(trades: pd.DataFrame, equity_df: pd.DataFrame) -> dict:
    """Calcula todas las métricas clave a partir de trades y la curva de equity."""
    # Extraemos la serie de equity
    equity = equity_df['Equity']

    # Capital inicial y final
    ini = equity.iloc[0]
    fin = equity.iloc[-1]

    # Total P&L y crecimiento
    total_profit = fin - ini
    growth = fin / ini - 1

    # CAGR
    days = (equity.index[-1] - equity.index[0]).days or 1
    cagr = (fin / ini) ** (365.0 / days) - 1

    # Drawdowns
    cummax = equity.cummax()
    dd_rel = (equity - cummax) / cummax
    mdd_pct = dd_rel.min()
    mdd_abs = (equity - cummax).min()

    # Sharpe Ratio (retornos diarios)
    daily = equity.pct_change().dropna()
    std_dev = daily.std()
    sharpe = (daily.mean() / std_dev * np.sqrt(252)) if std_dev > 0 else 0.0

    # Estadísticas de trades
    n = len(trades)
    wins   = trades[trades['Profit'] > 0]
    losses = trades[trades['Profit'] < 0]
    win_rate = len(wins) / n if n > 0 else 0.0
    avg_dur = (trades['Exit Date'] - trades['Entry Date']).dt.days.mean() if n > 0 else 0.0
    avg_ret_trade = trades['Profit %'].mean() if 'Profit %' in trades else 0.0

    gross_profit = wins['Profit'].sum()
    gross_loss   = abs(losses['Profit'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.nan

    avg_win = wins['Profit %'].mean() if not wins.empty else 0.0
    avg_loss = abs(losses['Profit %'].mean()) if not losses.empty else 0.0
    payoff = avg_win / avg_loss if avg_loss > 0 else np.nan

    return {
        "Beneficio Total":       total_profit,
        "Crecimiento Capital":   growth,
        "CAGR":                  cagr,
        "Max Drawdown %":        mdd_pct,
        "Max Drawdown $":        mdd_abs,
        "Sharpe Ratio":          sharpe,
        "Total Operaciones":     n,
        "% Ganadoras":           win_rate,
        "Duración Media (días)": avg_dur,
        "Retorno Medio/Op. (%)": avg_ret_trade,
        "Factor de Beneficio":   pf,
        "Ratio Payoff":          payoff
    }


# --- Sidebar ---
st.sidebar.header("📁 Carga de Datos")
trades_file = st.sidebar.file_uploader("Reporte PRT (CSV)", type=["csv","txt"])
initial_cap = st.sidebar.number_input("Capital Inicial", value=10000.0, min_value=0.0, step=100.0, format="%.2f")

if not trades_file:
    st.sidebar.warning("Por favor, sube primero el CSV de operaciones.")
    st.stop()

# --- Procesamiento ---
trades = load_prt_trades(trades_file)
equity = compute_equity(trades, initial_cap)
metrics = calculate_metrics(trades, equity)

# --- Pestañas ---
tabs = st.tabs(["📊 Resumen Métricas","📈 Equity & Drawdown","📝 Operaciones (Trades)"])

with tabs[0]:
    st.header("📋 Resumen de Métricas")
    # Mostrar en cuatro columnas
    cols = st.columns(4)
    keys = list(metrics.keys())
    for i, key in enumerate(keys):
        val = metrics[key]
        # formateo
        if "Ratio" in key or "Sharpe" in key or "Payoff" in key or "CAGR" in key or "Retorno" in key or "Crecimiento" in key:
            disp = f"{val*100:.2f}%" if "Max Drawdown" not in key else f"{val*100:.2f}%"
        elif "Total Operaciones" in key:
            disp = f"{int(val)}"
        else:
            disp = f"${val:,.2f}"
        cols[i%4].metric(label=key, value=disp)

with tabs[1]:
    st.header("📈 Curva de Equity y Drawdown")
    fig, ax = plt.subplots(1,2, figsize=(12,4))
    # Equity
    ax[0].plot(equity.index, equity['Equity'], linewidth=2)
    ax[0].set_title("Curva de Equity")
    ax[0].set_xlabel("Fecha"); ax[0].set_ylabel("Equity")
    ax[0].grid(True)
    # Drawdown %
    cummax = equity['Equity'].cummax()
    dd = (equity['Equity'] - cummax)/cummax * 100
    ax[1].fill_between(dd.index, dd, 0, color='red')
    ax[1].set_title("Drawdown (%)")
    ax[1].set_xlabel("Fecha"); ax[1].set_ylabel("Drawdown %")
    ax[1].grid(True)
    st.pyplot(fig)

with tabs[2]:
    st.header("📝 Detalle de Operaciones")
    st.dataframe(
        trades[['Entry Date','Exit Date','Side','Profit','Profit %','MFE','MAE']]
        .reset_index(drop=True),
        use_container_width=True
    )
