import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# --- Configuración de página ---
st.set_page_config(
    page_title="📊 Backtest PRT Statistics",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 Backtest PRT Statistics")
st.markdown("Carga el reporte de operaciones de ProRealTime y calcula automáticamente la curva de equity y las métricas clave.")

# --- Funciones de carga y cálculo ---
@st.cache_data
def load_prt_trades(file):
    """Carga y limpia el CSV de operaciones exportado por ProRealTime."""
    df = pd.read_csv(file, sep='\t', decimal=',')
    df = df.rename(columns={
        'Fecha entrada': 'Entry Date',
        'Fecha salida':  'Exit Date',
        'Tipo':          'Side',
        'Rdto Abs':      'Profit',
        'Ganancia unitaria': 'Profit %',
        'MFE':           'MFE',
        'MAE':           'MAE'
    })
    # Parseo de fechas (día primero, mes abreviado en español)
    df['Entry Date'] = pd.to_datetime(df['Entry Date'], format='%d %b %Y, %H:%M:%S', dayfirst=True)
    df['Exit Date']  = pd.to_datetime(df['Exit Date'],  format='%d %b %Y, %H:%M:%S', dayfirst=True)
    # Profit % → decimal
    df['Profit %'] = (
        df['Profit %']
          .str.replace('%','')
          .str.replace(',','.')
          .astype(float) / 100
    )
    # Profit absoluto → float
    df['Profit'] = (
        df['Profit']
          .str.replace('[^0-9,.-]', '', regex=True)
          .str.replace('.','', regex=False)   # miles
          .str.replace(',','.', regex=False)  # decimal
          .astype(float)
    )
    return df

def compute_equity_series(trades_df: pd.DataFrame, initial_cap: float) -> pd.DataFrame:
    """Construye la serie de equity a partir de los trades y capital inicial."""
    df = trades_df.sort_values('Exit Date').copy()
    df['Equity'] = initial_cap + df['Profit'].cumsum()
    equity = pd.DataFrame({
        'Date':   [df['Entry Date'].min()] + df['Exit Date'].tolist(),
        'Equity': [initial_cap] + df['Equity'].tolist()
    })
    return equity

def max_drawdown(equity: pd.Series) -> float:
    """Calcula el Max Drawdown (mínimo porcentaje)."""
    cum_max = equity.cummax()
    drawdown = (equity - cum_max) / cum_max
    return drawdown.min()

def annualized_return(equity: pd.Series, dates: pd.Series) -> float:
    """Calcula el CAGR sobre la serie de equity."""
    days = (dates.iloc[-1] - dates.iloc[0]).days
    if days <= 0:
        return 0.0
    total_ret = equity.iloc[-1] / equity.iloc[0]
    return total_ret ** (365.0 / days) - 1

def sharpe_ratio(equity: pd.Series, rf: float = 0.0) -> float:
    """Calcula el Sharpe Ratio asumiendo retornos diarios."""
    daily = equity.pct_change().dropna()
    if daily.std() == 0:
        return 0.0
    return (daily.mean() - rf/252) / daily.std() * np.sqrt(252)

# --- Sidebar: carga y parámetros ---
st.sidebar.header("📁 Carga de Datos y Parámetros")
trades_file = st.sidebar.file_uploader("Operaciones (CSV PRT)", type=["csv","txt"])
initial_cap = st.sidebar.number_input("Capital Inicial", min_value=0.0, value=10000.0, step=100.0, format="%.2f")

if not trades_file:
    st.info("👈 Por favor, sube el reporte de operaciones exportado por ProRealTime.")
    st.stop()

# --- Procesamiento ---
trades = load_prt_trades(trades_file)
equity_df = compute_equity_series(trades, initial_cap)
dates = equity_df['Date']
eq_values = equity_df['Equity']

cagr = annualized_return(eq_values, dates)
mdd  = max_drawdown(eq_values)
shp  = sharpe_ratio(eq_values)

# --- Visualización ---
st.subheader("🏷️ Resumen de Métricas")
col1, col2, col3 = st.columns(3)
col1.metric("CAGR",           f"{cagr*100:.2f}%")
col2.metric("Max Drawdown",   f"{mdd*100:.2f}%")
col3.metric("Sharpe Ratio",   f"{shp:.2f}")

st.subheader("📈 Curva de Equity")
fig, ax = plt.subplots()
ax.plot(dates, eq_values, linewidth=2)
ax.set_xlabel("Fecha")
ax.set_ylabel("Equity")
ax.grid(True)
st.pyplot(fig)

st.subheader("📑 Detalle de Operaciones")
st.dataframe(
    trades[['Entry Date','Exit Date','Side','Profit','Profit %','MFE','MAE']]
    .reset_index(drop=True),
    use_container_width=True
)

# --- Exportar Informe Excel ---
def generate_excel(trades_df, equity_df, metrics: dict):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        trades_df.to_excel(writer, sheet_name="Operaciones", index=False)
        equity_df.to_excel(writer, sheet_name="Curva Equity", index=False)
        pd.DataFrame.from_dict(metrics, orient='index', columns=['Valor']).to_excel(writer, sheet_name="Métricas")
    buf.seek(0)
    return buf

metrics_dict = {
    "CAGR":           f"{cagr:.6f}",
    "Max Drawdown %": f"{mdd:.6f}",
    "Sharpe Ratio":   f"{shp:.4f}"
}

excel_buf = generate_excel(trades, equity_df, metrics_dict)
st.sidebar.download_button(
    "📥 Descargar Informe Excel",
    excel_buf,
    file_name="backtest_prt_stats.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
