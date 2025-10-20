# 📊 Sistema de Análisis Técnico en Python

Conjunto modular de scripts en **Python** para análisis técnico bursátil, inspirado en la metodología de  
📘 *John J. Murphy – “Análisis técnico de los mercados financieros” (versión castellana)*.

---

## 🧩 Estructura del proyecto

| Archivo | Función principal |
|----------|-------------------|
| `functions_trading.py` | Cálculo de indicadores técnicos (SMA, EMA, RSI, MACD, ATR, Ichimoku, etc.) |
| `functions_plot.py` | Representación gráfica con `mplfinance` |
| `run_trading.py` | Ejecución principal: descarga de datos, iteración de tickers y generación de gráficos |
| `indicadores_uso.md` | Guía detallada de uso, parámetros e interpretación (versión extendida) |

---

## 🔍 Indicadores incluidos

| Categoría | Indicadores |
|------------|--------------|
| **Tendencia** | SMA / EMA / Envelopes / Ichimoku / Parabolic SAR / ADX / Keltner / Donchian / Chandelier / Pivots / Fibonacci |
| **Momentum** | RSI / MACD / Momentum / Estocástico / Williams %R / CCI / DPO |
| **Volumen** | OBV / Acumulación-Distribución / Chaikin Money Flow / Oscilador de Volumen |
| **Volatilidad** | ATR / Bandas de Bollinger |
| **Complementarios** | Retrocesos y extensiones de Fibonacci / Niveles ATR / Cruces dorados y muertes |

---

## 🎨 Visualización

- **Velas tipo Yahoo Finance** (`style='yahoo'`)
- Indicadores superpuestos o en subgráficos automáticos
- Marcadores visuales:
  - 🔼 *Golden Cross* → verde  
  - 🔽 *Death Cross* → rojo  
  - ❌ *Stops/Targets ATR* → rojo (stop) / verde (objetivo)
- Ventanas bloqueantes (`block=True`) para inspección manual
- Volumen activo en todos los gráficos (`volume=True`)

---

## 🧠 Extensiones previstas

- **Dashboard interactivo (Streamlit/Dash)** con zoom, selección de indicadores y hover dinámico  
- **Módulo de backtesting** (Backtrader o VectorBT) para simulación de estrategias  
- Exportación de resultados a Excel / Google Sheets

---

## ⚙️ Requisitos

pip install yfinance mplfinance pandas numpy matplotlib -->

