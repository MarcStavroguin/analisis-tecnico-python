# 📘 Guía de indicadores técnicos
**Referencia cruzada entre el código del programa y el libro de John J. Murphy _Análisis técnico de los mercados financieros_**  
*(Incluye descripción de colores, estilos de línea y marcadores utilizados en las gráficas)*

| Nº | Función / Plot en el programa | Indicador según John J. Murphy (versión castellana) | Parámetros estándar | Color / estilo en gráfico | Descripción técnica según Murphy | Interpretación práctica |
|----|-------------------------------|------------------------------------------------------|---------------------|----------------------------|----------------------------------|--------------------------|
| **1** | `plot_asset()` | **Medias Móviles Simples (SMA)** + **Cruces dorados/muertes (Golden/Death Cross)** + **Niveles ATR** | SMA 50 / SMA 200 | SMA50: **naranja** • SMA200: **púrpura** • Golden: triángulo **verde ↑** • Death: triángulo **rojo ↓** • Niveles ATR: ❌ rojo (stop) / ❌ verde (objetivo) | Dos medias de distinta longitud; el cruce al alza o a la baja genera señales de cambio de tendencia. | Golden Cross → tendencia alcista. Death Cross → tendencia bajista. ATR delimita zonas de riesgo/beneficio. |
| **2** | `plot_asset_ema()` | **Medias Móviles Exponenciales (EMA)** + **Cruces Golden/Death** + **Niveles ATR** | EMA 12 / EMA 26 | EMA12: **naranja** • EMA26: **púrpura** • Cruces y niveles ATR como en SMA | Igual que las SMA, pero las EMA ponderan más el precio reciente. | Cruce 12/26 (c/p) muy usado por traders activos. |
| **3** | `plot_asset_macd()` | **Convergencia/Divergencia de Medias Móviles (MACD)** | (12, 26, 9) | MACD: **azul** • Señal: **naranja** • Histograma: **verde** (≥0) / **rojo** (<0) | Diferencia entre dos EMAs y su señal. | Cruce MACD↑Señal → compra; MACD↓Señal → venta; divergencias anticipan giros. |
| **4** | `plot_asset_rsi()` | **Índice de Fuerza Relativa (RSI)** | (14) | Línea **púrpura** • Guías: **70/30** gris | Oscilador de momentum de Wilder. | RSI > 70 sobrecompra; < 30 sobreventa; divergencias relevantes. |
| **5** | `plot_asset_bbands()` | **Bandas de Bollinger** | SMA 20 ± 2σ | Media: **azul** • Bandas: **gris** | Canal de volatilidad con media y desviación típica. | Compresión → posibles rupturas; expansión → volatilidad alta. |
| **6** | `plot_asset_atr()` | **Rango Verdadero Medio (ATR)** | (14) | Línea **marrón** | Volatilidad (no dirección). | ATR alto = mercado agitado; útil para stops. |
| **7** | `plot_asset_momentum_smooth()` | **Oscilador de Momentum (M = V − Vₓ)** + **SMA del Momentum** | 10 + SMA(10) | Momentum: **naranja** • SMA: **púrpura** | Velocidad del cambio de precios, con suavizado. | Cruce Momentum/SMA confirma o debilita el impulso. |
| **8** | `plot_asset_adx()` | **Índice Direccional Medio (ADX)** + **+DI / −DI** | (14) | ADX: **azul** • +DI: **verde** • −DI: **rojo** • Guías **20/25** gris | Mide la **fuerza** de tendencia; +DI/−DI indican dirección. | ADX>20-25 = tendencia válida; +DI>−DI alcista; lo contrario bajista. |
| **9** | `plot_asset_psar()` | **Parabolic SAR** | paso 0.02, máx 0.2 | SAR alcista: puntos **verdes** (bajo precio) • SAR bajista: **rojos** (sobre precio) | Indicador tendencial con puntos “stop and reverse”. | Puntos bajo precio → tramo alcista; giro de puntos → posible cambio. |
| **10** | `plot_asset_stochastic()` | **Estocástico %K/%D (lento)** | (14, 3, 3) | %K: **azul** • %D: **naranja** • Guías **80/20** gris | Oscilador de posición del cierre dentro del rango. | %K cruza %D desde abajo → compra (mejor si <20); desde arriba → venta (si >80). |
| **11** | `plot_asset_williams_r()` | **Williams %R** | (14) | Línea **azul** • Guías **−20/−80** gris • Línea central **−50** gris claro | Oscilador de −100 a 0 basado en HH/LL. | Lecturas > −20 = sobrecompra; < −80 = sobreventa. |
| **12** | `plot_asset_cci()` | **Índice de Canal de Materias Primas (CCI)** | (20) | Línea **azul** • Guías **±100** gris • **0** gris claro | Desviación del TP (Typical Price) respecto a su media. | > +100 sobrecompra; < −100 sobreventa; rupturas de ±100 dan señales. |
| **13** | `plot_asset_dpo()` | **Oscilador de Precio Detrendido (DPO)** | (20) | Línea **azul** • **0** gris | Elimina la tendencia para resaltar ciclos. | Cruces con 0 y picos locales para detectar fatiga o inicio de ciclo. |
| **14** | `plot_asset_obv()` | **On-Balance Volume (OBV)** (+SMA opcional) | MA (20) opcional | OBV: **azul** • OBV_SMA: **naranja** | Acumula volumen según dirección del cierre. | Confirmación de tendencia: precio y OBV deben acompañarse. |
| **15** | `plot_asset_ad_line()` | **Línea de Acumulación/Distribución (A/D)** (+SMA opcional) | MA (20) opcional | A/D: **azul** • A/D_SMA: **naranja** | CLV pondera volumen según cierre alto/bajo del rango. | Divergencias entre A/D y precio alertan sobre fortaleza/debilidad. |
| **16** | `plot_asset_cmf()` | **Chaikin Money Flow (CMF)** | (20) | Línea **azul** • Guías **0** gris y **±0.1** gris claro | Flujo de dinero acumulado por ventana de 20 periodos. | > 0.1 presión compradora; < −0.1 vendedora; cruces de 0 relevantes. |
| **17** | `plot_asset_vo()` | **Oscilador de Volumen (VO)** | EMA 14/28 (% o abs.) | Barras **verdes/rojas** según signo • Línea **azul** • Guía **0** gris | Diferencia entre medias de volumen corto/largo. | Cambios de signo → aceleraciones de interés; útil con rupturas. |
| **18** | `plot_asset_keltner()` | **Canales de Keltner** | EMA 20 ± ATR(10) × 2.0 | Media: **azul** • Bordes: **gris** | Canal basado en ATR sobre una EMA central. | Toques exteriores en tendencia suelen ser continuaciones; salidas pueden marcar reversión. |
| **19** | `plot_asset_donchian()` | **Canales de Donchian** | (20) | Superior/Inferior: **gris** • Media: **azul** | Envolvente por máximos y mínimos del periodo. | Rupturas de canal definen señales tipo *Turtle Trading*. |
| **20** | `plot_asset_chandelier()` | **Chandelier Exit** | ATR 22, ×3, HL 22 | CE_long: **verde** • CE_short: **rojo** | Stop dinámico basado en ATR y HH/LL. | Sirve como salida seguimiento de tendencia o trailing stop. |
| **21** | `plot_asset_pivots_classic/fibonacci/camarilla()` | **Puntos Pivote (Clásicos / Fibonacci / Camarilla)** | Diario desplazado 1 barra | PP: **azul** • R: **rojo** • S: **verde** | Niveles intradía/diarios de referencia. | R/S actúan como resistencias/soportes; rupturas y re-test definen zonas clave. |
| **22** | `plot_asset_fibo_retracements()` | **Retrocesos de Fibonacci** | Lookback 100 (niveles: 0–100%) | Líneas **verdes** etiquetadas | Líneas horizontales en niveles clásicos (0, 23.6, 38.2, 50, 61.8, 78.6, 100). | Zonas 38.2–61.8 % como áreas de pullback probables. |
| **23** | `plot_asset_fibo_extensions()` | **Extensiones de Fibonacci** | Lookback 100 (1.272/1.618/2.618) | Líneas **naranja** etiquetadas | Proyecciones por encima/debajo del tramo base. | Objetivos y “measured moves” en tendencias fuertes. |
| **24** | `plot_asset_envelopes()` | **Medias Móviles Envolventes (Envelopes)** | SMA 20 ± 5% (ejemplo) | Media: **azul** • Bandas: **gris** | Bandas fijas porcentuales alrededor de una media. | Precio fuera de banda = sobreextensión; reentrada ≈ corrección a la media. |
| **25** | `plot_asset_ichimoku()` | **Ichimoku Kinko Hyo** | (9, 26, 52), desplazamiento 26 | Tenkan: **naranja** • Kijun: **púrpura** • Senkou A: **verde** • Senkou B: **rojo** • Chikou: **azul** | Sistema tendencial con líneas adelantadas y retrasadas. | Precio sobre la nube = sesgo alcista; cruces Tenkan/Kijun confirman; nube indica soporte/resistencia dinámica. |

---

### ⚙️ Parámetros internos relevantes

| Función base | Propósito | Parámetros principales | Comentario técnico |
|---------------|------------|------------------------|--------------------|
| `levels_from_entries(df, entries, atr_mult, rr)` | Niveles stop/objetivo por ATR | `atr_mult=1.5`, `rr=1.5` | Se alinea con el índice de `df` antes de graficar. |
| `adx(df, length)` | ADX y DI’s | `length=14` | Suavizado tipo Wilder (RMA vía EWM). |
| `parabolic_sar(df, step, max_step)` | Puntos SAR | `step=0.02`, `max_step=0.2` | Implementación iterativa con “clamping”. |
| `stochastic_kd(df, length, k_smooth, d_smooth)` | %K/%D | `(14, 3, 3)` | Versión “slow” por doble suavizado. |
| `williams_r(df, length)` | Williams %R | `length=14` | Rango −100 a 0; evita divisiones por cero. |
| `cci(df, length, c)` | CCI | `length=20`, `c=0.015` | Mean Deviation clásica de Lambert. |
| `dpo(df, length)` | DPO | `length=20` | `shift=floor(length/2)+1`. |
| `obv(df)` | OBV | — | Acumula volumen según signo del cierre. |
| `ad_line(df)` | A/D Line | — | CLV = (2C−H−L)/(H−L). |
| `cmf(df, length)` | Chaikin Money Flow | `length=20` | Media móvil de flujo/volumen. |
| `volume_oscillator(df, short, long, ma, pct)` | VO | `14/28`, `ma="ema"`, `%` | Versión porcentual o absoluta; barras ±. |
| `keltner_channels(df, length_ma, length_atr, multiplier, ma)` | Keltner | `EMA 20`, `ATR 10`, `x2.0` | Media central configurable (SMA/EMA). |
| `donchian_channels(df, length)` | Donchian | `20` | Upper=max(H), Lower=min(L). |
| `chandelier_exit(df, length_atr, multiplier, length_hl)` | CE | `22, x3, 22` | Stops dinámicos long/short. |
| `pivot_points(df, method)` | Pivots | `"classic"`, `"fibonacci"`, `"camarilla"` | Desplazados 1 día (usa H/L/C previos). |
| `fibonacci_retracements(df, length, levels)` | Retrocesos Fib | `length=100` | Columnas constantes alineadas a `df.index`. |
| `fibonacci_extensions(df, length, ratios)` | Extensiones Fib | `length=100` | Ratios típicos 1.272/1.618/2.618. |
| `moving_average_envelopes(df, length, pct, ma)` | Envelopes | `SMA 20`, `±5%` | Bandas fijas sobre media elegida. |
| `ichimoku(df, tenkan, kijun, senkou_b, displacement)` | Ichimoku | `9, 26, 52`, `disp 26` | Calcula Tenkan/Kijun; Senkou A/B adelantadas; Chikou atrasada. |
---

### 🧩 Recomendaciones de visualización

- **Subplots secundarios:** RSI, MACD, Momentum, ATR, ADX/DI, CMF/VO, CCI/DPO, OBV/A/D.  
- **Superpuestos sobre las velas:** SMA/EMA, Bollinger, Keltner, Donchian, Envelopes, Chandelier, Pivots, Fibonacci, PSAR, Ichimoku.  
- **Marcadores visuales:**
  - 🔼 *Golden Cross* → **verde**
  - 🔽 *Death Cross* → **rojo**
  - ❌ *Stops/Targets ATR* → **rojo** (stop) / **verde** (objetivo)
- **Volumen:** activado en todos los gráficos (`volume=True`).
- **Estilo base:** `"yahoo"` con formato limpio, fondo claro u oscuro según entorno.

---

📖 **Referencia principal:**  
Murphy, John J. *Análisis técnico de los mercados financieros* (versión castellana).  
Clasificación e interpretación de indicadores según metodología clásica.

---

🧠 **Notas finales:**
- Todas las funciones están vectorizadas y alineadas por índice (`df.index`) para evitar errores de tamaño.  
- Cada plot se ejecuta con `block=True` para permitir inspección manual de las gráficas.  
- El conjunto de indicadores cubre más del **90 %** de los presentados en el texto de Murphy, integrando tanto indicadores de **tendencia**, **momentum**, **volumen** como **volatilidad**.  
- El programa está preparado para ampliarse con módulos de **backtesting** o **dashboards interactivos (Streamlit/Dash)** sin modificar su estructura base.

---

✳️ **Autoría técnica y documentación:**  
Desarrollo e integración por *Marc Stavroguin*, con estructura modular Python (`functions_trading.py`, `functions_plot.py`, `run_trading.py`).  
Documentación adaptada al castellano y referenciada según la edición española del manual de Murphy.
