import pandas as pd
import numpy as np
import yfinance as yf
import mplfinance as mpf
import matplotlib.pyplot as plt

# ===========================
#     CARGA DE DATOS
# ===========================


def get_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Descarga datos OHLCV con yfinance.history()."""
    df = yf.Ticker(ticker).history(period=period)
    if df is None or df.empty:
        return pd.DataFrame()
    # Aseguramos columnas estándar esperadas por mplfinance
    expected = {'Open', 'High', 'Low', 'Close', 'Volume'}
    if not expected.issubset(df.columns):
        # Reetiquetado defensivo si vinieran con otra capitalización
        rename = {c: c.title() for c in df.columns if c.title() in expected}
        df = df.rename(columns=rename)
    return df

# ===========================
#     INDICADORES BÁSICOS
# ===========================


def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """Media móvil simple vectorizada."""
    return series.rolling(window=window, min_periods=1).mean()


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    """Media móvil exponencial."""
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range (versión simple, media móvil simple)."""
    high, low, close = df['High'], df['Low'], df['Close']
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length, min_periods=1).mean()

# ===========================
#     SEÑALES (CRUCES)
# ===========================


def golden_death_crosses(sma_short: pd.Series, sma_long: pd.Series):
    """Devuelve índices (golden, death) usando cambios de signo sin bucles."""
    sma_short, sma_long = sma_short.align(sma_long, join='inner')
    prev = (sma_short - sma_long).shift(1)
    curr = (sma_short - sma_long)
    golden = (prev <= 0) & (curr > 0)
    death = (prev >= 0) & (curr < 0)
    return sma_short.index[golden.fillna(False)], sma_short.index[death.fillna(False)]

# ===========================
#     NIVELES STOP / LIMIT
# ===========================


def levels_from_entries(df: pd.DataFrame, entries: pd.Series, atr_mult: float = 1.5, rr: float = 1.5) -> pd.DataFrame:
    """Calcula precio de entrada, stop (ATR*atr_mult) y objetivo (RR*ATR*atr_mult)."""
    _atr = atr(df, 14)
    entry_price = df['Close'].where(entries)
    atr_at_entry = _atr.where(entries)

    stop_price = (entry_price - atr_mult*atr_at_entry).round(2)
    limit_price = (entry_price + rr*atr_mult*atr_at_entry).round(2)

    out = pd.DataFrame({
        'entry_price': entry_price,
        'stop_price': stop_price,
        'limit_price': limit_price
    }, index=df.index)
    return out

# === Indicadores avanzados  ===


def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index (RSI) clásico de Wilder."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=length, min_periods=1).mean()
    avg_loss = loss.rolling(window=length, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(series: pd.Series, short=12, long=26, signal=9):
    """MACD line y Signal line."""
    ema_short = series.ewm(span=short, adjust=False).mean()
    ema_long = series.ewm(span=long, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(series: pd.Series, length=20, n_std=2):
    """Bandas de Bollinger: media ± n desviaciones estándar."""
    ma = series.rolling(window=length).mean()
    std = series.rolling(window=length).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return ma, upper, lower


def momentum_smooth(df, period=10, smooth=10):
    """
    Devuelve Momentum (M = V - Vx) y su media móvil (SMA del Momentum).
    Añade las columnas:
      - 'Momentum'
      - 'Momentum_SMA'
    """
    # Reutilizamos momentum existente
    if 'Momentum' not in df.columns:
        df['Momentum'] = df['Close'] - df['Close'].shift(period)

    #  SMA acepta una Serie; reutilizamos la misma firma
    df['Momentum_SMA'] = sma(df['Momentum'], window=smooth)
    return df


def adx(df: pd.DataFrame, length: int = 14):
    """
    ADX (Average Directional Index) de Wilder.
    Devuelve tres Series alineadas con el índice de df:
      - ADX(length) : fuerza de la tendencia (no direccional)
      - +DI         : Directional Indicator positivo
      - -DI         : Directional Indicator negativo
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    # True Range
    tr_components = pd.concat([
        (high - low),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1)
    tr = tr_components.max(axis=1)

    # +DM y -DM (según definición de Wilder)
    up_move = high.diff()
    down_move = -low.diff()  # low_{t-1} - low_t
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) &
                        (down_move > 0), down_move, 0.0)

    # Suavizados de Wilder (RMA) vía EWM(alpha=1/length)
    alpha = 1 / float(length)
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smoothed = pd.Series(plus_dm, index=df.index).ewm(
        alpha=alpha, adjust=False).mean()
    minus_dm_smoothed = pd.Series(minus_dm, index=df.index).ewm(
        alpha=alpha, adjust=False).mean()

    # +DI y -DI
    plus_di = 100 * (plus_dm_smoothed / atr)
    minus_di = 100 * (minus_dm_smoothed / atr)

    # DX y ADX
    di_sum = (plus_di + minus_di)
    di_diff = (plus_di - minus_di).abs()
    dx = 100 * (di_diff / di_sum.replace(0, np.nan))
    adx_series = dx.ewm(alpha=alpha, adjust=False).mean()

    plus_di = plus_di.rename('+DI')
    minus_di = minus_di.rename('-DI')
    adx_series = adx_series.rename(f'ADX({length})')

    return adx_series, plus_di, minus_di


def parabolic_sar(df: pd.DataFrame, step: float = 0.02, max_step: float = 0.2) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Parabolic SAR (Welles Wilder).
    Devuelve tres series:
      - psar: valor del SAR en cada vela
      - psar_bull: SAR cuando la tendencia es alcista (otros puntos NaN)
      - psar_bear: SAR cuando la tendencia es bajista (otros puntos NaN)

    Parámetros:
      step: factor de aceleración inicial (AF)
      max_step: AF máximo (clásico 0.2)

    Notas:
      - Implementación iterativa (el SAR depende del estado previo).
      - Ajustes 'clamping' a los últimos 2 lows/highs según la dirección.
    """
    if df is None or df.empty:
        return pd.Series(dtype='float64'), pd.Series(dtype='float64'), pd.Series(dtype='float64')

    high = df['High'].to_numpy()
    low = df['Low'].to_numpy()
    n = len(df)

    psar = np.zeros(n, dtype=float)
    bull = np.zeros(n, dtype=bool)  # True = tendencia alcista, False = bajista

    # Inicialización: detectamos dirección por las dos primeras velas
    # Si no, asumimos alcista por defecto.
    if n >= 2:
        bull[0] = True if high[1] >= high[0] else False
    else:
        bull[0] = True

    # EP (extreme point) y AF (acceleration factor)
    af = step
    if bull[0]:
        ep = high[0]
        psar[0] = low[0]
    else:
        ep = low[0]
        psar[0] = high[0]

    # Segundo punto (usamos también datos de la 1ª vela)
    if n >= 2:
        if bull[0]:
            ep = max(ep, high[1])
            psar[1] = min(low[0], low[1])  # clamp inicial
            bull[1] = True
        else:
            ep = min(ep, low[1])
            psar[1] = max(high[0], high[1])
            bull[1] = False

    # Iteración principal
    for i in range(2, n):
        prev_psar = psar[i-1]
        prev_bull = bull[i-1]

        # Cálculo del SAR proyectado
        sar = prev_psar + af * (ep - prev_psar)

        if prev_bull:
            # clamp: no puede estar por encima de lows previos
            sar = min(sar, low[i-1], low[i-2])

            # ¿nuevo EP (máximo)?
            if high[i] > ep:
                ep = high[i]
                af = min(af + step, max_step)

            # ¿cambio de tendencia?
            if low[i] < sar:
                # giro a bajista
                bull[i] = False
                psar[i] = ep  # al girar, SAR se iguala al EP
                af = step
                ep = low[i]
            else:
                # seguimos alcistas
                bull[i] = True
                psar[i] = sar
        else:
            # clamp: no puede estar por debajo de highs previos
            sar = max(sar, high[i-1], high[i-2])

            # ¿nuevo EP (mínimo)?
            if low[i] < ep:
                ep = low[i]
                af = min(af + step, max_step)

            # ¿cambio de tendencia?
            if high[i] > sar:
                # giro a alcista
                bull[i] = True
                psar[i] = ep
                af = step
                ep = high[i]
            else:
                # seguimos bajistas
                bull[i] = False
                psar[i] = sar

    psar_series = pd.Series(psar, index=df.index,
                            name=f'PSAR({step},{max_step})')
    psar_bull = psar_series.where(bull, other=np.nan).rename('PSAR_bull')
    psar_bear = psar_series.where(~pd.Series(
        bull, index=df.index), other=np.nan).rename('PSAR_bear')
    return psar_series, psar_bull, psar_bear


def stochastic_kd(df: pd.DataFrame, length: int = 14, k_smooth: int = 3, d_smooth: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Índice Estocástico %K/%D (versión 'slow' por suavizado).
    %K_fast  = 100 * (Close - LL(length)) / (HH(length) - LL(length))
    %K       = SMA(%K_fast, k_smooth)
    %D       = SMA(%K, d_smooth)

    Devuelve:
      - k (%K suavizada), con nombre: Stoch %K(length,k_smooth)
      - d (%D),           con nombre: Stoch %D(length,k_smooth)
    """
    if df is None or df.empty:
        return (pd.Series(dtype='float64', name=f"Stoch %K({length},{k_smooth})"),
                pd.Series(dtype='float64', name=f"Stoch %D({length},{k_smooth})"))

    low_roll = df['Low'].rolling(window=length, min_periods=1).min()
    high_roll = df['High'].rolling(window=length, min_periods=1).max()
    denom = (high_roll - low_roll).replace(0,
                                           np.nan)  # evita división por cero

    k_fast = 100.0 * (df['Close'] - low_roll) / denom
    k = k_fast.rolling(window=k_smooth, min_periods=1).mean()
    d = k.rolling(window=d_smooth, min_periods=1).mean()

    k = k.clip(lower=0, upper=100).rename(f"Stoch %K({length},{k_smooth})")
    d = d.clip(lower=0, upper=100).rename(f"Stoch %D({length},{k_smooth})")
    return k, d


def williams_r(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """
    Williams %R (rango: -100 a 0).
    Fórmula: %R = -100 * (HH(length) - Close) / (HH(length) - LL(length))
    Umbrales típicos: sobrecompra ≈ -20, sobreventa ≈ -80.
    """
    if df is None or df.empty:
        return pd.Series(dtype='float64', name=f"Williams %R({length})")

    high_roll = df['High'].rolling(window=length, min_periods=1).max()
    low_roll = df['Low'].rolling(window=length,  min_periods=1).min()
    denom = (high_roll - low_roll).replace(0,
                                           np.nan)  # evita división por cero

    wr = -100.0 * (high_roll - df['Close']) / denom
    wr = wr.clip(lower=-100, upper=0).rename(f"Williams %R({length})")
    return wr


def cci(df: pd.DataFrame, length: int = 20, c: float = 0.015) -> pd.Series:
    """
    Commodity Channel Index (CCI).
    Fórmula: CCI = (TP - SMA(TP)) / (c * MeanDeviation(TP))
      TP = (High + Low + Close) / 3
      c  = 0.015 (convención habitual)
    Rango típico: ±100 (sobrecompra/sobreventa).
    """
    if df is None or df.empty:
        return pd.Series(dtype='float64', name=f"CCI({length})")

    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    sma_tp = tp.rolling(window=length, min_periods=1).mean()

    # Desviación media: media de |TP - SMA(TP)| en la ventana
    mean_dev = (tp - sma_tp).abs().rolling(window=length, min_periods=1).mean()

    denom = (c * mean_dev).replace(0, np.nan)
    cci_val = ((tp - sma_tp) / denom).rename(f"CCI({length})")
    return cci_val


def dpo(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """
    Detrended Price Oscillator (DPO)
    Definición estándar:
      shift = floor(length / 2) + 1
      DPO(t) = Close(t - shift) - SMA_length(t)

    Notas:
      - Oscila alrededor de 0 (elimina tendencia para resaltar ciclos).
      - Totalmente vectorizado; evita errores con df vacío o divisiones por cero.
    """
    if df is None or df.empty:
        return pd.Series(dtype='float64', name=f"DPO({length})")

    close = df['Close']
    sma_len = close.rolling(window=length, min_periods=1).mean()
    shift = int(np.floor(length / 2) + 1)
    close_shifted = close.shift(shift)

    dpo_series = (close_shifted - sma_len).rename(f"DPO({length})")
    return dpo_series


def obv(df: pd.DataFrame) -> pd.Series:
    """
    On-Balance Volume (OBV).
    Regla:
      - Si Close_t > Close_{t-1}  → OBV_t = OBV_{t-1} + Volume_t
      - Si Close_t < Close_{t-1}  → OBV_t = OBV_{t-1} - Volume_t
      - Si Close_t = Close_{t-1}  → OBV_t = OBV_{t-1}
    Devuelve una Serie alineada con el índice de df, comenzando en 0.
    """
    if df is None or df.empty:
        return pd.Series(dtype='float64', name="OBV")

    close_diff = df['Close'].diff()
    step = np.where(close_diff > 0, df['Volume'],
                    np.where(close_diff < 0, -df['Volume'], 0))
    obv_series = pd.Series(step, index=df.index).fillna(0).cumsum()
    return obv_series.rename("OBV")


def ad_line(df: pd.DataFrame) -> pd.Series:
    """
    Acumulación/Distribución (A/D Line, Chaikin A/D).
    CLV = [(Close - Low) - (High - Close)] / (High - Low)
        = (2*Close - High - Low) / (High - Low)
    A/D_t = A/D_{t-1} + (CLV_t * Volume_t)

    Notas:
      - Si High == Low, tomamos CLV = 0 para evitar división por cero.
      - Devuelve una Serie alineada con el índice de df, comenzando en 0.
    """
    if df is None or df.empty:
        return pd.Series(dtype='float64', name="A/D")

    high = df['High']
    low = df['Low']
    close = df['Close']
    vol = df['Volume']

    range_hl = (high - low)
    # CLV con guardas (cuando no hay rango, definimos 0)
    clv = (2*close - high - low) / range_hl.replace(0, np.nan)
    clv = clv.fillna(0.0)

    ad = (clv * vol).fillna(0.0).cumsum()
    return ad.rename("A/D")


def cmf(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """
    Chaikin Money Flow (CMF).
    CLV = [(Close - Low) - (High - Close)] / (High - Low) = (2*Close - High - Low) / (High - Low)
    CMF = SUM_{length}(CLV * Volume) / SUM_{length}(Volume)

    - Rango típico ~ [-1, 1]
    - Manejo seguro de High==Low (CLV=0) y volúmenes cero.
    """
    if df is None or df.empty:
        return pd.Series(dtype='float64', name=f"CMF({length})")

    high = df['High']
    low = df['Low']
    close = df['Close']
    vol = df['Volume']

    range_hl = (high - low).replace(0, np.nan)
    clv = ((2*close - high - low) / range_hl).fillna(0.0)

    money_flow = (clv * vol)
    mf_sum = money_flow.rolling(window=length, min_periods=1).sum()
    vol_sum = vol.rolling(
        window=length, min_periods=1).sum().replace(0, np.nan)

    cmf_series = (mf_sum / vol_sum).rename(f"CMF({length})")
    return cmf_series


def volume_oscillator(df: pd.DataFrame, short: int = 14, long: int = 28, ma: str = "ema", pct: bool = True) -> pd.Series:
    """
    Volumen Oscillator (VO)
    Definiciones comunes:
      VO = MA_short(Volume) - MA_long(Volume)                 (pct=False)
      VO% = 100 * [MA_short(Volume) - MA_long(Volume)] / MA_long(Volume)   (pct=True)

    Parámetros:
      short: ventana corta
      long:  ventana larga (debe ser > short; si no, se intercambian)
      ma:    "ema" o "sma"
      pct:   True → %; False → valor absoluto

    Robustez:
      - Si df está vacío → Serie vacía con nombre.
      - Evita /0 sustituyendo denominador 0 por NaN.
    """
    if df is None or df.empty:
        name = f"VO({'EMA' if ma == 'ema' else 'SMA'} {short},{long}" + \
            (",%" if pct else "") + ")"
        return pd.Series(dtype='float64', name=name)

    v = df['Volume'].astype(float)

    s, L = (short, long) if short <= long else (long, short)
    if ma.lower() == "ema":
        v_short = v.ewm(span=s, adjust=False).mean()
        v_long = v.ewm(span=L, adjust=False).mean()
        tag = "EMA"
    else:
        v_short = v.rolling(window=s, min_periods=1).mean()
        v_long = v.rolling(window=L, min_periods=1).mean()
        tag = "SMA"

    diff = v_short - v_long
    if pct:
        denom = v_long.replace(0, np.nan)
        vo = (100.0 * diff / denom).rename(f"VO({tag} {s},{L},%)")
    else:
        vo = diff.rename(f"VO({tag} {s},{L})")
    return vo


def keltner_channels(df: pd.DataFrame,
                     length_ma: int = 20,
                     length_atr: int = 10,
                     multiplier: float = 2.0,
                     ma: str = "ema") -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Canales de Keltner:
      Medio = MA(Close, length_ma)   [EMA por defecto]
      Superior = Medio + multiplier * ATR(length_atr)
      Inferior = Medio - multiplier * ATR(length_atr)

    Robusto:
      - Devuelve Series vacías si df está vacío.
      - Sin bucles ni iat[]: todo rolling/EWM.
    """
    if df is None or df.empty:
        mid = pd.Series(dtype='float64',
                        name=f"KC mid({ma.upper()},{length_ma})")
        up = pd.Series(dtype='float64',
                       name=f"KC up({length_ma},{length_atr},{multiplier})")
        lo = pd.Series(dtype='float64',
                       name=f"KC low({length_ma},{length_atr},{multiplier})")
        return mid, up, lo

    close = df['Close']
    # media central
    if ma.lower() == "ema":
        mid = close.ewm(span=length_ma, adjust=False).mean().rename(
            f"KC mid(EMA,{length_ma})")
    else:
        mid = close.rolling(window=length_ma, min_periods=1).mean().rename(
            f"KC mid(SMA,{length_ma})")

    # ATR para el ancho
    atr_series = atr(df, length=length_atr)
    up = (mid + multiplier *
          atr_series).rename(f"KC up({length_ma},{length_atr},{multiplier})")
    lo = (mid - multiplier *
          atr_series).rename(f"KC low({length_ma},{length_atr},{multiplier})")
    return mid, up, lo


def donchian_channels(df: pd.DataFrame, length: int = 20) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Donchian Channels (DC):
      Upper = rolling max(High, length)
      Lower = rolling min(Low,  length)
      Middle = (Upper + Lower)/2
    """
    if df is None or df.empty:
        up = pd.Series(dtype='float64', name=f"DC up({length})")
        lo = pd.Series(dtype='float64', name=f"DC low({length})")
        mid = pd.Series(dtype='float64', name=f"DC mid({length})")
        return mid, up, lo

    up = df['High'].rolling(
        window=length, min_periods=1).max().rename(f"DC up({length})")
    lo = df['Low'] .rolling(
        window=length, min_periods=1).min().rename(f"DC low({length})")
    mid = ((up + lo) / 2.0).rename(f"DC mid({length})")
    return mid, up, lo


def chandelier_exit(df: pd.DataFrame,
                    length_atr: int = 22,
                    multiplier: float = 3.0,
                    length_hl: int | None = None) -> tuple[pd.Series, pd.Series]:
    """
    Chandelier Exit (CE)
      CE_long  = HH(length_hl) - ATR(length_atr) * multiplier
      CE_short = LL(length_hl) + ATR(length_atr) * multiplier

    Notas:
      - Por defecto length_hl = length_atr (si no se pasa).
      - Usa la función atr(df, length_atr) ya definida en este módulo.
      - Devuelve dos Series alineadas con df.index: (CE_long, CE_short)
    """
    if df is None or df.empty:
        return (
            pd.Series(dtype='float64',
                      name=f"CE long({length_atr},{multiplier})"),
            pd.Series(dtype='float64',
                      name=f"CE short({length_atr},{multiplier})"),
        )

    if length_hl is None:
        length_hl = length_atr

    hh = df['High'].rolling(window=length_hl, min_periods=1).max()
    ll = df['Low'] .rolling(window=length_hl, min_periods=1).min()
    atr_series = atr(df, length=length_atr)

    ce_long = (hh - multiplier *
               atr_series).rename(f"CE long({length_hl},{length_atr},x{multiplier})")
    ce_short = (ll + multiplier *
                atr_series).rename(f"CE short({length_hl},{length_atr},x{multiplier})")
    return ce_long, ce_short


def pivot_points(df: pd.DataFrame,
                 method: str = "classic") -> pd.DataFrame:
    """
    Puntos Pivote diarios, desplazados 1 barra (se usan los del día previo).
    method: "classic" | "fibonacci" | "camarilla"

    Devuelve DataFrame con columnas:
      - Siempre: PP, R1..R3, S1..S3
      - Camarilla añade R4 y S4
    """
    if df is None or df.empty:
        cols = ['PP', 'R1', 'R2', 'R3', 'S1', 'S2', 'S3']
        out = pd.DataFrame(
            columns=cols, index=df.index if df is not None else None)
        return out

    H = df['High'].shift(1)
    L = df['Low'].shift(1)
    C = df['Close'].shift(1)
    rng = (H - L)

    pp = ((H + L + C) / 3.0)

    if method.lower() == "classic":
        r1 = 2*pp - L
        s1 = 2*pp - H
        r2 = pp + rng
        s2 = pp - rng
        r3 = H + 2*(pp - L)
        s3 = L - 2*(H - pp)
        out = pd.DataFrame({'PP': pp, 'R1': r1, 'R2': r2,
                           'R3': r3, 'S1': s1, 'S2': s2, 'S3': s3})

    elif method.lower() == "fibonacci":
        r1 = pp + 0.382*rng
        s1 = pp - 0.382*rng
        r2 = pp + 0.618*rng
        s2 = pp - 0.618*rng
        r3 = pp + 1.000*rng
        s3 = pp - 1.000*rng
        out = pd.DataFrame({'PP': pp, 'R1': r1, 'R2': r2,
                           'R3': r3, 'S1': s1, 'S2': s2, 'S3': s3})

    elif method.lower() == "camarilla":
        k = 1.1  # factor clásico Camarilla
        r1 = C + (k/12.0)*rng
        s1 = C - (k/12.0)*rng
        r2 = C + (k/6.0)*rng
        s2 = C - (k/6.0)*rng
        r3 = C + (k/4.0)*rng
        s3 = C - (k/4.0)*rng
        r4 = C + (k/2.0)*rng
        s4 = C - (k/2.0)*rng
        out = pd.DataFrame({'PP': pp, 'R1': r1, 'R2': r2, 'R3': r3,
                           'S1': s1, 'S2': s2, 'S3': s3, 'R4': r4, 'S4': s4})
    else:
        raise ValueError(
            "method debe ser 'classic', 'fibonacci' o 'camarilla'")

    # Evita valores inválidos cuando no hay rango/valores previos
    out = out.replace([np.inf, -np.inf], np.nan)
    out.index = df.index
    return out


def fibonacci_retracements(df: pd.DataFrame, length: int = 100, levels: list[float] | None = None) -> pd.DataFrame:
    """
    Retrocesos de Fibonacci sobre el último tramo [length] velas.
    Devuelve un DataFrame con columnas constantes alineadas a df.index.
    """
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)

    levels = levels or [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    _df = df.tail(max(2, length))
    hi = _df['High'].max()
    lo = _df['Low'].min()
    if pd.isna(hi) or pd.isna(lo) or hi == lo:
        return pd.DataFrame(index=df.index)

    rng = hi - lo
    out = {}
    for lv in levels:
        val = hi - rng * lv  # 0% en el máximo, 100% en el mínimo
        out[f'Fib {int(lv*100)}%'] = pd.Series(val, index=df.index)

    return pd.DataFrame(out, index=df.index)


def fibonacci_extensions(df: pd.DataFrame, length: int = 100, ratios: list[float] | None = None) -> pd.DataFrame:
    """
    Extensiones de Fibonacci (por arriba y por abajo) sobre el último tramo.
    ratios > 1.0 (ej.: 1.272, 1.618, 2.618)
    Devuelve un DataFrame con columnas constantes alineadas a df.index.
    """
    if df is None or df.empty:
        return pd.DataFrame(index=df.index if df is not None else None)

    ratios = ratios or [1.272, 1.618, 2.618]
    _df = df.tail(max(2, length))
    hi = _df['High'].max()
    lo = _df['Low'].min()
    if pd.isna(hi) or pd.isna(lo) or hi == lo:
        return pd.DataFrame(index=df.index)

    rng = hi - lo
    out = {}
    for r in ratios:
        # Extensiones alcistas (por encima del máximo)
        up = lo + rng * r
        # Extensiones bajistas (por debajo del mínimo)
        dn = hi - rng * r
        out[f'Ext ↑ {r:.3f}'] = pd.Series(up, index=df.index)
        out[f'Ext ↓ {r:.3f}'] = pd.Series(dn, index=df.index)

    return pd.DataFrame(out, index=df.index)


def moving_average_envelopes(df: pd.DataFrame,
                             length: int = 20,
                             pct: float = 0.05,
                             ma: str = "sma",
                             price_col: str = "Close") -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Medias móviles envolventes (Envelopes) alrededor de una MA.
      - Medio = MA(price_col, length)
      - Superior = Medio * (1 + pct)
      - Inferior = Medio * (1 - pct)

    Parámetros:
      length: ventana de la media.
      pct   : ancho como fracción (0.05 = 5%).
      ma    : "sma" o "ema".
      price_col: columna de precio a usar (por defecto 'Close').

    Devuelve: (mid, up, lo) como Series alineadas con df.index.
    """
    if df is None or df.empty or price_col not in df.columns:
        mid = pd.Series(dtype='float64',
                        name=f"Env mid({ma.upper()},{length})")
        up = pd.Series(dtype='float64', name=f"Env sup({int(pct*100)}%)")
        lo = pd.Series(dtype='float64', name=f"Env inf({int(pct*100)}%)")
        return mid, up, lo

    price = df[price_col].astype(float)

    if ma.lower() == "ema":
        mid = price.ewm(span=length, adjust=False).mean().rename(
            f"Env mid(EMA,{length})")
    else:
        mid = price.rolling(window=length, min_periods=1).mean().rename(
            f"Env mid(SMA,{length})")

    up = (mid * (1.0 + float(pct))).rename(f"Env sup({int(pct*100)}%)")
    lo = (mid * (1.0 - float(pct))).rename(f"Env inf({int(pct*100)}%)")
    return mid, up, lo


def ichimoku(df: pd.DataFrame,
             tenkan: int = 9,
             kijun: int = 26,
             senkou_b: int = 52,
             displacement: int = 26) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Ichimoku Kinko Hyo (parámetros clásicos 9/26/52, desplazamiento 26):
      - Tenkan-sen (línea de conversión) = (HH(tenkan) + LL(tenkan)) / 2
      - Kijun-sen  (línea base)          = (HH(kijun)  + LL(kijun))  / 2
      - Senkou Span A (leading A)        = (Tenkan + Kijun) / 2   --> desplazada +displacement
      - Senkou Span B (leading B)        = (HH(senkou_b) + LL(senkou_b)) / 2  --> desplazada +displacement
      - Chikou Span (lagging)            = Close --> desplazada -displacement
    Devuelve: (tenkan, kijun, senkou_a, senkou_b_line, chikou)
    """
    if df is None or df.empty:
        idx = df.index if df is not None else None
        nan = pd.Series(dtype='float64', index=idx)
        return nan, nan, nan, nan, nan

    high = df['High']
    low = df['Low']
    close = df['Close']

    hh_t = high.rolling(window=tenkan, min_periods=1).max()
    ll_t = low .rolling(window=tenkan, min_periods=1).min()
    tenkan_sen = ((hh_t + ll_t) / 2.0).rename(f"Tenkan({tenkan})")

    hh_k = high.rolling(window=kijun, min_periods=1).max()
    ll_k = low .rolling(window=kijun, min_periods=1).min()
    kijun_sen = ((hh_k + ll_k) / 2.0).rename(f"Kijun({kijun})")

    senkou_a = ((tenkan_sen + kijun_sen) /
                2.0).shift(displacement).rename("Senkou A")

    hh_b = high.rolling(window=senkou_b, min_periods=1).max()
    ll_b = low .rolling(window=senkou_b, min_periods=1).min()
    senkou_b_line = (
        (hh_b + ll_b) / 2.0).shift(displacement).rename(f"Senkou B({senkou_b})")

    chikou = close.shift(-displacement).rename("Chikou")

    return tenkan_sen, kijun_sen, senkou_a, senkou_b_line, chikou
