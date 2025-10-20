import mplfinance as mpf
import pandas as pd
from functions_trading import (
    sma, ema, atr, rsi, macd, bollinger_bands,
    momentum_smooth, adx, parabolic_sar, golden_death_crosses, levels_from_entries,
    stochastic_kd, williams_r, cci, dpo, obv, ad_line,
    cmf, volume_oscillator, keltner_channels, donchian_channels, chandelier_exit, pivot_points,
    fibonacci_extensions, fibonacci_retracements, moving_average_envelopes, ichimoku,
)
# ===========================
#     FUNCIONES AUXILIARES


def make_addplots(df: pd.DataFrame,
                  sma_c: pd.Series,
                  sma_l: pd.Series,
                  golden_idx: pd.DatetimeIndex,
                  death_idx: pd.DatetimeIndex,
                  levels: pd.DataFrame | None = None):
    """Construye la lista de addplots para mplfinance."""
    ap = [
        mpf.make_addplot(sma_c, panel=0, width=1.0, color='orange',
                         label=sma_c.name if sma_c.name else 'SMA corta'),
        mpf.make_addplot(sma_l, panel=0, width=1.0, color='purple',
                         label=sma_l.name if sma_l.name else 'SMA larga'),
    ]

    # Marcadores de cruces
    price_at_golden = pd.Series(index=df.index, dtype='float64')
    price_at_death = pd.Series(index=df.index, dtype='float64')
    price_at_golden.loc[golden_idx] = df['Close'].loc[golden_idx]
    price_at_death.loc[death_idx] = df['Close'].loc[death_idx]

    ap += [
        mpf.make_addplot(price_at_golden, panel=0, type='scatter',
                         markersize=80, marker='^', color='g'),
        mpf.make_addplot(price_at_death,  panel=0, type='scatter',
                         markersize=80, marker='v', color='r')
    ]

    # Niveles (opcional): dibujamos scatter en la vela de entrada para visualizar stop/limit
    if levels is not None:
        stop_scatter = levels['stop_price']
        limit_scatter = levels['limit_price']
        ap += [
            mpf.make_addplot(stop_scatter,  panel=0, type='scatter',
                             markersize=30, marker='x', color='red'),
            mpf.make_addplot(limit_scatter, panel=0, type='scatter',
                             markersize=30, marker='x', color='green'),
        ]

    return ap

# ===========================
#     PLOTS PRINCIPALES


def plot_asset(df: pd.DataFrame,
               ticker: str,
               sma_short_len: int = 50,
               sma_long_len: int = 200,
               style: str = 'yahoo',
               show_levels: bool = True):
    """Representa velas + volumen + SMAs + cruces y, opcionalmente, stop/limit por ATR."""
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    sma_c = sma(df['Close'], sma_short_len).rename(f'SMA{sma_short_len}')
    sma_l = sma(df['Close'], sma_long_len).rename(f'SMA{sma_long_len}')
    golden_idx, death_idx = golden_death_crosses(sma_c, sma_l)

    levels = None
    if show_levels:
        entries = pd.Series(False, index=df.index)
        entries.loc[golden_idx] = True
        levels = levels_from_entries(df, entries)

    ap = make_addplots(df, sma_c, sma_l, golden_idx, death_idx, levels)

    mpf.plot(
        df,
        type='candle',
        style=style,
        title=f"{ticker} | Velas + SMA{sma_short_len}/{sma_long_len} | Golden/Death + niveles ATR",
        volume=True,
        addplot=ap,
        figratio=(16, 9),
        figscale=1.2,
        tight_layout=True,
        block=True,
    )


def plot_asset_momentum_smooth(df, ticker, period=10, smooth=10, style='yahoo'):
    """Ventana 7: Oscilador de Momento (M = V - Vx) + SMA del Momentum"""
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    _df = df.copy()
    _df = momentum_smooth(_df, period=period, smooth=smooth)

    ap = [
        mpf.make_addplot(_df['Momentum'], panel=1, width=1.0,
                         color='orange', label=f'Momentum({period})'),
        mpf.make_addplot(_df['Momentum_SMA'], panel=1, width=1.0,
                         color='purple', label=f'SMA Momentum({smooth})'),
    ]

    mpf.plot(
        _df,
        type='candle',
        style=style,
        title=f"{ticker} | Momentum (V - V{period}) + SMA({smooth})",
        addplot=ap,
        volume=True,
        figratio=(16, 9),
        figscale=1.2,
        tight_layout=True,
        block=True
    )


def plot_asset_rsi(df: pd.DataFrame, ticker: str, length: int = 14, style: str = 'yahoo'):
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    r = rsi(df['Close'], length).rename(f'RSI({length})')

    ap = [
        mpf.make_addplot(r, panel=1, color='purple', width=1.0, label=r.name),
        mpf.make_addplot(pd.Series(70, index=df.index),
                         panel=1, color='gray', width=0.8),
        mpf.make_addplot(pd.Series(30, index=df.index),
                         panel=1, color='gray', width=0.8),
    ]
    mpf.plot(
        df, type='candle', style=style,
        title=f"{ticker} | RSI({length})",
        addplot=ap, volume=True,
        figratio=(16, 9), figscale=1.2,
        tight_layout=True, block=True
    )

# MACD


def plot_asset_macd(df: pd.DataFrame, ticker: str,
                    short: int = 12, long: int = 26, signal: int = 9,
                    style: str = 'yahoo'):
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    macd_line, signal_line, hist = macd(df['Close'], short, long, signal)
    macd_line = macd_line.rename(f"MACD({short},{long},{signal})")
    signal_line = signal_line.rename("Señal MACD")
    # barras positivas y negativas separadas para color
    hist_pos = hist.where(hist >= 0)
    hist_neg = hist.where(hist < 0)

    ap = [
        mpf.make_addplot(hist_pos, panel=1, type='bar',
                         color='green', alpha=0.5, label='Hist +'),
        mpf.make_addplot(hist_neg, panel=1, type='bar',
                         color='red',   alpha=0.5, label='Hist -'),
        mpf.make_addplot(macd_line,   panel=1, color='blue',
                         width=1.0, label=macd_line.name),
        mpf.make_addplot(signal_line, panel=1, color='orange',
                         width=1.0, label=signal_line.name),
    ]
    mpf.plot(
        df, type='candle', style=style,
        title=f"{ticker} | MACD {short},{long},{signal}",
        addplot=ap, volume=True,
        figratio=(16, 9), figscale=1.2,
        tight_layout=True, block=True
    )

# Bandas de Bollinger


def plot_asset_bbands(df: pd.DataFrame, ticker: str,
                      length: int = 20, n_std: int = 2, style: str = 'yahoo'):
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    mid, upper, lower = bollinger_bands(df['Close'], length, n_std)
    mid = mid.rename(f'BB media({length})')
    upper = upper.rename(f'BB sup({length},{n_std}σ)')
    lower = lower.rename(f'BB inf({length},{n_std}σ)')

    ap = [
        mpf.make_addplot(mid,   panel=0, color='blue',
                         width=1.0, label=mid.name),
        mpf.make_addplot(upper, panel=0, color='gray',
                         width=1.0, label=upper.name),
        mpf.make_addplot(lower, panel=0, color='gray',
                         width=1.0, label=lower.name),
    ]
    mpf.plot(
        df, type='candle', style=style,
        title=f"{ticker} | Bandas de Bollinger ({length}, {n_std}σ)",
        addplot=ap, volume=True,
        figratio=(16, 9), figscale=1.2,
        tight_layout=True, block=True
    )

# ATR


def plot_asset_atr(df: pd.DataFrame, ticker: str, length: int = 14, style: str = 'yahoo'):
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    a = atr(df, length).rename(f'ATR({length})')
    ap = [mpf.make_addplot(a, panel=1, color='brown', width=1.0, label=a.name)]
    mpf.plot(
        df, type='candle', style=style,
        title=f"{ticker} | ATR({length})",
        addplot=ap, volume=True,
        figratio=(16, 9), figscale=1.2,
        tight_layout=True, block=True
    )

# ADX


def plot_asset_adx(df: pd.DataFrame, ticker: str, length: int = 14, style: str = 'yahoo'):
    """
    Dibuja velas + panel inferior con ADX y +DI/-DI.
    Convención de colores:
      - ADX: azul
      - +DI: verde
      - -DI: rojo
      - Niveles guía: 20 y 25 en gris
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    adx_series, plus_di, minus_di = adx(df, length)

    ap = [
        mpf.make_addplot(adx_series, panel=1, color='blue',
                         width=1.0, label=adx_series.name),
        mpf.make_addplot(plus_di,    panel=1, color='green',
                         width=1.0, label=plus_di.name),
        mpf.make_addplot(minus_di,   panel=1, color='red',
                         width=1.0, label=minus_di.name),
        mpf.make_addplot(pd.Series(20, index=df.index),
                         panel=1, color='gray', width=0.8),
        mpf.make_addplot(pd.Series(25, index=df.index),
                         panel=1, color='gray', width=0.8),
    ]

    mpf.plot(
        df,
        type='candle',
        style=style,
        title=f"{ticker} | ADX({length}) + DI+/DI-",
        addplot=ap,
        volume=True,
        figratio=(16, 9),
        figscale=1.2,
        tight_layout=True,
        block=True
    )

# Parabolic SAR


def plot_asset_psar(df: pd.DataFrame, ticker: str, step: float = 0.02, max_step: float = 0.2, style: str = 'yahoo'):
    """
    Plot: velas + Parabolic SAR.
    Convención de colores:
      - Puntos SAR alcistas (debajo del precio): verde
      - Puntos SAR bajistas (encima del precio): rojo
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    psar, psar_bull, psar_bear = parabolic_sar(
        df, step=step, max_step=max_step)

    ap = [
        mpf.make_addplot(psar_bull, panel=0, type='scatter', markersize=20,
                         marker='.', color='green', label=psar_bull.name),
        mpf.make_addplot(psar_bear, panel=0, type='scatter', markersize=20,
                         marker='.', color='red',   label=psar_bear.name),
    ]

    mpf.plot(
        df,
        type='candle',
        style=style,
        title=f"{ticker} | Parabolic SAR (step={step}, max={max_step})",
        addplot=ap,
        volume=True,
        figratio=(16, 9),
        figscale=1.2,
        tight_layout=True,
        block=True
    )


def plot_asset_stochastic(df: pd.DataFrame, ticker: str,
                          length: int = 14, k_smooth: int = 3, d_smooth: int = 3,
                          style: str = "yahoo"):
    """
    Velas + panel Estocástico %K/%D con niveles 20/80.
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    k, d = stochastic_kd(
        df, length=length, k_smooth=k_smooth, d_smooth=d_smooth)

    ap = [
        mpf.make_addplot(k, panel=1, color='blue',  width=1.0, label=k.name),
        mpf.make_addplot(d, panel=1, color='orange', width=1.0, label=d.name),
        mpf.make_addplot(pd.Series(80, index=df.index),
                         panel=1, color='gray', width=0.8),
        mpf.make_addplot(pd.Series(20, index=df.index),
                         panel=1, color='gray', width=0.8),
    ]

    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=f"{ticker} | Estocástico %K/%D ({length},{k_smooth},{d_smooth})",
        addplot=ap, figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )


def plot_asset_williams_r(df: pd.DataFrame, ticker: str,
                          length: int = 14, style: str = 'yahoo'):
    """
    Velas + panel Williams %R con niveles -20/-80 (sobrecompra/sobreventa).
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    wr = williams_r(df, length=length)

    ap = [
        mpf.make_addplot(wr, panel=1, color='blue', width=1.0, label=wr.name),
        mpf.make_addplot(pd.Series(-20, index=df.index),
                         panel=1, color='gray', width=0.8),
        mpf.make_addplot(pd.Series(-80, index=df.index),
                         panel=1, color='gray', width=0.8),
        mpf.make_addplot(pd.Series(-50, index=df.index), panel=1,
                         color='lightgray', width=0.6),  # guía central opcional
    ]

    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=f"{ticker} | Williams %R ({length})",
        addplot=ap, figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )


def plot_asset_cci(df: pd.DataFrame, ticker: str, length: int = 20, style: str = "yahoo"):
    """
    Velas + panel CCI con niveles guía ±100.
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    cci_series = cci(df, length=length)

    ap = [
        mpf.make_addplot(cci_series, panel=1, color='blue',
                         width=1.0, label=cci_series.name),
        mpf.make_addplot(pd.Series(100, index=df.index),
                         panel=1, color='gray',  width=0.8),
        mpf.make_addplot(pd.Series(-100, index=df.index),
                         panel=1, color='gray',  width=0.8),
        mpf.make_addplot(pd.Series(0, index=df.index),
                         panel=1, color='lightgray', width=0.6),
    ]

    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=f"{ticker} | CCI ({length})",
        addplot=ap, figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )


def plot_asset_dpo(df: pd.DataFrame, ticker: str, length: int = 20, style: str = "yahoo"):
    """
    Velas + panel con DPO y línea 0.
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    dpo_series = dpo(df, length=length)

    ap = [
        mpf.make_addplot(dpo_series, panel=1, color='blue',
                         width=1.0, label=dpo_series.name),
        mpf.make_addplot(pd.Series(0, index=df.index),
                         panel=1, color='gray', width=0.8),
    ]

    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=f"{ticker} | DPO ({length})",
        addplot=ap, figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )


def plot_asset_obv(df: pd.DataFrame, ticker: str, ma: int | None = None, style: str = "yahoo"):
    """
    Velas + panel OBV. Opcional: media móvil del OBV (ma).
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    obv_series = obv(df)

    ap = [mpf.make_addplot(obv_series, panel=1,
                           color='blue', width=1.0, label="OBV")]
    if ma and ma > 1:
        obv_ma = obv_series.rolling(
            window=ma, min_periods=1).mean().rename(f"OBV_SMA({ma})")
        ap.append(mpf.make_addplot(obv_ma, panel=1,
                  color='orange', width=1.0, label=obv_ma.name))

    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=f"{ticker} | OBV" + (f" + SMA({ma})" if ma else ""),
        addplot=ap, figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )


def plot_asset_ad_line(df: pd.DataFrame, ticker: str, ma: int | None = None, style: str = "yahoo"):
    """
    Velas + panel con A/D Line. Opcional: media móvil de A/D (ma).
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    ad = ad_line(df)

    ap = [mpf.make_addplot(ad, panel=1, color='blue', width=1.0, label="A/D")]
    if ma and ma > 1:
        ad_ma = ad.rolling(window=ma, min_periods=1).mean().rename(
            f"A/D_SMA({ma})")
        ap.append(mpf.make_addplot(ad_ma, panel=1,
                  color='orange', width=1.0, label=ad_ma.name))

    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=f"{ticker} | A/D Line" + (f" + SMA({ma})" if ma else ""),
        addplot=ap, figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )


def plot_asset_cmf(df: pd.DataFrame, ticker: str, length: int = 20, style: str = "yahoo"):
    """
    Velas + panel Chaikin Money Flow (CMF) con líneas guía 0 y ±0.1.
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    cmf_series = cmf(df, length=length)

    ap = [
        mpf.make_addplot(cmf_series, panel=1, color='blue',
                         width=1.0, label=cmf_series.name),
        mpf.make_addplot(pd.Series(0, index=df.index),
                         panel=1, color='gray',      width=0.8),
        mpf.make_addplot(pd.Series(0.1, index=df.index),
                         panel=1, color='lightgray', width=0.8),
        mpf.make_addplot(pd.Series(-0.1, index=df.index),
                         panel=1, color='lightgray', width=0.8),
    ]

    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=f"{ticker} | Chaikin Money Flow (CMF {length})",
        addplot=ap, figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )


def plot_asset_vo(df: pd.DataFrame, ticker: str,
                  short: int = 14, long: int = 28,
                  ma: str = "ema", pct: bool = True,
                  style: str = "yahoo"):
    """
    Velas + panel Volumen Oscillator (VO). Dibuja línea 0 y barras positivas/negativas.
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    vo = volume_oscillator(df, short=short, long=long, ma=ma, pct=pct)

    # Para resaltar signo, separamos en positivo/negativo si es % o absoluto
    vo_pos = vo.where(vo >= 0)
    vo_neg = vo.where(vo < 0)

    ap = [
        mpf.make_addplot(vo_pos, panel=1, type='bar',
                         color='green', alpha=0.5, label=vo.name + " +"),
        mpf.make_addplot(vo_neg, panel=1, type='bar',
                         color='red',   alpha=0.5, label=vo.name + " -"),
        mpf.make_addplot(vo,     panel=1, color='blue',
                         width=1.0, label=vo.name),
        mpf.make_addplot(pd.Series(0, index=df.index),
                         panel=1, color='gray', width=0.8),
    ]

    title = f"{ticker} | Volumen Oscillator ({ma.upper()} {min(short, long)},{max(short, long)}" + (
        ", %" if pct else "") + ")"
    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=title, addplot=ap, figratio=(16, 9), figscale=1.2,
        tight_layout=True, block=True
    )


def plot_asset_keltner(df: pd.DataFrame, ticker: str,
                       length_ma: int = 20, length_atr: int = 10,
                       multiplier: float = 2.0, ma: str = "ema",
                       style: str = "yahoo"):
    """
    Velas + Canales de Keltner (EMA±ATR*mult) en el panel principal.
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    mid, up, lo = keltner_channels(df,
                                   length_ma=length_ma,
                                   length_atr=length_atr,
                                   multiplier=multiplier,
                                   ma=ma)

    ap = [
        mpf.make_addplot(mid, panel=0, color='blue',
                         width=1.0, label=mid.name),
        mpf.make_addplot(up,  panel=0, color='gray',
                         width=1.0, label=up.name),
        mpf.make_addplot(lo,  panel=0, color='gray',
                         width=1.0, label=lo.name),
    ]

    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=f"{ticker} | Keltner Channels ({ma.upper()} {length_ma}, ATR {length_atr}, x{multiplier})",
        addplot=ap, figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )


def plot_asset_donchian(df: pd.DataFrame, ticker: str, length: int = 20, style: str = "yahoo"):
    """
    Velas + Donchian Channels en el panel principal.
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    mid, up, lo = donchian_channels(df, length=length)

    ap = [
        mpf.make_addplot(up,  panel=0, color='gray', width=1.0, label=up.name),
        mpf.make_addplot(lo,  panel=0, color='gray', width=1.0, label=lo.name),
        mpf.make_addplot(mid, panel=0, color='blue',
                         width=1.0, label=mid.name),
    ]

    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=f"{ticker} | Donchian Channels ({length})",
        addplot=ap, figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )


def plot_asset_chandelier(df: pd.DataFrame, ticker: str,
                          length_atr: int = 22, multiplier: float = 3.0,
                          length_hl: int | None = None, style: str = "yahoo"):
    """
    Velas + Chandelier Exit (bandas long/short) en el panel principal.
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    ce_long, ce_short = chandelier_exit(df,
                                        length_atr=length_atr,
                                        multiplier=multiplier,
                                        length_hl=length_hl)

    ap = [
        mpf.make_addplot(ce_long,  panel=0, color='green',
                         width=1.0, label=ce_long.name),
        mpf.make_addplot(ce_short, panel=0, color='red',
                         width=1.0, label=ce_short.name),
    ]

    title = f"{ticker} | Chandelier Exit (ATR {length_atr}, x{multiplier}, HL {length_hl or length_atr})"
    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=title, addplot=ap, figratio=(16, 9), figscale=1.2,
        tight_layout=True, block=True
    )


def _addplot_pivot_lines(df, pivots: pd.DataFrame, panel: int = 0):
    """Construye líneas horizontales por día para PP, R/S…"""
    ap = []
    # Colores: PP azul; R en rojo tenue; S en verde tenue
    if 'PP' in pivots:
        ap.append(mpf.make_addplot(
            pivots['PP'], panel=panel, color='blue', width=0.8, label='PP'))
    for col in pivots.columns:
        if col.startswith('R'):
            ap.append(mpf.make_addplot(
                pivots[col], panel=panel, color='red', width=0.7, alpha=0.7, label=col))
        if col.startswith('S'):
            ap.append(mpf.make_addplot(
                pivots[col], panel=panel, color='green', width=0.7, alpha=0.7, label=col))
    return ap


def plot_asset_pivots(df: pd.DataFrame, ticker: str,
                      method: str = "classic",
                      style: str = "yahoo"):
    """
    Velas + Puntos Pivote (Clásicos/Fibonacci/Camarilla) desplazados 1 día.
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    piv = pivot_points(df, method=method)

    ap = _addplot_pivot_lines(df, piv, panel=0)

    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=f"{ticker} | Pivot Points ({method.title()})",
        addplot=ap, figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )

# Azúcar sintáctico: tres envoltorios si te gusta llamarlos directos


def plot_asset_pivots_classic(df, ticker, style="yahoo"):
    return plot_asset_pivots(df, ticker, method="classic", style=style)


def plot_asset_pivots_fibonacci(df, ticker, style="yahoo"):
    return plot_asset_pivots(df, ticker, method="fibonacci", style=style)


def plot_asset_pivots_camarilla(df, ticker, style="yahoo"):
    return plot_asset_pivots(df, ticker, method="camarilla", style=style)


def plot_asset_fibo_retracements(df: pd.DataFrame, ticker: str,
                                 length: int = 100,
                                 levels: list[float] | None = None,
                                 style: str = "yahoo"):
    """
    Velas + Retrocesos de Fibonacci (líneas horizontales).
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    fib = fibonacci_retracements(df, length=length, levels=levels)
    if fib.empty:
        print(f"[WARN] Fibo retracements vacíos para {ticker}")
        return

    ap = []
    for col in fib.columns:
        ap.append(mpf.make_addplot(
            fib[col], panel=0, color='green', width=0.9, alpha=0.8, label=col))

    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=f"{ticker} | Fibonacci Retracements (lookback={length})",
        addplot=ap, figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )


def plot_asset_fibo_extensions(df: pd.DataFrame, ticker: str,
                               length: int = 100,
                               ratios: list[float] | None = None,
                               style: str = "yahoo"):
    """
    Velas + Extensiones de Fibonacci (arriba/abajo).
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    fibx = fibonacci_extensions(df, length=length, ratios=ratios)
    if fibx.empty:
        print(f"[WARN] Fibo extensions vacías para {ticker}")
        return

    ap = []
    for col in fibx.columns:
        ap.append(mpf.make_addplot(
            fibx[col], panel=0, color='orange', width=0.9, alpha=0.8, label=col))

    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=f"{ticker} | Fibonacci Extensions (lookback={length})",
        addplot=ap, figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )


def plot_asset_envelopes(df: pd.DataFrame, ticker: str,
                         length: int = 20, pct: float = 0.05,
                         ma: str = "sma", price_col: str = "Close",
                         style: str = "yahoo"):
    """
    Velas + Moving Average Envelopes (bandas a ±pct sobre la MA).
    Convención:
      - Media central: azul
      - Banda superior/inferior: gris
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    mid, up, lo = moving_average_envelopes(
        df, length=length, pct=pct, ma=ma, price_col=price_col)

    ap = [
        mpf.make_addplot(mid, panel=0, color='blue',
                         width=1.0, label=mid.name),
        mpf.make_addplot(up,  panel=0, color='gray', width=1.0, label=up.name),
        mpf.make_addplot(lo,  panel=0, color='gray', width=1.0, label=lo.name),
    ]

    pct_txt = f"{pct*100:.1f}%".rstrip('0').rstrip('.')  # 0.05 -> "5%"
    title = f"{ticker} | Envelopes ({ma.upper()} {length}, ±{pct_txt})"
    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=title, addplot=ap,
        figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )


def plot_asset_ichimoku(df: pd.DataFrame, ticker: str,
                        tenkan: int = 9, kijun: int = 26,
                        senkou_b: int = 52, displacement: int = 26,
                        style: str = "yahoo"):
    """
    Velas + Ichimoku (Tenkan, Kijun, Senkou A/B desplazadas + Chikou retrasada).
    Colores:
      - Tenkan: naranja
      - Kijun : morado
      - Senkou A: verde
      - Senkou B: rojo
      - Chikou: azul (retrasada)
    Nota: mpltfinance no rellena nativamente la 'nube'; trazamos líneas.
    """
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    tenkan_sen, kijun_sen, senkou_a, senkou_b_line, chikou = ichimoku(
        df, tenkan=tenkan, kijun=kijun, senkou_b=senkou_b, displacement=displacement
    )

    ap = [
        mpf.make_addplot(tenkan_sen,    panel=0, color='orange',
                         width=1.0, label=tenkan_sen.name),
        mpf.make_addplot(kijun_sen,     panel=0, color='purple',
                         width=1.0, label=kijun_sen.name),
        mpf.make_addplot(senkou_a,      panel=0, color='green',
                         width=1.0, label=senkou_a.name),
        mpf.make_addplot(senkou_b_line, panel=0, color='red',
                         width=1.0, label=senkou_b_line.name),
        mpf.make_addplot(chikou,        panel=0, color='blue',
                         width=1.0, label=chikou.name),
    ]

    title = f"{ticker} | Ichimoku (Tenkan {tenkan}, Kijun {kijun}, SenkouB {senkou_b}, disp {displacement})"
    mpf.plot(
        df, type='candle', style=style, volume=True,
        title=title, addplot=ap,
        figratio=(16, 9), figscale=1.2, tight_layout=True, block=True
    )
