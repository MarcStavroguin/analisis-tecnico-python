from functions_trading import (get_data, ema,
                               golden_death_crosses,
                               levels_from_entries,
                               )
from functions_plot import (
    make_addplots, plot_asset, plot_asset_macd, plot_asset_rsi,
    plot_asset_bbands, plot_asset_atr, plot_asset_momentum_smooth,
    plot_asset_adx, plot_asset_psar, plot_asset_stochastic,
    plot_asset_williams_r, plot_asset_cci, plot_asset_dpo, plot_asset_obv, plot_asset_ad_line, plot_asset_cmf, plot_asset_vo, plot_asset_keltner, plot_asset_donchian,
    plot_asset_chandelier, plot_asset_pivots_classic, plot_asset_pivots_fibonacci, plot_asset_pivots_camarilla, plot_asset_fibo_retracements, plot_asset_fibo_extensions,
    plot_asset_envelopes, plot_asset_ichimoku,
)

import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt

# ===========================
#     PARÁMETROS
# ===========================

PERIODO = "1y"
SMA_CORTA = 50
SMA_LARGA = 200
STYLE = "yahoo"
SHOW_LEVELS = True  # si quieres ocultar niveles ATR, pon False

# ===========================
#     PLOTS ADICIONALES
# ===========================


def plot_asset_ema(df, ticker, ema_short_len=12, ema_long_len=26, style=STYLE, show_levels=SHOW_LEVELS):
    """Ventana 2: Velas + EMAs + cruces + (opcional) niveles ATR."""
    if df is None or df.empty:
        print(f"[WARN] Sin datos para: {ticker}")
        return

    ema_c = ema(df['Close'], ema_short_len).rename(f'EMA{ema_short_len}')
    ema_l = ema(df['Close'], ema_long_len).rename(f'EMA{ema_long_len}')
    golden_idx, death_idx = golden_death_crosses(ema_c, ema_l)

    levels = None
    if show_levels:
        entries = pd.Series(False, index=df.index)
        entries.loc[golden_idx] = True
        levels = levels_from_entries(df, entries)
        # Alinear explícitamente con el índice de df
        levels = levels.reindex(df.index)

    try:
        ap = make_addplots(df, ema_c, ema_l, golden_idx, death_idx, levels)
        mpf.plot(
            df,
            type='candle',
            style=style,
            title=f"{ticker} | Velas + EMA {ema_short_len}/{ema_long_len} | Golden/Death + niveles ATR",
            volume=True,
            addplot=ap,
            figratio=(16, 9),
            figscale=1.2,
            tight_layout=True,
            block=True  # <- así no se cierra enseguida la ventana
        )
    except Exception as e:
        print(f"[ERROR] al graficar {ticker} EMA: {e}")


# ===========================
#     EJECUCIÓN
# ===========================


def main():
    # Leemos tickers de scope.txt (uno por línea)
    with open('scope.txt', 'r', encoding='utf-8') as f:
        scope = [l.strip() for l in f.read().splitlines() if l.strip()]

    for activo in scope:
        try:
            df = get_data(activo, period=PERIODO)

            # 1) Velas + SMA 50/200 + cruces + niveles ATR
            plot_asset(df, activo, sma_short_len=SMA_CORTA,
                       sma_long_len=SMA_LARGA, style=STYLE, show_levels=SHOW_LEVELS)

            # 2) Velas + EMA 12/26 + cruces + niveles ATR
            plot_asset_ema(df, activo, ema_short_len=12,
                           ema_long_len=26, style=STYLE, show_levels=SHOW_LEVELS)
            # 3) RSI (14)
            plot_asset_rsi(df, activo, length=14, style=STYLE)

            # 3b) Estocástico %K/%D (14,3,3)
            plot_asset_stochastic(df, activo, length=14,
                                  k_smooth=3, d_smooth=3, style=STYLE)

            # 3c) Williams %R (14)
            plot_asset_williams_r(df, activo, length=14, style=STYLE)

            # 3d) CCI (20)
            plot_asset_cci(df, activo, length=20, style=STYLE)

            # 3e) DPO (20)
            plot_asset_dpo(df, activo, length=20, style=STYLE)

            # 3f) OBV (+SMA opcional)
            # usa ma=None si no quieres la media
            plot_asset_obv(df, activo, ma=20, style=STYLE)

            # 3g) A/D Line (+SMA opcional)
            # usa ma=None si no quieres suavizado
            plot_asset_ad_line(df, activo, ma=20, style=STYLE)

            # 3h) Chaikin Money Flow (CMF)
            plot_asset_cmf(df, activo, length=20, style=STYLE)

            # 3i) Volumen Oscillator (EMA 14/28, %)
            plot_asset_vo(df, activo, short=14, long=28,
                          ma="ema", pct=True, style=STYLE)

            # 4) MACD (12,26,9)
            plot_asset_macd(df, activo, short=12, long=26,
                            signal=9, style=STYLE)

            # 5) Bandas de Bollinger (20, 2σ)
            plot_asset_bbands(df, activo, length=20, n_std=2, style=STYLE)

            # 5a) Envelopes (SMA 20, ±5%)
            plot_asset_envelopes(df, activo, length=20,
                                 pct=0.05, ma="sma", style=STYLE)

            # 5b) Canales de Keltner (EMA 20, ATR 10, x2)
            plot_asset_keltner(
                df, activo, length_ma=20, length_atr=10, multiplier=2.0, ma="ema", style=STYLE)

            # 5c) Donchian Channels (20)
            plot_asset_donchian(df, activo, length=20, style=STYLE)

            # 5d) Chandelier Exit (ATR 22, x3, HL 22)
            plot_asset_chandelier(df, activo, length_atr=22,
                                  multiplier=3.0, length_hl=22, style=STYLE)

            # 5e) Pivotes Clásicos / Fibonacci / Camarilla
            plot_asset_pivots_classic(df, activo, style=STYLE)
            plot_asset_pivots_fibonacci(df, activo, style=STYLE)
            plot_asset_pivots_camarilla(df, activo, style=STYLE)

            # 6) ATR (14)
            plot_asset_atr(df, activo, length=14, style=STYLE)

            # 7) Momentum (10) + SMA(10)
            plot_asset_momentum_smooth(
                df, activo, period=10, smooth=10, style=STYLE)

            # 8) ADX (14) + DI+/DI-
            plot_asset_adx(df, activo, length=14, style=STYLE)

            # 9) Parabolic SAR (0.02, 0.2)
            plot_asset_psar(df, activo, step=0.02, max_step=0.2, style=STYLE)

            # 10) Fibonacci (lookback 100 velas)
            plot_asset_fibo_retracements(df, activo, length=100, style=STYLE)
            plot_asset_fibo_extensions(df, activo, length=100, style=STYLE)

            # 11) Ichimoku (9,26,52) con desplazamiento 26
            plot_asset_ichimoku(df, activo, tenkan=9, kijun=26,
                                senkou_b=52, displacement=26, style=STYLE)

        except Exception as e:
            print(f"[ERROR] {activo}: {e}")


if __name__ == '__main__':
    main()
