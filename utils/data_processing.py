import pandas as pd
import ta

def process_data(df):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    df['50_MA'] = df['Close'].rolling(window=50).mean()
    df['200_MA'] = df['Close'].rolling(window=200).mean()
    df['500_MA'] = df['Close'].rolling(window=500).mean()
    df['EMA'] = ta.trend.EMAIndicator(df['Close'], window=20).ema_indicator()
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['Parabolic_SAR'] = ta.trend.PSARIndicator(df['High'], df['Low'], df['Close']).psar()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    df['VWAP'] = ta.volume.VolumeWeightedAveragePrice(df['High'], df['Low'], df['Close'], df['Volume']).volume_weighted_average_price()
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20)
    df['BollingerUpper'] = bollinger.bollinger_hband()
    df['BollingerLower'] = bollinger.bollinger_lband()
    adx = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14)
    df['ADX'] = adx.adx()

    df['RSI_Trend'] = (df['RSI'] > 50).astype(int)
    df['MACD_Trend'] = ((df['MACD'] > df['MACD_Signal']) & (df['MACD'] > 0)).astype(int)
    df['PSAR_Trend'] = (df['Parabolic_SAR'] < df['Close']).astype(int)
    df['OBV_Trend'] = (df['OBV'].diff() > 0).astype(int)
    df['VWAP_Trend'] = (df['Close'] > df['VWAP']).astype(int)
    df['ADX_Trend'] = (df['ADX'] > 25).astype(int)
    df['Trend'] = (df['500_MA'] > df['200_MA']).astype(int)

    df.dropna(inplace=True)
    return df
