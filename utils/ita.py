
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

def avwap(dataframe:DataFrame,period:int, **kwargs):
    df = dataframe.copy()
    if 'tp' not in df.columns:
        df['tp'] = ta.TYPPRICE(df)
    df['tpv'] = df.tp*df.volume
    df['hhvbar'] = hhv(df['high'], period)
    df['llvbar'] = llv(df['low'], period)
    # integer indices
    df['ema20'] =  df.apply(lambda row: df.iloc[row['hhvbar']: row.name+1, df.columns.get_loc('tpv')].sum()/df.iloc[row['hhvbar']:row.name+1,df.columns.get_loc('volume')].sum()
        if pd.notna(row['hhvbar']) else np.nan, axis=1)
    df['ema89'] = df.apply(lambda row: df.iloc[row['llvbar']: row.name+1, df.columns.get_loc('tpv')].sum()/ df.iloc[row['llvbar']:row.name+1, df.columns.get_loc('volume')].sum()
                if pd.notna(row['llvbar']) else np.nan, axis=1)
    return df[['ema20', 'ema89']]

def ema3(dataframe: DataFrame,timeperiod:int, **kwargs):
    df = dataframe.copy()
    period = 50 if timeperiod == 50 else 89
    df[['ema144', 'ema233']] = avwap(df, period)
    df['cross'] = np.where(
        (df['ema144']> 0.00) & (df['ema233']> 0.00), 
        np.where(
            (df['close'] > df['ema144']),
            'up',
            np.where(
                (df['close']< df['ema233']),
                'down',
                np.NaN
            )
        ),
        np.NaN) 
    return df[['ema144', 'ema233', 'cross']]
    
def umacd(dataframe:DataFrame):
    df = dataframe.copy()
    df['umacd'] = np.where(
        (df['ema144'] < df['upper_band']) & (df['ema233'] > df['lower_band']),
        True,False 
    )
    return df['umacd']

def bband(dataframe: DataFrame, multiplier= 1.5, timeperiod= 89):
    df = dataframe.copy()
    median = (df['close'].rolling(window= timeperiod).median()).values
    std_dev = (df['close'].rolling(window= timeperiod).std()).values
    upper_band = median + multiplier*std_dev
    lower_band = median - multiplier*std_dev
    return pd.DataFrame({
        'upper_band': upper_band,
        'lower_band': lower_band
    })

def cross(dataframe: DataFrame):
    df = dataframe.copy()
    df['cross_above'] = np.where(
        # qtpylib.crossed_above(df['ema89'], df['lower_band']), 1, 0
        qtpylib.crossed_above(df['ema233'], df['lower_band']), 1, 0

    )
    df['cross_below'] = np.where(
        # qtpylib.crossed_below(df['ema144'], df['upper_band']), 1, 0
        qtpylib.crossed_below(df['ema144'], df['upper_band']), 1, 0
    )
    return df[['cross_above', 'cross_below']]

def check_cross(s:Series, check_value, timeperiod:int):
    value_exists = s.rolling(window=timeperiod).apply(lambda x: check_value in x, raw= True).fillna(0)
    return value_exists.astype(bool)

def hhv(price: pd.Series, period:int)-> Series:
    return price.rolling(window=period).apply(lambda x: x.idxmax()).convert_dtypes(convert_integer=True)

def llv(price: pd.Series, period:int) -> Series:
    return price.rolling(window=period).apply(lambda x: x.idxmin()).convert_dtypes(convert_integer=True)

