
import pandas as pd
import numpy as np
def get_rsi_divergence(df: pd.DataFrame, width=5) -> pd.Series:
    df['divergence'] = np.nan
    lower_barrier = 30
    upper_barrier = 70
    width = 5
    # Bullish Divergence
    for i in range(len(df) - width):  # 确保有足够的数据进行比较
        if df.iloc[i]['rsi'] < lower_barrier:

            for a in range(i + 1, min(i + width + 1, len(df))):
                if df.iloc[a]['rsi'] > lower_barrier:
                    
                    for r in range(a + 1, min(a + width + 1, len(df))):
                        if df.iloc[r]['rsi'] < lower_barrier and df.iloc[r]['rsi'] > df.iloc[i]['rsi'] and df.iloc[r]['Close'] < df.iloc[i]['Close']:
                            df.loc[r, 'divergence'] = 1
                            break

    # Bearish Divergence
    for i in range(len(df) - width):  # 确保有足够的数据进行比较
        if df.iloc[i, 'rsi'] > upper_barrier:
            for a in range(i + 1, min(i + width + 1, len(df))):
                if df.iloc[a, 'rsi'] < upper_barrier:
                    for r in range(a + 1, min(a + width + 1, len(df))):
                        if df.iloc[r, 'rsi'] > upper_barrier and df.iloc[r, 'rsi'] < df.iloc[i, 'rsi'] and df.iloc[r, 'Close'] > df.iloc[i, 'Close']:
                            df.loc[r, 'divergence'] = 0
                            break

    # 填充未检测到背离的行
    df['divergence'].fillna(0, inplace=True)
    return df['divergence']