import numpy as np
from scipy.signal import argrelmax, argrelmin

def calc_high_idx(index, high_idx):
    high_idx_v = np.full(len(index), 0)

    start_idx = high_idx[0]
    for i, idx in enumerate(high_idx):
        if i == 0:
            continue
        high_idx_v[start_idx: idx] = high_idx[i-1]
        start_idx = idx

    high_idx_v[start_idx:] = high_idx[-1]
    return high_idx_v


def assign_idx(df: pd.DataFrame, idx_arr, col: str) -> [float]:
    idx_v = np.full(len(df), 0)
    start_idx = idx_arr[0]
    print(f'{col}_idx_arr: {idx_arr}')
    for i, idx in enumerate(idx_arr):
        if i == 0:
            continue
        print(f'赋值df.iloc[idx_arr[i-1], df.columns.get_loc(col)]: {df.iloc[idx_arr[i-1], df.columns.get_loc(col)]}')
        idx_v[start_idx: idx] = df.iloc[idx_arr[i-1], df.columns.get_loc(col)]
        start_idx = idx
    idx_v[start_idx:] = df.iloc[idx_arr[-1], df.columns.get_loc(col)]
    return idx_v
    
def get_range(dataframe: pd.DataFrame):
    df = dataframe.copy()
    if 'range_top' not in df.columns:
        df['range_top'] = np.nan
    if 'range_bottom' not in df.columns:
        df['range_bottom'] = np.nan
    high_idx = argrelmax(df['high'].values, order= 2)[0]
    low_idx = argrelmin(df['low'].values, order= 2)[0]
    # assign idx_v to dataframe
    df['range_top'] = assign_idx(df, high_idx, 'high')
    df['range_bottom'] = assign_idx(df, low_idx, 'low')
    return df[['range_top', 'range_bottom']]