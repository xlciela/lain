import numpy as np
def custom_func(x):
    # 忽略NaN值
    print('hello')
    not_nan = ~np.isnan(x)
    if np.sum(not_nan) == 0:
        return np.nan  # 如果窗口中全是NaN，则返回NaN
    # 使用NumPy的索引方式获取最后一个元素
    last_value = x[-1]
    print(last_value)
    return np.sum(x[not_nan] < last_value) / np.sum(not_nan)