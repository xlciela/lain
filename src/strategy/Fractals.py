from enum import Enum
from typing import NamedTuple
import pandas as pd
import numpy as np
import matplotlib as mp


class FractalType(Enum):
    BULL = 1
    BEAR = 2


class Fractal(NamedTuple):
    time: str
    type: enumerate[str]
    value: int


class Fractals:
    def __init__(self, data: pd.DataFrame, period: int = 2, size: int = 100):
        self.period = period
        self.df = data.copy()  # pd.DataFrame
        self.size: int = size
        self.bears  = None
        self.bulls = None
        # self.fractals = None
        self.fractal_top_avg: float
        self.fractal_bottom_avg: float
        self.isInitialized = False

    # initialize fractals
    def initialize(self, data) -> None:
        # this function should be executed only once when the strategy is initialized
        if not self.isInitialized:
            self.loc_frac(data)
            loc_bears, loc_bulls = self.loc_frac(data)
            self.bears = self.df.loc[loc_bears]
            self.bulls = self.df.loc[loc_bulls]
        self.isInitialized = True

    # update and maintain a fixed-size fractals, the fractal_top_avg, fractal_bottom_avg
    def update(self, data) -> None: # consumer
        # 1. update self.bears, self.bulls
        # 2. update self.fractal_top_avg, self.fractal_bottom_avg
        # 3. if new fractal is formed return (type: bull | bear),  notify the observers
        pass
    

    # loc_frac -> pd.Series[bool]
    def loc_frac(self, data):

        # default [-2, -1, 1, 2]
        periods = [p for p in range(-self.period, self.period + 1) if p != 0]

        # the curretn may overlook the fractal: h0< h1< h2= h3> h4
        # modified?: highs = [data['high'] >= data['high'].shift(p) for p in periods] # highs: list[pd.Series[bool]]
        highs = [data['high'] > data['high'].shift(p) for p in periods]
        loc_bears = pd.Series(np.logical_and.reduce(
            highs), index=data.index)  # bears: pd.Series[bool]

        lows = [data['low'] < data['low'].shift(p) for p in periods]
        loc_bulls = pd.Series(np.logical_and.reduce(lows), index=data.index)

        return loc_bears, loc_bulls

    # extract fractals from df
    def get_bears(self) -> None:
        pass

    def getBulls(self) -> list[Fractal]:
        pass

    def get_fractals(self) -> list[Fractal]:
        pass

# my_fractal = Fractal(time='2022-12-31', type= 2, value=100)
