import logging
from numpy.lib import math
from functools import reduce
from freqtrade import data
from freqtrade.strategy import (
    IStrategy, 
    informative,
    DecimalParameter,
    IntParameter
)
from datetime import datetime
from typing import Optional
import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence.trade_model import Trade

class Lainv35opt(IStrategy):
    INTERFACE_VERSION: int = 3
    minimal_roi = {  
        "720": -1,
        "360": 0.08,
        "180": 0.12,
        "90": 0.15,
        "60": 0.3,
        "0": 0.378
        }
    stoploss = -0.3
    timeframe = "15m"
    use_exit_signal= True 
    can_short = True
    process_only_new_candles= True
    startup_candle_count =  200
    position_adjustment_enable: True
    @informative('1h')
    def populate_indicators_1h(self, df:DataFrame, metadata: dict) -> DataFrame:
        df['tband_top'] = ta.EMA(df['high'], timeperiod=89)
        df['tband_down'] = ta.EMA(df['low'], timeperiod=89)
        df['cci']= ta.CCI(df, timeperiod= 55)
        fast = ta.EMA(df['close'], timeperiod= 89)
        df['fs'] = abs(ta.LINEARREG_SLOPE(fast, timeperiod= 5))
        df['fsr'] = df['fs']- df['fs'].shift(1)
        return df
    @informative('4h')
    def populate_indicators_4h(self, df:DataFrame, metadata: dict) -> DataFrame:
        df['slow'] = ta.EMA(df['close'], timeperiod= 100)
        df['slow_slope'] = ta.LINEARREG_SLOPE(df['slow'], timeperiod= 5)
        frames = [df]
        for val in self.havwap_period.range:
            frames.append(self.avwapmark(df, val))
        df = pd.concat(frames, axis= 1)
        return df
    # 15m
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['tp'] = ta.TYPPRICE(df)
        df['adx'] = ta.ADX(df, timeperiod= 14)
        frames = [df]
        for val in self.channel_multiplier.range:
            frames.append(self.get_median_channel(df, val))
        for val in self.avwap_period.range:
            frames.append(self.avwap(df, val))
        for val in self.avwapL_period.range:
            frames.append(self.avwapmark(df,val))
        df = pd.concat(frames, axis=1)
        df[['cross_above', 'cross_below']] = self.find_reg(df)
        for val in self.lookback_period.range:
            df[f'cross_above_e_{val}'] = self.lookback(df['cross_above'], 1, val)
            df[f'cross_below_e_{val}'] = self.lookback(df['cross_below'], 1, val)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        long_conds = []
        short_conds = []
        df['enter_tag'] = '' 

        bull_avwap_break = (
            (df[f'avwapx_{self.havwap_period.value}_4h'] == 'up') &
            (df[f'avwapx_{self.avwapL_period.value}'] == 'up') &
            (df['cci_1h'] > 0) & 
            (df['close'] > df['tband_top_1h']) &
            ((df['low'].shift(1) <= df['tband_top_1h']) | (df['low'].shift(2) <= df['tband_top_1h'])) &
            (df['volume'] > 0)            
        )
        bear_avwap_break = (
            (df[f'avwapx_{self.havwap_period.value}_4h'] == 'down') &
            (df[f'avwapx_{self.avwapL_period.value}'] == 'down') &
            (df['cci_1h'] < 0) &
            (df['close'] < df['tband_down_1h']) &
            ((df['high'].shift(1) > df['tband_down_1h']) | (df['high'].shift(2) > df['tband_down_1h'])) &
            (df['volume'] > 0)
        )
        bull_reg = (
            (df['adx'] < 24) &
            (df['cci_1h'] < -100) & 
            qtpylib.crossed_above(df['close'], df[f'avwap_upper_L_{self.avwapL_period.value}']) &
            (df[f'cross_above_e_{self.lookback_period.value}']) &
            (df['volume'] > 0)
        )
        bear_reg = (
            (df['adx'] < 24) &
            (df['cci_1h'] > 100) & 
            qtpylib.crossed_below(df['close'], df[f'avwap_lower_L_{self.avwapL_period.value}']) &
            (df[f'avwapx_{self.avwapL_period.value}'] == 'up') &
            (df[f'cross_below_e_{self.lookback_period.value}']) &
            (df['volume'] > 0)
        )
        df.loc[bull_avwap_break, 'enter_tag']+= 'entry1'
        df.loc[bull_reg, 'enter_tag']+= 'reg'
        df.loc[bear_avwap_break, 'enter_tag']+= 'entry1'
        df.loc[bear_reg, 'enter_tag']+= 'reg'

        # TODO: slope+@re
        bull_plus = (
            (df['enter_tag'] == 'reg') &
            (df[f'avwapx_{self.avwapL_period.value}'] == 'up') &
            (df['slow_slope_4h'] < 0) 
        )
        bear_plus = (
            (df['enter_tag'] == 'reg') &
            (df[f'avwapx_{self.avwapL_period.value}'] == 'down') &
            (df['slow_slope_4h'] > 0) 
        )
        df.loc[bull_plus, 'enter_tag'] += 'slope+'
        df.loc[bear_plus, 'enter_tag'] += 'slope+'

        long_conds.extend([bull_avwap_break, bull_reg])
        short_conds.extend([bear_avwap_break, bear_reg])

        if long_conds:
            df.loc[reduce(lambda x, y: x| y, long_conds), 'enter_long'] = 1
        if short_conds:
            df.loc[reduce(lambda x, y: x| y, short_conds), 'enter_short'] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conds = []
        exit_short_conds = []
        df['exit_tag'] = '' 

        df['sideways'] = self.get_context(df)

        bull2bear = (
            (~df[f'sideways']) &
            (df[f'avwapx_{self.avwapL_period.value}'] == 'down') & 
            (df['volume'] > 0)
        )
        bear2bull = (
            (~df[f'sideways']) &
            (df[f'avwapx_{self.avwapL_period.value}'] == 'up') & 
            (df['volume'] > 0)
        )
        exit_short = (df['enter_long'] == 1)
        exit_long = (df['enter_short'] == 1) 
        df.loc[bull2bear, 'exit_tag']+= '& context'
        df.loc[bear2bull, 'exit_tag']+= '& context'
        df.loc[exit_short, 'exit_tag']+= '& opposite_signal'
        df.loc[exit_long, 'exit_tag']+= '& opposite_signal'
        exit_long_conds.extend([bull2bear, exit_long])
        exit_short_conds.extend([bear2bull, exit_short])
        if exit_long_conds:
            df.loc[reduce(lambda x, y: x| y, exit_long_conds), 'exit_long'] = 1
        if exit_short_conds:
            df.loc[reduce(lambda x, y: x| y, exit_short_conds), 'exit_short'] = 1
        return df
    #custom_exit
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,):
        if 'reg' in trade.enter_tag: 
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            current_candle = df.iloc[-1].squeeze()
            side = -1 if trade.is_short else 1
            if side == 1:
                if current_candle[f'avwap_lower_L_{self.avwapL_period.value}'] < current_candle[f'lower_band_{self.channel_multiplier.value}']:
                    return 'reg_exit' 
            else:
                if current_candle[f'avwap_upper_L_{self.avwapL_period.value}'] > current_candle[f'upper_band_{self.channel_multiplier.value}']:
                    return 'reg_exit' 
        return  False
    
    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        if 'entry1' in trade.enter_tag and exit_reason == 'roi':
            return False
        return True

    @staticmethod
    def hhvbar(price: pd.Series, period:int)-> Series:
        return price.rolling(window=period).apply(lambda x: x.idxmax()).convert_dtypes(convert_integer=True) # the index of the certain window

    @staticmethod
    def llvbar(price: pd.Series, period:int) -> Series:
        return price.rolling(window=period).apply(lambda x: x.idxmin()).convert_dtypes(convert_integer=True)
        
    def avwap(self, dataframe:DataFrame,period:int) -> DataFrame:
        df = dataframe.copy()
        if 'tp' not in df.columns:
            df['tp'] = ta.TYPPRICE(df)
        df['tpv'] = df.tp*df.volume
        df['hhvbar'] = self.hhvbar(df['high'], period)
        df['llvbar'] = self.llvbar(df['low'], period)

        avwap_upper =  df.apply(lambda row: df.iloc[row['hhvbar']: row.name+1, df.columns.get_loc('tpv')].sum()/df.iloc[row['hhvbar']:row.name+1,df.columns.get_loc('volume')].sum()
            if pd.notna(row['hhvbar']) else np.nan, axis=1)
        avwap_lower = df.apply(lambda row: df.iloc[row['llvbar']: row.name+1, df.columns.get_loc('tpv')].sum()/ df.iloc[row['llvbar']:row.name+1, df.columns.get_loc('volume')].sum()
                    if pd.notna(row['llvbar']) else np.nan, axis=1)

        return pd.DataFrame({
            f'avwap_upper_{period}': avwap_upper,
            f'avwap_lower_{period}': avwap_lower 
        })
    
    def avwapmark(self, dataframe: DataFrame,timeperiod:int, **kwargs) -> DataFrame:
        df = dataframe.copy()
        sub_df = self.avwap(df, period=timeperiod)
        sub_df.columns = [f'avwap_upper_L_{timeperiod}', f'avwap_lower_L_{timeperiod}']
        sub_df[f'avwapx_{timeperiod}'] = np.where(
            (sub_df[f'avwap_upper_L_{timeperiod}']>0.00) & (sub_df[f'avwap_lower_L_{timeperiod}']>0.00), 
            np.where(
                (df['close'] > sub_df[f'avwap_upper_L_{timeperiod}']),
                'up',
                np.where(
                    (df['close']< sub_df[f'avwap_lower_L_{timeperiod}']),
                    'down',
                    np.NaN
                )
            ),
            np.NaN) 
        return sub_df

    @staticmethod
    def get_median_channel(dataframe: DataFrame, multiplier= 1.5, timeperiod= 89) -> DataFrame:
        df = dataframe.copy()
        median = (df['close'].rolling(window= timeperiod).median()).values
        std_dev = (df['close'].rolling(window= timeperiod).std()).values
        upper_band = median + multiplier*std_dev
        lower_band = median - multiplier*std_dev
        return pd.DataFrame({
            f'upper_band_{multiplier}': upper_band,
            f'lower_band_{multiplier}': lower_band
        })
    
    def adjust_trade_position( self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,):
        side = -1 if trade.is_short else 1
        df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        current_candle = df.iloc[-1].squeeze()
        prev_candle = df.iloc[-2].squeeze()
        if trade.nr_of_successful_exits == 0 and current_profit > self.tp_threshold.val:
            if side == 1:
                if current_candle['close'] < current_candle[f'avwap_lower_{self.avwap_period.value}'] and prev_candle['close'] > prev_candle[f'avwap_lower_{self.avwap_period.value}']:
                    return -(trade.stake_amount/2)
            else:
                if current_candle['close'] > current_candle[f'avwap_upper_{self.avwap_period.value}'] and prev_candle['close'] < prev_candle[f'avwap_upper_{self.avwap_period.value}']:
                    return -(trade.stake_amount/2)
        if trade.nr_of_successful_exits == 1:
            if side == 1:
                if current_candle['close'] < current_candle[f'avwap_lower_L_{self.avwapL_period.value}'] and prev_candle['close'] > prev_candle[f'avwap_lower_L_{self.avwapL_period.value}']:
                    return -(trade.stake_amount)
            else:
                if current_candle['close'] > current_candle[f'avwap_upper_L_{self.avwapL_period.value}'] and prev_candle['close'] < prev_candle[f'avwap_upper_L_{self.avwapL_period.value}']:
                    return -(trade.stake_amount) 

    def leverage(self, pair: str, current_time: datetime, current_rate: float,proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        max_leverage = 5 
        return max_leverage
    
    def get_context(self, dataframe:DataFrame):
        df = dataframe.copy()
        df['sideways'] = np.where(
            (df[f'avwap_upper_L_{self.avwapL_period.value}'] < df[f'upper_band_{self.channel_multiplier.value}']) 
            & (df[f'avwap_lower_L_{self.avwapL_period.value}'] > df[f'lower_band_{self.channel_multiplier.value}']),
            True,False 
        )
        return df['sideways']
    
    def find_reg(self, dataframe: DataFrame):
        df = dataframe.copy()
        df['cross_above'] = np.where(
            qtpylib.crossed_above(df[f'avwap_lower_{self.avwapL_period.value}'], df[f'lower_band_{self.channel_multiplier.value}']), 1, 0
        )
        df['cross_below'] = np.where(
            qtpylib.crossed_below(df[f'avwap_upper_{self.avwapL_period.value}'], df[f'upper_band_{self.channel_multiplier.value}']), 1, 0
        )
        return df[['cross_above', 'cross_below']]





