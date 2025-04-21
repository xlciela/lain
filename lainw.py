import logging
from numpy.lib import math
from functools import reduce
from freqtrade import data
from freqtrade.strategy import IStrategy, informative
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy.interface import IStrategy
from freqtrade.persistence.trade_model import Trade
import utils.ita as ita

class Lainx37(IStrategy):

    INTERFACE_VERSION: int = 3
    # minimal_roi = {  
    #     "360": 0,
    #     "240": 0.1,
    #     "60": 0.15,
    #     "30": 0.2,
    #     "0": 0.3
    # }
    minimal_roi = {  
        "720": -1,
        "240": 0.1,
        "60": 0.2,
        "30": 0.3,
        "0": 0.378
        }
 
    stoploss = -0.3
    # use_custom_stoploss = True

    # trailing_stop = False
    # trailing_stop_positive = 0.05
    # trailing_stop_positive_offset = 0.8 # 0.16*5
    # trailing_only_offset_is_reached = False

    # Position Mgt
    position_adjustment_enable = True
    
    # priority: config.json > strategy
    timeframe = "15m"
    use_exit_signal= True 
    can_short = True
    process_only_new_candles= True

    # order
    # order_time_in_force={
    #     'buy':gtc,
    #     'sell': gtc
    # }

    startup_candle_count =  200

    @property
    def plot_config(self):
        plot_config = {}
        plot_config['main_plot'] = {
            'close_4h': {'color': 'purple'},
            'emah1_4h': {'color': 'orange', },
            'emah2_4h': {'color': 'blue'},
            'ema89': {'color': '#ff5252', 'fill_to': 'ema20', 'fill_color':'rgba(255, 82, 82, 0.1)'},
            'ema144': {'color': '#4caf50', 'fill_to': 'ema89', 'fill_color': 'rgba(76, 175, 80, 0.1)'},
            'ema_high_1h':{'color': 'blue'},
            'ema_low_1h':{'color': 'red'},
            'cross':{'type':'scatter'},
            'upper_band': {'color': '#7d25e8'},
            'lower_band': {'color': '#7d25e8'},
        }
        plot_config['subplots'] = {
            'CCI': {
                'cci_1h': {'type': 'line'}
            },
            "SLOPE": {
                'slow_slope_4h': {'type': 'line'},
                'fsr_1h': {'type': 'line'}
            }
        }
        return plot_config
    
    @informative('4h')
    def populate_indicators_4h(self, df:DataFrame, metadata: dict) -> DataFrame:
        df[['emah1', 'emah2', 'cross']] = ita.ema3(df, 50, slow= 144)
        df['slow'] = ta.EMA(df['close'], timeperiod= 100)
        df['slow_slope'] = ta.LINEARREG_SLOPE(df['slow'], timeperiod= 5)
        return df

    @informative('1h')
    def populate_indicators_1h(self, df:DataFrame, metadata: dict) -> DataFrame:
        df['ema_high'] = ta.EMA(df['high'], timeperiod= 89)
        df['ema_low'] = ta.EMA(df['low'], timeperiod= 89)
        df['cci']= ta.CCI(df, timeperiod= 55)
        fast = ta.EMA(df['close'], timeperiod= 89)
        df['fs'] = abs(ta.LINEARREG_SLOPE(fast, timeperiod= 7))
        df['fsr'] = df['fs']- df['fs'].shift(1)
        return df

    # 15m
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['tp'] = ta.TYPPRICE(df)
        df['adx'] = ta.ADX(df, timeperiod= 14)
        df[['ema20', 'ema89']] = ita.avwap(df, 20, slow= 89) 
        df[['ema144', 'ema233', 'cross']] = ita.ema3(df,144, slow= 233)
        df[['upper_band', 'lower_band']] = ita.bband(df, 1.5, 89)
        df['umacd'] = ita.umacd(df)
        df[['cross_above', 'cross_below']] = ita.cross(df)
        df['cross_above_e'] = ita.check_cross(df['cross_above'], 1, 8)
        df['cross_below_e'] = ita.check_cross(df['cross_below'], 1, 8)
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        long_conds = []
        short_conds = []
        df['enter_tag'] = '' 
        bull_break = (
            (df['cross_4h'] == 'up') &
            (df['cross'] == 'up') &
            (df['cci_1h'] > 0) &
            (df['close'] > df['ema_high_1h']) &
            ((df['low'].shift(1) < df['ema_high_1h']) | (df['low'].shift(2) < df['ema_high_1h'])) &
            (df['volume'] > 0)            
        )
        bear_break = (
            (df['cross_4h'] == 'down') &
            (df['cross'] == 'down') &
            (df['cci_1h'] < 0) &
            (df['close'] < df['ema_low_1h']) &
            ((df['high'].shift(1) > df['ema_low_1h']) | (df['high'].shift(2) > df['ema_low_1h'])) &
            (df['volume'] > 0)
        )
        df.loc[bull_break, 'enter_tag']+= 'entry1'
        df.loc[bear_break, 'enter_tag']+= 'entry1'
        bull_cross = (
            (df['adx'] < 24) &
            (df['cci_1h'] < -100) & 
            # (df['slow_slope_4h'] < 0) &
            qtpylib.crossed_above(df['close'], df['ema144']) &
            df['cross_above_e'] &
            (df['volume'] > 0)
        )
        bear_cross = (
            (df['adx'] < 24) &
            (df['cci_1h'] > 100) &
            (df['slow_slope_4h'] > 0) &
            qtpylib.crossed_below(df['close'], df['ema233']) &
            # df['cross_below_e'] &
            (df['volume'] > 0)
        )
        df.loc[bull_cross, 'enter_tag']+= 're'
        df.loc[bear_cross, 'enter_tag']+= 're'

        long_conds.extend([bull_break, bull_cross])
        short_conds.extend([bear_break, bear_cross])

        if long_conds:
            df.loc[reduce(lambda x, y: x| y, long_conds), 'enter_long'] = 1
        if short_conds:
            df.loc[reduce(lambda x, y: x| y, short_conds), 'enter_short'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conds = []
        exit_short_conds = []
        df['exit_tag'] = '' 
        bull2bear = (
            (~df['umacd']) &
            (df['cross'] == 'down') & 
            (df['volume'] > 0)
        )
        bear2bull = (
            (~df['umacd']) & 
            (df['cross'] == 'up') & 
            (df['volume'] > 0)
        )
        exit_short = (
            (df['enter_long'] == 1)
        )
        exit_long = (
            (df['enter_short'] == 1)
        )
        df.loc[bull2bear, 'exit_tag']+= '& context'
        df.loc[bear2bull, 'exit_tag']+= '& context'
        df.loc[exit_long, 'exit_tag']+= '& opposite_signal'
        df.loc[exit_short, 'exit_tag']+= '& opposite_signal'

        exit_long_conds.extend([bull2bear, exit_long])
        exit_short_conds.extend([bear2bull, exit_short])

        if exit_long_conds:
            df.loc[reduce(lambda x, y: x| y, exit_long_conds), 'exit_long'] = 1
        if exit_short_conds:
            df.loc[reduce(lambda x, y: x| y, exit_short_conds), 'exit_short'] = 1
        return df
    
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,):
        # TODO:
        if trade.enter_tag == 're':
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            current_candle = df.iloc[-1].squeeze()
            side = -1 if trade.is_short else 1
            if side == 1:
                # if current_candle['ema233'] < current_candle['lower_band']:
                if current_candle['cross'] == 'down':
                    return 'reg_exit'
            else:
                # if current_candle['ema144'] > current_candle['upper_band']:
                if current_candle['cross'] == 'up':
                    return 'reg_exit'
        return False

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
        if 're' in trade.enter_tag and exit_reason=='roi': 
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            current_candle = df.iloc[-1].squeeze()
            signal_name = 'enter_short' if trade.is_short else 'enter_long'
            lasted_candles = math.floor((current_time- trade.open_date_utc).total_seconds()/900)
            if len(df) > lasted_candles:
                if any(df.iloc[-lasted_candles+1:][signal_name] == 1):
                    signal_override = df.iloc[-lasted_candles+1:]['enter_tag'].str.contains('entry1', na= False).any()
                    return not signal_override
        return True

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
        if trade.nr_of_successful_exits == 0 and current_profit > 1:
            if side == 1:
                if current_candle['close'] < current_candle['ema89'] and prev_candle['close'] > prev_candle['ema89']:
                    return -(trade.stake_amount/2)
            else:
                if current_candle['close'] > current_candle['ema20'] and prev_candle['close'] < prev_candle['ema20']:
                    return -(trade.stake_amount/2)

        if trade.nr_of_successful_exits == 1:
            if side == 1:
                if current_candle['close'] < current_candle['ema233'] and prev_candle['close'] > prev_candle['ema233']:
                    return -(trade.stake_amount)
            else:
                if current_candle['close'] > current_candle['ema144'] and prev_candle['close'] < prev_candle['ema144']:
                    return -(trade.stake_amount) 
        

    def leverage(self, pair: str, current_time: datetime, current_rate: float,proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        max_leverage = 5
        return max_leverage

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                                proposed_stake: float, min_stake: Optional[float], max_stake: float,
                                leverage: float, entry_tag: Optional[str], side: str,
                                **kwargs) -> float:
        f = 0.02
        capital = self.wallets.get_total_stake_amount() 
        if capital > 20000:
            f = 0.01
        if capital > 100000:
            f = 0.005
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        current_candle = df.iloc[-1].squeeze()
        if current_candle['fsr_1h'] > 0:
            f = f*2
        if 'entry1' in entry_tag:  
            if side == 'long':
                if current_candle['slow_slope_4h'] > 0: 
                    f = f*1.5
                if current_candle['cci_1h'] > 70:
                    f = f*2
            if side == 'short':  
                if current_candle['slow_slope_4h'] < 0:
                    f = f*1.5
                if current_candle['cci_1h'] < -70:
                    f = f*2
        if entry_tag == 're':
            f = f*2
        return max(capital*f, 100)