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
import tools

class Lainv11(IStrategy):

    INTERFACE_VERSION: int = 3
    ''' ROI table: 
    minimal Return On Investment (ROI) a trade should reach before exiting, 
    independent from the exit signal.
    超参数优化 hyp
    '''
    minimal_roi = {  
        "1440": 0.1,
        "360": 0.15,
        "90": 0.2,
        "60": 0.3,
        "30": 0.5,
        "0": 1
        }
    stoploss = -0.3
    # use_custom_stoploss = True
    position_adjustment_enable = True
    timeframe = "15m"
    use_exit_signal= True 
    can_short = True
    process_only_new_candles= True
    startup_candle_count =  200
    @property
    def plot_config(self):
        plot_config = {}
        plot_config['main_plot'] = {
            'avwap_upper_L': {'color': '#ff5252', 'fill_to': 'avwap_upper', 'fill_color':'rgba(255, 82, 82, 0.1)'},
            'avwap_lower_L': {'color': '#4caf50', 'fill_to': 'avwap_lower', 'fill_color': 'rgba(76, 175, 80, 0.1)'},
            'tband_top_1h':{'color': 'blue'},
            'tband_down_1h':{'color': 'red'},
            'avwapx':{'type':'scatter'},
            'upper_band': {'color': '#7d25e8'},
            'lower_band': {'color': '#7d25e8'},
        }
        plot_config['subplots'] = {
            # 'pass_reg_filter': {
            #     'pass_reg_filter': {'type': 'line'}
            # }
        }
        return plot_config

    ''' 
    define informative higher timeframe for each pair on same method
    available in populate_indicators as tband_top_1h, tband_dwn_1h
    '''
    @informative('1h')
    def populate_indicators_1h(self, df:DataFrame, metadata: dict) -> DataFrame:
        df['tband_top'] = ta.EMA(df['high'], timeperiod= 89)
        df['tband_down'] = ta.EMA(df['low'], timeperiod= 89)
        df['htf'] = ta.EMA(df['close'], timeperiod= 144)
        return df
    @informative('4h')
    def populate_indicators_4h(self, df:DataFrame, metadata: dict) -> DataFrame:
        df[['havwap_upper', 'havwap_lower', 'havwapx']] = avwapmark(df, 50)  
        df['slow'] = ta.EMA(df['close'], timeperiod= 100)
        df['slow_slope'] = ta.LINEARREG_SLOPE(df['slow'], timeperiod= 5)
        return df

    # 15m
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['tp'] = ta.TYPPRICE(df)
        # df['atr'] = ta.ATR(df, timeperiod= 14)
        # df[['atr_upper', 'atr_lower']] = self.atr_band(df, 1)
        df['ss_abs'] = abs(df['slow_slope_4h'])
        df[['avwap_upper', 'avwap_lower']] = avwap(df, 22) 
        df[['avwap_upper_L', 'avwap_lower_L', 'avwapx']] = avwapmark(df,88)
        df[['upper_band', 'lower_band']] = get_median_channel(df, 2, 89)
        # df[['deviation','bull_quantile', 'bear_quantile', 'pass_reg_filter']] = regression_ema(df, 144)
        # df['rsi'] = ta.RSI(df, timeperiod = 14)
        df['sideways'] = get_context(df)
        df[['cross_above', 'cross_below']] = find_reg(df)
        df['cross_above_e'] = lookback(df['cross_above'], 1, 5) 
        df['cross_below_e'] = lookback(df['cross_below'], 1, 5)
        return df
    """ 
        开仓
        1. entry1
        2. entry2
    """
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        long_conds = []
        short_conds = []
        df['enter_tag'] = '' 
        bull_avwap_break = (
            (df['havwapx_4h'] == 'up') &
            (df['avwapx'] == 'up') &
            (df['close'] > df['tband_top_1h']) &
            ((df['low'].shift(1) < df['tband_top_1h']) | (df['low'].shift(2) < df['tband_top_1h'])) &
            (df['volume'] > 0)            
        )
        bear_avwap_break = (
            (df['havwapx_4h'] == 'down') &
            (df['avwapx'] == 'down') &
            (df['close'] < df['tband_down_1h']) &
            ((df['high'].shift(1) > df['tband_down_1h']) | (df['high'].shift(2) > df['tband_down_1h'])) &
            (df['volume'] > 0)
        )
        bull_reg = (
            (df['ss_abs'] < 0.1) & # TODO: test
            # (df['pass_reg_filter'] == 1) & # TODO: test
            (df['avwapx'] == 'up') &
            df['cross_above_e'] &
            (df['volume'] > 0)
        )
        bear_reg = (
            (df['ss_abs'] < 0.1) & # TODO: test
            # (df['pass_reg_filter'] == -1) &
            (df['avwapx'] == 'down') &
            df['cross_below_e'] &
            (df['volume'] > 0)
        )
        df.loc[bull_avwap_break, 'enter_tag']+= 'entry1'
        df.loc[bear_avwap_break, 'enter_tag']+= 'entry1'
        df.loc[bull_reg, 'enter_tag']+= 'reg'
        df.loc[bear_reg, 'enter_tag']+= 'reg'
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
        bull2bear = (
            (~df['sideways']) &
            (df['avwapx'] == 'down') &
            (df['volume'] > 0)
        )
        bear2bull = (
            (~df['sideways']) &
            (df['avwapx'] == 'up') &
            (df['volume'] > 0)
        )
        exit_long = (
            (df['enter_short'] == 1)
        )
        exit_short = (
            (df['enter_long'] == 1)
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
        if trade.enter_tag == 'reg':
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            current_candle = df.iloc[-1].squeeze()
            prev_candle = df.iloc[-2].squeeze()
            lasted_candles = math.floor((current_time- trade.open_date_utc).total_seconds()/900)
            side = -1 if trade.is_short else 1
            if side == 1:
                if current_candle['avwap_lower_L'] < current_candle['lower_band']: # lagging
                    return 'reg_exit'
            else:
                if current_candle['avwap_upper_L'] > current_candle['upper_band']:
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
        """
        confirm_trade_exit() can prevent stoploss exits, 
        causing significant losses as this would ignore stoploss exits. 
        confirm_trade_exit() will not be called for Liquidations - as liquidations are forced by the exchange, 
        and therefore cannot be rejected.
        """
        if 'entry1' in trade.enter_tag and exit_reason == 'roi':
            return False
        if trade.enter_tag == 'reg' and exit_reason=='roi': 
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            current_candle = df.iloc[-1].squeeze()
            signal_name = 'enter_short' if trade.is_short else 'enter_long'
            lasted_candles = math.floor((current_time- trade.open_date_utc).total_seconds()/900)
            if lasted_candles> 48 and len(df) > 48:
                if any(df.iloc[-lasted_candles+1:][signal_name] == 1):
                    return False
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
        if trade.nr_of_successful_exits == 0 and current_profit > 0.2:
            if side == 1:
                if current_candle['close'] < current_candle['avwap_lower'] and prev_candle['close'] > prev_candle['avwap_lower']:
                    return -(trade.stake_amount/2)
            else:
                if current_candle['close'] > current_candle['avwap_upper'] and prev_candle['close'] < prev_candle['avwap_upper']:
                    return -(trade.stake_amount/2)
        if trade.nr_of_successful_exits == 1:
            if side == 1:
                if current_candle['close'] < current_candle['avwap_lower_L'] and prev_candle['close'] > prev_candle['avwap_lower_L']:
                    return -(trade.stake_amount)
            else:
                if current_candle['close'] > current_candle['avwap_upper_L'] and prev_candle['close'] < prev_candle['avwap_upper_L']:
                    return -(trade.stake_amount) 
        

    def leverage(self, pair: str, current_time: datetime, current_rate: float,proposed_leverage: float, max_leverage: float, side: str, **kwargs) -> float:
        max_leverage = 5 
        return max_leverage
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str,
                            **kwargs) -> float:
        f = 0.04
        capital = self.wallets.get_total_stake_amount() 
        if capital > 7000:
            f = 0.03
        if capital > 10000:
            f = 0.02
        if capital > 30000:
            f = 0.01
        return capital*f





