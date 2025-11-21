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

class Lainlfw(IStrategy):

    INTERFACE_VERSION: int = 3
    # stoploss = -0.3
    # use_custom_stoploss = True

    # trailing_stop = False
    # trailing_stop_positive = 0.05
    # trailing_stop_positive_offset = 0.8 # 0.16*5
    # trailing_only_offset_is_reached = False

    # Position Mgt
    position_adjustment_enable = True
    
    # priority: config.json > strategy
    timeframe = "1h"
    use_exit_signal= True 
    # can_short = False
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
        }
        plot_config['subplots'] = {
            'CCI': {
                'cci_1h': {'type': 'line'}
            },
            "SLOPE": {
                'slow_slope': {'type': 'line'},
            }
        }
        return plot_config

    def bot_loop_start(self, **kwargs) -> None:
        """
        Called at the start of the bot iteration (one loop).
        Might be used to perform pair-independent tasks
        (e.g. gather some remote resource for comparison)
        :param **kwargs: Ensure to keep this here so updates to this won't break your strategy.
        """
        if self.config['runmode'].value in ('live', 'dry_run'):
            # Assign this to the class by using self.*
            # can then be used by populate_* methods
            self.cust_sentiment = requests.get('https://some_remote_source.example.com') 

    # ... populate_* methods
    @informative('D')
    def populate_indicators_D(self, df:DataFrame, metadata: dict) -> DataFrame:
        df[['emah1', 'emah2', 'cross']] = ita.ema3(df, 50, slow= 144)
        return df

    @informative('4h')
    def populate_indicators_4h(self, df:DataFrame, metadata: dict) -> DataFrame:
        df['ema_high'] = ta.EMA(df['high'], timeperiod= 89)
        df['ema_low'] = ta.EMA(df['low'], timeperiod= 89)
        return df

    # 1h 
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['tp'] = ta.TYPPRICE(df)
        df[['ema20', 'ema89']] = ita.avwap(df, 20, slow= 89) 
        df[['ema144', 'ema233', 'cross']] = ita.ema3(df,144, slow= 233)
        # df[['upper_band', 'lower_band']] = ita.bband(df, 1.5, 89)
        df['umacd'] = ita.umacd(df)

        # Check if the entry already exists
        if not metadata["pair"] in self.cust_sentiment:
            # Create empty entry for this pair
            self.cust_sentiment[metadata["pair"]] = {}

        if "crosstime" in self.cust_sentiment[metadata["pair"]]:
            self.cust_sentiment[metadata["pair"]]["crosstime"] += 1
        else:
            self.cust_sentiment[metadata["pair"]]["crosstime"] = 1
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        long_conds = []
        short_conds = []
        df['enter_tag'] = '' 
        bull_break = (
            (df['cross'] == 'up') &
            # (df['cci_1h'] > 0) &
            (df['close'] > df['ema_high_1h']) &
            ((df['low'].shift(1) < df['ema_high_1h']) | (df['low'].shift(2) < df['ema_high_1h'])) &
            (df['volume'] > 0)            
        )
        bear_break = (
            (df['cross'] == 'down') &
            # (df['cci_1h'] < 0) &
            (df['close'] < df['ema_low_1h']) &
            ((df['high'].shift(1) > df['ema_low_1h']) | (df['high'].shift(2) > df['ema_low_1h'])) &
            (df['volume'] > 0)
        )
        df.loc[bull_break, 'enter_tag']+= 'entry1'
        df.loc[bear_break, 'enter_tag']+= 'entry1'

        long_conds.append(bull_break)
        short_conds.append(bear_break)

        if long_conds:
            df.loc[reduce(lambda x, y: x| y, long_conds), 'enter_long'] = 1
        if short_conds:
            df.loc[reduce(lambda x, y: x| y, short_conds), 'enter_short'] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        exit_long_conds = []
        df['exit_tag'] = '' 
        bull2bear = (
            (~df['umacd']) &
            (df['cross'] == 'down') & 
            (df['volume'] > 0)
        )
        exit_long = (
            (df['enter_short'] == 1)
        )
        df.loc[bull2bear, 'exit_tag']+= '& context'
        df.loc[bear2bull, 'exit_tag']+= '& context'
        df.loc[exit_long, 'exit_tag']+= '& opposite_signal'

        exit_long_conds.extend([bull2bear, exit_long])

        if exit_long_conds:
            df.loc[reduce(lambda x, y: x| y, exit_long_conds), 'exit_long'] = 1
        return df
    
    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,):
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
        
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                                proposed_stake: float, min_stake: Optional[float], max_stake: float,
                                leverage: float, entry_tag: Optional[str], side: str,
                                **kwargs) -> float:
        f = 0.01
        capital = self.wallets.get_total_stake_amount() 
        return max(capital*f, 10)