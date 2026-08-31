# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pandas import DataFrame
from typing import Optional, Union, Dict

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    BooleanParameter,
    CategoricalParameter,
    DecimalParameter,
    IntParameter,
    RealParameter,
)

import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


class FastScalperPro(IStrategy):
    """
    ================================================================================
    FastScalperPro - High-Frequency Sniper Scalper (1m Timeframe)
    ================================================================================
    Built for:
    - High trade frequency (20-50 quick trades per day)
    - Quick in & out (5 to 20 minutes average trade duration)
    - High Win Rate target (70% - 80%+)
    - Fast compounding for small capital (50$)
    
    Logic:
    1. Trend Confirmation (15m): 15m Close > 15m EMA 50 (Trade only with momentum)
    2. 1m Trigger:
       - Price bounces off Lower Bollinger Band (Oversold Dip)
       - Stochastic RSI K crosses above D in Oversold zone (< 25)
       - RSI recovers above 35
       - Volume surge (Volume > 1.3x Volume SMA 15)
    3. Quick Sniper Exits:
       - Immediate Take-Profit at +1.2% to +1.8%
       - Trailing Stop locks profit after +0.8%
       - Tight Hard Stoploss at -1.6%
    """

    INTERFACE_VERSION = 3

    # Fast 1-minute execution timeframe
    timeframe = "1m"
    informative_timeframe = "15m"

    can_short: bool = False
    startup_candle_count: int = 150
    process_only_new_candles = True

    # -------------------------------------------------------------------------
    # Sniper Risk Management & Quick Take-Profit
    # -------------------------------------------------------------------------
    # Tight Stoploss (-1.6%)
    stoploss = -0.016

    # Trailing Stop: Activates quickly at +0.8% profit, trails tightly by 0.35%
    trailing_stop = True
    trailing_stop_positive = 0.0035         # 0.35% trailing gap
    trailing_stop_positive_offset = 0.008   # Triggers once +0.8% is hit
    trailing_only_offset_is_reached = True

    # Quick Sniper ROI ladder
    minimal_roi = {
        "0": 0.018,      # 1.8% immediate sniper profit
        "10": 0.012,     # 1.2% profit after 10 mins
        "20": 0.008,     # 0.8% profit after 20 mins
        "40": 0.005      # 0.5% profit after 40 mins
    }

    use_exit_signal = True
    exit_profit_only = True
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False
    }

    order_time_in_force = {
        "entry": "GTC",
        "exit": "GTC"
    }

    # Protections
    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 3
        },
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 30,
            "trade_limit": 2,
            "stop_duration_candles": 30,
            "only_per_pair": True
        }
    ]

    # -------------------------------------------------------------------------
    # Informative Higher Timeframe (15 Min) Trend Filter
    # -------------------------------------------------------------------------
    @informative("15m")
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    # -------------------------------------------------------------------------
    # Base 1-Minute Scalping Indicators
    # -------------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # EMAs
        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)

        # RSI & Stochastic RSI (Fast momentum)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=3, fastd_period=3)
        dataframe["stoch_k"] = stoch_rsi["fastk"]
        dataframe["stoch_d"] = stoch_rsi["fastd"]

        # Bollinger Bands for Dip-Buying
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2.0)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )

        # Volume Moving Average
        dataframe["volume_mean_15"] = dataframe["volume"].rolling(window=15).mean()

        return dataframe

    # -------------------------------------------------------------------------
    # Entry Signals (Quick Buy Trigger)
    # -------------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # 1. Higher Timeframe 15m Trend Confirmation (Micro Bullish bias)
        conditions.append(dataframe["close_15m"] > dataframe["ema_50_15m"])
        conditions.append(dataframe["rsi_15m"] > 45)

        # 2. Oversold Bounce on 1m (Dip Buying)
        conditions.append(dataframe["close"] <= dataframe["bb_lowerband"] * 1.002) # Touched / near lower band
        conditions.append(dataframe["rsi"] < 45)
        conditions.append(dataframe["stoch_k"] < 30)                               # Stoch RSI oversold
        conditions.append(dataframe["stoch_k"] > dataframe["stoch_d"])             # Stoch Bullish Crossover

        # 3. Volume Surge (Buyer presence)
        conditions.append(dataframe["volume"] > (dataframe["volume_mean_15"] * 1.15))
        conditions.append(dataframe["volume"] > 0)

        if conditions:
            dataframe.loc[
                np.logical_and.reduce(conditions),
                "enter_long"
            ] = 1

        return dataframe

    # -------------------------------------------------------------------------
    # Exit Signals (Fast Take-Profit & Reversal)
    # -------------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # Exit when price hits upper Bollinger Band with overbought RSI
        hit_upper_bb = dataframe["close"] >= dataframe["bb_upperband"]
        rsi_overbought = dataframe["rsi"] > 70
        stoch_overbought = (dataframe["stoch_k"] > 80) & (dataframe["stoch_k"] < dataframe["stoch_d"])

        conditions.append(hit_upper_bb | (rsi_overbought & stoch_overbought))
        conditions.append(dataframe["volume"] > 0)

        if conditions:
            dataframe.loc[
                np.logical_and.reduce(conditions),
                "exit_long"
            ] = 1

        return dataframe
