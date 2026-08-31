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


class TrendMasterPro(IStrategy):
    """
    ================================================================================
    TrendMasterPro - Institutional Multi-Timeframe Trend Following Strategy
    ================================================================================
    Designed for High Win-rate, strict Risk Management, and Capital Preservation.
    
    Key Mechanisms:
    1. Macro 1h Higher Timeframe Trend Filter:
       - 1h EMA 50 > 1h EMA 200 (Macro Golden Cross)
       - 1h Close > 1h EMA 200 (Above major support)
       - 1h ADX > 20 (Strong market trend confirmation)
    2. Micro 5m Entry Triggers:
       - 5m EMA 9 > 5m EMA 21 (Short-term bullish momentum)
       - 5m RSI Pullback into value zone (40 - 65)
       - 5m Volume Spike (Volume > 1.2x Volume SMA 20)
       - Price above 5m EMA 50
    3. Institutional Exits & Risk Protection:
       - Hard Stoploss: -4.0%
       - Trailing Stop: Activates at +2.5% profit, trails by 1.2%
       - Dynamic ROI Take-Profit Ladder
       - Custom Indicator Exit (EMA breakdown or RSI > 75 overbought reversal)
    """

    INTERFACE_VERSION = 3

    # Base Timeframe
    timeframe = "5m"
    informative_timeframe = "1h"

    # Spot (can_short = False) or Futures (can_short = True)
    can_short: bool = False

    # Startup candles to warm up 1h EMA 200 (200 * 12 = 2400 5m candles or 200 1h candles)
    startup_candle_count: int = 250

    # Process only new candles for speed and clean signals
    process_only_new_candles = True

    # -------------------------------------------------------------------------
    # Risk Management & ROI
    # -------------------------------------------------------------------------
    # Hard Stoploss (-4%)
    stoploss = -0.04

    # Trailing Stoploss (Locks in profit when price pumps)
    trailing_stop = True
    trailing_stop_positive = 0.012          # 1.2% trailing distance
    trailing_stop_positive_offset = 0.025   # Activates once +2.5% profit is reached
    trailing_only_offset_is_reached = True

    # Dynamic ROI Ladder (Takes profit automatically based on trade duration)
    minimal_roi = {
        "0": 0.05,       # 5.0% profit immediate exit
        "30": 0.03,      # 3.0% profit after 30 mins
        "60": 0.02,      # 2.0% profit after 60 mins
        "120": 0.012     # 1.2% profit after 2 hours
    }

    # Exit Signals
    use_exit_signal = True
    exit_profit_only = True                 # Exit signals trigger only if trade is in profit
    ignore_roi_if_entry_signal = False

    # Order Types
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

    # Protections (Modern Freqtrade style)
    protections = [
        {
            "method": "CooldownPeriod",
            "stop_duration_candles": 5
        },
        {
            "method": "StoplossGuard",
            "lookback_period_candles": 60,
            "trade_limit": 2,
            "stop_duration_candles": 60,
            "only_per_pair": True
        },
        {
            "method": "MaxDrawdown",
            "lookback_period_candles": 240,
            "trade_limit": 4,
            "stop_duration_candles": 120,
            "max_allowed_drawdown": 0.10
        }
    ]

    # -------------------------------------------------------------------------
    # Hyperoptable Parameters (Can be tuned with 'freqtrade hyperopt')
    # -------------------------------------------------------------------------
    # Buy / Entry Parameters
    buy_rsi_min = IntParameter(30, 50, default=40, space="buy", optimize=True)
    buy_rsi_max = IntParameter(55, 75, default=65, space="buy", optimize=True)
    buy_volume_multiplier = DecimalParameter(1.0, 2.5, default=1.2, decimals=1, space="buy", optimize=True)
    buy_adx_1h_min = IntParameter(15, 35, default=20, space="buy", optimize=True)

    # Sell / Exit Parameters
    sell_rsi = IntParameter(70, 85, default=75, space="sell", optimize=True)

    # -------------------------------------------------------------------------
    # Informative Higher Timeframe (1 Hour) Indicators
    # -------------------------------------------------------------------------
    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Macro EMAs
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)
        
        # 1h RSI & ADX (Trend Strength)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)

        return dataframe

    # -------------------------------------------------------------------------
    # Base Timeframe (5 Min) Indicators
    # -------------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Moving Averages (5m)
        dataframe["ema_9"] = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema_21"] = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema_50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_200"] = ta.EMA(dataframe, timeperiod=200)

        # Momentum & Oscillators
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        
        # Volume Moving Average
        dataframe["volume_mean_20"] = dataframe["volume"].rolling(window=20).mean()

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]
        dataframe["bb_percent"] = (
            (dataframe["close"] - dataframe["bb_lowerband"]) /
            (dataframe["bb_upperband"] - dataframe["bb_lowerband"])
        )

        # Average True Range (ATR) for volatility
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        return dataframe

    # -------------------------------------------------------------------------
    # Entry Signals (Long)
    # -------------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # 1. Macro 1h Trend Conditions (Market Alignment)
        conditions.append(dataframe["ema_50_1h"] > dataframe["ema_200_1h"])       # Macro Golden Cross
        conditions.append(dataframe["close_1h"] > dataframe["ema_200_1h"])        # Price above Macro 200 EMA
        conditions.append(dataframe["adx_1h"] >= self.buy_adx_1h_min.value)       # Macro Trend is strong

        # 2. Micro 5m Trend Conditions (Local Momentum)
        conditions.append(dataframe["ema_9"] > dataframe["ema_21"])               # Fast EMA cross
        conditions.append(dataframe["close"] > dataframe["ema_50"])               # Price above 50 EMA
        
        # 3. Pullback / Value Zone Trigger (Avoid buying overbought peaks)
        conditions.append(dataframe["rsi"] >= self.buy_rsi_min.value)
        conditions.append(dataframe["rsi"] <= self.buy_rsi_max.value)

        # 4. Volume Spike Confirmation (Institutional backing)
        conditions.append(
            dataframe["volume"] > (dataframe["volume_mean_20"] * self.buy_volume_multiplier.value)
        )

        # 5. Volume sanity check
        conditions.append(dataframe["volume"] > 0)

        if conditions:
            dataframe.loc[
                np.logical_and.reduce(conditions),
                "enter_long"
            ] = 1

        return dataframe

    # -------------------------------------------------------------------------
    # Exit Signals (Sell / Take-Profit / Trend Breakdown)
    # -------------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []

        # 1. Overbought RSI Reversal Signal
        rsi_overbought = (dataframe["rsi"] > self.sell_rsi.value) & (
            qtpylib.crossed_below(dataframe["rsi"], self.sell_rsi.value)
        )

        # 2. Fast EMA Breakdown (EMA 9 crossed below EMA 21)
        ema_breakdown = qtpylib.crossed_below(dataframe["ema_9"], dataframe["ema_21"])

        conditions.append(rsi_overbought | ema_breakdown)
        conditions.append(dataframe["volume"] > 0)

        if conditions:
            dataframe.loc[
                np.logical_and.reduce(conditions),
                "exit_long"
            ] = 1

        return dataframe
