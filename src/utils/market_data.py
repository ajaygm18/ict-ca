"""
Market Data Module for ICT Trading Framework

This module handles data loading, processing, and indicator calculation
for the ICT trading strategies.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, Any, List
import ta
from datetime import datetime, time
import pytz


class MarketDataLoader:
    """
    Handles loading and processing of market data from various sources
    """
    
    def __init__(self, timezone: str = 'US/Eastern'):
        """
        Initialize the data loader
        
        Args:
            timezone: Trading timezone (default: US/Eastern for ICT)
        """
        self.timezone = pytz.timezone(timezone)
    
    def load_from_yahoo(self, symbol: str, start_date: str, end_date: str, 
                       interval: str = '5m') -> pd.DataFrame:
        """
        Load data from Yahoo Finance
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD=X')
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            interval: Data interval (1m, 5m, 15m, 1h, 1d)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")
            
            # Clean column names
            data.columns = [col.lower() for col in data.columns]
            
            # Ensure timezone awareness
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            data.index = data.index.tz_convert(self.timezone)
            
            return data
            
        except Exception as e:
            raise Exception(f"Error loading data from Yahoo Finance: {str(e)}")
    
    def load_from_csv(self, file_path: str, datetime_column: str = 'datetime') -> pd.DataFrame:
        """
        Load data from CSV file
        
        Args:
            file_path: Path to CSV file
            datetime_column: Name of datetime column
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = pd.read_csv(file_path)
            
            # Set datetime index
            data[datetime_column] = pd.to_datetime(data[datetime_column])
            data.set_index(datetime_column, inplace=True)
            
            # Ensure timezone awareness
            if data.index.tz is None:
                data.index = data.index.tz_localize(self.timezone)
            
            # Clean column names
            data.columns = [col.lower() for col in data.columns]
            
            return data
            
        except Exception as e:
            raise Exception(f"Error loading data from CSV: {str(e)}")
    
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators required for ICT strategies
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with added indicators
        """
        df = data.copy()
        
        # Average True Range (ATR) - for volatility filter
        df['atr'] = ta.volatility.AverageTrueRange(
            high=df['high'], low=df['low'], close=df['close'], window=14
        ).average_true_range()
        
        # Simple Moving Averages
        df['sma_20'] = ta.trend.SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = ta.trend.SMAIndicator(close=df['close'], window=50).sma_indicator()
        
        # Exponential Moving Averages
        df['ema_20'] = ta.trend.EMAIndicator(close=df['close'], window=20).ema_indicator()
        
        # RSI for momentum confirmation
        df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
        
        # MACD for trend confirmation
        macd = ta.trend.MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bollinger.bollinger_hband()
        df['bb_middle'] = bollinger.bollinger_mavg()
        df['bb_lower'] = bollinger.bollinger_lband()
        
        return df
    
    def resample_data(self, data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to different timeframe
        
        Args:
            data: OHLCV DataFrame
            timeframe: Target timeframe (e.g., '15T', '1H', '1D')
            
        Returns:
            Resampled DataFrame
        """
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        resampled = data.resample(timeframe).agg(ohlc_dict)
        resampled.dropna(inplace=True)
        
        return resampled


class SessionDetector:
    """
    Detects trading sessions based on ICT methodology
    """
    
    def __init__(self, timezone: str = 'US/Eastern'):
        """
        Initialize session detector
        
        Args:
            timezone: Trading timezone
        """
        self.timezone = pytz.timezone(timezone)
        
        # ICT Trading Sessions (EST)
        self.sessions = {
            'ny_killzone': {'start': time(10, 0), 'end': time(11, 0)},
            'pre_market': {'start': time(2, 0), 'end': time(7, 0)},
            'ny_open': {'start': time(9, 30), 'end': time(10, 0)},
            'power_hour': {'start': time(14, 0), 'end': time(15, 0)},
            'london_ny_overlap': {'start': time(8, 0), 'end': time(11, 0)},
            'afternoon': {'start': time(13, 0), 'end': time(16, 0)}
        }
    
    def is_in_session(self, timestamp: pd.Timestamp, session_name: str) -> bool:
        """
        Check if timestamp is within specified session
        
        Args:
            timestamp: Timestamp to check
            session_name: Name of the session
            
        Returns:
            True if timestamp is in session
        """
        if session_name not in self.sessions:
            return False
        
        session = self.sessions[session_name]
        current_time = timestamp.time()
        
        # Handle sessions that cross midnight
        if session['start'] > session['end']:
            return current_time >= session['start'] or current_time <= session['end']
        else:
            return session['start'] <= current_time <= session['end']
    
    def add_session_markers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add session marker columns to data
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with session columns
        """
        df = data.copy()
        
        for session_name in self.sessions.keys():
            df[f'is_{session_name}'] = df.index.to_series().apply(
                lambda x: self.is_in_session(x, session_name)
            )
        
        return df
    
    def get_session_data(self, data: pd.DataFrame, session_name: str) -> pd.DataFrame:
        """
        Filter data for specific session
        
        Args:
            data: OHLCV DataFrame
            session_name: Name of the session
            
        Returns:
            Filtered DataFrame for the session
        """
        session_mask = data.index.to_series().apply(
            lambda x: self.is_in_session(x, session_name)
        )
        return data[session_mask]


def calculate_pip_value(symbol: str, price: float) -> float:
    """
    Calculate pip value for different currency pairs
    
    Args:
        symbol: Currency pair symbol
        price: Current price
        
    Returns:
        Pip value
    """
    # Simplified pip calculation
    if 'JPY' in symbol.upper():
        return 0.01  # JPY pairs
    else:
        return 0.0001  # Major pairs
    

def price_to_pips(price_diff: float, symbol: str) -> float:
    """
    Convert price difference to pips
    
    Args:
        price_diff: Price difference
        symbol: Currency pair symbol
        
    Returns:
        Difference in pips
    """
    pip_value = calculate_pip_value(symbol, 0)
    return price_diff / pip_value