"""
Turtle Soup Strategy - Strategy 10

The Turtle Soup strategy is designed to fade breakouts that fail, capitalizing on
false breakouts beyond 20-day highs or lows. This strategy identifies when price
breaks beyond recent highs/lows but then quickly reverses, trapping breakout traders.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from .base_strategy import BaseICTStrategy, Signal, TradeDirection

class TurtleSoupStrategy(BaseICTStrategy):
    """
    Turtle Soup Strategy Implementation
    
    This strategy identifies failed breakouts beyond 20-day highs/lows and trades
    the reversal, capturing trapped breakout traders and institutional fading.
    
    Entry Conditions:
    - Price breaks beyond 20-day high/low
    - Reversal occurs within specified time window (usually same day)
    - Volume confirms the fake-out pattern
    - Price returns inside the previous range
    
    Exit Conditions:
    - Price reaches opposite end of the range
    - Stop loss beyond the false breakout level
    - Time-based exit if no movement within session
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lookback_period = config.get('lookback_period', 20)
        self.reversal_window = config.get('reversal_window', 4)  # hours for reversal
        self.min_breakout_size = config.get('min_breakout_size', 0.001)  # minimum breakout %
        self.volume_confirmation = config.get('volume_confirmation', True)
        self.same_day_only = config.get('same_day_only', True)
        
    def setup_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Setup indicators for Turtle Soup pattern detection"""
        df = data.copy()
        
        # Calculate rolling highs and lows
        df['high_20'] = df['high'].rolling(self.lookback_period).max()
        df['low_20'] = df['low'].rolling(self.lookback_period).min()
        
        # Shift to get previous period's levels
        df['prev_high_20'] = df['high_20'].shift(1)
        df['prev_low_20'] = df['low_20'].shift(1)
        
        # Identify breakouts
        df['breakout_high'] = df['high'] > df['prev_high_20']
        df['breakout_low'] = df['low'] < df['prev_low_20']
        
        # Calculate range and volatility
        df['range_size'] = df['prev_high_20'] - df['prev_low_20']
        df['atr'] = self._calculate_atr(df, 14)
        df['volatility_adj'] = df['atr'] / df['close']
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['high_volume'] = df['volume_ratio'] > 1.5
        
        # Session and time analysis
        df['session'] = self._identify_session(df)
        df['hour'] = pd.to_datetime(df.index).hour
        df['date'] = pd.to_datetime(df.index).date
        
        return df
        
    def identify_entry_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Identify Turtle Soup entry opportunities"""
        signals = []
        
        if len(data) < self.lookback_period + 10:
            return signals
            
        i = self.lookback_period
        while i < len(data) - self.reversal_window:
            current_bar = data.iloc[i]
            
            # Check for breakout
            if current_bar['breakout_high'] or current_bar['breakout_low']:
                signal = self._check_turtle_soup_setup(data, i)
                if signal:
                    signals.append(signal)
                    # Skip ahead to avoid overlapping signals
                    i += self.reversal_window
                    continue
                    
            i += 1
            
        return signals
        
    def _check_turtle_soup_setup(self, data: pd.DataFrame, breakout_idx: int) -> Optional[Signal]:
        """Check if a breakout qualifies as a Turtle Soup setup"""
        breakout_bar = data.iloc[breakout_idx]
        prev_high = breakout_bar['prev_high_20']
        prev_low = breakout_bar['prev_low_20']
        
        # Determine breakout direction
        is_upside_breakout = breakout_bar['breakout_high']
        is_downside_breakout = breakout_bar['breakout_low']
        
        if not (is_upside_breakout or is_downside_breakout):
            return None
            
        # Calculate breakout size
        if is_upside_breakout:
            breakout_size = (breakout_bar['high'] - prev_high) / prev_high
            breakout_level = breakout_bar['high']
        else:
            breakout_size = (prev_low - breakout_bar['low']) / prev_low
            breakout_level = breakout_bar['low']
            
        # Check minimum breakout size
        if breakout_size < self.min_breakout_size:
            return None
            
        # Look for reversal within the window
        reversal_idx = self._find_reversal(data, breakout_idx, is_upside_breakout)
        if reversal_idx is None:
            return None
            
        reversal_bar = data.iloc[reversal_idx]
        
        # Check same day requirement
        if self.same_day_only:
            breakout_date = data.index[breakout_idx].date()
            reversal_date = data.index[reversal_idx].date()
            if breakout_date != reversal_date:
                return None
                
        # Volume confirmation
        if self.volume_confirmation:
            if not (breakout_bar['high_volume'] or reversal_bar['high_volume']):
                return None
                
        # Setup trade parameters
        if is_upside_breakout:
            # Failed upside breakout - go short
            direction = TradeDirection.SHORT
            entry_price = reversal_bar['close']
            stop_loss = breakout_level + (breakout_level * 0.002)  # Small buffer above breakout
            take_profit = prev_low  # Target bottom of range
            
        else:
            # Failed downside breakout - go long
            direction = TradeDirection.LONG
            entry_price = reversal_bar['close']
            stop_loss = breakout_level - (breakout_level * 0.002)  # Small buffer below breakout
            take_profit = prev_high  # Target top of range
            
        # Validate risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward / risk < 1.0:  # Minimum 1:1 R:R
            return None
            
        return Signal(
            timestamp=reversal_bar.name,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.7,
            strategy="TURTLE_SOUP",
            metadata={
                'breakout_level': breakout_level,
                'breakout_size': breakout_size,
                'range_high': prev_high,
                'range_low': prev_low,
                'volume_ratio': reversal_bar['volume_ratio'],
                'session': reversal_bar['session'],
                'reversal_candles': reversal_idx - breakout_idx
            }
        )
        
    def _find_reversal(self, data: pd.DataFrame, breakout_idx: int, is_upside_breakout: bool) -> Optional[int]:
        """Find reversal within the specified window"""
        end_idx = min(breakout_idx + self.reversal_window, len(data))
        
        if is_upside_breakout:
            # Look for price returning below the previous 20-day high
            prev_high = data.iloc[breakout_idx]['prev_high_20']
            
            for i in range(breakout_idx + 1, end_idx):
                if data.iloc[i]['close'] < prev_high:
                    # Confirm with bearish price action
                    if self._confirm_bearish_reversal(data, i):
                        return i
                        
        else:
            # Look for price returning above the previous 20-day low
            prev_low = data.iloc[breakout_idx]['prev_low_20']
            
            for i in range(breakout_idx + 1, end_idx):
                if data.iloc[i]['close'] > prev_low:
                    # Confirm with bullish price action
                    if self._confirm_bullish_reversal(data, i):
                        return i
                        
        return None
        
    def _confirm_bearish_reversal(self, data: pd.DataFrame, idx: int) -> bool:
        """Confirm bearish reversal pattern"""
        if idx < 1:
            return False
            
        current = data.iloc[idx]
        previous = data.iloc[idx-1]
        
        # Look for bearish engulfing or strong bearish candle
        bearish_candle = current['close'] < current['open']
        strong_decline = current['close'] < previous['low']
        high_volume = current['volume_ratio'] > 1.2
        
        return bearish_candle and (strong_decline or high_volume)
        
    def _confirm_bullish_reversal(self, data: pd.DataFrame, idx: int) -> bool:
        """Confirm bullish reversal pattern"""
        if idx < 1:
            return False
            
        current = data.iloc[idx]
        previous = data.iloc[idx-1]
        
        # Look for bullish engulfing or strong bullish candle
        bullish_candle = current['close'] > current['open']
        strong_advance = current['close'] > previous['high']
        high_volume = current['volume_ratio'] > 1.2
        
        return bullish_candle and (strong_advance or high_volume)
        
    def identify_exit_signals(self, data: pd.DataFrame, open_trades: List) -> List[Dict]:
        """Identify exit signals for open Turtle Soup trades"""
        exit_signals = []
        
        for trade in open_trades:
            current_bar = data.iloc[-1]
            current_price = current_bar['close']
            
            # Check for time-based exit (end of session)
            if self._should_exit_on_time(current_bar, trade):
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'time_exit',
                    'timestamp': current_bar.name
                })
                continue
                
            # Check for breakout level retest (failed turtle soup)
            if self._check_level_retest(current_bar, trade):
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'level_retest',
                    'timestamp': current_bar.name
                })
                
        return exit_signals
        
    def _should_exit_on_time(self, current_bar: pd.Series, trade: Dict) -> bool:
        """Check if trade should be exited based on time"""
        # Exit at end of session if not much progress
        if current_bar['session'] in ['Close', 'Asian']:
            return True
            
        # Exit if in trade for more than 8 hours
        entry_time = pd.to_datetime(trade['timestamp'])
        current_time = pd.to_datetime(current_bar.name)
        hours_in_trade = (current_time - entry_time).total_seconds() / 3600
        
        return hours_in_trade > 8
        
    def _check_level_retest(self, current_bar: pd.Series, trade: Dict) -> bool:
        """Check if price is retesting the breakout level"""
        breakout_level = trade['metadata']['breakout_level']
        current_price = current_bar['close']
        
        # If price moves back to breakout level, exit (failed turtle soup)
        distance_to_level = abs(current_price - breakout_level) / breakout_level
        
        return distance_to_level < 0.001  # Within 0.1% of breakout level
        
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information"""
        return {
            'name': 'Turtle Soup',
            'description': 'Fades failed breakouts beyond 20-day highs/lows',
            'timeframes': ['1h', '4h'],
            'sessions': ['London', 'New York'],
            'risk_level': 'Medium',
            'parameters': {
                'lookback_period': self.lookback_period,
                'reversal_window': self.reversal_window,
                'min_breakout_size': self.min_breakout_size,
                'volume_confirmation': self.volume_confirmation,
                'same_day_only': self.same_day_only
            }
        }