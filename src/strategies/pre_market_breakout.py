"""
PRE-MARKET BREAKOUT Strategy - ICT Trading Framework

Session: Define pre-market range (2:00–7:00 EST)
Setup: Detect consolidation range
Entry: Breakout trade at NY open (9:30 EST)
Stop-Loss: Opposite side of range
Take-Profit: 2× risk or next liquidity
Filter: Avoid trades if ATR < threshold (low volatility)
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from ..strategies.base_strategy import BaseICTStrategy, Signal, SignalType, TradeDirection
from ..utils.market_data import SessionDetector


class PreMarketBreakoutStrategy(BaseICTStrategy):
    """
    Pre-Market Breakout Strategy Implementation
    
    This strategy identifies consolidation ranges during pre-market hours
    and trades breakouts at the NY market open with proper risk management.
    """
    
    def _set_default_parameters(self):
        """Set Pre-Market Breakout specific parameters"""
        self.parameters.setdefault('pre_market_session', 'pre_market')
        self.parameters.setdefault('entry_session', 'ny_open')
        self.parameters.setdefault('min_range_pips', 10)
        self.parameters.setdefault('max_range_pips', 50)
        self.parameters.setdefault('min_atr_threshold', 0.0010)
        self.parameters.setdefault('breakout_confirmation_pips', 3)
        self.parameters.setdefault('range_lookback_hours', 5)  # Look back 5 hours for range
        
        # Initialize components
        self.session_detector = SessionDetector()
        self.daily_ranges = {}  # Store daily pre-market ranges
    
    def check_entry_conditions(self, data: pd.DataFrame, index: int) -> Optional[Signal]:
        """
        Check Pre-Market Breakout entry conditions at specific index
        
        Args:
            data: OHLCV price data
            index: Current data index
            
        Returns:
            Trading signal if conditions met, None otherwise
        """
        current_time = data.index[index]
        
        # Filter 1: Must be at NY open session for entry
        if not self.session_detector.is_in_session(current_time, self.parameters['entry_session']):
            return None
        
        # Filter 2: Must have valid pre-market range for today
        current_date = current_time.date()
        pre_market_range = self._get_pre_market_range(data, index, current_date)
        
        if not pre_market_range:
            return None
        
        range_high, range_low, range_size = pre_market_range
        
        # Filter 3: Range size must be within acceptable limits
        range_pips = range_size / 0.0001
        if (range_pips < self.parameters['min_range_pips'] or 
            range_pips > self.parameters['max_range_pips']):
            return None
        
        # Filter 4: Check ATR volatility filter
        if not self._check_volatility_filter(data, index):
            return None
        
        # Filter 5: Check for breakout
        current_price = data.iloc[index]['close']
        current_high = data.iloc[index]['high']
        current_low = data.iloc[index]['low']
        
        breakout_buffer = self.parameters['breakout_confirmation_pips'] * 0.0001
        
        breakout_direction = None
        entry_price = current_price
        
        # Check for bullish breakout
        if current_high > range_high + breakout_buffer:
            breakout_direction = 'up'
            entry_price = range_high + breakout_buffer
        
        # Check for bearish breakout
        elif current_low < range_low - breakout_buffer:
            breakout_direction = 'down'
            entry_price = range_low - breakout_buffer
        
        if not breakout_direction:
            return None
        
        # Calculate stop loss and take profit
        stop_loss = self._calculate_breakout_stop_loss(
            range_high, range_low, breakout_direction
        )
        take_profit = self._calculate_breakout_take_profit(
            data, index, entry_price, stop_loss, breakout_direction
        )
        
        trade_direction = TradeDirection.LONG if breakout_direction == 'up' else TradeDirection.SHORT
        signal_type = SignalType.BUY if trade_direction == TradeDirection.LONG else SignalType.SELL
        
        return Signal(
            timestamp=current_time,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.75,
            reason=f"Pre-Market Breakout: {breakout_direction} breakout of {range_pips:.1f} pip range",
            metadata={
                'breakout_direction': breakout_direction,
                'range_high': range_high,
                'range_low': range_low,
                'range_size_pips': range_pips,
                'pre_market_session': self.parameters['pre_market_session']
            }
        )
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate all Pre-Market Breakout signals for the dataset"""
        signals = []
        
        # First pass: identify all pre-market ranges
        self._identify_pre_market_ranges(data)
        
        # Second pass: check for breakout entries
        for i in range(len(data)):
            signal = self.check_entry_conditions(data, i)
            if signal:
                signals.append(signal)
        
        self.signals = signals
        return signals
    
    def _identify_pre_market_ranges(self, data: pd.DataFrame):
        """Identify pre-market consolidation ranges for each day"""
        self.daily_ranges = {}
        
        for date in data.index.date:
            if date not in self.daily_ranges:
                pre_market_data = self._get_pre_market_data(data, date)
                if len(pre_market_data) > 0:
                    range_high = pre_market_data['high'].max()
                    range_low = pre_market_data['low'].min()
                    range_size = range_high - range_low
                    
                    self.daily_ranges[date] = {
                        'high': range_high,
                        'low': range_low,
                        'size': range_size
                    }
    
    def _get_pre_market_data(self, data: pd.DataFrame, target_date) -> pd.DataFrame:
        """Get pre-market session data for specific date"""
        day_data = data[data.index.date == target_date]
        
        pre_market_mask = day_data.index.to_series().apply(
            lambda x: self.session_detector.is_in_session(x, self.parameters['pre_market_session'])
        )
        
        return day_data[pre_market_mask]
    
    def _get_pre_market_range(self, data: pd.DataFrame, index: int, 
                             current_date) -> Optional[Tuple[float, float, float]]:
        """Get pre-market range for current date"""
        if current_date in self.daily_ranges:
            range_info = self.daily_ranges[current_date]
            return (range_info['high'], range_info['low'], range_info['size'])
        
        # Fallback: calculate range on the fly
        pre_market_data = self._get_pre_market_data(data, current_date)
        if len(pre_market_data) > 0:
            range_high = pre_market_data['high'].max()
            range_low = pre_market_data['low'].min()
            range_size = range_high - range_low
            return (range_high, range_low, range_size)
        
        return None
    
    def _calculate_breakout_stop_loss(self, range_high: float, range_low: float, 
                                    breakout_direction: str) -> float:
        """Calculate stop loss on opposite side of range"""
        if breakout_direction == 'up':
            # For long trades, stop loss below range low
            return range_low
        else:
            # For short trades, stop loss above range high
            return range_high
    
    def _calculate_breakout_take_profit(self, data: pd.DataFrame, index: int, 
                                      entry_price: float, stop_loss: float, 
                                      breakout_direction: str) -> float:
        """Calculate take profit: 2x risk or next liquidity level"""
        risk = abs(entry_price - stop_loss)
        default_target = entry_price + (risk * self.risk_reward_ratio) if breakout_direction == 'up' else entry_price - (risk * self.risk_reward_ratio)
        
        # Look for next liquidity level
        lookback = min(100, index)
        if lookback > 0:
            recent_data = data.iloc[index-lookback:index]
            
            if breakout_direction == 'up':
                # Find next resistance level
                resistance_levels = recent_data['high'].rolling(window=10).max()
                next_liquidity = resistance_levels.max()
                if next_liquidity > entry_price:
                    return min(next_liquidity, default_target * 1.5)  # Cap at 1.5x default
            else:
                # Find next support level
                support_levels = recent_data['low'].rolling(window=10).min()
                next_liquidity = support_levels.min()
                if next_liquidity < entry_price:
                    return max(next_liquidity, default_target * 1.5)  # Cap at 1.5x default
        
        return default_target
    
    def _check_time_filter(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within NY open session for entry"""
        return self.session_detector.is_in_session(timestamp, self.parameters['entry_session'])
    
    def _check_volatility_filter(self, data: pd.DataFrame, index: int) -> bool:
        """Enhanced volatility filter for pre-market breakout"""
        if 'atr' not in data.columns:
            return True
        
        current_atr = data.iloc[index]['atr']
        min_threshold = self.parameters.get('min_atr_threshold', 0.0010)
        
        return current_atr >= min_threshold