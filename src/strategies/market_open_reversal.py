"""
MARKET OPEN REVERSAL Strategy - ICT Trading Framework

Session: First 30 mins after NY open (9:30–10:00 EST)
Setup: Look for false breakout of pre-market high/low
Entry: Enter opposite direction after breakout fails + strong rejection candle
Stop-Loss: Beyond fakeout wick
Take-Profit: FVG or OB target
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from ..strategies.base_strategy import BaseICTStrategy, Signal, SignalType, TradeDirection
from ..utils.ict_patterns import ICTPatternDetector
from ..utils.market_data import SessionDetector


class MarketOpenReversalStrategy(BaseICTStrategy):
    """
    Market Open Reversal Strategy Implementation
    
    This strategy targets false breakouts during the first 30 minutes of NY open,
    entering reversal trades when breakouts fail with strong rejection signals.
    """
    
    def _set_default_parameters(self):
        """Set Market Open Reversal specific parameters"""
        self.parameters.setdefault('entry_session', 'ny_open')
        self.parameters.setdefault('pre_market_session', 'pre_market')
        self.parameters.setdefault('min_rejection_body_percent', 0.6)  # 60% rejection of candle body
        self.parameters.setdefault('fakeout_confirmation_pips', 5)
        self.parameters.setdefault('min_wick_size_pips', 8)
        self.parameters.setdefault('max_time_in_breakout_minutes', 15)
        self.parameters.setdefault('rejection_candle_lookback', 3)
        
        # Initialize components
        self.session_detector = SessionDetector()
        self.pattern_detector = ICTPatternDetector()
        self.daily_ranges = {}
    
    def check_entry_conditions(self, data: pd.DataFrame, index: int) -> Optional[Signal]:
        """
        Check Market Open Reversal entry conditions at specific index
        
        Args:
            data: OHLCV price data
            index: Current data index
            
        Returns:
            Trading signal if conditions met, None otherwise
        """
        if index < 10:
            return None
        
        current_time = data.index[index]
        
        # Filter 1: Must be in NY open session
        if not self.session_detector.is_in_session(current_time, self.parameters['entry_session']):
            return None
        
        # Filter 2: Must have pre-market range for reference
        current_date = current_time.date()
        pre_market_range = self._get_pre_market_range(data, current_date)
        
        if not pre_market_range:
            return None
        
        range_high, range_low = pre_market_range
        
        # Filter 3: Check for false breakout pattern
        false_breakout = self._detect_false_breakout(data, index, range_high, range_low)
        
        if not false_breakout:
            return None
        
        breakout_direction, fakeout_high, fakeout_low = false_breakout
        
        # Filter 4: Check for strong rejection candle
        rejection_signal = self._detect_rejection_candle(data, index, breakout_direction)
        
        if not rejection_signal:
            return None
        
        # Determine trade direction (opposite of false breakout)
        trade_direction = TradeDirection.SHORT if breakout_direction == 'up' else TradeDirection.LONG
        
        # Calculate entry levels
        entry_price = data.iloc[index]['close']
        stop_loss = self._calculate_reversal_stop_loss(
            fakeout_high, fakeout_low, breakout_direction
        )
        take_profit = self._calculate_reversal_take_profit(
            data, index, entry_price, stop_loss, trade_direction
        )
        
        signal_type = SignalType.BUY if trade_direction == TradeDirection.LONG else SignalType.SELL
        
        return Signal(
            timestamp=current_time,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.8,
            reason=f"Market Open Reversal: False {breakout_direction} breakout + strong rejection",
            metadata={
                'false_breakout_direction': breakout_direction,
                'fakeout_high': fakeout_high,
                'fakeout_low': fakeout_low,
                'pre_market_high': range_high,
                'pre_market_low': range_low,
                'session': self.parameters['entry_session']
            }
        )
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate all Market Open Reversal signals for the dataset"""
        signals = []
        
        # First pass: identify all pre-market ranges
        self._identify_pre_market_ranges(data)
        
        # Second pass: check for reversal entries
        for i in range(len(data)):
            signal = self.check_entry_conditions(data, i)
            if signal:
                signals.append(signal)
        
        self.signals = signals
        return signals
    
    def _identify_pre_market_ranges(self, data: pd.DataFrame):
        """Identify pre-market ranges for each day"""
        self.daily_ranges = {}
        
        for date in data.index.date:
            if date not in self.daily_ranges:
                pre_market_data = self._get_pre_market_data(data, date)
                if len(pre_market_data) > 0:
                    range_high = pre_market_data['high'].max()
                    range_low = pre_market_data['low'].min()
                    
                    self.daily_ranges[date] = {
                        'high': range_high,
                        'low': range_low
                    }
    
    def _get_pre_market_data(self, data: pd.DataFrame, target_date) -> pd.DataFrame:
        """Get pre-market session data for specific date"""
        day_data = data[data.index.date == target_date]
        
        pre_market_mask = day_data.index.to_series().apply(
            lambda x: self.session_detector.is_in_session(x, self.parameters['pre_market_session'])
        )
        
        return day_data[pre_market_mask]
    
    def _get_pre_market_range(self, data: pd.DataFrame, current_date) -> Optional[Tuple[float, float]]:
        """Get pre-market range for current date"""
        if current_date in self.daily_ranges:
            range_info = self.daily_ranges[current_date]
            return (range_info['high'], range_info['low'])
        
        # Fallback: calculate range on the fly
        pre_market_data = self._get_pre_market_data(data, current_date)
        if len(pre_market_data) > 0:
            range_high = pre_market_data['high'].max()
            range_low = pre_market_data['low'].min()
            return (range_high, range_low)
        
        return None
    
    def _detect_false_breakout(self, data: pd.DataFrame, index: int, 
                              range_high: float, range_low: float) -> Optional[Tuple[str, float, float]]:
        """
        Detect false breakout of pre-market range
        
        Args:
            data: OHLCV DataFrame
            index: Current index
            range_high: Pre-market high
            range_low: Pre-market low
            
        Returns:
            Tuple of (breakout_direction, fakeout_high, fakeout_low) if detected
        """
        confirmation_pips = self.parameters['fakeout_confirmation_pips'] * 0.0001
        max_time_minutes = self.parameters['max_time_in_breakout_minutes']
        
        # Look back for recent breakout attempt
        lookback_periods = min(max_time_minutes, index)
        
        for i in range(1, lookback_periods + 1):
            past_index = index - i
            past_candle = data.iloc[past_index]
            current_candle = data.iloc[index]
            
            # Check for failed upside breakout
            if (past_candle['high'] > range_high + confirmation_pips and
                current_candle['close'] < range_high):
                
                # Find the extent of the fakeout move
                fakeout_high = max(data.iloc[past_index:index+1]['high'])
                fakeout_low = min(data.iloc[past_index:index+1]['low'])
                
                return ('up', fakeout_high, fakeout_low)
            
            # Check for failed downside breakout
            elif (past_candle['low'] < range_low - confirmation_pips and
                  current_candle['close'] > range_low):
                
                # Find the extent of the fakeout move
                fakeout_high = max(data.iloc[past_index:index+1]['high'])
                fakeout_low = min(data.iloc[past_index:index+1]['low'])
                
                return ('down', fakeout_high, fakeout_low)
        
        return None
    
    def _detect_rejection_candle(self, data: pd.DataFrame, index: int, 
                               breakout_direction: str) -> bool:
        """
        Detect strong rejection candle pattern
        
        Args:
            data: OHLCV DataFrame
            index: Current index
            breakout_direction: Direction of the false breakout
            
        Returns:
            True if strong rejection detected
        """
        lookback = self.parameters['rejection_candle_lookback']
        
        for i in range(max(0, lookback)):
            candle_index = index - i
            candle = data.iloc[candle_index]
            
            body_size = abs(candle['close'] - candle['open'])
            total_range = candle['high'] - candle['low']
            
            if total_range == 0:
                continue
            
            if breakout_direction == 'up':
                # Look for bearish rejection (long upper wick)
                upper_wick = candle['high'] - max(candle['open'], candle['close'])
                wick_size_pips = upper_wick / 0.0001
                
                # Check rejection criteria
                if (candle['close'] < candle['open'] and  # Bearish candle
                    upper_wick / total_range > self.parameters['min_rejection_body_percent'] and
                    wick_size_pips >= self.parameters['min_wick_size_pips']):
                    return True
                    
            else:
                # Look for bullish rejection (long lower wick)
                lower_wick = min(candle['open'], candle['close']) - candle['low']
                wick_size_pips = lower_wick / 0.0001
                
                # Check rejection criteria
                if (candle['close'] > candle['open'] and  # Bullish candle
                    lower_wick / total_range > self.parameters['min_rejection_body_percent'] and
                    wick_size_pips >= self.parameters['min_wick_size_pips']):
                    return True
        
        return False
    
    def _calculate_reversal_stop_loss(self, fakeout_high: float, fakeout_low: float, 
                                    breakout_direction: str) -> float:
        """Calculate stop loss beyond the fakeout wick"""
        buffer_pips = 3 * 0.0001  # 3 pip buffer
        
        if breakout_direction == 'up':
            # For short trades, stop above fakeout high
            return fakeout_high + buffer_pips
        else:
            # For long trades, stop below fakeout low
            return fakeout_low - buffer_pips
    
    def _calculate_reversal_take_profit(self, data: pd.DataFrame, index: int, 
                                      entry_price: float, stop_loss: float, 
                                      trade_direction: TradeDirection) -> float:
        """Calculate take profit targeting FVG or OB levels"""
        # Default risk-reward target
        risk = abs(entry_price - stop_loss)
        default_target = (entry_price + (risk * self.risk_reward_ratio) 
                         if trade_direction == TradeDirection.LONG 
                         else entry_price - (risk * self.risk_reward_ratio))
        
        # Look for ICT targets (FVG or Order Blocks)
        try:
            # Check for nearby Fair Value Gaps
            fvgs = self.pattern_detector.detect_fair_value_gaps(
                data.iloc[max(0, index-50):index+1]
            )
            
            if fvgs:
                for fvg in reversed(fvgs):  # Check most recent first
                    if trade_direction == TradeDirection.LONG:
                        if fvg.top > entry_price and fvg.fvg_type.value == 1:  # Bullish FVG above
                            return min(fvg.bottom, default_target * 1.5)
                    else:
                        if fvg.bottom < entry_price and fvg.fvg_type.value == -1:  # Bearish FVG below
                            return max(fvg.top, default_target * 1.5)
            
            # Check for Order Blocks
            order_blocks = self.pattern_detector.detect_order_blocks(
                data.iloc[max(0, index-30):index+1]
            )
            
            if order_blocks:
                for ob in reversed(order_blocks):  # Check most recent first
                    if trade_direction == TradeDirection.LONG:
                        if ob.low > entry_price and ob.ob_type.value == 1:  # Bullish OB above
                            return min(ob.low, default_target * 1.2)
                    else:
                        if ob.high < entry_price and ob.ob_type.value == -1:  # Bearish OB below
                            return max(ob.high, default_target * 1.2)
        
        except Exception:
            # Fall back to default if pattern detection fails
            pass
        
        return default_target
    
    def _check_time_filter(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within NY open session"""
        return self.session_detector.is_in_session(timestamp, self.parameters['entry_session'])