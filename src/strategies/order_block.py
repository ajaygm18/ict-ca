"""
ORDER BLOCK Strategy - ICT Trading Framework

Setup: Identify bullish/bearish OB (last down candle before up move or last up candle before down move)
Entry: On retest of OB
Stop-Loss: Beyond OB wick
Take-Profit: Liquidity or opposite OB
Filter: Only valid if OB causes market structure shift
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from ..strategies.base_strategy import BaseICTStrategy, Signal, SignalType, TradeDirection
from ..utils.ict_patterns import ICTPatternDetector, OBType, OrderBlock
from ..utils.market_data import SessionDetector


class OrderBlockStrategy(BaseICTStrategy):
    """
    Order Block Strategy Implementation
    
    This strategy identifies Order Blocks and enters trades on retests
    of these institutional levels with proper confirmation.
    """
    
    def _set_default_parameters(self):
        """Set Order Block specific parameters"""
        self.parameters.setdefault('ob_lookback_periods', 30)
        self.parameters.setdefault('max_ob_age_candles', 50)
        self.parameters.setdefault('min_structure_shift_pips', 10)
        self.parameters.setdefault('retest_confirmation_candles', 2)
        self.parameters.setdefault('min_ob_size_pips', 8)
        self.parameters.setdefault('max_distance_from_ob_pips', 20)
        self.parameters.setdefault('require_structure_shift', True)
        
        # Initialize components
        self.pattern_detector = ICTPatternDetector()
        self.session_detector = SessionDetector()
        self.active_order_blocks = []  # Track active untested OBs
    
    def check_entry_conditions(self, data: pd.DataFrame, index: int) -> Optional[Signal]:
        """
        Check Order Block entry conditions at specific index
        
        Args:
            data: OHLCV price data
            index: Current data index
            
        Returns:
            Trading signal if conditions met, None otherwise
        """
        if index < self.parameters['ob_lookback_periods']:
            return None
        
        current_time = data.index[index]
        current_candle = data.iloc[index]
        
        # Update active Order Blocks list
        self._update_active_order_blocks(data, index)
        
        # Filter 1: Check for retest of active Order Block
        ob_retest = self._check_order_block_retest(current_candle, index)
        
        if not ob_retest:
            return None
        
        target_ob, retest_type = ob_retest
        
        # Filter 2: Confirm structure shift (if required)
        if self.parameters['require_structure_shift']:
            if not self._confirm_structure_shift(data, target_ob, index):
                return None
        
        # Filter 3: Check for retest confirmation
        if not self._confirm_retest_signal(data, index, target_ob.ob_type):
            return None
        
        # Calculate entry levels
        entry_price = current_candle['close']
        stop_loss = self._calculate_ob_stop_loss(target_ob)
        take_profit = self._calculate_ob_take_profit(
            data, index, entry_price, stop_loss, target_ob.ob_type
        )
        
        trade_direction = (TradeDirection.LONG if target_ob.ob_type == OBType.BULLISH 
                          else TradeDirection.SHORT)
        signal_type = SignalType.BUY if trade_direction == TradeDirection.LONG else SignalType.SELL
        
        return Signal(
            timestamp=current_time,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.85,
            reason=f"Order Block: {target_ob.ob_type.name} OB retest entry",
            metadata={
                'ob_type': target_ob.ob_type.name,
                'ob_high': target_ob.high,
                'ob_low': target_ob.low,
                'ob_age': index - target_ob.index,
                'retest_type': retest_type
            }
        )
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate all Order Block signals for the dataset"""
        signals = []
        self.active_order_blocks = []  # Reset active OBs
        
        for i in range(len(data)):
            signal = self.check_entry_conditions(data, i)
            if signal:
                signals.append(signal)
        
        self.signals = signals
        return signals
    
    def _update_active_order_blocks(self, data: pd.DataFrame, index: int):
        """Update the list of active (untested) Order Blocks"""
        # Add new Order Blocks
        lookback = min(self.parameters['ob_lookback_periods'], index)
        
        if lookback > 10:
            analysis_data = data.iloc[index-lookback:index+1]
            new_obs = self.pattern_detector.detect_order_blocks(analysis_data)
            
            for ob in new_obs:
                # Adjust index to global data frame
                ob.index += (index - lookback)
                
                # Filter by minimum size
                ob_size_pips = (ob.high - ob.low) / 0.0001
                if ob_size_pips >= self.parameters['min_ob_size_pips']:
                    self.active_order_blocks.append(ob)
        
        # Remove old or tested Order Blocks
        max_age = self.parameters['max_ob_age_candles']
        current_high = data.iloc[index]['high']
        current_low = data.iloc[index]['low']
        
        self.active_order_blocks = [
            ob for ob in self.active_order_blocks
            if (index - ob.index <= max_age and  # Not too old
                not ob.retested)  # Not yet retested
        ]
        
        # Mark Order Blocks as retested if price has been through them
        for ob in self.active_order_blocks:
            if not ob.retested:
                if (current_low <= ob.high and current_high >= ob.low):
                    ob.retested = True
                    ob.retest_index = index
    
    def _check_order_block_retest(self, current_candle, index: int) -> Optional[tuple]:
        """
        Check if current price is retesting an active Order Block
        
        Args:
            current_candle: Current price candle
            index: Current index
            
        Returns:
            Tuple of (target_ob, retest_type) if retest detected
        """
        current_high = current_candle['high']
        current_low = current_candle['low']
        current_close = current_candle['close']
        
        max_distance = self.parameters['max_distance_from_ob_pips'] * 0.0001
        
        for ob in self.active_order_blocks:
            if ob.retested:
                continue
            
            # Check if current candle is interacting with Order Block
            if (current_low <= ob.high and current_high >= ob.low):
                
                # Determine if this is a valid retest based on OB type
                if ob.ob_type == OBType.BULLISH:
                    # Bullish OB: look for bounce from support level
                    if (current_low <= ob.high and 
                        current_close >= ob.low and
                        abs(current_low - ob.low) <= max_distance):
                        return (ob, 'bullish_retest')
                
                else:  # Bearish OB
                    # Bearish OB: look for rejection from resistance level
                    if (current_high >= ob.low and 
                        current_close <= ob.high and
                        abs(current_high - ob.high) <= max_distance):
                        return (ob, 'bearish_retest')
        
        return None
    
    def _confirm_structure_shift(self, data: pd.DataFrame, order_block: OrderBlock, 
                               current_index: int) -> bool:
        """
        Confirm that the Order Block caused a market structure shift
        
        Args:
            data: OHLCV DataFrame
            order_block: The Order Block to verify
            current_index: Current index
            
        Returns:
            True if structure shift is confirmed
        """
        ob_index = order_block.index
        min_shift_pips = self.parameters['min_structure_shift_pips'] * 0.0001
        
        # Look at price action after the Order Block formation
        post_ob_data = data.iloc[ob_index:min(ob_index + 20, current_index)]
        
        if len(post_ob_data) < 5:
            return False
        
        if order_block.ob_type == OBType.BULLISH:
            # Check if price moved significantly higher after bullish OB
            max_high_after = post_ob_data['high'].max()
            structure_shift = max_high_after > order_block.high + min_shift_pips
            return structure_shift
        
        else:  # Bearish OB
            # Check if price moved significantly lower after bearish OB
            min_low_after = post_ob_data['low'].min()
            structure_shift = min_low_after < order_block.low - min_shift_pips
            return structure_shift
    
    def _confirm_retest_signal(self, data: pd.DataFrame, index: int, 
                             ob_type: OBType) -> bool:
        """
        Confirm that the retest provides a valid entry signal
        
        Args:
            data: OHLCV DataFrame
            index: Current index
            ob_type: Type of Order Block being retested
            
        Returns:
            True if retest signal is confirmed
        """
        confirmation_candles = self.parameters['retest_confirmation_candles']
        
        if index < confirmation_candles:
            return False
        
        recent_candles = data.iloc[index-confirmation_candles+1:index+1]
        
        if ob_type == OBType.BULLISH:
            # For bullish OB retest, look for buying pressure
            bullish_closes = (recent_candles['close'] > recent_candles['open']).sum()
            bullish_bias = (recent_candles['close'] > recent_candles['low']).sum()
            
            return (bullish_closes >= confirmation_candles // 2 and 
                   bullish_bias >= confirmation_candles)
        
        else:  # Bearish OB
            # For bearish OB retest, look for selling pressure
            bearish_closes = (recent_candles['close'] < recent_candles['open']).sum()
            bearish_bias = (recent_candles['close'] < recent_candles['high']).sum()
            
            return (bearish_closes >= confirmation_candles // 2 and 
                   bearish_bias >= confirmation_candles)
    
    def _calculate_ob_stop_loss(self, order_block: OrderBlock) -> float:
        """Calculate stop loss beyond Order Block wick"""
        buffer_pips = 5 * 0.0001  # 5 pip buffer
        
        if order_block.ob_type == OBType.BULLISH:
            # For long trades, stop below OB low
            return order_block.low - buffer_pips
        else:
            # For short trades, stop above OB high
            return order_block.high + buffer_pips
    
    def _calculate_ob_take_profit(self, data: pd.DataFrame, index: int, 
                                entry_price: float, stop_loss: float, 
                                ob_type: OBType) -> float:
        """Calculate take profit targeting liquidity or opposite OB"""
        risk = abs(entry_price - stop_loss)
        default_target = (entry_price + (risk * self.risk_reward_ratio) 
                         if ob_type == OBType.BULLISH 
                         else entry_price - (risk * self.risk_reward_ratio))
        
        # Look for liquidity targets or opposite Order Blocks
        try:
            lookback = min(100, index)
            analysis_data = data.iloc[index-lookback:index+1]
            
            # Find swing levels as liquidity targets
            if ob_type == OBType.BULLISH:
                # For long trades, target recent swing highs
                swing_highs = []
                for i in range(5, len(analysis_data) - 5):
                    if self._is_swing_high(analysis_data, i):
                        swing_high = analysis_data.iloc[i]['high']
                        if swing_high > entry_price:
                            swing_highs.append(swing_high)
                
                if swing_highs:
                    nearest_target = min(swing_highs)
                    return min(nearest_target, default_target * 2.0)
            
            else:  # Bearish OB
                # For short trades, target recent swing lows
                swing_lows = []
                for i in range(5, len(analysis_data) - 5):
                    if self._is_swing_low(analysis_data, i):
                        swing_low = analysis_data.iloc[i]['low']
                        if swing_low < entry_price:
                            swing_lows.append(swing_low)
                
                if swing_lows:
                    nearest_target = max(swing_lows)
                    return max(nearest_target, default_target * 2.0)
            
            # Look for opposite Order Blocks
            order_blocks = self.pattern_detector.detect_order_blocks(analysis_data)
            
            for other_ob in reversed(order_blocks):
                if ob_type == OBType.BULLISH and other_ob.ob_type == OBType.BEARISH:
                    if other_ob.low > entry_price:
                        return min(other_ob.low, default_target * 1.5)
                elif ob_type == OBType.BEARISH and other_ob.ob_type == OBType.BULLISH:
                    if other_ob.high < entry_price:
                        return max(other_ob.high, default_target * 1.5)
        
        except Exception:
            # Fall back to default if analysis fails
            pass
        
        return default_target
    
    def _is_swing_high(self, data: pd.DataFrame, index: int) -> bool:
        """Check if index represents a swing high"""
        if index < 2 or index >= len(data) - 2:
            return False
        
        current_high = data.iloc[index]['high']
        return (current_high > data.iloc[index-1]['high'] and
                current_high > data.iloc[index+1]['high'] and
                current_high > data.iloc[index-2]['high'] and
                current_high > data.iloc[index+2]['high'])
    
    def _is_swing_low(self, data: pd.DataFrame, index: int) -> bool:
        """Check if index represents a swing low"""
        if index < 2 or index >= len(data) - 2:
            return False
        
        current_low = data.iloc[index]['low']
        return (current_low < data.iloc[index-1]['low'] and
                current_low < data.iloc[index+1]['low'] and
                current_low < data.iloc[index-2]['low'] and
                current_low < data.iloc[index+2]['low'])