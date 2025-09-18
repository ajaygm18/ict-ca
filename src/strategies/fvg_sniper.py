"""
FVG SNIPER Strategy - ICT Trading Framework

Setup: Identify Fair Value Gaps (3-candle pattern: low of 1st > high of 3rd for bullish FVG)
Entry: Enter on retracement into FVG
Stop-Loss: Beyond FVG boundary
Take-Profit: Nearest liquidity pool or OB
Filter: Align with HTF bias (daily direction)
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from ..strategies.base_strategy import BaseICTStrategy, Signal, SignalType, TradeDirection
from ..utils.ict_patterns import ICTPatternDetector, FVGType
from ..utils.market_data import SessionDetector


class FVGSniperStrategy(BaseICTStrategy):
    """
    FVG Sniper Strategy Implementation
    
    This strategy identifies Fair Value Gaps and enters trades on retracements
    into these gaps, aligned with higher timeframe bias.
    """
    
    def _set_default_parameters(self):
        """Set FVG Sniper specific parameters"""
        self.parameters.setdefault('min_fvg_size_pips', 5)
        self.parameters.setdefault('max_fvg_age_candles', 20)
        self.parameters.setdefault('htf_lookback_periods', 100)
        self.parameters.setdefault('retracement_confirmation_periods', 3)
        self.parameters.setdefault('fvg_fill_threshold', 0.5)  # 50% fill threshold
        self.parameters.setdefault('require_htf_alignment', True)
        self.parameters.setdefault('min_liquidity_distance_pips', 15)
        
        # Initialize components
        self.pattern_detector = ICTPatternDetector()
        self.session_detector = SessionDetector()
        self.active_fvgs = []  # Track active unfilled FVGs
    
    def check_entry_conditions(self, data: pd.DataFrame, index: int) -> Optional[Signal]:
        """
        Check FVG Sniper entry conditions at specific index
        
        Args:
            data: OHLCV price data
            index: Current data index
            
        Returns:
            Trading signal if conditions met, None otherwise
        """
        if index < 10:
            return None
        
        current_time = data.index[index]
        current_candle = data.iloc[index]
        
        # Update active FVGs list
        self._update_active_fvgs(data, index)
        
        # Filter 1: Check for retracement into active FVG
        fvg_entry = self._check_fvg_retracement(current_candle, index)
        
        if not fvg_entry:
            return None
        
        target_fvg, entry_type = fvg_entry
        
        # Filter 2: Check Higher Timeframe bias alignment
        if self.parameters['require_htf_alignment']:
            htf_bias = self._determine_htf_bias(data, index)
            
            if not self._is_aligned_with_htf(target_fvg.fvg_type, htf_bias):
                return None
        
        # Filter 3: Check for retracement confirmation
        if not self._confirm_retracement_entry(data, index, target_fvg.fvg_type):
            return None
        
        # Calculate entry levels
        entry_price = current_candle['close']
        stop_loss = self._calculate_fvg_stop_loss(target_fvg, entry_type)
        take_profit = self._calculate_fvg_take_profit(
            data, index, entry_price, stop_loss, target_fvg.fvg_type
        )
        
        trade_direction = (TradeDirection.LONG if target_fvg.fvg_type == FVGType.BULLISH 
                          else TradeDirection.SHORT)
        signal_type = SignalType.BUY if trade_direction == TradeDirection.LONG else SignalType.SELL
        
        return Signal(
            timestamp=current_time,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.85,
            reason=f"FVG Sniper: {target_fvg.fvg_type.name} FVG retracement entry",
            metadata={
                'fvg_type': target_fvg.fvg_type.name,
                'fvg_top': target_fvg.top,
                'fvg_bottom': target_fvg.bottom,
                'fvg_age': index - target_fvg.end_index,
                'entry_type': entry_type
            }
        )
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate all FVG Sniper signals for the dataset"""
        signals = []
        self.active_fvgs = []  # Reset active FVGs
        
        for i in range(len(data)):
            signal = self.check_entry_conditions(data, i)
            if signal:
                signals.append(signal)
        
        self.signals = signals
        return signals
    
    def _update_active_fvgs(self, data: pd.DataFrame, index: int):
        """Update the list of active (unfilled) FVGs"""
        # Add new FVGs
        if index >= 3:
            new_fvgs = self.pattern_detector.detect_fair_value_gaps(
                data.iloc[index-2:index+1],
                self.parameters['min_fvg_size_pips']
            )
            
            for fvg in new_fvgs:
                # Adjust indices to global data frame
                fvg.start_index += (index - 2)
                fvg.end_index += (index - 2)
                self.active_fvgs.append(fvg)
        
        # Remove old or filled FVGs
        max_age = self.parameters['max_fvg_age_candles']
        current_price = data.iloc[index]['close']
        current_high = data.iloc[index]['high']
        current_low = data.iloc[index]['low']
        
        self.active_fvgs = [
            fvg for fvg in self.active_fvgs
            if (index - fvg.end_index <= max_age and  # Not too old
                not self._is_fvg_filled(fvg, current_high, current_low))  # Not filled
        ]
    
    def _is_fvg_filled(self, fvg, current_high: float, current_low: float) -> bool:
        """Check if FVG is filled beyond threshold"""
        fvg_range = fvg.top - fvg.bottom
        fill_threshold = fvg_range * self.parameters['fvg_fill_threshold']
        
        if fvg.fvg_type == FVGType.BULLISH:
            # Bullish FVG filled if price comes down significantly into the gap
            if current_low <= fvg.top - fill_threshold:
                return True
        else:
            # Bearish FVG filled if price comes up significantly into the gap
            if current_high >= fvg.bottom + fill_threshold:
                return True
        
        return False
    
    def _check_fvg_retracement(self, current_candle, index: int) -> Optional[tuple]:
        """
        Check if current price is retracing into an active FVG
        
        Args:
            current_candle: Current price candle
            index: Current index
            
        Returns:
            Tuple of (target_fvg, entry_type) if retracement detected
        """
        current_high = current_candle['high']
        current_low = current_candle['low']
        
        for fvg in self.active_fvgs:
            # Check if current candle is interacting with FVG
            if fvg.fvg_type == FVGType.BULLISH:
                # Bullish FVG: look for price retracing into gap from above
                if (current_low <= fvg.top and current_high >= fvg.bottom):
                    return (fvg, 'bullish_retracement')
            
            else:  # Bearish FVG
                # Bearish FVG: look for price retracing into gap from below
                if (current_high >= fvg.bottom and current_low <= fvg.top):
                    return (fvg, 'bearish_retracement')
        
        return None
    
    def _determine_htf_bias(self, data: pd.DataFrame, index: int) -> str:
        """
        Determine Higher Timeframe bias using daily structure
        
        Args:
            data: OHLCV DataFrame
            index: Current index
            
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        lookback = min(self.parameters['htf_lookback_periods'], index)
        
        if lookback < 20:
            return 'neutral'
        
        htf_data = data.iloc[index-lookback:index+1]
        
        # Simple HTF bias using moving averages and recent highs/lows
        if 'sma_20' in htf_data.columns and 'sma_50' in htf_data.columns:
            current_sma20 = htf_data['sma_20'].iloc[-1]
            current_sma50 = htf_data['sma_50'].iloc[-1]
            
            # Price position relative to MAs
            current_price = htf_data['close'].iloc[-1]
            
            if current_price > current_sma20 > current_sma50:
                return 'bullish'
            elif current_price < current_sma20 < current_sma50:
                return 'bearish'
        
        # Fallback: use recent swing structure
        recent_highs = htf_data['high'].rolling(window=10).max()
        recent_lows = htf_data['low'].rolling(window=10).min()
        
        if recent_highs.iloc[-1] > recent_highs.iloc[-10]:
            return 'bullish'
        elif recent_lows.iloc[-1] < recent_lows.iloc[-10]:
            return 'bearish'
        
        return 'neutral'
    
    def _is_aligned_with_htf(self, fvg_type: FVGType, htf_bias: str) -> bool:
        """Check if FVG type aligns with HTF bias"""
        if htf_bias == 'neutral':
            return True  # Allow trades in neutral bias
        
        if fvg_type == FVGType.BULLISH and htf_bias == 'bullish':
            return True
        elif fvg_type == FVGType.BEARISH and htf_bias == 'bearish':
            return True
        
        return False
    
    def _confirm_retracement_entry(self, data: pd.DataFrame, index: int, 
                                 fvg_type: FVGType) -> bool:
        """
        Confirm that the retracement is valid for entry
        
        Args:
            data: OHLCV DataFrame
            index: Current index
            fvg_type: Type of FVG being traded
            
        Returns:
            True if retracement is confirmed
        """
        confirmation_periods = self.parameters['retracement_confirmation_periods']
        
        if index < confirmation_periods:
            return False
        
        recent_candles = data.iloc[index-confirmation_periods+1:index+1]
        
        if fvg_type == FVGType.BULLISH:
            # For bullish FVG, look for buying pressure after retracement
            bullish_candles = (recent_candles['close'] > recent_candles['open']).sum()
            return bullish_candles >= confirmation_periods // 2
        
        else:  # Bearish FVG
            # For bearish FVG, look for selling pressure after retracement
            bearish_candles = (recent_candles['close'] < recent_candles['open']).sum()
            return bearish_candles >= confirmation_periods // 2
    
    def _calculate_fvg_stop_loss(self, fvg, entry_type: str) -> float:
        """Calculate stop loss beyond FVG boundary"""
        buffer_pips = 3 * 0.0001  # 3 pip buffer
        
        if fvg.fvg_type == FVGType.BULLISH:
            # For long trades, stop below FVG bottom
            return fvg.bottom - buffer_pips
        else:
            # For short trades, stop above FVG top
            return fvg.top + buffer_pips
    
    def _calculate_fvg_take_profit(self, data: pd.DataFrame, index: int, 
                                 entry_price: float, stop_loss: float, 
                                 fvg_type: FVGType) -> float:
        """Calculate take profit targeting nearest liquidity pool or Order Block"""
        risk = abs(entry_price - stop_loss)
        default_target = (entry_price + (risk * self.risk_reward_ratio) 
                         if fvg_type == FVGType.BULLISH 
                         else entry_price - (risk * self.risk_reward_ratio))
        
        # Look for nearby liquidity pools or Order Blocks
        try:
            lookback = min(50, index)
            analysis_data = data.iloc[index-lookback:index+1]
            
            # Find Order Blocks in the direction of trade
            order_blocks = self.pattern_detector.detect_order_blocks(analysis_data)
            
            min_distance_pips = self.parameters['min_liquidity_distance_pips']
            min_distance = min_distance_pips * 0.0001
            
            for ob in reversed(order_blocks):  # Most recent first
                if fvg_type == FVGType.BULLISH:
                    # For long trades, look for bullish OB above entry
                    if (ob.ob_type.value == 1 and 
                        ob.low > entry_price + min_distance):
                        return min(ob.low, default_target * 1.5)
                else:
                    # For short trades, look for bearish OB below entry
                    if (ob.ob_type.value == -1 and 
                        ob.high < entry_price - min_distance):
                        return max(ob.high, default_target * 1.5)
            
            # Look for swing levels as liquidity targets
            if fvg_type == FVGType.BULLISH:
                swing_highs = analysis_data['high'].rolling(window=10).max()
                target_high = swing_highs.max()
                if target_high > entry_price + min_distance:
                    return min(target_high, default_target * 2.0)
            else:
                swing_lows = analysis_data['low'].rolling(window=10).min()
                target_low = swing_lows.min()
                if target_low < entry_price - min_distance:
                    return max(target_low, default_target * 2.0)
        
        except Exception:
            # Fall back to default if analysis fails
            pass
        
        return default_target