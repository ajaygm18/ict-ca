"""
POWER HOUR Strategy - ICT Trading Framework

Session: 14:00–15:00 EST
Setup: Detect continuation or reversal moves into major liquidity pools
Entry: Enter after sweep + displacement
Stop-Loss: Beyond sweep wick
Take-Profit: Next liquidity pool or daily range expansion
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from ..strategies.base_strategy import BaseICTStrategy, Signal, SignalType, TradeDirection
from ..utils.ict_patterns import ICTPatternDetector, detect_displacement
from ..utils.market_data import SessionDetector


class PowerHourStrategy(BaseICTStrategy):
    """
    Power Hour Strategy Implementation
    
    This strategy targets liquidity sweeps during the Power Hour session (14:00-15:00 EST),
    entering trades after sweep confirmation with displacement towards major liquidity pools.
    """
    
    def _set_default_parameters(self):
        """Set Power Hour specific parameters"""
        self.parameters.setdefault('session', 'power_hour')
        self.parameters.setdefault('liquidity_lookback_periods', 50)
        self.parameters.setdefault('min_displacement_candles', 3)
        self.parameters.setdefault('sweep_confirmation_pips', 3)
        self.parameters.setdefault('daily_range_multiplier', 1.5)
        self.parameters.setdefault('min_liquidity_age_periods', 10)
        self.parameters.setdefault('continuation_vs_reversal_threshold', 0.7)
        
        # Initialize components
        self.session_detector = SessionDetector()
        self.pattern_detector = ICTPatternDetector()
        self.daily_levels = {}
    
    def check_entry_conditions(self, data: pd.DataFrame, index: int) -> Optional[Signal]:
        """
        Check Power Hour entry conditions at specific index
        
        Args:
            data: OHLCV price data
            index: Current data index
            
        Returns:
            Trading signal if conditions met, None otherwise
        """
        if index < self.parameters['liquidity_lookback_periods']:
            return None
        
        current_time = data.index[index]
        
        # Filter 1: Must be in Power Hour session
        if not self.session_detector.is_in_session(current_time, self.parameters['session']):
            return None
        
        # Filter 2: Identify major liquidity pools
        liquidity_pools = self._identify_major_liquidity_pools(data, index)
        
        if not liquidity_pools:
            return None
        
        # Filter 3: Check for liquidity sweep
        liquidity_sweep = self._detect_power_hour_sweep(data, index, liquidity_pools)
        
        if not liquidity_sweep:
            return None
        
        sweep_direction, swept_level, sweep_type = liquidity_sweep
        
        # Filter 4: Check for displacement after sweep
        displacement_direction = 'down' if sweep_direction == 'up' else 'up'
        
        if not detect_displacement(data, index, displacement_direction, 
                                 self.parameters['min_displacement_candles']):
            return None
        
        # Filter 5: Determine if this is continuation or reversal
        trade_bias = self._determine_trade_bias(data, index, swept_level, sweep_direction)
        
        # Calculate entry and risk levels
        entry_price = data.iloc[index]['close']
        stop_loss = self._calculate_power_hour_stop_loss(
            entry_price, swept_level, sweep_direction
        )
        take_profit = self._calculate_power_hour_take_profit(
            data, index, entry_price, stop_loss, trade_bias, liquidity_pools
        )
        
        trade_direction = TradeDirection.LONG if trade_bias == 'bullish' else TradeDirection.SHORT
        signal_type = SignalType.BUY if trade_direction == TradeDirection.LONG else SignalType.SELL
        
        return Signal(
            timestamp=current_time,
            signal_type=signal_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.8,
            reason=f"Power Hour: {sweep_type} sweep + {displacement_direction} displacement, bias: {trade_bias}",
            metadata={
                'sweep_direction': sweep_direction,
                'swept_level': swept_level,
                'sweep_type': sweep_type,
                'trade_bias': trade_bias,
                'displacement_direction': displacement_direction,
                'session': self.parameters['session']
            }
        )
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate all Power Hour signals for the dataset"""
        signals = []
        
        # Pre-calculate daily levels for efficiency
        self._calculate_daily_levels(data)
        
        for i in range(len(data)):
            signal = self.check_entry_conditions(data, i)
            if signal:
                signals.append(signal)
        
        self.signals = signals
        return signals
    
    def _calculate_daily_levels(self, data: pd.DataFrame):
        """Pre-calculate daily high/low levels for reference"""
        self.daily_levels = {}
        
        for date in data.index.date:
            day_data = data[data.index.date == date]
            if len(day_data) > 0:
                self.daily_levels[date] = {
                    'high': day_data['high'].max(),
                    'low': day_data['low'].min(),
                    'open': day_data.iloc[0]['open'],
                    'range': day_data['high'].max() - day_data['low'].min()
                }
    
    def _identify_major_liquidity_pools(self, data: pd.DataFrame, index: int) -> List[Dict]:
        """
        Identify major liquidity pools (significant highs/lows)
        
        Args:
            data: OHLCV DataFrame
            index: Current index
            
        Returns:
            List of liquidity pool dictionaries
        """
        lookback = self.parameters['liquidity_lookback_periods']
        min_age = self.parameters['min_liquidity_age_periods']
        
        analysis_data = data.iloc[max(0, index-lookback):index]
        liquidity_pools = []
        
        # Find significant highs (resistance/liquidity)
        for i in range(min_age, len(analysis_data) - min_age):
            current_high = analysis_data.iloc[i]['high']
            
            # Check if this is a significant high
            if self._is_significant_level(analysis_data, i, 'high'):
                age = len(analysis_data) - i
                strength = self._calculate_level_strength(analysis_data, i, current_high, 'high')
                
                liquidity_pools.append({
                    'price': current_high,
                    'type': 'resistance',
                    'age': age,
                    'strength': strength,
                    'index': i
                })
        
        # Find significant lows (support/liquidity)
        for i in range(min_age, len(analysis_data) - min_age):
            current_low = analysis_data.iloc[i]['low']
            
            # Check if this is a significant low
            if self._is_significant_level(analysis_data, i, 'low'):
                age = len(analysis_data) - i
                strength = self._calculate_level_strength(analysis_data, i, current_low, 'low')
                
                liquidity_pools.append({
                    'price': current_low,
                    'type': 'support',
                    'age': age,
                    'strength': strength,
                    'index': i
                })
        
        # Sort by strength and age (older and stronger levels are more significant)
        liquidity_pools.sort(key=lambda x: (x['strength'], x['age']), reverse=True)
        
        return liquidity_pools[:10]  # Return top 10 most significant levels
    
    def _is_significant_level(self, data: pd.DataFrame, index: int, level_type: str) -> bool:
        """Check if a price level is significant (forms a swing high/low)"""
        if index < 3 or index >= len(data) - 3:
            return False
        
        if level_type == 'high':
            current = data.iloc[index]['high']
            return (current > data.iloc[index-1]['high'] and 
                   current > data.iloc[index+1]['high'] and
                   current > data.iloc[index-2]['high'] and 
                   current > data.iloc[index+2]['high'])
        else:
            current = data.iloc[index]['low']
            return (current < data.iloc[index-1]['low'] and 
                   current < data.iloc[index+1]['low'] and
                   current < data.iloc[index-2]['low'] and 
                   current < data.iloc[index+2]['low'])
    
    def _calculate_level_strength(self, data: pd.DataFrame, index: int, 
                                price: float, level_type: str) -> float:
        """Calculate the strength of a liquidity level based on touches and reactions"""
        touches = 0
        reactions = 0
        
        # Count touches and reactions after the level formation
        for i in range(index + 1, len(data)):
            if level_type == 'high':
                # Check for touches of resistance
                if abs(data.iloc[i]['high'] - price) <= 0.0001 * 3:  # Within 3 pips
                    touches += 1
                    # Check for reaction (rejection)
                    if data.iloc[i]['close'] < data.iloc[i]['high'] - 0.0001 * 5:  # 5 pip rejection
                        reactions += 1
            else:
                # Check for touches of support
                if abs(data.iloc[i]['low'] - price) <= 0.0001 * 3:  # Within 3 pips
                    touches += 1
                    # Check for reaction (bounce)
                    if data.iloc[i]['close'] > data.iloc[i]['low'] + 0.0001 * 5:  # 5 pip bounce
                        reactions += 1
        
        # Strength based on touches and reaction rate
        if touches == 0:
            return 0.0
        
        reaction_rate = reactions / touches if touches > 0 else 0
        strength = min(touches * 0.2 + reaction_rate * 0.8, 1.0)
        
        return strength
    
    def _detect_power_hour_sweep(self, data: pd.DataFrame, index: int, 
                               liquidity_pools: List[Dict]) -> Optional[Tuple[str, float, str]]:
        """
        Detect liquidity sweep of major pools during Power Hour
        
        Args:
            data: OHLCV DataFrame
            index: Current index
            liquidity_pools: List of identified liquidity pools
            
        Returns:
            Tuple of (sweep_direction, swept_level, sweep_type) if detected
        """
        current_candle = data.iloc[index]
        confirmation_pips = self.parameters['sweep_confirmation_pips'] * 0.0001
        
        for pool in liquidity_pools:
            pool_price = pool['price']
            pool_type = pool['type']
            
            # Check for resistance (high) sweep
            if (pool_type == 'resistance' and 
                current_candle['high'] > pool_price + confirmation_pips):
                return ('up', pool_price, 'resistance')
            
            # Check for support (low) sweep
            elif (pool_type == 'support' and 
                  current_candle['low'] < pool_price - confirmation_pips):
                return ('down', pool_price, 'support')
        
        return None
    
    def _determine_trade_bias(self, data: pd.DataFrame, index: int, 
                            swept_level: float, sweep_direction: str) -> str:
        """
        Determine if the trade should be continuation or reversal based on context
        
        Args:
            data: OHLCV DataFrame
            index: Current index
            swept_level: Price level that was swept
            sweep_direction: Direction of the sweep
            
        Returns:
            'bullish' or 'bearish' trade bias
        """
        current_date = data.index[index].date()
        
        # Get daily context
        if current_date in self.daily_levels:
            daily_info = self.daily_levels[current_date]
            daily_range = daily_info['range']
            daily_mid = daily_info['low'] + (daily_range / 2)
            current_price = data.iloc[index]['close']
            
            # Position within daily range
            range_position = (current_price - daily_info['low']) / daily_range if daily_range > 0 else 0.5
            
            # Bias logic:
            # - If sweep occurs in upper part of daily range, bias towards reversal (bearish)
            # - If sweep occurs in lower part of daily range, bias towards reversal (bullish)
            # - Consider overall daily trend and momentum
            
            threshold = self.parameters['continuation_vs_reversal_threshold']
            
            if sweep_direction == 'up':
                # Upward sweep
                if range_position > threshold:
                    return 'bearish'  # Reversal bias (expect downward move)
                else:
                    return 'bullish'  # Continuation bias (expect continued upward move)
            else:
                # Downward sweep
                if range_position < (1 - threshold):
                    return 'bullish'  # Reversal bias (expect upward move)
                else:
                    return 'bearish'  # Continuation bias (expect continued downward move)
        
        # Default: trade opposite to sweep direction (reversal bias)
        return 'bearish' if sweep_direction == 'up' else 'bullish'
    
    def _calculate_power_hour_stop_loss(self, entry_price: float, swept_level: float, 
                                      sweep_direction: str) -> float:
        """Calculate stop loss beyond the sweep wick"""
        buffer_pips = 5 * 0.0001  # 5 pip buffer
        
        if sweep_direction == 'up':
            return swept_level + buffer_pips
        else:
            return swept_level - buffer_pips
    
    def _calculate_power_hour_take_profit(self, data: pd.DataFrame, index: int, 
                                        entry_price: float, stop_loss: float, 
                                        trade_bias: str, liquidity_pools: List[Dict]) -> float:
        """Calculate take profit targeting next liquidity pool or daily range expansion"""
        risk = abs(entry_price - stop_loss)
        
        # Default target based on risk-reward
        default_target = (entry_price + (risk * self.risk_reward_ratio) 
                         if trade_bias == 'bullish' 
                         else entry_price - (risk * self.risk_reward_ratio))
        
        # Look for next liquidity pool in trade direction
        for pool in liquidity_pools:
            pool_price = pool['price']
            
            if trade_bias == 'bullish' and pool_price > entry_price:
                # Target higher liquidity for long trades
                if pool['type'] == 'resistance':
                    return min(pool_price, default_target * 1.5)
            elif trade_bias == 'bearish' and pool_price < entry_price:
                # Target lower liquidity for short trades
                if pool['type'] == 'support':
                    return max(pool_price, default_target * 1.5)
        
        # Check for daily range expansion targets
        current_date = data.index[index].date()
        if current_date in self.daily_levels:
            daily_info = self.daily_levels[current_date]
            daily_range = daily_info['range']
            expansion_target = daily_range * self.parameters['daily_range_multiplier']
            
            if trade_bias == 'bullish':
                range_expansion_target = daily_info['high'] + (expansion_target * 0.382)  # 38.2% extension
                return min(range_expansion_target, default_target * 2.0)
            else:
                range_expansion_target = daily_info['low'] - (expansion_target * 0.382)  # 38.2% extension
                return max(range_expansion_target, default_target * 2.0)
        
        return default_target
    
    def _check_time_filter(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within Power Hour session"""
        return self.session_detector.is_in_session(timestamp, self.parameters['session'])