"""
SILVER BULLET Strategy - ICT Trading Framework

Session: NY Killzone (10:00–11:00 EST)
Setup: Identify liquidity sweep of prior swing high/low within Killzone
Entry: Enter in opposite direction after sweep + displacement
Stop-Loss: Beyond the liquidity sweep wick
Take-Profit: Target next liquidity pool (previous high/low)
Filter: Confirm displacement via FVG
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from ..strategies.base_strategy import BaseICTStrategy, Signal, SignalType, TradeDirection
from ..utils.ict_patterns import ICTPatternDetector, detect_displacement
from ..utils.market_data import SessionDetector


class SilverBulletStrategy(BaseICTStrategy):
    """
    Silver Bullet Strategy Implementation
    
    This strategy targets liquidity sweeps during the NY Killzone session,
    entering in the opposite direction after confirmation via displacement and FVG.
    """
    
    def _set_default_parameters(self):
        """Set Silver Bullet specific parameters"""
        self.parameters.setdefault('session', 'ny_killzone')
        self.parameters.setdefault('lookback_periods', 20)
        self.parameters.setdefault('min_displacement_candles', 3)
        self.parameters.setdefault('min_fvg_size_pips', 5)
        self.parameters.setdefault('liquidity_sweep_buffer_pips', 2)
        
        # Initialize components
        self.session_detector = SessionDetector()
        self.pattern_detector = ICTPatternDetector()
    
    def check_entry_conditions(self, data: pd.DataFrame, index: int) -> Optional[Signal]:
        """
        Check Silver Bullet entry conditions at specific index
        
        Args:
            data: OHLCV price data
            index: Current data index
            
        Returns:
            Trading signal if conditions met, None otherwise
        """
        if index < self.parameters['lookback_periods']:
            return None
        
        current_time = data.index[index]
        
        # Filter 1: Must be in NY Killzone session
        if not self.session_detector.is_in_session(current_time, self.parameters['session']):
            return None
        
        # Filter 2: Check for liquidity sweep
        liquidity_sweep = self._detect_liquidity_sweep(data, index)
        if not liquidity_sweep:
            return None
        
        sweep_direction, sweep_price = liquidity_sweep
        
        # Filter 3: Check for displacement in opposite direction
        opposite_direction = 'down' if sweep_direction == 'up' else 'up'
        if not detect_displacement(data, index, opposite_direction, 
                                 self.parameters['min_displacement_candles']):
            return None
        
        # Filter 4: Confirm with Fair Value Gap
        fvgs = self.pattern_detector.detect_fair_value_gaps(
            data.iloc[index-5:index+1], 
            self.parameters['min_fvg_size_pips']
        )
        
        if not fvgs:
            return None
        
        # Check if latest FVG aligns with trade direction
        latest_fvg = fvgs[-1]
        trade_direction = TradeDirection.SHORT if sweep_direction == 'up' else TradeDirection.LONG
        
        if trade_direction == TradeDirection.LONG and latest_fvg.fvg_type.value != 1:
            return None
        if trade_direction == TradeDirection.SHORT and latest_fvg.fvg_type.value != -1:
            return None
        
        # Determine entry price and levels
        entry_price = data.iloc[index]['close']
        stop_loss = self._calculate_silver_bullet_stop_loss(
            entry_price, sweep_price, sweep_direction
        )
        take_profit = self._calculate_silver_bullet_take_profit(
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
            reason=f"Silver Bullet: {sweep_direction} liquidity sweep + {opposite_direction} displacement + FVG confirmation",
            metadata={
                'sweep_direction': sweep_direction,
                'sweep_price': sweep_price,
                'fvg_type': latest_fvg.fvg_type.name,
                'session': self.parameters['session']
            }
        )
    
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Generate all Silver Bullet signals for the dataset"""
        signals = []
        
        for i in range(len(data)):
            signal = self.check_entry_conditions(data, i)
            if signal:
                signals.append(signal)
        
        self.signals = signals
        return signals
    
    def _detect_liquidity_sweep(self, data: pd.DataFrame, index: int) -> Optional[tuple]:
        """
        Detect liquidity sweep of recent swing highs/lows
        
        Args:
            data: OHLCV DataFrame
            index: Current index
            
        Returns:
            Tuple of (sweep_direction, sweep_price) if sweep detected, None otherwise
        """
        lookback = self.parameters['lookback_periods']
        if index < lookback:
            return None
        
        current_candle = data.iloc[index]
        recent_data = data.iloc[index-lookback:index]
        
        # Find recent swing high and low
        swing_high_idx = recent_data['high'].idxmax()
        swing_low_idx = recent_data['low'].idxmin()
        
        swing_high = recent_data.loc[swing_high_idx, 'high']
        swing_low = recent_data.loc[swing_low_idx, 'low']
        
        buffer_pips = self.parameters['liquidity_sweep_buffer_pips'] * 0.0001
        
        # Check for high liquidity sweep (bearish sweep)
        if current_candle['high'] > swing_high + buffer_pips:
            return ('up', swing_high)
        
        # Check for low liquidity sweep (bullish sweep)
        if current_candle['low'] < swing_low - buffer_pips:
            return ('down', swing_low)
        
        return None
    
    def _calculate_silver_bullet_stop_loss(self, entry_price: float, sweep_price: float, 
                                         sweep_direction: str) -> float:
        """Calculate stop loss beyond the liquidity sweep wick"""
        buffer_pips = self.parameters['liquidity_sweep_buffer_pips'] * 0.0001
        
        if sweep_direction == 'up':
            # For short trades, stop loss above the sweep high
            return sweep_price + buffer_pips
        else:
            # For long trades, stop loss below the sweep low
            return sweep_price - buffer_pips
    
    def _calculate_silver_bullet_take_profit(self, data: pd.DataFrame, index: int, 
                                           entry_price: float, stop_loss: float, 
                                           trade_direction: TradeDirection) -> float:
        """Calculate take profit targeting next liquidity pool"""
        lookback = self.parameters['lookback_periods']
        
        # Find next liquidity target (previous swing in opposite direction)
        recent_data = data.iloc[max(0, index-lookback*2):index]
        
        if trade_direction == TradeDirection.LONG:
            # Target previous swing high
            target_price = recent_data['high'].max()
            # Ensure reasonable risk-reward
            risk = abs(entry_price - stop_loss)
            min_reward = entry_price + (risk * self.risk_reward_ratio)
            return max(target_price, min_reward)
        else:
            # Target previous swing low
            target_price = recent_data['low'].min()
            # Ensure reasonable risk-reward
            risk = abs(entry_price - stop_loss)
            min_reward = entry_price - (risk * self.risk_reward_ratio)
            return min(target_price, min_reward)
    
    def _check_time_filter(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within NY Killzone session"""
        return self.session_detector.is_in_session(timestamp, self.parameters['session'])