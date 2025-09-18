"""
Power of 3 Strategy - Strategy 11

The Power of 3 strategy is based on ICT's concept that each session follows a
three-phase pattern: Accumulation, Manipulation, and Distribution. This strategy
identifies these phases and trades the distribution move after manipulation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base_strategy import BaseICTStrategy, Signal, TradeDirection

class PowerOf3Strategy(BaseICTStrategy):
    """
    Power of 3 Strategy Implementation
    
    This strategy identifies the three phases of institutional trading:
    1. Accumulation - Consolidation and position building
    2. Manipulation - False moves to trigger stops and create liquidity
    3. Distribution - The real directional move where institutions profit
    
    Entry Conditions:
    - Clear accumulation phase identified (range-bound price action)
    - Manipulation phase occurs (fake breakout or stop hunt)
    - Distribution phase begins (strong directional move)
    - Volume confirms institutional participation
    
    Exit Conditions:
    - Distribution phase exhausted (momentum divergence)
    - Next session begins (new Power of 3 cycle)
    - Stop loss beyond manipulation extreme
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.accumulation_min_bars = config.get('accumulation_min_bars', 8)
        self.manipulation_max_bars = config.get('manipulation_max_bars', 4)
        self.distribution_min_move = config.get('distribution_min_move', 0.5)  # % of daily range
        self.volume_threshold = config.get('volume_threshold', 1.3)
        self.range_compression_threshold = config.get('range_compression_threshold', 0.7)
        
    def setup_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Setup indicators for Power of 3 detection"""
        df = data.copy()
        
        # Session identification
        df['session'] = self._identify_session(df)
        df['session_start'] = df['session'] != df['session'].shift(1)
        
        # Range and volatility analysis
        df['range'] = df['high'] - df['low']
        df['range_ma'] = df['range'].rolling(20).mean()
        df['range_ratio'] = df['range'] / df['range_ma']
        df['atr'] = self._calculate_atr(df, 14)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum and structure
        df['momentum'] = df['close'].pct_change(5)
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # Swing levels for structure analysis
        df['swing_high'] = self._identify_swing_highs(df, 3)
        df['swing_low'] = self._identify_swing_lows(df, 3)
        
        # Daily levels for context
        df['daily_high'] = df.groupby(df.index.date)['high'].transform('max')
        df['daily_low'] = df.groupby(df.index.date)['low'].transform('min')
        df['daily_range'] = df['daily_high'] - df['daily_low']
        
        return df
        
    def identify_entry_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Identify Power of 3 entry opportunities"""
        signals = []
        
        if len(data) < 50:
            return signals
            
        # Analyze each session for Power of 3 pattern
        sessions = self._identify_session_boundaries(data)
        
        for session_start, session_end in sessions:
            if session_end - session_start < self.accumulation_min_bars + 5:
                continue
                
            signal = self._analyze_power_of_3_session(data, session_start, session_end)
            if signal:
                signals.append(signal)
                
        return signals
        
    def _identify_session_boundaries(self, data: pd.DataFrame) -> List[Tuple[int, int]]:
        """Identify session start and end indices"""
        sessions = []
        session_starts = data[data['session_start']].index
        
        for i, start_time in enumerate(session_starts[:-1]):
            start_idx = data.index.get_loc(start_time)
            end_idx = data.index.get_loc(session_starts[i + 1])
            sessions.append((start_idx, end_idx))
            
        return sessions
        
    def _analyze_power_of_3_session(self, data: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[Signal]:
        """Analyze a session for Power of 3 pattern"""
        session_data = data.iloc[start_idx:end_idx]
        
        if len(session_data) < self.accumulation_min_bars + 5:
            return None
            
        # Phase 1: Identify Accumulation
        accumulation_phase = self._identify_accumulation_phase(session_data)
        if not accumulation_phase:
            return None
            
        acc_start, acc_end = accumulation_phase
        acc_high = session_data.iloc[acc_start:acc_end]['high'].max()
        acc_low = session_data.iloc[acc_start:acc_end]['low'].min()
        acc_range = acc_high - acc_low
        
        # Phase 2: Identify Manipulation
        manip_start = acc_end
        manipulation_phase = self._identify_manipulation_phase(
            session_data, manip_start, acc_high, acc_low
        )
        if not manipulation_phase:
            return None
            
        manip_direction, manip_extreme, manip_end = manipulation_phase
        
        # Phase 3: Identify Distribution
        dist_start = manip_end
        if dist_start >= len(session_data) - 2:
            return None
            
        distribution_signal = self._identify_distribution_phase(
            session_data, dist_start, manip_direction, manip_extreme, acc_range
        )
        
        if not distribution_signal:
            return None
            
        # Convert relative indices to absolute indices
        signal_idx = start_idx + dist_start
        if signal_idx >= len(data):
            return None
            
        entry_bar = data.iloc[signal_idx]
        
        # Setup trade based on distribution direction
        if manip_direction == 'up':
            # Manipulation was upward, expect downward distribution
            direction = TradeDirection.SHORT
            entry_price = entry_bar['close']
            stop_loss = manip_extreme + (acc_range * 0.1)
            take_profit = acc_low - (acc_range * 0.5)
            
        else:
            # Manipulation was downward, expect upward distribution
            direction = TradeDirection.LONG
            entry_price = entry_bar['close']
            stop_loss = manip_extreme - (acc_range * 0.1)
            take_profit = acc_high + (acc_range * 0.5)
            
        # Validate risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward / risk < 1.0:
            return None
            
        return Signal(
            timestamp=entry_bar.name,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.75,
            strategy="POWER_OF_3",
            metadata={
                'accumulation_range': acc_range,
                'manipulation_direction': manip_direction,
                'manipulation_extreme': manip_extreme,
                'session': entry_bar['session'],
                'volume_ratio': entry_bar['volume_ratio'],
                'phase_timing': {
                    'accumulation': (acc_start, acc_end),
                    'manipulation': (manip_start, manip_end),
                    'distribution_start': dist_start
                }
            }
        )
        
    def _identify_accumulation_phase(self, session_data: pd.DataFrame) -> Optional[Tuple[int, int]]:
        """Identify the accumulation phase of the session"""
        if len(session_data) < self.accumulation_min_bars:
            return None
            
        # Look for initial range-bound price action
        for end_bar in range(self.accumulation_min_bars, len(session_data) // 2):
            start_bar = 0
            phase_data = session_data.iloc[start_bar:end_bar]
            
            # Check for range compression
            phase_high = phase_data['high'].max()
            phase_low = phase_data['low'].min()
            phase_range = phase_high - phase_low
            
            avg_range = phase_data['range'].mean()
            
            # Accumulation should have compressed ranges
            range_compression = phase_range / (avg_range * len(phase_data))
            
            if range_compression < self.range_compression_threshold:
                # Check for lack of clear direction
                price_change = abs(phase_data['close'].iloc[-1] - phase_data['close'].iloc[0])
                relative_change = price_change / phase_range
                
                if relative_change < 0.3:  # Price hasn't moved much relative to range
                    return (start_bar, end_bar)
                    
        return None
        
    def _identify_manipulation_phase(self, session_data: pd.DataFrame, start_idx: int, 
                                   acc_high: float, acc_low: float) -> Optional[Tuple[str, float, int]]:
        """Identify the manipulation phase"""
        if start_idx >= len(session_data) - 2:
            return None
            
        end_search = min(start_idx + self.manipulation_max_bars, len(session_data))
        
        for i in range(start_idx, end_search):
            current_bar = session_data.iloc[i]
            
            # Check for breakout above accumulation range
            if current_bar['high'] > acc_high:
                # Upward manipulation
                if self._confirm_manipulation(session_data, i, 'up', acc_high):
                    return ('up', current_bar['high'], i + 1)
                    
            # Check for breakdown below accumulation range
            elif current_bar['low'] < acc_low:
                # Downward manipulation
                if self._confirm_manipulation(session_data, i, 'down', acc_low):
                    return ('down', current_bar['low'], i + 1)
                    
        return None
        
    def _confirm_manipulation(self, session_data: pd.DataFrame, manip_idx: int, 
                            direction: str, level: float) -> bool:
        """Confirm manipulation with volume and reversal"""
        if manip_idx >= len(session_data) - 1:
            return False
            
        manip_bar = session_data.iloc[manip_idx]
        
        # Volume confirmation
        volume_spike = manip_bar['volume_ratio'] > self.volume_threshold
        
        # Look for quick reversal (within 1-2 bars)
        reversal_bars = session_data.iloc[manip_idx:manip_idx+3]
        
        if direction == 'up':
            # After upward manipulation, look for bearish reversal
            quick_reversal = (reversal_bars['close'] < level).any()
        else:
            # After downward manipulation, look for bullish reversal
            quick_reversal = (reversal_bars['close'] > level).any()
            
        return volume_spike and quick_reversal
        
    def _identify_distribution_phase(self, session_data: pd.DataFrame, start_idx: int,
                                   manip_direction: str, manip_extreme: float, 
                                   acc_range: float) -> bool:
        """Identify the beginning of distribution phase"""
        if start_idx >= len(session_data) - 1:
            return False
            
        # Look for strong move in opposite direction of manipulation
        distribution_bars = session_data.iloc[start_idx:start_idx+3]
        
        if manip_direction == 'up':
            # Expect strong downward move
            strong_move = (distribution_bars['close'] < manip_extreme - acc_range * 0.2).any()
            volume_confirmation = (distribution_bars['volume_ratio'] > 1.2).any()
            
        else:
            # Expect strong upward move
            strong_move = (distribution_bars['close'] > manip_extreme + acc_range * 0.2).any()
            volume_confirmation = (distribution_bars['volume_ratio'] > 1.2).any()
            
        return strong_move and volume_confirmation
        
    def identify_exit_signals(self, data: pd.DataFrame, open_trades: List) -> List[Dict]:
        """Identify exit signals for Power of 3 trades"""
        exit_signals = []
        
        for trade in open_trades:
            current_bar = data.iloc[-1]
            current_price = current_bar['close']
            
            # Exit at session end (new Power of 3 cycle)
            if current_bar['session_start']:
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'session_end',
                    'timestamp': current_bar.name
                })
                continue
                
            # Check for momentum exhaustion
            if self._check_momentum_exhaustion(data, trade):
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'momentum_exhaustion',
                    'timestamp': current_bar.name
                })
                
        return exit_signals
        
    def _check_momentum_exhaustion(self, data: pd.DataFrame, trade: Dict) -> bool:
        """Check for momentum exhaustion in distribution phase"""
        recent_momentum = data['momentum'].tail(3).mean()
        current_rsi = data['rsi'].iloc[-1]
        
        if trade['direction'] == TradeDirection.LONG:
            return current_rsi > 75 and recent_momentum < 0.001
        else:
            return current_rsi < 25 and recent_momentum > -0.001
            
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information"""
        return {
            'name': 'Power of 3',
            'description': 'Trades the three institutional phases: Accumulation, Manipulation, Distribution',
            'timeframes': ['15m', '1h'],
            'sessions': ['London', 'New York'],
            'risk_level': 'Medium-High',
            'parameters': {
                'accumulation_min_bars': self.accumulation_min_bars,
                'manipulation_max_bars': self.manipulation_max_bars,
                'distribution_min_move': self.distribution_min_move,
                'volume_threshold': self.volume_threshold,
                'range_compression_threshold': self.range_compression_threshold
            }
        }