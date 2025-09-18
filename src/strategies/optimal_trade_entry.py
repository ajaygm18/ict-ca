"""
Optimal Trade Entry Strategy - Strategy 15

The Optimal Trade Entry (OTE) strategy is based on ICT's concept of entering trades
at optimal retracement levels (typically 61.8% to 78.6% Fibonacci) after a strong
directional move. This strategy focuses on high-probability entries with excellent
risk-to-reward ratios.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .base_strategy import BaseICTStrategy, Signal, TradeDirection

class OptimalTradeEntryStrategy(BaseICTStrategy):
    """
    Optimal Trade Entry (OTE) Strategy Implementation
    
    This strategy identifies optimal entry points using:
    1. Strong directional move identification (impulse wave)
    2. Retracement analysis to OTE zone (61.8% - 78.6%)
    3. Confluence factors (FVG, Order Blocks, session times)
    4. Entry triggers within the OTE zone
    
    Entry Conditions:
    - Clear impulse move identified (strong directional move)
    - Price retraces to OTE zone (61.8% - 78.6% Fibonacci)
    - Confluence factors present (FVG, OB, key levels)
    - Entry trigger activated (rejection, engulfing, etc.)
    - Volume confirms institutional interest
    
    Exit Conditions:
    - Target reached (measured move or key levels)
    - Stop loss beyond OTE zone
    - Retracement exceeds 78.6% (invalidated setup)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_impulse_size = config.get('min_impulse_size', 0.008)  # 0.8% minimum impulse
        self.ote_zone_start = config.get('ote_zone_start', 0.618)      # 61.8% Fibonacci
        self.ote_zone_end = config.get('ote_zone_end', 0.786)          # 78.6% Fibonacci
        self.impulse_lookback = config.get('impulse_lookback', 20)     # Bars to look for impulse
        self.confluence_min_score = config.get('confluence_min_score', 0.6)
        self.entry_trigger_bars = config.get('entry_trigger_bars', 3)
        
    def setup_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Setup indicators for OTE analysis"""
        df = data.copy()
        
        # Session and time analysis
        df['session'] = self._identify_session(df)
        df['hour'] = pd.to_datetime(df.index).hour
        df['killzone'] = self._identify_killzones(df)
        
        # Swing point identification for impulse detection
        df['swing_high'] = self._identify_swing_highs(df, 5)
        df['swing_low'] = self._identify_swing_lows(df, 5)
        
        # Price structure analysis
        df['atr'] = self._calculate_atr(df, 14)
        df['range'] = df['high'] - df['low']
        df['momentum'] = df['close'].pct_change(5)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # ICT pattern detection
        df = self._detect_ict_patterns(df)
        
        # Fibonacci levels (will be calculated dynamically for each impulse)
        df['fib_382'] = np.nan
        df['fib_500'] = np.nan
        df['fib_618'] = np.nan
        df['fib_786'] = np.nan
        
        return df
        
    def _identify_killzones(self, data: pd.DataFrame) -> pd.Series:
        """Identify ICT killzones (high probability trading windows)"""
        killzones = pd.Series('None', index=data.index)
        
        # London Killzone: 02:00-05:00 EST
        london_kz = (data['hour'] >= 2) & (data['hour'] <= 5)
        killzones[london_kz] = 'London_KZ'
        
        # New York Killzone: 07:00-10:00 EST  
        ny_kz = (data['hour'] >= 7) & (data['hour'] <= 10)
        killzones[ny_kz] = 'NY_KZ'
        
        # London Close Killzone: 10:00-12:00 EST
        london_close_kz = (data['hour'] >= 10) & (data['hour'] <= 12)
        killzones[london_close_kz] = 'London_Close_KZ'
        
        return killzones
        
    def _detect_ict_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Detect ICT patterns for confluence"""
        df = data.copy()
        
        # Fair Value Gaps (simplified detection)
        df['fvg_bullish'] = (
            (df['low'].shift(1) > df['high'].shift(-1)) &  # Gap between bars
            (df['close'] > df['open'])  # Current bar is bullish
        )
        
        df['fvg_bearish'] = (
            (df['high'].shift(1) < df['low'].shift(-1)) &  # Gap between bars
            (df['close'] < df['open'])  # Current bar is bearish
        )
        
        # Order Blocks (simplified - strong reversal candles)
        df['order_block_bullish'] = (
            (df['close'] > df['open']) &  # Bullish candle
            (df['volume_ratio'] > 1.5) &  # High volume
            (df['range'] > df['atr'] * 1.5)  # Large range
        )
        
        df['order_block_bearish'] = (
            (df['close'] < df['open']) &  # Bearish candle
            (df['volume_ratio'] > 1.5) &  # High volume
            (df['range'] > df['atr'] * 1.5)  # Large range
        )
        
        return df
        
    def identify_entry_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Identify OTE entry opportunities"""
        signals = []
        
        if len(data) < 50:
            return signals
            
        # Look for impulse moves and subsequent retracements
        impulse_moves = self._identify_impulse_moves(data)
        
        for impulse in impulse_moves:
            signal = self._analyze_ote_setup(data, impulse)
            if signal:
                signals.append(signal)
                
        return signals
        
    def _identify_impulse_moves(self, data: pd.DataFrame) -> List[Dict]:
        """Identify strong directional impulse moves"""
        impulse_moves = []
        
        # Find swing highs and lows
        swing_highs = data[data['swing_high'] == True]
        swing_lows = data[data['swing_low'] == True]
        
        # Combine and sort swing points
        swing_points = []
        
        for idx, row in swing_highs.iterrows():
            swing_points.append({'timestamp': idx, 'price': row['high'], 'type': 'high'})
            
        for idx, row in swing_lows.iterrows():
            swing_points.append({'timestamp': idx, 'price': row['low'], 'type': 'low'})
            
        swing_points.sort(key=lambda x: x['timestamp'])
        
        # Analyze consecutive swing points for impulse moves
        for i in range(len(swing_points) - 1):
            current_swing = swing_points[i]
            next_swing = swing_points[i + 1]
            
            # Calculate impulse characteristics
            price_diff = abs(next_swing['price'] - current_swing['price'])
            price_change_pct = price_diff / current_swing['price']
            
            # Check if this qualifies as an impulse
            if price_change_pct >= self.min_impulse_size:
                # Determine direction
                if next_swing['price'] > current_swing['price']:
                    direction = 'bullish'
                else:
                    direction = 'bearish'
                    
                impulse = {
                    'start_time': current_swing['timestamp'],
                    'end_time': next_swing['timestamp'],
                    'start_price': current_swing['price'],
                    'end_price': next_swing['price'],
                    'direction': direction,
                    'size_pct': price_change_pct,
                    'start_idx': data.index.get_loc(current_swing['timestamp']),
                    'end_idx': data.index.get_loc(next_swing['timestamp'])
                }
                
                impulse_moves.append(impulse)
                
        return impulse_moves
        
    def _analyze_ote_setup(self, data: pd.DataFrame, impulse: Dict) -> Optional[Signal]:
        """Analyze OTE setup for a given impulse move"""
        end_idx = impulse['end_idx']
        
        # Calculate Fibonacci retracement levels
        fib_levels = self._calculate_fibonacci_levels(impulse)
        
        # Look for retracement to OTE zone
        ote_entry = self._find_ote_entry(data, impulse, fib_levels, end_idx)
        if not ote_entry:
            return None
            
        entry_idx, entry_price, retracement_level = ote_entry
        entry_bar = data.iloc[entry_idx]
        
        # Check confluence factors
        confluence_score = self._calculate_confluence_score(data, entry_idx, impulse['direction'])
        if confluence_score < self.confluence_min_score:
            return None
            
        # Look for entry trigger
        entry_trigger = self._identify_entry_trigger(data, entry_idx, impulse['direction'])
        if not entry_trigger:
            return None
            
        # Setup trade parameters
        if impulse['direction'] == 'bullish':
            direction = TradeDirection.LONG
            stop_loss = fib_levels['fib_786'] - (impulse['end_price'] - impulse['start_price']) * 0.05
            # Target: Measured move or next resistance
            take_profit = impulse['end_price'] + (impulse['end_price'] - impulse['start_price'])
            
        else:  # bearish
            direction = TradeDirection.SHORT
            stop_loss = fib_levels['fib_786'] + (impulse['start_price'] - impulse['end_price']) * 0.05
            # Target: Measured move or next support
            take_profit = impulse['end_price'] - (impulse['start_price'] - impulse['end_price'])
            
        # Validate risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward / risk < 2.0:  # Minimum 2:1 R:R for OTE
            return None
            
        return Signal(
            timestamp=entry_bar.name,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=min(0.9, confluence_score),
            strategy="OPTIMAL_TRADE_ENTRY",
            metadata={
                'impulse_direction': impulse['direction'],
                'impulse_size_pct': impulse['size_pct'],
                'retracement_level': retracement_level,
                'confluence_score': confluence_score,
                'entry_trigger': entry_trigger,
                'killzone': entry_bar['killzone'],
                'session': entry_bar['session'],
                'fib_levels': fib_levels,
                'volume_ratio': entry_bar['volume_ratio']
            }
        )
        
    def _calculate_fibonacci_levels(self, impulse: Dict) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels for impulse move"""
        start_price = impulse['start_price']
        end_price = impulse['end_price']
        
        if impulse['direction'] == 'bullish':
            # For bullish impulse, retracements go down from the high
            fib_levels = {
                'fib_236': end_price - (end_price - start_price) * 0.236,
                'fib_382': end_price - (end_price - start_price) * 0.382,
                'fib_500': end_price - (end_price - start_price) * 0.500,
                'fib_618': end_price - (end_price - start_price) * 0.618,
                'fib_786': end_price - (end_price - start_price) * 0.786
            }
        else:
            # For bearish impulse, retracements go up from the low
            fib_levels = {
                'fib_236': end_price + (start_price - end_price) * 0.236,
                'fib_382': end_price + (start_price - end_price) * 0.382,
                'fib_500': end_price + (start_price - end_price) * 0.500,
                'fib_618': end_price + (start_price - end_price) * 0.618,
                'fib_786': end_price + (start_price - end_price) * 0.786
            }
            
        return fib_levels
        
    def _find_ote_entry(self, data: pd.DataFrame, impulse: Dict, fib_levels: Dict, 
                       start_idx: int) -> Optional[Tuple[int, float, str]]:
        """Find entry within OTE zone"""
        # Define OTE zone boundaries
        ote_upper = fib_levels['fib_618']
        ote_lower = fib_levels['fib_786']
        
        if impulse['direction'] == 'bearish':
            ote_upper, ote_lower = ote_lower, ote_upper  # Swap for bearish
            
        # Look for price entering OTE zone
        search_end = min(start_idx + 50, len(data))  # Look up to 50 bars ahead
        
        for i in range(start_idx + 1, search_end):
            current_bar = data.iloc[i]
            current_price = current_bar['close']
            
            # Check if price is in OTE zone
            if impulse['direction'] == 'bullish':
                in_ote_zone = ote_lower <= current_price <= ote_upper
            else:
                in_ote_zone = ote_upper <= current_price <= ote_lower
                
            if in_ote_zone:
                # Determine which Fibonacci level we're closest to
                closest_level = 'fib_618'
                min_distance = abs(current_price - fib_levels['fib_618'])
                
                for level_name, level_price in fib_levels.items():
                    distance = abs(current_price - level_price)
                    if distance < min_distance:
                        min_distance = distance
                        closest_level = level_name
                        
                return (i, current_price, closest_level)
                
        return None
        
    def _calculate_confluence_score(self, data: pd.DataFrame, entry_idx: int, 
                                  impulse_direction: str) -> float:
        """Calculate confluence score for OTE entry"""
        entry_bar = data.iloc[entry_idx]
        confluence_score = 0.0
        
        # Factor 1: Killzone timing (0.3 points)
        if entry_bar['killzone'] != 'None':
            confluence_score += 0.3
            
        # Factor 2: Fair Value Gap presence (0.25 points)
        if impulse_direction == 'bullish' and entry_bar['fvg_bullish']:
            confluence_score += 0.25
        elif impulse_direction == 'bearish' and entry_bar['fvg_bearish']:
            confluence_score += 0.25
            
        # Factor 3: Order Block proximity (0.25 points)
        # Check for order blocks in recent bars
        lookback_range = max(0, entry_idx - 10)
        recent_data = data.iloc[lookback_range:entry_idx]
        
        if impulse_direction == 'bullish':
            recent_ob = recent_data['order_block_bullish'].any()
        else:
            recent_ob = recent_data['order_block_bearish'].any()
            
        if recent_ob:
            confluence_score += 0.25
            
        # Factor 4: Volume confirmation (0.2 points)
        if entry_bar['volume_ratio'] > 1.2:
            confluence_score += 0.2
            
        return confluence_score
        
    def _identify_entry_trigger(self, data: pd.DataFrame, entry_idx: int, 
                              impulse_direction: str) -> Optional[str]:
        """Identify entry trigger pattern"""
        # Look for trigger in next few bars
        search_end = min(entry_idx + self.entry_trigger_bars, len(data))
        
        for i in range(entry_idx, search_end):
            current_bar = data.iloc[i]
            
            if impulse_direction == 'bullish':
                # Look for bullish triggers
                if self._is_bullish_engulfing(data, i):
                    return 'bullish_engulfing'
                elif self._is_hammer_pattern(current_bar):
                    return 'hammer'
                elif current_bar['close'] > current_bar['open']:
                    return 'bullish_candle'
                    
            else:  # bearish
                # Look for bearish triggers
                if self._is_bearish_engulfing(data, i):
                    return 'bearish_engulfing'
                elif self._is_shooting_star_pattern(current_bar):
                    return 'shooting_star'
                elif current_bar['close'] < current_bar['open']:
                    return 'bearish_candle'
                    
        return None
        
    def _is_bullish_engulfing(self, data: pd.DataFrame, idx: int) -> bool:
        """Check for bullish engulfing pattern"""
        if idx < 1:
            return False
            
        current = data.iloc[idx]
        previous = data.iloc[idx-1]
        
        return (previous['close'] < previous['open'] and  # Previous bearish
                current['close'] > current['open'] and   # Current bullish
                current['open'] < previous['close'] and  # Opens below previous close
                current['close'] > previous['open'])     # Closes above previous open
                
    def _is_bearish_engulfing(self, data: pd.DataFrame, idx: int) -> bool:
        """Check for bearish engulfing pattern"""
        if idx < 1:
            return False
            
        current = data.iloc[idx]
        previous = data.iloc[idx-1]
        
        return (previous['close'] > previous['open'] and  # Previous bullish
                current['close'] < current['open'] and   # Current bearish
                current['open'] > previous['close'] and  # Opens above previous close
                current['close'] < previous['open'])     # Closes below previous open
                
    def _is_hammer_pattern(self, bar: pd.Series) -> bool:
        """Check for hammer candlestick pattern"""
        body_size = abs(bar['close'] - bar['open'])
        lower_shadow = bar['open'] - bar['low'] if bar['close'] > bar['open'] else bar['close'] - bar['low']
        upper_shadow = bar['high'] - bar['close'] if bar['close'] > bar['open'] else bar['high'] - bar['open']
        
        return (lower_shadow > 2 * body_size and  # Long lower shadow
                upper_shadow < body_size * 0.5)   # Small upper shadow
                
    def _is_shooting_star_pattern(self, bar: pd.Series) -> bool:
        """Check for shooting star candlestick pattern"""
        body_size = abs(bar['close'] - bar['open'])
        lower_shadow = bar['open'] - bar['low'] if bar['close'] > bar['open'] else bar['close'] - bar['low']
        upper_shadow = bar['high'] - bar['close'] if bar['close'] > bar['open'] else bar['high'] - bar['open']
        
        return (upper_shadow > 2 * body_size and  # Long upper shadow
                lower_shadow < body_size * 0.5)   # Small lower shadow
                
    def identify_exit_signals(self, data: pd.DataFrame, open_trades: List) -> List[Dict]:
        """Identify exit signals for OTE trades"""
        exit_signals = []
        
        for trade in open_trades:
            current_bar = data.iloc[-1]
            current_price = current_bar['close']
            
            # Check for invalidation (retracement beyond 78.6%)
            if self._check_ote_invalidation(data, trade):
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'ote_invalidation',
                    'timestamp': current_bar.name
                })
                continue
                
            # Check for momentum shift
            if self._check_momentum_shift(data, trade):
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'momentum_shift',
                    'timestamp': current_bar.name
                })
                
        return exit_signals
        
    def _check_ote_invalidation(self, data: pd.DataFrame, trade: Dict) -> bool:
        """Check if OTE setup is invalidated"""
        current_price = data['close'].iloc[-1]
        fib_levels = trade['metadata']['fib_levels']
        impulse_direction = trade['metadata']['impulse_direction']
        
        # Check if price moved beyond 78.6% level (invalidation)
        if impulse_direction == 'bullish' and current_price < fib_levels['fib_786']:
            return True
        elif impulse_direction == 'bearish' and current_price > fib_levels['fib_786']:
            return True
            
        return False
        
    def _check_momentum_shift(self, data: pd.DataFrame, trade: Dict) -> bool:
        """Check for momentum shift against trade direction"""
        recent_momentum = data['momentum'].tail(3).mean()
        
        if trade['direction'] == TradeDirection.LONG and recent_momentum < -0.005:
            return True
        elif trade['direction'] == TradeDirection.SHORT and recent_momentum > 0.005:
            return True
            
        return False
        
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information"""
        return {
            'name': 'Optimal Trade Entry (OTE)',
            'description': 'Enters at optimal Fibonacci retracement levels (61.8%-78.6%) after impulse moves',
            'timeframes': ['15m', '1h', '4h'],
            'sessions': ['London', 'New York'],
            'risk_level': 'Medium',
            'parameters': {
                'min_impulse_size': self.min_impulse_size,
                'ote_zone_start': self.ote_zone_start,
                'ote_zone_end': self.ote_zone_end,
                'impulse_lookback': self.impulse_lookback,
                'confluence_min_score': self.confluence_min_score,
                'entry_trigger_bars': self.entry_trigger_bars
            }
        }