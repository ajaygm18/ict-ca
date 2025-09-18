"""
Afternoon Reversal Strategy - Strategy 14

The Afternoon Reversal strategy targets reversals that occur during the NY afternoon
session, typically between 2-4 PM EST. This strategy capitalizes on profit-taking,
institutional rebalancing, and position adjustments that create reversal opportunities.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from .base_strategy import BaseICTStrategy, Signal, TradeDirection

class AfternoonReversalStrategy(BaseICTStrategy):
    """
    Afternoon Reversal Strategy Implementation
    
    This strategy identifies reversal opportunities during the NY afternoon session:
    1. Morning/midday trend exhaustion analysis
    2. Key level confluence identification
    3. Reversal pattern recognition (divergences, exhaustion)
    4. Volume analysis for institutional participation
    
    Entry Conditions:
    - Clear trend established during morning session
    - Price approaching key resistance/support levels
    - Reversal signals (divergence, exhaustion patterns)
    - Volume confirms institutional interest
    - Time window: 2-4 PM EST (afternoon session)
    
    Exit Conditions:
    - Reversal target reached (50-78.6% retracement)
    - End of session approaching (5 PM EST)
    - Trend resumption signals
    - Stop loss beyond reversal level
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.afternoon_start_hour = config.get('afternoon_start_hour', 14)  # 2 PM EST
        self.afternoon_end_hour = config.get('afternoon_end_hour', 16)      # 4 PM EST
        self.trend_min_hours = config.get('trend_min_hours', 4)             # Minimum trend duration
        self.reversal_confirmation_bars = config.get('reversal_confirmation_bars', 3)
        self.volume_divergence_threshold = config.get('volume_divergence_threshold', 0.8)
        self.rsi_reversal_levels = config.get('rsi_reversal_levels', {'overbought': 75, 'oversold': 25})
        
    def setup_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Setup indicators for afternoon reversal analysis"""
        df = data.copy()
        
        # Time and session analysis
        df['hour'] = pd.to_datetime(df.index).hour
        df['session'] = self._identify_session(df)
        df['is_afternoon'] = ((df['hour'] >= self.afternoon_start_hour) & 
                             (df['hour'] <= self.afternoon_end_hour) & 
                             (df['session'] == 'New York'))
        
        # Trend analysis indicators
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['trend_direction'] = np.where(df['ema_20'] > df['ema_50'], 1, -1)
        df['trend_strength'] = abs(df['ema_20'] - df['ema_50']) / df['close']
        
        # Momentum and reversal indicators
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['macd'], df['macd_signal'], df['macd_histogram'] = self._calculate_macd(df['close'])
        df['momentum'] = df['close'].pct_change(5)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(15).mean()
        
        # Price structure and levels
        df['atr'] = self._calculate_atr(df, 14)
        df['swing_high'] = self._identify_swing_highs(df, 5)
        df['swing_low'] = self._identify_swing_lows(df, 5)
        
        # Daily levels for context
        df['daily_high'] = df.groupby(df.index.date)['high'].transform('max')
        df['daily_low'] = df.groupby(df.index.date)['low'].transform('min')
        df['daily_vwap'] = self._calculate_daily_vwap(df)
        
        # Retracement levels
        df['fib_618'] = self._calculate_fibonacci_levels(df, 0.618)
        df['fib_50'] = self._calculate_fibonacci_levels(df, 0.5)
        df['fib_382'] = self._calculate_fibonacci_levels(df, 0.382)
        
        return df
        
    def identify_entry_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Identify afternoon reversal entry opportunities"""
        signals = []
        
        if len(data) < 100:
            return signals
            
        # Look for reversal setups during afternoon hours
        afternoon_indices = data[data['is_afternoon']].index
        
        for timestamp in afternoon_indices:
            idx = data.index.get_loc(timestamp)
            
            if idx < 50 or idx >= len(data) - 5:  # Need sufficient data
                continue
                
            signal = self._analyze_afternoon_reversal_setup(data, idx)
            if signal:
                signals.append(signal)
                
        return signals
        
    def _analyze_afternoon_reversal_setup(self, data: pd.DataFrame, current_idx: int) -> Optional[Signal]:
        """Analyze afternoon reversal setup at given index"""
        current_bar = data.iloc[current_idx]
        
        # Validate afternoon timing
        if not current_bar['is_afternoon']:
            return None
            
        # Analyze established trend
        trend_analysis = self._analyze_established_trend(data, current_idx)
        if not trend_analysis:
            return None
            
        trend_direction, trend_strength, trend_start_idx = trend_analysis
        
        # Check for reversal confluence
        reversal_confluence = self._check_reversal_confluence(data, current_idx, trend_direction)
        if not reversal_confluence:
            return None
            
        confluence_score, key_levels = reversal_confluence
        
        # Look for reversal signals
        reversal_signals = self._identify_reversal_signals(data, current_idx, trend_direction)
        if not reversal_signals:
            return None
            
        signal_strength, reversal_type = reversal_signals
        
        # Confirm with volume analysis
        volume_confirmation = self._analyze_reversal_volume(data, current_idx, trend_direction)
        if not volume_confirmation:
            return None
            
        # Setup trade parameters
        if trend_direction == 1:  # Bullish trend reversing to bearish
            direction = TradeDirection.SHORT
            entry_price = current_bar['close']
            stop_loss = self._find_recent_high(data, current_idx, 10)
            
            # Target based on retracement levels
            target_level = self._calculate_reversal_target(data, current_idx, 'bearish', key_levels)
            take_profit = target_level
            
        else:  # Bearish trend reversing to bullish
            direction = TradeDirection.LONG
            entry_price = current_bar['close']
            stop_loss = self._find_recent_low(data, current_idx, 10)
            
            target_level = self._calculate_reversal_target(data, current_idx, 'bullish', key_levels)
            take_profit = target_level
            
        # Validate risk/reward
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward / risk < 1.0:  # Minimum 1:1 R:R
            return None
            
        # Calculate confidence based on multiple factors
        confidence = min(0.9, (confluence_score + signal_strength + volume_confirmation) / 3)
        
        return Signal(
            timestamp=current_bar.name,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            strategy="AFTERNOON_REVERSAL",
            metadata={
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'confluence_score': confluence_score,
                'reversal_type': reversal_type,
                'key_levels': key_levels,
                'volume_confirmation': volume_confirmation,
                'session': current_bar['session'],
                'rsi': current_bar['rsi'],
                'time_in_session': current_bar['hour']
            }
        )
        
    def _analyze_established_trend(self, data: pd.DataFrame, current_idx: int) -> Optional[tuple]:
        """Analyze if there's an established trend to reverse"""
        if current_idx < self.trend_min_hours * 4:  # Need enough history
            return None
            
        # Look back for trend establishment
        lookback_data = data.iloc[current_idx - self.trend_min_hours * 4:current_idx]
        
        # Trend direction consistency
        trend_consistency = (lookback_data['trend_direction'] == lookback_data['trend_direction'].iloc[-1]).mean()
        
        if trend_consistency < 0.7:  # Need at least 70% consistency
            return None
            
        current_trend = lookback_data['trend_direction'].iloc[-1]
        avg_trend_strength = lookback_data['trend_strength'].mean()
        
        if avg_trend_strength < 0.01:  # Minimum trend strength
            return None
            
        # Find trend start
        trend_start_idx = self._find_trend_start(data, current_idx, current_trend)
        
        return (current_trend, avg_trend_strength, trend_start_idx)
        
    def _find_trend_start(self, data: pd.DataFrame, current_idx: int, trend_direction: int) -> int:
        """Find where the current trend started"""
        for i in range(current_idx - 1, max(0, current_idx - 50), -1):
            if data.iloc[i]['trend_direction'] != trend_direction:
                return i + 1
        return max(0, current_idx - 50)
        
    def _check_reversal_confluence(self, data: pd.DataFrame, current_idx: int, 
                                 trend_direction: int) -> Optional[tuple]:
        """Check for confluence of reversal factors"""
        current_bar = data.iloc[current_idx]
        confluence_factors = []
        key_levels = []
        
        # Factor 1: Key level proximity
        daily_high = current_bar['daily_high']
        daily_low = current_bar['daily_low']
        current_price = current_bar['close']
        
        if trend_direction == 1:  # Bullish trend, look for resistance
            if abs(current_price - daily_high) / current_price < 0.005:
                confluence_factors.append(0.3)
                key_levels.append(('daily_high', daily_high))
                
        else:  # Bearish trend, look for support
            if abs(current_price - daily_low) / current_price < 0.005:
                confluence_factors.append(0.3)
                key_levels.append(('daily_low', daily_low))
                
        # Factor 2: Fibonacci levels
        fib_proximity = self._check_fibonacci_proximity(current_bar, trend_direction)
        if fib_proximity:
            confluence_factors.append(0.25)
            key_levels.append(fib_proximity)
            
        # Factor 3: VWAP proximity
        if abs(current_price - current_bar['daily_vwap']) / current_price < 0.003:
            confluence_factors.append(0.2)
            key_levels.append(('vwap', current_bar['daily_vwap']))
            
        # Factor 4: Previous swing levels
        swing_level = self._check_swing_level_proximity(data, current_idx, trend_direction)
        if swing_level:
            confluence_factors.append(0.25)
            key_levels.append(swing_level)
            
        confluence_score = sum(confluence_factors)
        
        if confluence_score >= 0.5:  # Minimum confluence required
            return (confluence_score, key_levels)
            
        return None
        
    def _check_fibonacci_proximity(self, current_bar: pd.Series, trend_direction: int) -> Optional[tuple]:
        """Check proximity to Fibonacci retracement levels"""
        current_price = current_bar['close']
        
        fib_levels = [
            ('fib_382', current_bar['fib_382']),
            ('fib_50', current_bar['fib_50']),
            ('fib_618', current_bar['fib_618'])
        ]
        
        for level_name, level_value in fib_levels:
            if not pd.isna(level_value) and abs(current_price - level_value) / current_price < 0.003:
                return (level_name, level_value)
                
        return None
        
    def _check_swing_level_proximity(self, data: pd.DataFrame, current_idx: int, 
                                   trend_direction: int) -> Optional[tuple]:
        """Check proximity to previous swing levels"""
        current_price = data.iloc[current_idx]['close']
        
        # Look back for swing levels
        lookback_data = data.iloc[max(0, current_idx-50):current_idx]
        
        if trend_direction == 1:  # Look for swing highs (resistance)
            swing_highs = lookback_data[lookback_data['swing_high'] == True]['high']
            for swing_high in swing_highs:
                if abs(current_price - swing_high) / current_price < 0.005:
                    return ('swing_high', swing_high)
                    
        else:  # Look for swing lows (support)
            swing_lows = lookback_data[lookback_data['swing_low'] == True]['low']
            for swing_low in swing_lows:
                if abs(current_price - swing_low) / current_price < 0.005:
                    return ('swing_low', swing_low)
                    
        return None
        
    def _identify_reversal_signals(self, data: pd.DataFrame, current_idx: int, 
                                 trend_direction: int) -> Optional[tuple]:
        """Identify reversal signals (divergences, exhaustion patterns)"""
        current_bar = data.iloc[current_idx]
        signal_strength = 0
        reversal_types = []
        
        # RSI divergence and extremes
        rsi_signal = self._check_rsi_reversal_signals(data, current_idx, trend_direction)
        if rsi_signal:
            signal_strength += rsi_signal
            reversal_types.append('rsi_divergence')
            
        # MACD divergence
        macd_signal = self._check_macd_divergence(data, current_idx, trend_direction)
        if macd_signal:
            signal_strength += macd_signal
            reversal_types.append('macd_divergence')
            
        # Momentum exhaustion
        momentum_signal = self._check_momentum_exhaustion(data, current_idx, trend_direction)
        if momentum_signal:
            signal_strength += momentum_signal
            reversal_types.append('momentum_exhaustion')
            
        if signal_strength >= 0.3:  # Minimum signal strength
            return (signal_strength, reversal_types)
            
        return None
        
    def _check_rsi_reversal_signals(self, data: pd.DataFrame, current_idx: int, 
                                  trend_direction: int) -> float:
        """Check RSI for reversal signals"""
        current_rsi = data.iloc[current_idx]['rsi']
        
        if trend_direction == 1:  # Bullish trend, look for overbought
            if current_rsi >= self.rsi_reversal_levels['overbought']:
                return 0.3
        else:  # Bearish trend, look for oversold
            if current_rsi <= self.rsi_reversal_levels['oversold']:
                return 0.3
                
        return 0
        
    def _check_macd_divergence(self, data: pd.DataFrame, current_idx: int, 
                             trend_direction: int) -> float:
        """Check MACD for divergence signals"""
        if current_idx < 10:
            return 0
            
        recent_data = data.iloc[current_idx-10:current_idx+1]
        
        # Simple divergence check (price vs MACD)
        price_direction = recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]
        macd_direction = recent_data['macd'].iloc[-1] - recent_data['macd'].iloc[0]
        
        if trend_direction == 1 and price_direction > 0 and macd_direction < 0:
            return 0.25  # Bearish divergence
        elif trend_direction == -1 and price_direction < 0 and macd_direction > 0:
            return 0.25  # Bullish divergence
            
        return 0
        
    def _check_momentum_exhaustion(self, data: pd.DataFrame, current_idx: int, 
                                 trend_direction: int) -> float:
        """Check for momentum exhaustion"""
        if current_idx < 5:
            return 0
            
        recent_momentum = data['momentum'].iloc[current_idx-4:current_idx+1].mean()
        
        if trend_direction == 1 and recent_momentum < 0.001:  # Bullish trend losing steam
            return 0.2
        elif trend_direction == -1 and recent_momentum > -0.001:  # Bearish trend losing steam
            return 0.2
            
        return 0
        
    def _analyze_reversal_volume(self, data: pd.DataFrame, current_idx: int, 
                               trend_direction: int) -> float:
        """Analyze volume for reversal confirmation"""
        current_bar = data.iloc[current_idx]
        
        # Volume spike confirmation
        volume_spike = current_bar['volume_ratio'] > 1.3
        
        # Volume trend divergence
        volume_trend = current_bar['volume_trend']
        volume_divergence = volume_trend < self.volume_divergence_threshold
        
        confirmation_score = 0
        
        if volume_spike:
            confirmation_score += 0.3
            
        if volume_divergence:
            confirmation_score += 0.2
            
        return confirmation_score if confirmation_score >= 0.3 else 0
        
    def _calculate_reversal_target(self, data: pd.DataFrame, current_idx: int, 
                                 reversal_direction: str, key_levels: List) -> float:
        """Calculate reversal target based on retracement levels"""
        current_price = data.iloc[current_idx]['close']
        
        # Use Fibonacci retracement as primary target
        if reversal_direction == 'bullish':
            fib_target = data.iloc[current_idx]['fib_618']
            if not pd.isna(fib_target) and fib_target > current_price:
                return fib_target
            # Fallback to daily VWAP
            return data.iloc[current_idx]['daily_vwap']
            
        else:  # bearish
            fib_target = data.iloc[current_idx]['fib_618']
            if not pd.isna(fib_target) and fib_target < current_price:
                return fib_target
            # Fallback to daily VWAP
            return data.iloc[current_idx]['daily_vwap']
            
    def _calculate_daily_vwap(self, data: pd.DataFrame) -> pd.Series:
        """Calculate daily Volume Weighted Average Price"""
        data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
        data['volume_price'] = data['typical_price'] * data['volume']
        
        daily_groups = data.groupby(data.index.date)
        daily_vwap = daily_groups.apply(
            lambda x: x['volume_price'].sum() / x['volume'].sum()
        )
        
        return data.index.to_series().dt.date.map(daily_vwap)
        
    def _calculate_fibonacci_levels(self, data: pd.DataFrame, ratio: float) -> pd.Series:
        """Calculate Fibonacci retracement levels"""
        # This is a simplified implementation
        # In practice, you'd identify swing highs/lows and calculate retracements
        daily_high = data.groupby(data.index.date)['high'].transform('max')
        daily_low = data.groupby(data.index.date)['low'].transform('min')
        
        return daily_low + (daily_high - daily_low) * ratio
        
    def _calculate_macd(self, close_prices: pd.Series, fast=12, slow=26, signal=9) -> tuple:
        """Calculate MACD indicator"""
        ema_fast = close_prices.ewm(span=fast).mean()
        ema_slow = close_prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
        
    def identify_exit_signals(self, data: pd.DataFrame, open_trades: List) -> List[Dict]:
        """Identify exit signals for afternoon reversal trades"""
        exit_signals = []
        
        for trade in open_trades:
            current_bar = data.iloc[-1]
            current_price = current_bar['close']
            
            # Exit near session close (5 PM EST)
            if current_bar['hour'] >= 17:
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'session_close',
                    'timestamp': current_bar.name
                })
                continue
                
            # Check for trend resumption
            if self._check_trend_resumption(data, trade):
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'trend_resumption',
                    'timestamp': current_bar.name
                })
                
        return exit_signals
        
    def _check_trend_resumption(self, data: pd.DataFrame, trade: Dict) -> bool:
        """Check if original trend is resuming"""
        original_trend = trade['metadata']['trend_direction']
        current_trend = data['trend_direction'].iloc[-1]
        
        # Check if trend has aligned with original direction for several bars
        recent_trend_consistency = (data['trend_direction'].tail(3) == original_trend).all()
        
        return recent_trend_consistency
        
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information"""
        return {
            'name': 'Afternoon Reversal',
            'description': 'Targets reversals during NY afternoon session with confluence analysis',
            'timeframes': ['15m', '1h'],
            'sessions': ['New York'],
            'risk_level': 'Medium-High',
            'parameters': {
                'afternoon_start_hour': self.afternoon_start_hour,
                'afternoon_end_hour': self.afternoon_end_hour,
                'trend_min_hours': self.trend_min_hours,
                'reversal_confirmation_bars': self.reversal_confirmation_bars,
                'volume_divergence_threshold': self.volume_divergence_threshold,
                'rsi_reversal_levels': self.rsi_reversal_levels
            }
        }