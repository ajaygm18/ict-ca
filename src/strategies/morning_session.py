"""
Morning Session Strategy - Strategy 13

The Morning Session strategy focuses on the first 2-4 hours of the London session,
capturing the initial directional bias and momentum as Asian session liquidity
gets absorbed and European institutional flow begins.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from .base_strategy import BaseICTStrategy, Signal, TradeDirection

class MorningSessionStrategy(BaseICTStrategy):
    """
    Morning Session Strategy Implementation
    
    This strategy capitalizes on the morning session dynamics:
    1. Asian session range analysis and liquidity identification
    2. London open reaction and initial bias establishment
    3. Momentum continuation or reversal patterns
    4. Volume confirmation of institutional participation
    
    Entry Conditions:
    - Clear break of Asian session range with volume
    - Morning session bias confirmed (first 2 hours)
    - Price shows intent to continue directional move
    - Volume supports institutional involvement
    
    Exit Conditions:
    - Target reached (opposite end of daily range)
    - Morning session momentum exhausted
    - Reversal patterns emerge
    - Mid-session consolidation begins
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.asian_session_min_hours = config.get('asian_session_min_hours', 4)
        self.morning_session_hours = config.get('morning_session_hours', 4)
        self.min_range_break = config.get('min_range_break', 0.0015)  # 0.15% minimum break
        self.volume_threshold = config.get('volume_threshold', 1.4)
        self.momentum_confirmation_bars = config.get('momentum_confirmation_bars', 3)
        
    def setup_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Setup indicators for morning session analysis"""
        df = data.copy()
        
        # Session identification with precise timing
        df['session'] = self._identify_session(df)
        df['hour'] = pd.to_datetime(df.index).hour
        df['date'] = pd.to_datetime(df.index).date
        
        # Session-specific indicators
        df['is_asian'] = df['session'] == 'Asian'
        df['is_london_open'] = (df['session'] == 'London') & (df['hour'] >= 8) & (df['hour'] <= 12)
        df['is_morning_session'] = df['is_london_open']
        
        # Asian session range calculation
        df = self._calculate_asian_range(df)
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_spike'] = df['volume_ratio'] > self.volume_threshold
        
        # Momentum indicators
        df['momentum_15m'] = df['close'].pct_change(1)
        df['momentum_1h'] = df['close'].pct_change(4)
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        # Price structure
        df['atr'] = self._calculate_atr(df, 14)
        df['range'] = df['high'] - df['low']
        df['range_percentile'] = df['range'].rolling(20).rank(pct=True)
        
        # Swing points for structure analysis
        df['swing_high'] = self._identify_swing_highs(df, 3)
        df['swing_low'] = self._identify_swing_lows(df, 3)
        
        return df
        
    def _calculate_asian_range(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate Asian session range for each day"""
        df = data.copy()
        
        # Initialize Asian range columns
        df['asian_high'] = np.nan
        df['asian_low'] = np.nan
        df['asian_range'] = np.nan
        df['asian_midpoint'] = np.nan
        
        # Group by date and calculate Asian session ranges
        for date in df['date'].unique():
            day_data = df[df['date'] == date]
            asian_data = day_data[day_data['is_asian']]
            
            if len(asian_data) >= self.asian_session_min_hours:
                asian_high = asian_data['high'].max()
                asian_low = asian_data['low'].min()
                asian_range = asian_high - asian_low
                asian_midpoint = (asian_high + asian_low) / 2
                
                # Apply to all bars of this date
                date_mask = df['date'] == date
                df.loc[date_mask, 'asian_high'] = asian_high
                df.loc[date_mask, 'asian_low'] = asian_low
                df.loc[date_mask, 'asian_range'] = asian_range
                df.loc[date_mask, 'asian_midpoint'] = asian_midpoint
                
        return df
        
    def identify_entry_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Identify morning session entry opportunities"""
        signals = []
        
        if len(data) < 50:
            return signals
            
        # Find London session opens
        london_opens = self._find_london_opens(data)
        
        for open_idx in london_opens:
            if open_idx + self.morning_session_hours * 4 >= len(data):  # Need enough data
                continue
                
            signal = self._analyze_morning_session_setup(data, open_idx)
            if signal:
                signals.append(signal)
                
        return signals
        
    def _find_london_opens(self, data: pd.DataFrame) -> List[int]:
        """Find indices where London session opens"""
        london_opens = []
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            if (row['session'] == 'London' and 
                row['hour'] == 8 and  # 8 AM London time
                not pd.isna(row['asian_high'])):  # Must have Asian range data
                london_opens.append(i)
                
        return london_opens
        
    def _analyze_morning_session_setup(self, data: pd.DataFrame, open_idx: int) -> Optional[Signal]:
        """Analyze morning session for trading setup"""
        london_open_bar = data.iloc[open_idx]
        
        # Validate Asian session data
        if pd.isna(london_open_bar['asian_high']) or pd.isna(london_open_bar['asian_low']):
            return None
            
        asian_high = london_open_bar['asian_high']
        asian_low = london_open_bar['asian_low']
        asian_range = london_open_bar['asian_range']
        
        # Look for range break in the first few hours
        morning_end = min(open_idx + self.morning_session_hours * 4, len(data))
        morning_data = data.iloc[open_idx:morning_end]
        
        # Identify range break
        range_break = self._identify_range_break(morning_data, asian_high, asian_low)
        if not range_break:
            return None
            
        break_direction, break_idx, break_price = range_break
        actual_break_idx = open_idx + break_idx
        break_bar = data.iloc[actual_break_idx]
        
        # Validate break strength
        if not self._validate_range_break(data, actual_break_idx, break_direction, asian_range):
            return None
            
        # Look for entry confirmation
        entry_signal = self._find_morning_entry(data, actual_break_idx, break_direction)
        if not entry_signal:
            return None
            
        entry_idx, entry_price = entry_signal
        entry_bar = data.iloc[entry_idx]
        
        # Setup trade parameters
        if break_direction == 'upward':
            direction = TradeDirection.LONG
            stop_loss = asian_low - (asian_range * 0.1)
            # Target based on daily range projection
            take_profit = asian_high + (asian_range * 1.5)
            
        else:  # downward
            direction = TradeDirection.SHORT
            stop_loss = asian_high + (asian_range * 0.1)
            take_profit = asian_low - (asian_range * 1.5)
            
        # Risk management validation
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward / risk < 1.2:  # Minimum 1.2:1 R:R
            return None
            
        return Signal(
            timestamp=entry_bar.name,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=0.7,
            strategy="MORNING_SESSION",
            metadata={
                'asian_high': asian_high,
                'asian_low': asian_low,
                'asian_range': asian_range,
                'break_direction': break_direction,
                'break_price': break_price,
                'volume_ratio': entry_bar['volume_ratio'],
                'session': entry_bar['session'],
                'momentum_1h': entry_bar['momentum_1h']
            }
        )
        
    def _identify_range_break(self, morning_data: pd.DataFrame, 
                            asian_high: float, asian_low: float) -> Optional[tuple]:
        """Identify break of Asian session range"""
        for i, (timestamp, bar) in enumerate(morning_data.iterrows()):
            # Check for upward break
            if bar['high'] > asian_high:
                break_size = (bar['high'] - asian_high) / asian_high
                if break_size >= self.min_range_break:
                    return ('upward', i, bar['high'])
                    
            # Check for downward break
            elif bar['low'] < asian_low:
                break_size = (asian_low - bar['low']) / asian_low
                if break_size >= self.min_range_break:
                    return ('downward', i, bar['low'])
                    
        return None
        
    def _validate_range_break(self, data: pd.DataFrame, break_idx: int, 
                            break_direction: str, asian_range: float) -> bool:
        """Validate the strength and quality of range break"""
        break_bar = data.iloc[break_idx]
        
        # Volume confirmation
        if not break_bar['volume_spike']:
            return False
            
        # Range expansion check
        if break_bar['range'] < asian_range * 0.3:  # Break should show some range expansion
            return False
            
        # Momentum confirmation
        if break_direction == 'upward':
            return break_bar['momentum_15m'] > 0.001
        else:
            return break_bar['momentum_15m'] < -0.001
            
    def _find_morning_entry(self, data: pd.DataFrame, break_idx: int, 
                          break_direction: str) -> Optional[tuple]:
        """Find optimal entry after range break"""
        # Look for entry in next few bars
        search_end = min(break_idx + self.momentum_confirmation_bars + 2, len(data))
        
        for i in range(break_idx, search_end):
            current_bar = data.iloc[i]
            
            # Confirm momentum continuation
            if not self._confirm_momentum_continuation(data, i, break_direction):
                continue
                
            # Price action confirmation
            if self._confirm_morning_entry_pattern(current_bar, break_direction):
                return (i, current_bar['close'])
                
        return None
        
    def _confirm_momentum_continuation(self, data: pd.DataFrame, idx: int, 
                                     break_direction: str) -> bool:
        """Confirm momentum is continuing in break direction"""
        if idx < self.momentum_confirmation_bars:
            return False
            
        recent_bars = data.iloc[idx-self.momentum_confirmation_bars:idx+1]
        
        if break_direction == 'upward':
            # Look for higher highs or consistent upward momentum
            higher_highs = (recent_bars['high'].diff() > 0).sum() >= 2
            positive_momentum = recent_bars['momentum_15m'].mean() > 0
            return higher_highs or positive_momentum
            
        else:  # downward
            # Look for lower lows or consistent downward momentum
            lower_lows = (recent_bars['low'].diff() < 0).sum() >= 2
            negative_momentum = recent_bars['momentum_15m'].mean() < 0
            return lower_lows or negative_momentum
            
    def _confirm_morning_entry_pattern(self, entry_bar: pd.Series, break_direction: str) -> bool:
        """Confirm entry with price action pattern"""
        if break_direction == 'upward':
            # Look for bullish candle or strong close
            return (entry_bar['close'] > entry_bar['open'] or 
                   entry_bar['close'] > (entry_bar['high'] + entry_bar['low']) / 2)
        else:
            # Look for bearish candle or weak close
            return (entry_bar['close'] < entry_bar['open'] or 
                   entry_bar['close'] < (entry_bar['high'] + entry_bar['low']) / 2)
                   
    def identify_exit_signals(self, data: pd.DataFrame, open_trades: List) -> List[Dict]:
        """Identify exit signals for morning session trades"""
        exit_signals = []
        
        for trade in open_trades:
            current_bar = data.iloc[-1]
            current_price = current_bar['close']
            
            # Exit at end of morning session (12 PM London)
            if current_bar['hour'] >= 12 and current_bar['session'] == 'London':
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'session_end',
                    'timestamp': current_bar.name
                })
                continue
                
            # Check for momentum exhaustion
            if self._check_morning_momentum_exhaustion(data, trade):
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'momentum_exhaustion',
                    'timestamp': current_bar.name
                })
                continue
                
            # Check for reversal patterns
            if self._check_morning_reversal(data, trade):
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'reversal_pattern',
                    'timestamp': current_bar.name
                })
                
        return exit_signals
        
    def _check_morning_momentum_exhaustion(self, data: pd.DataFrame, trade: Dict) -> bool:
        """Check for momentum exhaustion in morning session"""
        recent_momentum = data['momentum_15m'].tail(3).mean()
        current_rsi = data['rsi'].iloc[-1]
        
        if trade['direction'] == TradeDirection.LONG:
            return current_rsi > 75 or recent_momentum < 0.0005
        else:
            return current_rsi < 25 or recent_momentum > -0.0005
            
    def _check_morning_reversal(self, data: pd.DataFrame, trade: Dict) -> bool:
        """Check for reversal patterns during morning session"""
        if len(data) < 3:
            return False
            
        recent_bars = data.tail(3)
        
        if trade['direction'] == TradeDirection.LONG:
            # Look for bearish reversal patterns
            bearish_engulfing = (recent_bars['close'].iloc[-1] < recent_bars['open'].iloc[-2] and
                               recent_bars['open'].iloc[-1] > recent_bars['close'].iloc[-2])
            volume_climax = recent_bars['volume_ratio'].iloc[-1] > 2.0 and recent_bars['close'].iloc[-1] < recent_bars['open'].iloc[-1]
            return bearish_engulfing or volume_climax
            
        else:
            # Look for bullish reversal patterns
            bullish_engulfing = (recent_bars['close'].iloc[-1] > recent_bars['open'].iloc[-2] and
                               recent_bars['open'].iloc[-1] < recent_bars['close'].iloc[-2])
            volume_climax = recent_bars['volume_ratio'].iloc[-1] > 2.0 and recent_bars['close'].iloc[-1] > recent_bars['open'].iloc[-1]
            return bullish_engulfing or volume_climax
            
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information"""
        return {
            'name': 'Morning Session',
            'description': 'Captures morning session momentum after Asian range breaks',
            'timeframes': ['15m', '1h'],
            'sessions': ['London'],
            'risk_level': 'Medium',
            'parameters': {
                'asian_session_min_hours': self.asian_session_min_hours,
                'morning_session_hours': self.morning_session_hours,
                'min_range_break': self.min_range_break,
                'volume_threshold': self.volume_threshold,
                'momentum_confirmation_bars': self.momentum_confirmation_bars
            }
        }