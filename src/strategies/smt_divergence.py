"""
SMT Divergence Strategy - Strategy 9

Smart Money Technique (SMT) Divergence looks for divergences between correlated pairs
to identify when institutional money is moving contrary to retail expectations.
This strategy identifies divergence patterns and trades the correction.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from .base_strategy import BaseICTStrategy, Signal, TradeDirection

class SMTDivergenceStrategy(BaseICTStrategy):
    """
    SMT Divergence Strategy Implementation
    
    This strategy identifies Smart Money Technique divergences between correlated
    currency pairs or indices to spot institutional positioning against retail flow.
    
    Entry Conditions:
    - Divergence detected between correlated instruments
    - Price action confirmation (rejection candle or break of structure)
    - Volume confirmation showing institutional activity
    - Session alignment (preferably during major session overlaps)
    
    Exit Conditions:
    - Divergence corrected (instruments realign)
    - Stop loss hit (beyond recent swing high/low)
    - Take profit at predetermined R:R ratio
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.correlation_threshold = config.get('correlation_threshold', 0.7)
        self.divergence_periods = config.get('divergence_periods', 20)
        self.confirmation_candles = config.get('confirmation_candles', 3)
        self.volume_threshold = config.get('volume_threshold', 1.5)
        self.correlation_data = None
        
    def setup_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Setup indicators for SMT divergence detection"""
        df = data.copy()
        
        # Price momentum indicators
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        df['momentum'] = df['close'].pct_change(self.divergence_periods)
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price structure levels
        df['swing_high'] = self._identify_swing_highs(df, 5)
        df['swing_low'] = self._identify_swing_lows(df, 5)
        
        # Session identification
        df['session'] = self._identify_session(df)
        df['major_session'] = df['session'].isin(['London', 'New York', 'Overlap'])
        
        return df
        
    def identify_entry_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Identify SMT divergence entry opportunities"""
        signals = []
        
        if len(data) < self.divergence_periods + 10:
            return signals
            
        # Note: In real implementation, you would need correlation data
        # For this example, we'll simulate divergence detection
        for i in range(self.divergence_periods, len(data) - self.confirmation_candles):
            current_bar = data.iloc[i]
            
            # Skip if not in major session
            if not current_bar['major_session']:
                continue
                
            # Simulate divergence detection (in practice, compare with correlated instrument)
            price_momentum = data['momentum'].iloc[i]
            rsi_divergence = self._detect_rsi_divergence(data, i)
            volume_confirmation = current_bar['volume_ratio'] > self.volume_threshold
            
            if rsi_divergence and volume_confirmation:
                # Determine direction based on divergence type
                if rsi_divergence == 'bearish' and price_momentum > 0:
                    # Price making higher highs but RSI making lower highs
                    direction = TradeDirection.SHORT
                    entry_price = current_bar['close']
                    stop_loss = self._find_recent_high(data, i, 10)
                    
                elif rsi_divergence == 'bullish' and price_momentum < 0:
                    # Price making lower lows but RSI making higher lows
                    direction = TradeDirection.LONG
                    entry_price = current_bar['close']
                    stop_loss = self._find_recent_low(data, i, 10)
                    
                else:
                    continue
                    
                # Confirm with price action
                if self._confirm_price_action(data, i, direction):
                    take_profit = self._calculate_take_profit(
                        entry_price, stop_loss, self.risk_reward_ratio
                    )
                    
                    signal = Signal(
                        timestamp=current_bar.name,
                        direction=direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        confidence=0.75,
                        strategy="SMT_DIVERGENCE",
                        metadata={
                            'divergence_type': rsi_divergence,
                            'volume_ratio': current_bar['volume_ratio'],
                            'session': current_bar['session'],
                            'momentum': price_momentum
                        }
                    )
                    signals.append(signal)
                    
        return signals
        
    def identify_exit_signals(self, data: pd.DataFrame, open_trades: List) -> List[Dict]:
        """Identify exit signals for open SMT divergence trades"""
        exit_signals = []
        
        for trade in open_trades:
            current_price = data['close'].iloc[-1]
            current_bar = data.iloc[-1]
            
            # Check for divergence correction
            divergence_corrected = self._check_divergence_correction(data, trade)
            
            # Check for momentum shift
            momentum_shift = self._check_momentum_shift(data, trade)
            
            if divergence_corrected or momentum_shift:
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'divergence_corrected' if divergence_corrected else 'momentum_shift',
                    'timestamp': current_bar.name
                })
                
        return exit_signals
        
    def _detect_rsi_divergence(self, data: pd.DataFrame, current_idx: int) -> Optional[str]:
        """Detect RSI divergence patterns"""
        lookback = min(self.divergence_periods, current_idx)
        
        if lookback < 10:
            return None
            
        # Get recent price and RSI data
        prices = data['close'].iloc[current_idx-lookback:current_idx+1]
        rsi_values = data['rsi'].iloc[current_idx-lookback:current_idx+1]
        
        # Find recent highs and lows
        price_highs = prices[data['swing_high'].iloc[current_idx-lookback:current_idx+1] == True]
        price_lows = prices[data['swing_low'].iloc[current_idx-lookback:current_idx+1] == True]
        
        if len(price_highs) >= 2:
            # Check for bearish divergence (higher highs in price, lower highs in RSI)
            recent_price_highs = price_highs.tail(2)
            rsi_at_highs = rsi_values.loc[recent_price_highs.index]
            
            if (recent_price_highs.iloc[-1] > recent_price_highs.iloc[-2] and 
                rsi_at_highs.iloc[-1] < rsi_at_highs.iloc[-2]):
                return 'bearish'
                
        if len(price_lows) >= 2:
            # Check for bullish divergence (lower lows in price, higher lows in RSI)
            recent_price_lows = price_lows.tail(2)
            rsi_at_lows = rsi_values.loc[recent_price_lows.index]
            
            if (recent_price_lows.iloc[-1] < recent_price_lows.iloc[-2] and 
                rsi_at_lows.iloc[-1] > rsi_at_lows.iloc[-2]):
                return 'bullish'
                
        return None
        
    def _confirm_price_action(self, data: pd.DataFrame, idx: int, direction: TradeDirection) -> bool:
        """Confirm entry with price action patterns"""
        if idx + self.confirmation_candles >= len(data):
            return False
            
        confirmation_data = data.iloc[idx:idx+self.confirmation_candles]
        
        if direction == TradeDirection.LONG:
            # Look for bullish confirmation
            bullish_candles = (confirmation_data['close'] > confirmation_data['open']).sum()
            higher_lows = (confirmation_data['low'].diff() > 0).sum()
            return bullish_candles >= 2 or higher_lows >= 2
            
        else:  # SHORT
            # Look for bearish confirmation
            bearish_candles = (confirmation_data['close'] < confirmation_data['open']).sum()
            lower_highs = (confirmation_data['high'].diff() < 0).sum()
            return bearish_candles >= 2 or lower_highs >= 2
            
    def _check_divergence_correction(self, data: pd.DataFrame, trade: Dict) -> bool:
        """Check if the divergence has been corrected"""
        # In practice, this would compare with the correlated instrument
        # For simulation, we'll check if momentum has shifted significantly
        recent_momentum = data['momentum'].tail(5).mean()
        entry_momentum = trade['metadata'].get('momentum', 0)
        
        # If momentum has reversed significantly, consider divergence corrected
        if trade['direction'] == TradeDirection.LONG:
            return recent_momentum > abs(entry_momentum) * 0.5
        else:
            return recent_momentum < -abs(entry_momentum) * 0.5
            
    def _check_momentum_shift(self, data: pd.DataFrame, trade: Dict) -> bool:
        """Check for significant momentum shift"""
        current_rsi = data['rsi'].iloc[-1]
        
        if trade['direction'] == TradeDirection.LONG:
            return current_rsi > 70  # Overbought after bullish divergence
        else:
            return current_rsi < 30  # Oversold after bearish divergence
            
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information"""
        return {
            'name': 'SMT Divergence',
            'description': 'Trades Smart Money Technique divergences between correlated instruments',
            'timeframes': ['15m', '1h', '4h'],
            'sessions': ['London', 'New York', 'Overlap'],
            'risk_level': 'Medium-High',
            'parameters': {
                'correlation_threshold': self.correlation_threshold,
                'divergence_periods': self.divergence_periods,
                'confirmation_candles': self.confirmation_candles,
                'volume_threshold': self.volume_threshold
            }
        }