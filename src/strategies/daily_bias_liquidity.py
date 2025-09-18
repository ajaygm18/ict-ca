"""
Daily Bias Liquidity Strategy - Strategy 12

This strategy identifies the daily bias based on overnight price action and targets
liquidity pools formed by previous day's highs, lows, and key levels. It focuses
on how institutional traders position for daily ranges and liquidity grabs.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from .base_strategy import BaseICTStrategy, Signal, TradeDirection

class DailyBiasLiquidityStrategy(BaseICTStrategy):
    """
    Daily Bias Liquidity Strategy Implementation
    
    This strategy determines the daily bias based on:
    1. Overnight price action and gap analysis
    2. Previous day's high/low relationships
    3. Weekly/monthly context
    4. Liquidity pool identification and targeting
    
    Entry Conditions:
    - Clear daily bias established (bullish/bearish)
    - Liquidity pools identified above/below current price
    - Price shows intent to target specific liquidity
    - Volume confirms institutional participation
    
    Exit Conditions:
    - Liquidity pool reached and defended
    - Daily bias shifts (reversal signals)
    - End of session (daily cycle complete)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.gap_threshold = config.get('gap_threshold', 0.002)  # 0.2% minimum gap
        self.liquidity_strength = config.get('liquidity_strength', 3)  # touches to confirm level
        self.bias_confirmation_bars = config.get('bias_confirmation_bars', 4)
        self.max_daily_range_pct = config.get('max_daily_range_pct', 0.015)  # 1.5% max daily range
        self.volume_confirmation = config.get('volume_confirmation', True)
        
    def setup_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Setup indicators for daily bias and liquidity analysis"""
        df = data.copy()
        
        # Time and session analysis
        df['date'] = pd.to_datetime(df.index).date
        df['hour'] = pd.to_datetime(df.index).hour
        df['session'] = self._identify_session(df)
        df['is_overnight'] = df['session'] == 'Asian'
        
        # Daily OHLC levels
        daily_groups = df.groupby('date')
        df['daily_open'] = daily_groups['open'].transform('first')
        df['daily_high'] = daily_groups['high'].transform('max')
        df['daily_low'] = daily_groups['low'].transform('min')
        df['daily_close'] = daily_groups['close'].transform('last')
        
        # Previous day levels
        df['prev_daily_high'] = df['daily_high'].shift(1)
        df['prev_daily_low'] = df['daily_low'].shift(1)
        df['prev_daily_close'] = df['daily_close'].shift(1)
        
        # Gap analysis
        df['gap_size'] = (df['daily_open'] - df['prev_daily_close']) / df['prev_daily_close']
        df['gap_up'] = df['gap_size'] > self.gap_threshold
        df['gap_down'] = df['gap_size'] < -self.gap_threshold
        
        # Weekly and monthly context
        df['weekly_high'] = df['high'].rolling(35).max()  # ~1 week
        df['weekly_low'] = df['low'].rolling(35).min()
        df['monthly_high'] = df['high'].rolling(150).max()  # ~1 month
        df['monthly_low'] = df['low'].rolling(150).min()
        
        # Range analysis
        df['daily_range'] = df['daily_high'] - df['daily_low']
        df['avg_daily_range'] = df['daily_range'].rolling(20).mean()
        df['range_expansion'] = df['daily_range'] / df['avg_daily_range']
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Momentum and bias indicators
        df['momentum_1h'] = df['close'].pct_change(4)  # 4-hour momentum
        df['momentum_4h'] = df['close'].pct_change(16)  # 16-hour momentum
        df['rsi'] = self._calculate_rsi(df['close'], 14)
        
        return df
        
    def identify_entry_signals(self, data: pd.DataFrame) -> List[Signal]:
        """Identify daily bias liquidity entry opportunities"""
        signals = []
        
        if len(data) < 50:
            return signals
            
        # Analyze daily bias at market open
        daily_opens = self._find_daily_opens(data)
        
        for open_idx in daily_opens:
            if open_idx + 20 >= len(data):  # Need enough data after open
                continue
                
            signal = self._analyze_daily_bias_setup(data, open_idx)
            if signal:
                signals.append(signal)
                
        return signals
        
    def _find_daily_opens(self, data: pd.DataFrame) -> List[int]:
        """Find indices where new trading days begin"""
        daily_opens = []
        current_date = None
        
        for i, (timestamp, row) in enumerate(data.iterrows()):
            row_date = timestamp.date()
            if row_date != current_date and row['session'] in ['London', 'New York']:
                if current_date is not None:  # Skip first day
                    daily_opens.append(i)
                current_date = row_date
                
        return daily_opens
        
    def _analyze_daily_bias_setup(self, data: pd.DataFrame, open_idx: int) -> Optional[Signal]:
        """Analyze daily bias and liquidity setup"""
        open_bar = data.iloc[open_idx]
        
        # Determine daily bias
        daily_bias = self._determine_daily_bias(data, open_idx)
        if not daily_bias:
            return None
            
        bias_direction, bias_strength = daily_bias
        
        # Identify liquidity pools
        liquidity_pools = self._identify_liquidity_pools(data, open_idx)
        if not liquidity_pools:
            return None
            
        # Select target based on bias
        target_pool = self._select_target_pool(liquidity_pools, bias_direction, open_bar['close'])
        if not target_pool:
            return None
            
        # Look for entry opportunity
        entry_signal = self._find_bias_entry(data, open_idx, bias_direction, target_pool)
        if not entry_signal:
            return None
            
        entry_idx, entry_price = entry_signal
        entry_bar = data.iloc[entry_idx]
        
        # Setup trade parameters
        if bias_direction == 'bullish':
            direction = TradeDirection.LONG
            stop_loss = self._find_recent_low(data, entry_idx, 10)
            take_profit = target_pool['level']
            
        else:  # bearish
            direction = TradeDirection.SHORT
            stop_loss = self._find_recent_high(data, entry_idx, 10)
            take_profit = target_pool['level']
            
        # Validate setup
        if not self._validate_bias_setup(entry_bar, bias_direction, stop_loss, take_profit):
            return None
            
        return Signal(
            timestamp=entry_bar.name,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=bias_strength,
            strategy="DAILY_BIAS_LIQUIDITY",
            metadata={
                'daily_bias': bias_direction,
                'bias_strength': bias_strength,
                'target_pool': target_pool,
                'gap_size': open_bar['gap_size'],
                'session': entry_bar['session'],
                'volume_ratio': entry_bar['volume_ratio'],
                'range_context': entry_bar['range_expansion']
            }
        )
        
    def _determine_daily_bias(self, data: pd.DataFrame, open_idx: int) -> Optional[tuple]:
        """Determine the daily bias based on multiple factors"""
        open_bar = data.iloc[open_idx]
        
        bias_factors = []
        
        # Factor 1: Gap analysis
        if open_bar['gap_up']:
            bias_factors.append(('bullish', 0.3))
        elif open_bar['gap_down']:
            bias_factors.append(('bearish', 0.3))
            
        # Factor 2: Previous day's close vs range
        prev_close = open_bar['prev_daily_close']
        prev_high = open_bar['prev_daily_high']
        prev_low = open_bar['prev_daily_low']
        prev_range = prev_high - prev_low
        
        if prev_range > 0:
            close_position = (prev_close - prev_low) / prev_range
            if close_position > 0.7:
                bias_factors.append(('bullish', 0.2))
            elif close_position < 0.3:
                bias_factors.append(('bearish', 0.2))
                
        # Factor 3: Weekly/Monthly context
        weekly_position = (open_bar['close'] - open_bar['weekly_low']) / (open_bar['weekly_high'] - open_bar['weekly_low'])
        if weekly_position > 0.8:
            bias_factors.append(('bearish', 0.15))  # Near weekly high, expect reversal
        elif weekly_position < 0.2:
            bias_factors.append(('bullish', 0.15))  # Near weekly low, expect bounce
            
        # Factor 4: Momentum alignment
        if open_bar['momentum_4h'] > 0.01:
            bias_factors.append(('bullish', 0.25))
        elif open_bar['momentum_4h'] < -0.01:
            bias_factors.append(('bearish', 0.25))
            
        # Factor 5: RSI context
        if open_bar['rsi'] < 35:
            bias_factors.append(('bullish', 0.1))
        elif open_bar['rsi'] > 65:
            bias_factors.append(('bearish', 0.1))
            
        # Calculate weighted bias
        bullish_weight = sum(weight for direction, weight in bias_factors if direction == 'bullish')
        bearish_weight = sum(weight for direction, weight in bias_factors if direction == 'bearish')
        
        if bullish_weight > bearish_weight and bullish_weight > 0.5:
            return ('bullish', min(bullish_weight, 0.9))
        elif bearish_weight > bullish_weight and bearish_weight > 0.5:
            return ('bearish', min(bearish_weight, 0.9))
            
        return None
        
    def _identify_liquidity_pools(self, data: pd.DataFrame, open_idx: int) -> List[Dict]:
        """Identify liquidity pools based on previous levels"""
        liquidity_pools = []
        
        if open_idx < 50:
            return liquidity_pools
            
        # Look back for significant levels
        lookback_data = data.iloc[max(0, open_idx-100):open_idx]
        current_price = data.iloc[open_idx]['close']
        
        # Previous day highs/lows
        prev_highs = lookback_data[lookback_data['daily_high'] == lookback_data['daily_high']]['daily_high'].unique()
        prev_lows = lookback_data[lookback_data['daily_low'] == lookback_data['daily_low']]['daily_low'].unique()
        
        # Add significant levels as liquidity pools
        for high in prev_highs[-5:]:  # Last 5 daily highs
            if abs(high - current_price) / current_price > 0.001:  # At least 0.1% away
                touches = self._count_level_touches(lookback_data, high, 0.001)
                if touches >= self.liquidity_strength:
                    liquidity_pools.append({
                        'level': high,
                        'type': 'resistance',
                        'touches': touches,
                        'strength': min(touches / 5.0, 1.0)
                    })
                    
        for low in prev_lows[-5:]:  # Last 5 daily lows
            if abs(low - current_price) / current_price > 0.001:
                touches = self._count_level_touches(lookback_data, low, 0.001)
                if touches >= self.liquidity_strength:
                    liquidity_pools.append({
                        'level': low,
                        'type': 'support',
                        'touches': touches,
                        'strength': min(touches / 5.0, 1.0)
                    })
                    
        # Weekly levels
        weekly_high = lookback_data['weekly_high'].iloc[-1]
        weekly_low = lookback_data['weekly_low'].iloc[-1]
        
        if abs(weekly_high - current_price) / current_price > 0.005:
            liquidity_pools.append({
                'level': weekly_high,
                'type': 'weekly_resistance',
                'touches': 1,
                'strength': 0.8
            })
            
        if abs(weekly_low - current_price) / current_price > 0.005:
            liquidity_pools.append({
                'level': weekly_low,
                'type': 'weekly_support',
                'touches': 1,
                'strength': 0.8
            })
            
        return liquidity_pools
        
    def _select_target_pool(self, liquidity_pools: List[Dict], bias_direction: str, current_price: float) -> Optional[Dict]:
        """Select the most appropriate target liquidity pool"""
        if bias_direction == 'bullish':
            # Target resistance levels above current price
            candidates = [pool for pool in liquidity_pools 
                         if pool['level'] > current_price and pool['type'] in ['resistance', 'weekly_resistance']]
        else:
            # Target support levels below current price
            candidates = [pool for pool in liquidity_pools 
                         if pool['level'] < current_price and pool['type'] in ['support', 'weekly_support']]
            
        if not candidates:
            return None
            
        # Select closest strong level
        candidates.sort(key=lambda x: (abs(x['level'] - current_price), -x['strength']))
        return candidates[0]
        
    def _find_bias_entry(self, data: pd.DataFrame, open_idx: int, 
                        bias_direction: str, target_pool: Dict) -> Optional[tuple]:
        """Find entry point based on bias and target"""
        # Look for entry in the next few hours after bias determination
        search_end = min(open_idx + 20, len(data))
        
        for i in range(open_idx + self.bias_confirmation_bars, search_end):
            current_bar = data.iloc[i]
            
            # Check for bias confirmation
            if not self._confirm_bias_direction(data, i, bias_direction):
                continue
                
            # Volume confirmation
            if self.volume_confirmation and current_bar['volume_ratio'] < 1.2:
                continue
                
            # Price action confirmation
            if self._confirm_entry_price_action(data, i, bias_direction):
                return (i, current_bar['close'])
                
        return None
        
    def _confirm_bias_direction(self, data: pd.DataFrame, idx: int, bias_direction: str) -> bool:
        """Confirm bias with recent price action"""
        if idx < 4:
            return False
            
        recent_bars = data.iloc[idx-3:idx+1]
        
        if bias_direction == 'bullish':
            higher_lows = (recent_bars['low'].diff() > 0).sum() >= 2
            bullish_momentum = recent_bars['momentum_1h'].iloc[-1] > 0
            return higher_lows or bullish_momentum
            
        else:  # bearish
            lower_highs = (recent_bars['high'].diff() < 0).sum() >= 2
            bearish_momentum = recent_bars['momentum_1h'].iloc[-1] < 0
            return lower_highs or bearish_momentum
            
    def _confirm_entry_price_action(self, data: pd.DataFrame, idx: int, bias_direction: str) -> bool:
        """Confirm entry with price action pattern"""
        current_bar = data.iloc[idx]
        
        if bias_direction == 'bullish':
            return current_bar['close'] > current_bar['open']  # Bullish candle
        else:
            return current_bar['close'] < current_bar['open']  # Bearish candle
            
    def _validate_bias_setup(self, entry_bar: pd.Series, bias_direction: str, 
                           stop_loss: float, take_profit: float) -> bool:
        """Validate the complete bias setup"""
        entry_price = entry_bar['close']
        
        # Risk/reward validation
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if reward / risk < 1.5:  # Minimum 1.5:1 R:R
            return False
            
        # Range validation (don't trade if already extended)
        if entry_bar['range_expansion'] > 2.0:
            return False
            
        return True
        
    def _count_level_touches(self, data: pd.DataFrame, level: float, tolerance: float) -> int:
        """Count how many times price touched a level"""
        touches = 0
        level_range = level * tolerance
        
        for _, bar in data.iterrows():
            if bar['low'] <= level + level_range and bar['high'] >= level - level_range:
                touches += 1
                
        return touches
        
    def identify_exit_signals(self, data: pd.DataFrame, open_trades: List) -> List[Dict]:
        """Identify exit signals for daily bias trades"""
        exit_signals = []
        
        for trade in open_trades:
            current_bar = data.iloc[-1]
            current_price = current_bar['close']
            
            # Exit at end of day
            if current_bar['hour'] >= 21:  # 9 PM
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'end_of_day',
                    'timestamp': current_bar.name
                })
                continue
                
            # Check for bias reversal
            if self._check_bias_reversal(data, trade):
                exit_signals.append({
                    'trade_id': trade.get('id'),
                    'exit_price': current_price,
                    'exit_reason': 'bias_reversal',
                    'timestamp': current_bar.name
                })
                
        return exit_signals
        
    def _check_bias_reversal(self, data: pd.DataFrame, trade: Dict) -> bool:
        """Check for daily bias reversal"""
        recent_momentum = data['momentum_1h'].tail(3).mean()
        original_bias = trade['metadata']['daily_bias']
        
        if original_bias == 'bullish' and recent_momentum < -0.005:
            return True
        elif original_bias == 'bearish' and recent_momentum > 0.005:
            return True
            
        return False
        
    def get_strategy_info(self) -> Dict[str, Any]:
        """Return strategy information"""
        return {
            'name': 'Daily Bias Liquidity',
            'description': 'Trades daily bias targeting liquidity pools from previous levels',
            'timeframes': ['15m', '1h'],
            'sessions': ['London', 'New York'],
            'risk_level': 'Medium',
            'parameters': {
                'gap_threshold': self.gap_threshold,
                'liquidity_strength': self.liquidity_strength,
                'bias_confirmation_bars': self.bias_confirmation_bars,
                'max_daily_range_pct': self.max_daily_range_pct,
                'volume_confirmation': self.volume_confirmation
            }
        }