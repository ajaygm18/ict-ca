"""
ICT Pattern Detection Module

This module implements detection algorithms for key ICT concepts:
- Fair Value Gaps (FVG)
- Order Blocks (OB)
- Liquidity Sweeps
- Market Structure Shifts
- Breaker Blocks
- Optimal Trade Entry (OTE) zones
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FVGType(Enum):
    """Fair Value Gap types"""
    BULLISH = 1
    BEARISH = -1


class OBType(Enum):
    """Order Block types"""
    BULLISH = 1
    BEARISH = -1


class LiquidityType(Enum):
    """Liquidity pool types"""
    HIGH = 1
    LOW = -1


@dataclass
class FairValueGap:
    """Fair Value Gap structure"""
    start_index: int
    end_index: int
    top: float
    bottom: float
    fvg_type: FVGType
    filled: bool = False
    fill_index: Optional[int] = None


@dataclass
class OrderBlock:
    """Order Block structure"""
    index: int
    high: float
    low: float
    open: float
    close: float
    ob_type: OBType
    retested: bool = False
    retest_index: Optional[int] = None
    breaker: bool = False


@dataclass
class LiquidityPool:
    """Liquidity Pool structure"""
    index: int
    price: float
    liquidity_type: LiquidityType
    swept: bool = False
    sweep_index: Optional[int] = None


class ICTPatternDetector:
    """
    Detects ICT patterns in price data
    """
    
    def __init__(self):
        self.fvgs: List[FairValueGap] = []
        self.order_blocks: List[OrderBlock] = []
        self.liquidity_pools: List[LiquidityPool] = []
    
    def detect_fair_value_gaps(self, data: pd.DataFrame, min_gap_pips: float = 5) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps in price data
        
        FVG Rules:
        - Bullish FVG: Low of candle 1 > High of candle 3
        - Bearish FVG: High of candle 1 < Low of candle 3
        - Middle candle creates the gap through strong momentum
        
        Args:
            data: OHLCV DataFrame
            min_gap_pips: Minimum gap size in pips
            
        Returns:
            List of detected FVGs
        """
        fvgs = []
        
        for i in range(2, len(data)):
            candle1 = data.iloc[i-2]
            candle2 = data.iloc[i-1]  # Gap-creating candle
            candle3 = data.iloc[i]
            
            # Bullish FVG: Low of candle 1 > High of candle 3
            if candle1['low'] > candle3['high']:
                gap_size = candle1['low'] - candle3['high']
                if gap_size >= min_gap_pips * 0.0001:  # Convert pips to price
                    fvg = FairValueGap(
                        start_index=i-2,
                        end_index=i,
                        top=candle1['low'],
                        bottom=candle3['high'],
                        fvg_type=FVGType.BULLISH
                    )
                    fvgs.append(fvg)
            
            # Bearish FVG: High of candle 1 < Low of candle 3
            elif candle1['high'] < candle3['low']:
                gap_size = candle3['low'] - candle1['high']
                if gap_size >= min_gap_pips * 0.0001:  # Convert pips to price
                    fvg = FairValueGap(
                        start_index=i-2,
                        end_index=i,
                        top=candle3['low'],
                        bottom=candle1['high'],
                        fvg_type=FVGType.BEARISH
                    )
                    fvgs.append(fvg)
        
        return fvgs
    
    def detect_order_blocks(self, data: pd.DataFrame, lookback: int = 10) -> List[OrderBlock]:
        """
        Detect Order Blocks in price data
        
        OB Rules:
        - Bullish OB: Last down candle before strong up move
        - Bearish OB: Last up candle before strong down move
        - Must cause market structure shift
        
        Args:
            data: OHLCV DataFrame
            lookback: Lookback period for structure analysis
            
        Returns:
            List of detected Order Blocks
        """
        order_blocks = []
        
        for i in range(lookback, len(data) - 1):
            current_candle = data.iloc[i]
            next_candle = data.iloc[i + 1]
            
            # Check for bullish order block (last down candle before up move)
            if (current_candle['close'] < current_candle['open'] and  # Down candle
                next_candle['close'] > next_candle['open'] and      # Next is up candle
                next_candle['close'] > current_candle['high']):     # Breaks high
                
                # Confirm with strong momentum
                if self._check_momentum_shift(data, i, 'bullish', lookback):
                    ob = OrderBlock(
                        index=i,
                        high=current_candle['high'],
                        low=current_candle['low'],
                        open=current_candle['open'],
                        close=current_candle['close'],
                        ob_type=OBType.BULLISH
                    )
                    order_blocks.append(ob)
            
            # Check for bearish order block (last up candle before down move)
            elif (current_candle['close'] > current_candle['open'] and  # Up candle
                  next_candle['close'] < next_candle['open'] and      # Next is down candle
                  next_candle['close'] < current_candle['low']):      # Breaks low
                
                # Confirm with strong momentum
                if self._check_momentum_shift(data, i, 'bearish', lookback):
                    ob = OrderBlock(
                        index=i,
                        high=current_candle['high'],
                        low=current_candle['low'],
                        open=current_candle['open'],
                        close=current_candle['close'],
                        ob_type=OBType.BEARISH
                    )
                    order_blocks.append(ob)
        
        return order_blocks
    
    def detect_liquidity_sweeps(self, data: pd.DataFrame, lookback: int = 20) -> List[LiquidityPool]:
        """
        Detect liquidity sweeps of previous highs/lows
        
        Args:
            data: OHLCV DataFrame
            lookback: Lookback period for high/low detection
            
        Returns:
            List of detected liquidity pools
        """
        liquidity_pools = []
        
        for i in range(lookback, len(data)):
            current_high = data.iloc[i]['high']
            current_low = data.iloc[i]['low']
            
            # Look for recent highs/lows that might be liquidity
            recent_data = data.iloc[i-lookback:i]
            
            # Find significant highs (resistance levels)
            for j, row in recent_data.iterrows():
                if self._is_significant_high(recent_data, row.name - recent_data.index[0]):
                    # Check if current candle sweeps this high
                    if current_high > row['high']:
                        pool = LiquidityPool(
                            index=recent_data.index.get_loc(j),
                            price=row['high'],
                            liquidity_type=LiquidityType.HIGH,
                            swept=True,
                            sweep_index=i
                        )
                        liquidity_pools.append(pool)
            
            # Find significant lows (support levels)
            for j, row in recent_data.iterrows():
                if self._is_significant_low(recent_data, row.name - recent_data.index[0]):
                    # Check if current candle sweeps this low
                    if current_low < row['low']:
                        pool = LiquidityPool(
                            index=recent_data.index.get_loc(j),
                            price=row['low'],
                            liquidity_type=LiquidityType.LOW,
                            swept=True,
                            sweep_index=i
                        )
                        liquidity_pools.append(pool)
        
        return liquidity_pools
    
    def detect_market_structure_shift(self, data: pd.DataFrame, index: int) -> bool:
        """
        Detect if there's a market structure shift at given index
        
        Args:
            data: OHLCV DataFrame
            index: Index to check for structure shift
            
        Returns:
            True if structure shift detected
        """
        if index < 10:
            return False
        
        recent_data = data.iloc[index-10:index+1]
        
        # Simple structure shift detection
        # Look for break of recent high/low with momentum
        recent_high = recent_data['high'].max()
        recent_low = recent_data['low'].min()
        current_close = data.iloc[index]['close']
        
        # Check for bullish structure shift
        if current_close > recent_high:
            return True
        
        # Check for bearish structure shift
        if current_close < recent_low:
            return True
        
        return False
    
    def check_fvg_retest(self, fvg: FairValueGap, data: pd.DataFrame, 
                        current_index: int) -> bool:
        """
        Check if price is retesting a Fair Value Gap
        
        Args:
            fvg: FairValueGap to check
            data: OHLCV DataFrame
            current_index: Current data index
            
        Returns:
            True if price is in FVG zone
        """
        if fvg.filled:
            return False
        
        current_high = data.iloc[current_index]['high']
        current_low = data.iloc[current_index]['low']
        
        # Check if current candle overlaps with FVG
        if (current_low <= fvg.top and current_high >= fvg.bottom):
            return True
        
        return False
    
    def check_order_block_retest(self, ob: OrderBlock, data: pd.DataFrame, 
                                current_index: int) -> bool:
        """
        Check if price is retesting an Order Block
        
        Args:
            ob: OrderBlock to check
            data: OHLCV DataFrame
            current_index: Current data index
            
        Returns:
            True if price is retesting OB
        """
        current_high = data.iloc[current_index]['high']
        current_low = data.iloc[current_index]['low']
        
        # Check if current candle overlaps with Order Block
        if (current_low <= ob.high and current_high >= ob.low):
            return True
        
        return False
    
    def calculate_optimal_trade_entry(self, swing_high: float, swing_low: float) -> Dict[str, float]:
        """
        Calculate Optimal Trade Entry (OTE) zone using Fibonacci levels
        
        Args:
            swing_high: Recent swing high
            swing_low: Recent swing low
            
        Returns:
            Dictionary with OTE levels
        """
        range_size = swing_high - swing_low
        
        ote_levels = {
            'fib_618': swing_low + (range_size * 0.618),
            'fib_705': swing_low + (range_size * 0.705),
            'fib_786': swing_low + (range_size * 0.786),
            'ote_low': swing_low + (range_size * 0.618),  # 61.8%
            'ote_high': swing_low + (range_size * 0.786)  # 78.6%
        }
        
        return ote_levels
    
    def _check_momentum_shift(self, data: pd.DataFrame, index: int, 
                             direction: str, lookback: int) -> bool:
        """Check for strong momentum shift after order block"""
        if direction == 'bullish':
            # Check if subsequent candles show strong upward momentum
            next_candles = data.iloc[index+1:index+4]
            return (next_candles['close'] > next_candles['open']).sum() >= 2
        else:
            # Check if subsequent candles show strong downward momentum
            next_candles = data.iloc[index+1:index+4]
            return (next_candles['close'] < next_candles['open']).sum() >= 2
    
    def _is_significant_high(self, data: pd.DataFrame, index: int) -> bool:
        """Check if candle forms a significant high"""
        if index < 2 or index >= len(data) - 2:
            return False
        
        current_high = data.iloc[index]['high']
        left_high = data.iloc[index-1]['high']
        right_high = data.iloc[index+1]['high']
        
        return current_high > left_high and current_high > right_high
    
    def _is_significant_low(self, data: pd.DataFrame, index: int) -> bool:
        """Check if candle forms a significant low"""
        if index < 2 or index >= len(data) - 2:
            return False
        
        current_low = data.iloc[index]['low']
        left_low = data.iloc[index-1]['low']
        right_low = data.iloc[index+1]['low']
        
        return current_low < left_low and current_low < right_low


def detect_displacement(data: pd.DataFrame, index: int, direction: str, 
                       min_candles: int = 3) -> bool:
    """
    Detect strong price displacement (key ICT concept)
    
    Args:
        data: OHLCV DataFrame
        index: Starting index
        direction: 'up' or 'down'
        min_candles: Minimum number of consecutive candles
        
    Returns:
        True if displacement detected
    """
    if index + min_candles >= len(data):
        return False
    
    displacement_candles = data.iloc[index:index+min_candles]
    
    if direction == 'up':
        # Check for consecutive bullish candles with strong momentum
        bullish_candles = (displacement_candles['close'] > displacement_candles['open']).sum()
        strong_momentum = (displacement_candles['close'].iloc[-1] > 
                          displacement_candles['open'].iloc[0] * 1.001)  # 0.1% minimum move
        return bullish_candles >= min_candles - 1 and strong_momentum
    
    elif direction == 'down':
        # Check for consecutive bearish candles with strong momentum
        bearish_candles = (displacement_candles['close'] < displacement_candles['open']).sum()
        strong_momentum = (displacement_candles['close'].iloc[-1] < 
                          displacement_candles['open'].iloc[0] * 0.999)  # 0.1% minimum move
        return bearish_candles >= min_candles - 1 and strong_momentum
    
    return False


def check_confluence(data: pd.DataFrame, index: int, patterns: Dict[str, bool]) -> float:
    """
    Calculate confluence score based on multiple ICT patterns
    
    Args:
        data: OHLCV DataFrame
        index: Current index
        patterns: Dictionary of pattern confirmations
        
    Returns:
        Confluence score (0.0 to 1.0)
    """
    weights = {
        'fvg': 0.2,
        'order_block': 0.25,
        'liquidity_sweep': 0.2,
        'market_structure': 0.15,
        'displacement': 0.2
    }
    
    total_score = 0.0
    for pattern, confirmed in patterns.items():
        if pattern in weights and confirmed:
            total_score += weights[pattern]
    
    return min(total_score, 1.0)