"""
Base Strategy Class for ICT Trading Framework

This module provides the abstract base class that all ICT strategies inherit from.
It defines the common interface for entry/exit signals, risk management, and filters.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Trading signal types"""
    BUY = 1
    SELL = -1
    HOLD = 0


class TradeDirection(Enum):
    """Trade direction types"""
    LONG = 1
    SHORT = -1


@dataclass
class Trade:
    """Trade execution details"""
    entry_time: pd.Timestamp
    entry_price: float
    direction: TradeDirection
    stop_loss: float
    take_profit: float
    size: float
    strategy: str
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    exit_reason: Optional[str] = None


@dataclass
class Signal:
    """Trading signal with metadata"""
    timestamp: pd.Timestamp
    signal_type: SignalType
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str
    metadata: Dict[str, Any] = None


class BaseICTStrategy(ABC):
    """
    Abstract base class for all ICT trading strategies
    
    This class defines the common interface that all ICT strategies must implement:
    - Signal generation (entry/exit conditions)
    - Risk management (stop-loss, take-profit, position sizing)
    - Filters (time-based, volatility, confirmation filters)
    - Strategy parameters and configuration
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any] = None):
        """
        Initialize the strategy
        
        Args:
            name: Strategy name identifier
            parameters: Strategy-specific parameters
        """
        self.name = name
        self.parameters = parameters or {}
        self.trades: List[Trade] = []
        self.signals: List[Signal] = []
        
        # Default risk management parameters
        self.risk_percent = self.parameters.get('risk_percent', 0.02)
        self.risk_reward_ratio = self.parameters.get('risk_reward_ratio', 2.0)
        self.max_trades_per_day = self.parameters.get('max_trades_per_day', 3)
        
        # Initialize strategy-specific parameters
        self._set_default_parameters()
    
    @abstractmethod
    def _set_default_parameters(self):
        """Set strategy-specific default parameters"""
        pass
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on strategy logic
        
        Args:
            data: OHLCV price data with required indicators
            
        Returns:
            List of trading signals
        """
        pass
    
    @abstractmethod
    def check_entry_conditions(self, data: pd.DataFrame, index: int) -> Optional[Signal]:
        """
        Check if entry conditions are met at specific index
        
        Args:
            data: OHLCV price data
            index: Current data index to check
            
        Returns:
            Trading signal if conditions met, None otherwise
        """
        pass
    
    def check_exit_conditions(self, data: pd.DataFrame, trade: Trade, index: int) -> Optional[Tuple[float, str]]:
        """
        Check if exit conditions are met for an open trade
        
        Args:
            data: OHLCV price data
            trade: Open trade to check
            index: Current data index
            
        Returns:
            Tuple of (exit_price, exit_reason) if exit needed, None otherwise
        """
        current_price = data.iloc[index]['close']
        
        # Check stop-loss
        if trade.direction == TradeDirection.LONG:
            if current_price <= trade.stop_loss:
                return current_price, "Stop Loss"
            elif current_price >= trade.take_profit:
                return current_price, "Take Profit"
        else:
            if current_price >= trade.stop_loss:
                return current_price, "Stop Loss"
            elif current_price <= trade.take_profit:
                return current_price, "Take Profit"
        
        return None
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                              account_balance: float) -> float:
        """
        Calculate position size based on risk management rules
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            account_balance: Current account balance
            
        Returns:
            Position size in base currency units
        """
        risk_amount = account_balance * self.risk_percent
        price_diff = abs(entry_price - stop_loss)
        
        if price_diff == 0:
            return 0
        
        position_size = risk_amount / price_diff
        return position_size
    
    def calculate_stop_loss(self, entry_price: float, direction: TradeDirection, 
                          data: pd.DataFrame, index: int) -> float:
        """
        Calculate stop loss based on strategy-specific rules
        Default implementation uses recent swing levels
        
        Args:
            entry_price: Entry price
            direction: Trade direction
            data: Price data
            index: Current index
            
        Returns:
            Stop loss price
        """
        # Default stop loss calculation - can be overridden by strategies
        lookback = min(20, index)
        recent_data = data.iloc[index-lookback:index+1]
        
        if direction == TradeDirection.LONG:
            return recent_data['low'].min()
        else:
            return recent_data['high'].max()
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float, 
                            direction: TradeDirection) -> float:
        """
        Calculate take profit based on risk-reward ratio
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            direction: Trade direction
            
        Returns:
            Take profit price
        """
        risk_distance = abs(entry_price - stop_loss)
        reward_distance = risk_distance * self.risk_reward_ratio
        
        if direction == TradeDirection.LONG:
            return entry_price + reward_distance
        else:
            return entry_price - reward_distance
    
    def apply_filters(self, signal: Signal, data: pd.DataFrame, index: int) -> bool:
        """
        Apply strategy-specific filters to validate signals
        
        Args:
            signal: Generated signal
            data: Price data
            index: Current index
            
        Returns:
            True if signal passes all filters, False otherwise
        """
        # Default filters - can be overridden by strategies
        
        # Time-based filter
        if not self._check_time_filter(signal.timestamp):
            return False
        
        # Volatility filter
        if not self._check_volatility_filter(data, index):
            return False
        
        # Maximum trades per day filter
        if not self._check_max_trades_filter(signal.timestamp):
            return False
        
        return True
    
    def _check_time_filter(self, timestamp: pd.Timestamp) -> bool:
        """Check if timestamp is within allowed trading sessions"""
        # Default implementation - always allow
        # Override in specific strategies for session-based filtering
        return True
    
    def _check_volatility_filter(self, data: pd.DataFrame, index: int) -> bool:
        """Check if market volatility meets minimum requirements"""
        if 'atr' not in data.columns:
            return True  # Skip filter if ATR not available
        
        min_atr = self.parameters.get('min_atr', 0.001)
        current_atr = data.iloc[index]['atr']
        
        return current_atr >= min_atr
    
    def _check_max_trades_filter(self, timestamp: pd.Timestamp) -> bool:
        """Check if maximum trades per day limit is reached"""
        trades_today = [t for t in self.trades 
                       if t.entry_time.date() == timestamp.date()]
        return len(trades_today) < self.max_trades_per_day
    
    def update_parameters(self, new_parameters: Dict[str, Any]):
        """Update strategy parameters"""
        self.parameters.update(new_parameters)
        
        # Update risk management parameters
        self.risk_percent = self.parameters.get('risk_percent', self.risk_percent)
        self.risk_reward_ratio = self.parameters.get('risk_reward_ratio', self.risk_reward_ratio)
        self.max_trades_per_day = self.parameters.get('max_trades_per_day', self.max_trades_per_day)
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get strategy information and current parameters"""
        return {
            'name': self.name,
            'parameters': self.parameters,
            'total_trades': len(self.trades),
            'total_signals': len(self.signals)
        }
    
    def reset(self):
        """Reset strategy state for new backtest"""
        self.trades = []
        self.signals = []