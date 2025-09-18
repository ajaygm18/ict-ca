"""
Backtesting Engine for ICT Trading Framework

This module provides comprehensive backtesting capabilities including:
- Trade execution simulation
- Performance metrics calculation
- Risk analytics
- Visualization tools
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

from ..strategies.base_strategy import BaseICTStrategy, Trade, Signal, TradeDirection


@dataclass
class BacktestResults:
    """Container for backtest results and metrics"""
    trades: List[Trade]
    signals: List[Signal]
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    performance_metrics: Dict[str, Any]
    monthly_returns: pd.Series
    trade_analysis: Dict[str, Any]


class PerformanceCalculator:
    """Calculate trading performance metrics"""
    
    @staticmethod
    def calculate_metrics(trades: List[Trade], equity_curve: pd.Series, 
                         initial_capital: float = 10000) -> Dict[str, Any]:
        """
        Calculate comprehensive performance metrics
        
        Args:
            trades: List of executed trades
            equity_curve: Account equity over time
            initial_capital: Starting account balance
            
        Returns:
            Dictionary of performance metrics
        """
        if not trades or equity_curve.empty:
            return {}
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl and t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl and t.pnl < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # PnL calculations
        total_pnl = sum(t.pnl for t in trades if t.pnl is not None)
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        
        # Average trade metrics
        avg_win = gross_profit / len(winning_trades) if winning_trades else 0
        avg_loss = gross_loss / len(losing_trades) if losing_trades else 0
        avg_trade = total_pnl / total_trades if total_trades > 0 else 0
        
        # Risk metrics
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        expectancy = avg_trade
        
        # Equity curve analysis
        returns = equity_curve.pct_change().dropna()
        final_equity = equity_curve.iloc[-1]
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Sharpe ratio (annualized)
        if len(returns) > 0 and returns.std() > 0:
            daily_sharpe = returns.mean() / returns.std()
            sharpe_ratio = daily_sharpe * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        annual_return = (final_equity / initial_capital) ** (252 / len(equity_curve)) - 1
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Consecutive wins/losses
        consecutive_wins = PerformanceCalculator._max_consecutive(trades, True)
        consecutive_losses = PerformanceCalculator._max_consecutive(trades, False)
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_trade': avg_trade,
            'expectancy': expectancy,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'final_equity': final_equity,
            'consecutive_wins': consecutive_wins,
            'consecutive_losses': consecutive_losses
        }
    
    @staticmethod
    def _max_consecutive(trades: List[Trade], winning: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        if not trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if trade.pnl is None:
                continue
                
            is_winner = trade.pnl > 0
            if is_winner == winning:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive


class TradeSimulator:
    """Simulate trade execution during backtesting"""
    
    def __init__(self, initial_capital: float = 10000, commission: float = 0.0001):
        """
        Initialize trade simulator
        
        Args:
            initial_capital: Starting account balance
            commission: Commission per trade (as percentage of trade value)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.current_balance = initial_capital
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
    
    def execute_signal(self, signal: Signal, data: pd.DataFrame, index: int) -> Optional[Trade]:
        """
        Execute a trading signal
        
        Args:
            signal: Trading signal to execute
            data: Price data
            index: Current data index
            
        Returns:
            Trade object if executed, None if rejected
        """
        current_price = data.iloc[index]['close']
        
        # Calculate position size based on risk
        risk_amount = self.current_balance * 0.02  # 2% risk per trade
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        
        if stop_distance == 0:
            return None
        
        position_size = risk_amount / stop_distance
        
        # Create trade
        trade_direction = (TradeDirection.LONG if signal.signal_type.value == 1 
                          else TradeDirection.SHORT)
        
        trade = Trade(
            entry_time=signal.timestamp,
            entry_price=signal.entry_price,
            direction=trade_direction,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            size=position_size,
            strategy=signal.reason.split(':')[0] if ':' in signal.reason else 'Unknown'
        )
        
        # Add commission
        commission_cost = position_size * self.commission
        self.current_balance -= commission_cost
        
        self.open_trades.append(trade)
        return trade
    
    def check_exits(self, data: pd.DataFrame, index: int) -> List[Trade]:
        """
        Check for trade exits and close trades
        
        Args:
            data: Price data
            index: Current data index
            
        Returns:
            List of closed trades
        """
        closed_trades = []
        current_time = data.index[index]
        current_price = data.iloc[index]['close']
        current_high = data.iloc[index]['high']
        current_low = data.iloc[index]['low']
        
        for trade in self.open_trades[:]:  # Copy list to avoid modification during iteration
            exit_triggered = False
            exit_price = None
            exit_reason = None
            
            if trade.direction == TradeDirection.LONG:
                # Check stop loss
                if current_low <= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = "Stop Loss"
                    exit_triggered = True
                # Check take profit
                elif current_high >= trade.take_profit:
                    exit_price = trade.take_profit
                    exit_reason = "Take Profit"
                    exit_triggered = True
            
            else:  # SHORT
                # Check stop loss
                if current_high >= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = "Stop Loss"
                    exit_triggered = True
                # Check take profit
                elif current_low <= trade.take_profit:
                    exit_price = trade.take_profit
                    exit_reason = "Take Profit"
                    exit_triggered = True
            
            if exit_triggered:
                # Close the trade
                trade.exit_time = current_time
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                
                # Calculate PnL
                if trade.direction == TradeDirection.LONG:
                    trade.pnl = (exit_price - trade.entry_price) * trade.size
                else:
                    trade.pnl = (trade.entry_price - exit_price) * trade.size
                
                # Subtract commission
                commission_cost = trade.size * self.commission
                trade.pnl -= commission_cost
                
                # Update balance
                self.current_balance += trade.pnl
                
                # Move to closed trades
                self.open_trades.remove(trade)
                self.closed_trades.append(trade)
                closed_trades.append(trade)
        
        return closed_trades
    
    def get_current_equity(self) -> float:
        """Calculate current account equity including open trades"""
        equity = self.current_balance
        
        # Add unrealized PnL from open trades
        # Note: This would require current market price, simplified for now
        return equity


class Backtester:
    """Main backtesting engine"""
    
    def __init__(self, initial_capital: float = 10000):
        """
        Initialize backtester
        
        Args:
            initial_capital: Starting account balance
        """
        self.initial_capital = initial_capital
        self.trade_simulator = TradeSimulator(initial_capital)
    
    def run_backtest(self, strategy: BaseICTStrategy, data: pd.DataFrame) -> BacktestResults:
        """
        Run complete backtest for a strategy
        
        Args:
            strategy: Strategy to test
            data: Historical price data
            
        Returns:
            BacktestResults containing all metrics and analysis
        """
        # Reset strategy and simulator
        strategy.reset()
        self.trade_simulator = TradeSimulator(self.initial_capital)
        
        # Generate all signals
        signals = strategy.generate_signals(data)
        
        # Track equity over time
        equity_timeline = []
        timestamps = []
        
        # Process each data point
        signal_index = 0
        
        for i in range(len(data)):
            current_time = data.index[i]
            
            # Check for new signals
            while (signal_index < len(signals) and 
                   signals[signal_index].timestamp <= current_time):
                
                signal = signals[signal_index]
                
                # Apply strategy filters
                if strategy.apply_filters(signal, data, i):
                    self.trade_simulator.execute_signal(signal, data, i)
                
                signal_index += 1
            
            # Check for trade exits
            self.trade_simulator.check_exits(data, i)
            
            # Record equity
            current_equity = self.trade_simulator.get_current_equity()
            equity_timeline.append(current_equity)
            timestamps.append(current_time)
        
        # Create equity curve
        equity_curve = pd.Series(equity_timeline, index=timestamps)
        
        # Calculate drawdown
        peak = equity_curve.expanding().max()
        drawdown_curve = (equity_curve - peak) / peak
        
        # Calculate performance metrics
        all_trades = self.trade_simulator.closed_trades
        performance_metrics = PerformanceCalculator.calculate_metrics(
            all_trades, equity_curve, self.initial_capital
        )
        
        # Monthly returns analysis
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        
        # Trade analysis
        trade_analysis = self._analyze_trades(all_trades)
        
        return BacktestResults(
            trades=all_trades,
            signals=signals,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            performance_metrics=performance_metrics,
            monthly_returns=monthly_returns,
            trade_analysis=trade_analysis
        )
    
    def _calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate monthly returns from equity curve"""
        if equity_curve.empty:
            return pd.Series()
        
        monthly_equity = equity_curve.resample('ME').last()  # ME = Month End
        monthly_returns = monthly_equity.pct_change().dropna()
        
        return monthly_returns
    
    def _analyze_trades(self, trades: List[Trade]) -> Dict[str, Any]:
        """Perform detailed trade analysis"""
        if not trades:
            return {}
        
        # Trade duration analysis
        durations = []
        for trade in trades:
            if trade.entry_time and trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # Hours
                durations.append(duration)
        
        # PnL distribution
        pnls = [t.pnl for t in trades if t.pnl is not None]
        
        # Strategy performance breakdown
        strategy_performance = {}
        for trade in trades:
            strategy = trade.strategy
            if strategy not in strategy_performance:
                strategy_performance[strategy] = {'trades': 0, 'pnl': 0, 'wins': 0}
            
            strategy_performance[strategy]['trades'] += 1
            if trade.pnl:
                strategy_performance[strategy]['pnl'] += trade.pnl
                if trade.pnl > 0:
                    strategy_performance[strategy]['wins'] += 1
        
        return {
            'avg_duration_hours': np.mean(durations) if durations else 0,
            'median_duration_hours': np.median(durations) if durations else 0,
            'pnl_std': np.std(pnls) if pnls else 0,
            'strategy_breakdown': strategy_performance
        }


class BacktestVisualizer:
    """Create visualizations for backtest results"""
    
    @staticmethod
    def plot_equity_curve(results: BacktestResults, show_drawdown: bool = True) -> go.Figure:
        """
        Plot equity curve with optional drawdown
        
        Args:
            results: Backtest results
            show_drawdown: Whether to show drawdown subplot
            
        Returns:
            Plotly figure
        """
        if show_drawdown:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Equity Curve', 'Drawdown'],
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
        else:
            fig = go.Figure()
        
        # Equity curve
        fig.add_trace(
            go.Scatter(
                x=results.equity_curve.index,
                y=results.equity_curve.values,
                mode='lines',
                name='Equity',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1 if show_drawdown else None
        )
        
        if show_drawdown:
            # Drawdown
            fig.add_trace(
                go.Scatter(
                    x=results.drawdown_curve.index,
                    y=results.drawdown_curve.values * 100,
                    mode='lines',
                    name='Drawdown %',
                    line=dict(color='red', width=1),
                    fill='tonexty'
                ),
                row=2, col=1
            )
            
            fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
        
        fig.update_layout(
            title='Backtest Results',
            xaxis_title='Date',
            yaxis_title='Account Value',
            height=600 if show_drawdown else 400
        )
        
        return fig
    
    @staticmethod
    def plot_trade_analysis(results: BacktestResults) -> go.Figure:
        """
        Plot trade analysis charts
        
        Args:
            results: Backtest results
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['PnL Distribution', 'Monthly Returns', 'Trade Timeline', 'Win/Loss Ratio'],
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "pie"}]]
        )
        
        # PnL Distribution
        pnls = [t.pnl for t in results.trades if t.pnl is not None]
        if pnls:
            fig.add_trace(
                go.Histogram(x=pnls, name='PnL Distribution', nbinsx=20),
                row=1, col=1
            )
        
        # Monthly Returns
        if not results.monthly_returns.empty:
            fig.add_trace(
                go.Bar(
                    x=results.monthly_returns.index,
                    y=results.monthly_returns.values * 100,
                    name='Monthly Returns %'
                ),
                row=1, col=2
            )
        
        # Trade Timeline
        trade_times = [t.entry_time for t in results.trades if t.entry_time]
        trade_pnls = [t.pnl for t in results.trades if t.pnl is not None]
        
        if trade_times and trade_pnls:
            colors = ['green' if pnl > 0 else 'red' for pnl in trade_pnls]
            fig.add_trace(
                go.Scatter(
                    x=trade_times,
                    y=trade_pnls,
                    mode='markers',
                    marker=dict(color=colors),
                    name='Trades'
                ),
                row=2, col=1
            )
        
        # Win/Loss Ratio
        wins = len([t for t in results.trades if t.pnl and t.pnl > 0])
        losses = len([t for t in results.trades if t.pnl and t.pnl <= 0])
        
        if wins + losses > 0:
            fig.add_trace(
                go.Pie(
                    labels=['Wins', 'Losses'],
                    values=[wins, losses],
                    name='Win/Loss Ratio'
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title='Trade Analysis')
        return fig
    
    @staticmethod
    def print_performance_summary(results: BacktestResults):
        """Print formatted performance summary"""
        metrics = results.performance_metrics
        
        print("=" * 50)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 50)
        
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        print(f"Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Average Trade: ${metrics.get('avg_trade', 0):.2f}")
        print(f"Final Equity: ${metrics.get('final_equity', 0):.2f}")
        
        print("\n" + "=" * 50)