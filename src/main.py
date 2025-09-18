"""
Main ICT Trading Framework Interface

This module provides the primary interface for the ICT trading framework,
allowing users to easily run strategies, perform backtests, and analyze results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Type
import importlib
import os
from datetime import datetime

# Import framework components
from .utils.market_data import MarketDataLoader, SessionDetector
from .utils.ict_patterns import ICTPatternDetector
from .strategies.base_strategy import BaseICTStrategy
from .backtesting.backtester import Backtester, BacktestResults, BacktestVisualizer

# Import available strategies
from .strategies.silver_bullet import SilverBulletStrategy
from .strategies.pre_market_breakout import PreMarketBreakoutStrategy
from .strategies.market_open_reversal import MarketOpenReversalStrategy
from .strategies.power_hour import PowerHourStrategy
from .strategies.fvg_sniper import FVGSniperStrategy
from .strategies.order_block import OrderBlockStrategy
from .strategies.smt_divergence import SMTDivergenceStrategy
from .strategies.turtle_soup import TurtleSoupStrategy
from .strategies.power_of_3 import PowerOf3Strategy
from .strategies.daily_bias_liquidity import DailyBiasLiquidityStrategy
from .strategies.morning_session import MorningSessionStrategy
from .strategies.afternoon_reversal import AfternoonReversalStrategy
from .strategies.optimal_trade_entry import OptimalTradeEntryStrategy


class ICTFramework:
    """
    Main ICT Trading Framework Interface
    
    This class provides a unified interface for:
    - Loading and processing market data
    - Running ICT strategies
    - Performing backtests
    - Analyzing and visualizing results
    """
    
    def __init__(self, initial_capital: float = 10000):
        """
        Initialize the ICT Framework
        
        Args:
            initial_capital: Starting capital for backtests
        """
        self.initial_capital = initial_capital
        self.data_loader = MarketDataLoader()
        self.session_detector = SessionDetector()
        self.pattern_detector = ICTPatternDetector()
        self.backtester = Backtester(initial_capital)
        
        # Available strategies registry
        self.available_strategies = {
            'SILVER_BULLET': SilverBulletStrategy,
            'PRE_MARKET_BREAKOUT': PreMarketBreakoutStrategy,
            'MARKET_OPEN_REVERSAL': MarketOpenReversalStrategy,
            'POWER_HOUR': PowerHourStrategy,
            'FVG_SNIPER': FVGSniperStrategy,
            'ORDER_BLOCK': OrderBlockStrategy,
            'SMT_DIVERGENCE': SMTDivergenceStrategy,
            'TURTLE_SOUP': TurtleSoupStrategy,
            'POWER_OF_3': PowerOf3Strategy,
            'DAILY_BIAS_LIQUIDITY': DailyBiasLiquidityStrategy,
            'MORNING_SESSION': MorningSessionStrategy,
            'AFTERNOON_REVERSAL': AfternoonReversalStrategy,
            'OPTIMAL_TRADE_ENTRY': OptimalTradeEntryStrategy,
        }
        
        self.current_data: Optional[pd.DataFrame] = None
        self.last_results: Optional[BacktestResults] = None
    
    def load_data(self, symbol: str, start_date: str, end_date: str, 
                  source: str = 'yahoo', **kwargs) -> pd.DataFrame:
        """
        Load market data for analysis
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD=X')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            source: Data source ('yahoo' or 'csv')
            **kwargs: Additional arguments for data loading
            
        Returns:
            DataFrame with OHLCV data and indicators
        """
        if source.lower() == 'yahoo':
            interval = kwargs.get('interval', '5m')
            data = self.data_loader.load_from_yahoo(symbol, start_date, end_date, interval)
        
        elif source.lower() == 'csv':
            file_path = kwargs.get('file_path')
            if not file_path:
                raise ValueError("file_path required for CSV data source")
            data = self.data_loader.load_from_csv(file_path)
        
        else:
            raise ValueError(f"Unsupported data source: {source}")
        
        # Add technical indicators
        data = self.data_loader.add_technical_indicators(data)
        
        # Add session markers
        data = self.session_detector.add_session_markers(data)
        
        self.current_data = data
        return data
    
    def list_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.available_strategies.keys())
    
    def get_strategy_info(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get information about a specific strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with strategy information
        """
        if strategy_name not in self.available_strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {self.list_strategies()}")
        
        # Create temporary instance to get info
        strategy_class = self.available_strategies[strategy_name]
        temp_strategy = strategy_class(strategy_name)
        
        return {
            'name': strategy_name,
            'class': strategy_class.__name__,
            'description': strategy_class.__doc__,
            'default_parameters': temp_strategy.parameters
        }
    
    def run_strategy(self, strategy_name: str, symbol: str = None, 
                    start_date: str = None, end_date: str = None,
                    data: pd.DataFrame = None, parameters: Dict[str, Any] = None,
                    **data_kwargs) -> BacktestResults:
        """
        Run a complete strategy backtest
        
        Args:
            strategy_name: Name of the strategy to run
            symbol: Trading symbol (if loading new data)
            start_date: Start date (if loading new data)
            end_date: End date (if loading new data)
            data: Pre-loaded data (optional)
            parameters: Strategy parameters to override defaults
            **data_kwargs: Additional arguments for data loading
            
        Returns:
            BacktestResults object with complete analysis
        """
        # Validate strategy
        if strategy_name not in self.available_strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found. Available: {self.list_strategies()}")
        
        # Load data if not provided
        if data is None:
            if not all([symbol, start_date, end_date]):
                if self.current_data is not None:
                    data = self.current_data
                else:
                    raise ValueError("Either provide data or symbol/start_date/end_date")
            else:
                data = self.load_data(symbol, start_date, end_date, **data_kwargs)
        
        # Initialize strategy
        strategy_class = self.available_strategies[strategy_name]
        strategy = strategy_class(strategy_name, parameters)
        
        # Run backtest
        results = self.backtester.run_backtest(strategy, data)
        self.last_results = results
        
        return results
    
    def run_multiple_strategies(self, strategy_names: List[str], 
                              symbol: str = None, start_date: str = None, 
                              end_date: str = None, data: pd.DataFrame = None,
                              **data_kwargs) -> Dict[str, BacktestResults]:
        """
        Run multiple strategies and compare results
        
        Args:
            strategy_names: List of strategy names to run
            symbol: Trading symbol (if loading new data)
            start_date: Start date (if loading new data)
            end_date: End date (if loading new data)
            data: Pre-loaded data (optional)
            **data_kwargs: Additional arguments for data loading
            
        Returns:
            Dictionary mapping strategy names to their results
        """
        # Load data if not provided
        if data is None:
            if not all([symbol, start_date, end_date]):
                if self.current_data is not None:
                    data = self.current_data
                else:
                    raise ValueError("Either provide data or symbol/start_date/end_date")
            else:
                data = self.load_data(symbol, start_date, end_date, **data_kwargs)
        
        results = {}
        
        for strategy_name in strategy_names:
            try:
                print(f"Running {strategy_name}...")
                results[strategy_name] = self.run_strategy(
                    strategy_name, data=data
                )
                print(f"✓ {strategy_name} completed")
            
            except Exception as e:
                print(f"✗ {strategy_name} failed: {str(e)}")
                results[strategy_name] = None
        
        return results
    
    def optimize_strategy(self, strategy_name: str, parameter_grid: Dict[str, List],
                         symbol: str = None, start_date: str = None, 
                         end_date: str = None, data: pd.DataFrame = None,
                         optimization_metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search
        
        Args:
            strategy_name: Name of the strategy to optimize
            parameter_grid: Dictionary of parameters and their values to test
            symbol: Trading symbol (if loading new data)
            start_date: Start date (if loading new data)
            end_date: End date (if loading new data)
            data: Pre-loaded data (optional)
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            
        Returns:
            Dictionary with best parameters and results
        """
        # Load data if not provided
        if data is None:
            if not all([symbol, start_date, end_date]):
                if self.current_data is not None:
                    data = self.current_data
                else:
                    raise ValueError("Either provide data or symbol/start_date/end_date")
            else:
                data = self.load_data(symbol, start_date, end_date)
        
        # Generate parameter combinations
        from itertools import product
        
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        param_combinations = list(product(*param_values))
        
        best_score = -float('inf')
        best_params = None
        best_results = None
        all_results = []
        
        print(f"Optimizing {strategy_name} with {len(param_combinations)} parameter combinations...")
        
        for i, param_combo in enumerate(param_combinations):
            # Create parameter dictionary
            params = dict(zip(param_names, param_combo))
            
            try:
                # Run backtest with these parameters
                results = self.run_strategy(strategy_name, data=data, parameters=params)
                
                # Get optimization metric
                score = results.performance_metrics.get(optimization_metric, -float('inf'))
                
                all_results.append({
                    'parameters': params,
                    'score': score,
                    'results': results
                })
                
                # Check if this is the best so far
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_results = results
                
                if (i + 1) % 10 == 0:
                    print(f"Completed {i + 1}/{len(param_combinations)} combinations...")
            
            except Exception as e:
                print(f"Failed combination {params}: {str(e)}")
                continue
        
        print(f"Optimization completed. Best {optimization_metric}: {best_score:.4f}")
        
        return {
            'best_parameters': best_params,
            'best_score': best_score,
            'best_results': best_results,
            'all_results': all_results,
            'optimization_metric': optimization_metric
        }
    
    def show_performance(self, results: BacktestResults = None, 
                        show_plots: bool = True) -> None:
        """
        Display performance summary and plots
        
        Args:
            results: BacktestResults to display (uses last results if None)
            show_plots: Whether to show interactive plots
        """
        if results is None:
            results = self.last_results
        
        if results is None:
            print("No results to display. Run a strategy first.")
            return
        
        # Print performance summary
        BacktestVisualizer.print_performance_summary(results)
        
        if show_plots:
            # Show equity curve
            equity_fig = BacktestVisualizer.plot_equity_curve(results)
            equity_fig.show()
            
            # Show trade analysis
            trade_fig = BacktestVisualizer.plot_trade_analysis(results)
            trade_fig.show()
    
    def export_results(self, results: BacktestResults = None, 
                      file_path: str = None) -> str:
        """
        Export backtest results to files
        
        Args:
            results: BacktestResults to export (uses last results if None)
            file_path: Base file path (without extension)
            
        Returns:
            Path to exported files
        """
        if results is None:
            results = self.last_results
        
        if results is None:
            raise ValueError("No results to export. Run a strategy first.")
        
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = f"backtest_results_{timestamp}"
        
        # Export trades to CSV
        trades_data = []
        for trade in results.trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'direction': trade.direction.name,
                'size': trade.size,
                'pnl': trade.pnl,
                'stop_loss': trade.stop_loss,
                'take_profit': trade.take_profit,
                'exit_reason': trade.exit_reason,
                'strategy': trade.strategy
            })
        
        trades_df = pd.DataFrame(trades_data)
        trades_csv = f"{file_path}_trades.csv"
        trades_df.to_csv(trades_csv, index=False)
        
        # Export equity curve
        equity_csv = f"{file_path}_equity.csv"
        results.equity_curve.to_csv(equity_csv)
        
        # Export performance metrics
        metrics_csv = f"{file_path}_metrics.csv"
        metrics_df = pd.DataFrame([results.performance_metrics])
        metrics_df.to_csv(metrics_csv, index=False)
        
        print(f"Results exported to:")
        print(f"  Trades: {trades_csv}")
        print(f"  Equity: {equity_csv}")
        print(f"  Metrics: {metrics_csv}")
        
        return file_path
    
    def get_strategy_comparison(self, results_dict: Dict[str, BacktestResults]) -> pd.DataFrame:
        """
        Create comparison table for multiple strategy results
        
        Args:
            results_dict: Dictionary mapping strategy names to their results
            
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for strategy_name, results in results_dict.items():
            if results is None:
                continue
            
            metrics = results.performance_metrics
            comparison_data.append({
                'Strategy': strategy_name,
                'Total Trades': metrics.get('total_trades', 0),
                'Win Rate': f"{metrics.get('win_rate', 0):.2%}",
                'Total Return': f"{metrics.get('total_return', 0):.2%}",
                'Annual Return': f"{metrics.get('annual_return', 0):.2%}",
                'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'Profit Factor': f"{metrics.get('profit_factor', 0):.2f}",
                'Final Equity': f"${metrics.get('final_equity', 0):.2f}"
            })
        
        return pd.DataFrame(comparison_data)


# Convenience functions for quick access
def quick_backtest(strategy_name: str, symbol: str, start_date: str, 
                  end_date: str, initial_capital: float = 10000) -> BacktestResults:
    """
    Quick backtest function for single strategy
    
    Args:
        strategy_name: Name of strategy to test
        symbol: Trading symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital
        
    Returns:
        BacktestResults
    """
    framework = ICTFramework(initial_capital)
    return framework.run_strategy(strategy_name, symbol, start_date, end_date)


def compare_strategies(strategy_names: List[str], symbol: str, 
                      start_date: str, end_date: str, 
                      initial_capital: float = 10000) -> pd.DataFrame:
    """
    Compare multiple strategies
    
    Args:
        strategy_names: List of strategy names
        symbol: Trading symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        initial_capital: Starting capital
        
    Returns:
        Comparison DataFrame
    """
    framework = ICTFramework(initial_capital)
    results = framework.run_multiple_strategies(
        strategy_names, symbol, start_date, end_date
    )
    return framework.get_strategy_comparison(results)