"""
ICT Trading Framework - Basic Usage Examples

This script demonstrates how to use the ICT trading framework
for backtesting various Inner Circle Trader strategies.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.main import ICTFramework, quick_backtest, compare_strategies


def get_recent_date_range(days_back=30):
    """Get a recent date range for data loading"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def example_1_basic_backtest():
    """Example 1: Basic single strategy backtest"""
    print("=" * 60)
    print("EXAMPLE 1: Basic Strategy Backtest")
    print("=" * 60)
    
    # Initialize framework
    framework = ICTFramework(initial_capital=10000)
    
    # Get recent date range
    start_date, end_date = get_recent_date_range(30)
    print(f"Using date range: {start_date} to {end_date}")
    
    # Load data and run Silver Bullet strategy
    results = framework.run_strategy(
        strategy_name='SILVER_BULLET',
        symbol='EURUSD=X',
        start_date=start_date,
        end_date=end_date,
        interval='5m'
    )
    
    # Display results
    framework.show_performance(results, show_plots=False)
    
    print(f"Strategy generated {len(results.signals)} signals")
    print(f"Executed {len(results.trades)} trades")


def example_2_strategy_comparison():
    """Example 2: Compare multiple strategies"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Strategy Comparison")
    print("=" * 60)
    
    # Compare multiple strategies on recent data
    start_date, end_date = get_recent_date_range(45)  # Longer period for comparison
    comparison = compare_strategies(
        strategy_names=['SILVER_BULLET', 'FVG_SNIPER', 'ORDER_BLOCK'],
        symbol='EURUSD=X',
        start_date=start_date,
        end_date=end_date,
        initial_capital=10000
    )
    
    print("Strategy Comparison Results:")
    print(comparison.to_string(index=False))


def example_3_custom_parameters():
    """Example 3: Using custom strategy parameters"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Custom Parameters")
    print("=" * 60)
    
    framework = ICTFramework()
    
    # Custom parameters for FVG Sniper strategy
    custom_params = {
        'min_fvg_size_pips': 8,  # Larger minimum FVG size
        'risk_percent': 0.01,    # Lower risk per trade (1%)
        'risk_reward_ratio': 3.0, # Higher risk-reward ratio
        'require_htf_alignment': True
    }
    # Get recent date range for custom parameters test
    start_date, end_date = get_recent_date_range(30)
    
    results = framework.run_strategy(
        strategy_name='FVG_SNIPER',
        symbol='GBPUSD=X',
        start_date=start_date,
        end_date=end_date,
        parameters=custom_params
    )
    
    framework.show_performance(results, show_plots=False)


def example_4_optimization():
    """Example 4: Parameter optimization"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Parameter Optimization")
    print("=" * 60)
    
    framework = ICTFramework()
    
    # Define parameter grid for optimization
    param_grid = {
        'min_fvg_size_pips': [3, 5, 8, 10],
        'risk_reward_ratio': [1.5, 2.0, 2.5, 3.0],
        'max_fvg_age_candles': [15, 20, 25, 30]
    }
    
    print("Running parameter optimization...")
    start_date, end_date = get_recent_date_range(30)
    optimization_results = framework.optimize_strategy(
        strategy_name='FVG_SNIPER',
        parameter_grid=param_grid,
        symbol='EURUSD=X',
        start_date=start_date,
        end_date=end_date,
        optimization_metric='sharpe_ratio'
    )
    
    print(f"Best parameters: {optimization_results['best_parameters']}")
    print(f"Best Sharpe ratio: {optimization_results['best_score']:.3f}")


def example_5_data_analysis():
    """Example 5: Data loading and analysis"""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Data Analysis")
    print("=" * 60)
    
    framework = ICTFramework()
    
    # Get recent date range for analysis
    start_date, end_date = get_recent_date_range(20)  # Shorter period for detailed analysis
    
    # Load and analyze data
    data = framework.load_data(
        symbol='EURUSD=X',
        start_date=start_date,
        end_date=end_date,
        interval='15m'
    )
    
    print(f"Loaded {len(data)} candles")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Available columns: {list(data.columns)}")
    
    # Show session analysis
    ny_killzone_data = framework.session_detector.get_session_data(data, 'ny_killzone')
    print(f"NY Killzone periods: {len(ny_killzone_data)}")
    
    # Pattern detection example
    fvgs = framework.pattern_detector.detect_fair_value_gaps(data, min_gap_pips=5)
    print(f"Detected {len(fvgs)} Fair Value Gaps")


def example_6_export_results():
    """Example 6: Exporting results"""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Export Results")
    print("=" * 60)
    
    # Run a quick backtest with recent data
    start_date, end_date = get_recent_date_range(20)
    results = quick_backtest(
        strategy_name='SILVER_BULLET',
        symbol='EURUSD=X',
        start_date=start_date,
        end_date=end_date
    )
    
    # Export results
    framework = ICTFramework()
    framework.last_results = results
    
    export_path = framework.export_results(
        file_path='examples/silver_bullet_recent_results'
    )
    
    print(f"Results exported to: {export_path}")


def example_7_live_data_simulation():
    """Example 7: Simulate live trading signals"""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Live Signal Simulation")
    print("=" * 60)
    
    framework = ICTFramework()
    
    # Get recent data for live simulation
    start_date, end_date = get_recent_date_range(15)  # Short period for live simulation
    
    # Load recent data
    data = framework.load_data(
        symbol='EURUSD=X',
        start_date=start_date,
        end_date=end_date,
        interval='5m'
    )
    
    # Initialize strategy
    from src.strategies.fvg_sniper import FVGSniperStrategy
    strategy = FVGSniperStrategy('FVG_SNIPER')
    
    # Simulate live signal generation (last 100 candles)
    recent_data = data.tail(100)
    signals = []
    
    for i in range(50, len(recent_data)):
        signal = strategy.check_entry_conditions(recent_data, i)
        if signal:
            signals.append(signal)
            print(f"Signal at {signal.timestamp}: {signal.signal_type.name} - {signal.reason}")
    
    print(f"\nGenerated {len(signals)} signals in recent data")


if __name__ == "__main__":
    try:
        # Run all examples
        example_1_basic_backtest()
        example_2_strategy_comparison()
        example_3_custom_parameters()
        example_4_optimization()
        example_5_data_analysis()
        example_6_export_results()
        example_7_live_data_simulation()
        
        print("\n" + "=" * 60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure you have installed all required dependencies:")
        print("pip install -r requirements.txt")