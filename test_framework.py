#!/usr/bin/env python3
"""
Quick Test Script for ICT Trading Framework
This script demonstrates that all major errors have been fixed.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.main import ICTFramework


def get_recent_date_range(days_back=7):
    """Get a recent date range for testing"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')


def main():
    print("=" * 60)
    print("ICT TRADING FRAMEWORK - QUICK TEST")
    print("=" * 60)
    
    # Test 1: Framework initialization
    print("1. Testing framework initialization...")
    try:
        framework = ICTFramework(initial_capital=10000)
        print("   ✓ Framework initialized successfully")
    except Exception as e:
        print(f"   ✗ Framework initialization failed: {e}")
        return False
    
    # Test 2: Strategy listing
    print("\n2. Testing strategy availability...")
    try:
        strategies = framework.list_strategies()
        print(f"   ✓ Found {len(strategies)} available strategies:")
        for strategy in strategies[:3]:
            print(f"     - {strategy}")
        print(f"     ... and {len(strategies)-3} more")
    except Exception as e:
        print(f"   ✗ Strategy listing failed: {e}")
        return False
    
    # Test 3: Data loading
    print("\n3. Testing data loading...")
    try:
        start_date, end_date = get_recent_date_range(7)
        data = framework.load_data(
            symbol='EURUSD=X',
            start_date=start_date,
            end_date=end_date,
            interval='5m'
        )
        print(f"   ✓ Loaded {len(data)} data points from {start_date} to {end_date}")
        print(f"   ✓ Data columns: {len(data.columns)} (OHLCV + indicators + sessions)")
    except Exception as e:
        print(f"   ✗ Data loading failed: {e}")
        return False
    
    # Test 4: Strategy execution
    print("\n4. Testing strategy execution...")
    try:
        results = framework.run_strategy(
            strategy_name='SILVER_BULLET',
            symbol='EURUSD=X',
            start_date=start_date,
            end_date=end_date,
            interval='5m'
        )
        metrics = results.performance_metrics
        print(f"   ✓ Strategy executed successfully")
        print(f"   ✓ Generated {metrics.get('total_trades', 0)} trades")
        print(f"   ✓ Win rate: {metrics.get('win_rate', 0):.1%}")
    except Exception as e:
        print(f"   ✗ Strategy execution failed: {e}")
        return False
    
    # Test 5: Multiple strategy comparison
    print("\n5. Testing strategy comparison...")
    try:
        from src.main import compare_strategies
        comparison = compare_strategies(
            strategy_names=['SILVER_BULLET', 'FVG_SNIPER'],
            symbol='EURUSD=X',
            start_date=start_date,
            end_date=end_date,
            initial_capital=10000
        )
        print(f"   ✓ Compared {len(comparison)} strategies successfully")
    except Exception as e:
        print(f"   ✗ Strategy comparison failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED! The framework is working correctly.")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python examples/basic_usage.py")
    print("  python examples/advanced_usage.py")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)