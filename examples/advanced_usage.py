"""
Advanced ICT Framework Examples

This script demonstrates advanced usage patterns including:
- Custom strategy development
- Advanced pattern detection
- Portfolio-level analysis
- Risk management customization
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.main import ICTFramework
from src.strategies.base_strategy import BaseICTStrategy, Signal, SignalType, TradeDirection
from src.utils.ict_patterns import ICTPatternDetector
from src.utils.market_data import SessionDetector


class CustomICTStrategy(BaseICTStrategy):
    """
    Example custom ICT strategy combining multiple concepts
    
    This strategy demonstrates how to create your own ICT strategy
    by combining Fair Value Gaps, Order Blocks, and session filtering.
    """
    
    def _set_default_parameters(self):
        """Set custom strategy parameters"""
        self.parameters.setdefault('fvg_min_size_pips', 5)
        self.parameters.setdefault('ob_min_size_pips', 8)
        self.parameters.setdefault('confluence_required', True)
        self.parameters.setdefault('allowed_sessions', ['ny_killzone', 'power_hour'])
        
        # Initialize components
        self.pattern_detector = ICTPatternDetector()
        self.session_detector = SessionDetector()
    
    def check_entry_conditions(self, data: pd.DataFrame, index: int):
        """Custom entry logic combining FVG and OB confluence"""
        if index < 20:
            return None
        
        current_time = data.index[index]
        
        # Session filter
        in_session = any(
            self.session_detector.is_in_session(current_time, session)
            for session in self.parameters['allowed_sessions']
        )
        
        if not in_session:
            return None
        
        # Check for FVG and OB confluence
        recent_data = data.iloc[index-20:index+1]
        
        # Detect patterns
        fvgs = self.pattern_detector.detect_fair_value_gaps(
            recent_data, self.parameters['fvg_min_size_pips']
        )
        obs = self.pattern_detector.detect_order_blocks(recent_data)
        
        if not fvgs or not obs:
            return None
        
        # Look for confluence
        current_candle = data.iloc[index]
        
        for fvg in fvgs:
            for ob in obs:
                # Check if price is interacting with both FVG and OB
                if (self._check_fvg_interaction(current_candle, fvg) and
                    self._check_ob_interaction(current_candle, ob) and
                    fvg.fvg_type.value == ob.ob_type.value):  # Same direction
                    
                    # Create signal
                    trade_direction = TradeDirection.LONG if fvg.fvg_type.value == 1 else TradeDirection.SHORT
                    signal_type = SignalType.BUY if trade_direction == TradeDirection.LONG else SignalType.SELL
                    
                    return Signal(
                        timestamp=current_time,
                        signal_type=signal_type,
                        entry_price=current_candle['close'],
                        stop_loss=self._calculate_stop_loss(fvg, ob, trade_direction),
                        take_profit=self._calculate_take_profit(data, index, fvg, ob, trade_direction),
                        confidence=0.9,
                        reason=f"Custom ICT: {fvg.fvg_type.name} FVG + {ob.ob_type.name} OB confluence",
                        metadata={'fvg': fvg, 'ob': ob}
                    )
        
        return None
    
    def generate_signals(self, data: pd.DataFrame):
        """Generate all signals for the dataset"""
        signals = []
        for i in range(len(data)):
            signal = self.check_entry_conditions(data, i)
            if signal:
                signals.append(signal)
        
        self.signals = signals
        return signals
    
    def _check_fvg_interaction(self, candle, fvg):
        """Check if candle interacts with FVG"""
        return (candle['low'] <= fvg.top and candle['high'] >= fvg.bottom)
    
    def _check_ob_interaction(self, candle, ob):
        """Check if candle interacts with Order Block"""
        return (candle['low'] <= ob.high and candle['high'] >= ob.low)
    
    def _calculate_stop_loss(self, fvg, ob, direction):
        """Calculate stop loss based on FVG and OB levels"""
        if direction == TradeDirection.LONG:
            return min(fvg.bottom, ob.low) - 0.0001 * 5  # 5 pip buffer
        else:
            return max(fvg.top, ob.high) + 0.0001 * 5  # 5 pip buffer
    
    def _calculate_take_profit(self, data, index, fvg, ob, direction):
        """Calculate take profit based on next liquidity level"""
        entry_price = data.iloc[index]['close']
        stop_loss = self._calculate_stop_loss(fvg, ob, direction)
        risk = abs(entry_price - stop_loss)
        
        # Default 2:1 risk reward
        if direction == TradeDirection.LONG:
            return entry_price + (risk * self.risk_reward_ratio)
        else:
            return entry_price - (risk * self.risk_reward_ratio)


def advanced_example_1_custom_strategy():
    """Demonstrate custom strategy development"""
    print("=" * 60)
    print("ADVANCED EXAMPLE 1: Custom Strategy Development")
    print("=" * 60)
    
    framework = ICTFramework()
    
    # Add our custom strategy to the framework
    framework.available_strategies['CUSTOM_ICT'] = CustomICTStrategy
    
    # Run backtest with custom strategy
    results = framework.run_strategy(
        strategy_name='CUSTOM_ICT',
        symbol='EURUSD=X',
        start_date='2023-10-01',
        end_date='2023-12-31'
    )
    
    framework.show_performance(results, show_plots=False)
    
    print(f"Custom strategy generated {len(results.signals)} signals")


def advanced_example_2_multi_timeframe():
    """Multi-timeframe analysis example"""
    print("\n" + "=" * 60)
    print("ADVANCED EXAMPLE 2: Multi-Timeframe Analysis")
    print("=" * 60)
    
    framework = ICTFramework()
    
    # Load different timeframes
    timeframes = ['5m', '15m', '1h']
    results_by_tf = {}
    
    for tf in timeframes:
        print(f"Running Silver Bullet on {tf} timeframe...")
        
        results = framework.run_strategy(
            strategy_name='SILVER_BULLET',
            symbol='EURUSD=X',
            start_date='2023-11-01',
            end_date='2023-12-31',
            interval=tf
        )
        
        results_by_tf[tf] = results
        
        # Quick metrics
        metrics = results.performance_metrics
        print(f"  {tf}: {metrics.get('total_trades', 0)} trades, "
              f"{metrics.get('win_rate', 0):.1%} win rate, "
              f"{metrics.get('total_return', 0):.1%} return")


def advanced_example_3_risk_management():
    """Advanced risk management customization"""
    print("\n" + "=" * 60)
    print("ADVANCED EXAMPLE 3: Risk Management Customization")
    print("=" * 60)
    
    framework = ICTFramework()
    
    # Test different risk levels
    risk_levels = [0.005, 0.01, 0.02, 0.03]  # 0.5%, 1%, 2%, 3%
    
    print("Testing different risk levels:")
    for risk in risk_levels:
        params = {
            'risk_percent': risk,
            'max_trades_per_day': 2,
            'risk_reward_ratio': 2.5
        }
        
        results = framework.run_strategy(
            strategy_name='FVG_SNIPER',
            symbol='EURUSD=X',
            start_date='2023-11-01',
            end_date='2023-12-31',
            parameters=params
        )
        
        metrics = results.performance_metrics
        print(f"  {risk*100:.1f}% risk: {metrics.get('total_return', 0):.1%} return, "
              f"{metrics.get('max_drawdown', 0):.1%} max DD")


def advanced_example_4_session_analysis():
    """Detailed session performance analysis"""
    print("\n" + "=" * 60)
    print("ADVANCED EXAMPLE 4: Session Performance Analysis")
    print("=" * 60)
    
    framework = ICTFramework()
    
    # Load data
    data = framework.load_data(
        symbol='EURUSD=X',
        start_date='2023-10-01',
        end_date='2023-12-31',
        interval='5m'
    )
    
    # Analyze different sessions
    sessions = ['ny_killzone', 'power_hour', 'london_ny_overlap']
    
    for session in sessions:
        # Filter data for this session
        session_data = framework.session_detector.get_session_data(data, session)
        
        if len(session_data) > 0:
            # Run strategy on session data
            results = framework.run_strategy(
                strategy_name='SILVER_BULLET',
                data=session_data
            )
            
            metrics = results.performance_metrics
            print(f"{session}:")
            print(f"  Periods: {len(session_data)}")
            print(f"  Trades: {metrics.get('total_trades', 0)}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.1%}")
            print(f"  Return: {metrics.get('total_return', 0):.1%}")
            print()


def advanced_example_5_pattern_statistics():
    """Pattern detection and statistics"""
    print("\n" + "=" * 60)
    print("ADVANCED EXAMPLE 5: Pattern Statistics")
    print("=" * 60)
    
    framework = ICTFramework()
    
    # Load data
    data = framework.load_data(
        symbol='EURUSD=X',
        start_date='2023-11-01',
        end_date='2023-12-31',
        interval='15m'
    )
    
    # Detect patterns
    detector = ICTPatternDetector()
    
    # Fair Value Gaps
    fvgs = detector.detect_fair_value_gaps(data, min_gap_pips=3)
    bullish_fvgs = [f for f in fvgs if f.fvg_type.value == 1]
    bearish_fvgs = [f for f in fvgs if f.fvg_type.value == -1]
    
    print(f"Fair Value Gaps detected: {len(fvgs)}")
    print(f"  Bullish: {len(bullish_fvgs)}")
    print(f"  Bearish: {len(bearish_fvgs)}")
    
    # Order Blocks
    obs = detector.detect_order_blocks(data)
    bullish_obs = [o for o in obs if o.ob_type.value == 1]
    bearish_obs = [o for o in obs if o.ob_type.value == -1]
    
    print(f"\nOrder Blocks detected: {len(obs)}")
    print(f"  Bullish: {len(bullish_obs)}")
    print(f"  Bearish: {len(bearish_obs)}")
    
    # Liquidity sweeps
    liquidity_pools = detector.detect_liquidity_sweeps(data)
    high_sweeps = [l for l in liquidity_pools if l.liquidity_type.value == 1]
    low_sweeps = [l for l in liquidity_pools if l.liquidity_type.value == -1]
    
    print(f"\nLiquidity sweeps detected: {len(liquidity_pools)}")
    print(f"  High sweeps: {len(high_sweeps)}")
    print(f"  Low sweeps: {len(low_sweeps)}")


def advanced_example_6_portfolio_simulation():
    """Portfolio-level multi-strategy simulation"""
    print("\n" + "=" * 60)
    print("ADVANCED EXAMPLE 6: Portfolio Simulation")
    print("=" * 60)
    
    framework = ICTFramework(initial_capital=50000)  # Larger account
    
    # Simulate portfolio allocation across strategies
    strategies = ['SILVER_BULLET', 'FVG_SNIPER', 'ORDER_BLOCK']
    allocation_per_strategy = 0.33  # 33% each
    
    portfolio_results = {}
    total_portfolio_value = framework.initial_capital
    
    for strategy in strategies:
        strategy_capital = framework.initial_capital * allocation_per_strategy
        
        # Create sub-framework for this strategy
        sub_framework = ICTFramework(strategy_capital)
        
        results = sub_framework.run_strategy(
            strategy_name=strategy,
            symbol='EURUSD=X',
            start_date='2023-09-01',
            end_date='2023-12-31'
        )
        
        portfolio_results[strategy] = results
        
        # Calculate strategy contribution
        final_value = results.performance_metrics.get('final_equity', strategy_capital)
        strategy_return = (final_value - strategy_capital) / strategy_capital
        
        print(f"{strategy}:")
        print(f"  Allocated: ${strategy_capital:,.2f}")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Return: {strategy_return:.2%}")
        print()
    
    # Calculate portfolio performance
    total_final_value = sum(
        r.performance_metrics.get('final_equity', 0) 
        for r in portfolio_results.values()
    )
    portfolio_return = (total_final_value - framework.initial_capital) / framework.initial_capital
    
    print(f"Portfolio Summary:")
    print(f"  Initial Value: ${framework.initial_capital:,.2f}")
    print(f"  Final Value: ${total_final_value:,.2f}")
    print(f"  Total Return: {portfolio_return:.2%}")


if __name__ == "__main__":
    try:
        # Run advanced examples
        advanced_example_1_custom_strategy()
        advanced_example_2_multi_timeframe()
        advanced_example_3_risk_management()
        advanced_example_4_session_analysis()
        advanced_example_5_pattern_statistics()
        advanced_example_6_portfolio_simulation()
        
        print("\n" + "=" * 60)
        print("ALL ADVANCED EXAMPLES COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error running advanced examples: {str(e)}")
        import traceback
        traceback.print_exc()