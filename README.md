# ICT-Base Algorithmic Trading Framework

A comprehensive Python trading framework implementing 15 Inner Circle Trader (ICT) strategies with modular architecture, backtesting capabilities, and performance analytics.

## 🚀 Features

- **15 ICT Strategies**: All 15 core strategies fully implemented and ready to use
  - Session-based: SILVER_BULLET, PRE_MARKET_BREAKOUT, MARKET_OPEN_REVERSAL, POWER_HOUR
  - Pattern-based: FVG_SNIPER, ORDER_BLOCK, BREAKER_BLOCK, REJECTION_BLOCK  
  - Advanced: SMT_DIVERGENCE, TURTLE_SOUP, POWER_OF_3, DAILY_BIAS_LIQUIDITY
  - Timing-based: MORNING_SESSION, AFTERNOON_REVERSAL, OPTIMAL_TRADE_ENTRY
- **Modular Design**: Each strategy is a separate module with clear entry/exit conditions
- **Risk Management**: Stop-loss, take-profit, position sizing for all strategies
- **Session Detection**: NY Killzone, London-NY overlap, Power Hour, Pre-market sessions
- **ICT Pattern Detection**: Fair Value Gaps (FVG), Order Blocks (OB), Liquidity Sweeps
- **Backtesting Engine**: Comprehensive performance metrics, equity curves, trade logging
- **Data Support**: CSV files, Yahoo Finance API integration
- **Visualization**: Interactive charts and performance analysis
- **Parameter Optimization**: Grid search optimization for strategy parameters

## 📦 Installation

```bash
# Clone the repository
git clone <repository-url>
cd ict-base

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

```python
from src.main import ICTFramework

# Initialize framework
framework = ICTFramework(initial_capital=10000)

# Run a strategy
results = framework.run_strategy(
    strategy_name="FVG_SNIPER",
    symbol="EURUSD=X",
    start_date="2023-01-01",
    end_date="2023-12-31"
)

# View performance
framework.show_performance(results)
```

## 📁 Project Structure

```
ict-base/
├── src/
│   ├── strategies/         # ICT strategy implementations
│   │   ├── base_strategy.py
│   │   ├── silver_bullet.py
│   │   ├── fvg_sniper.py
│   │   ├── order_block.py
│   │   └── ...
│   ├── utils/             # Market data and ICT utilities
│   │   ├── market_data.py
│   │   └── ict_patterns.py
│   ├── backtesting/       # Backtesting engine
│   │   └── backtester.py
│   └── main.py           # Main framework interface
├── examples/             # Usage examples
│   ├── basic_usage.py
│   └── advanced_usage.py
├── data/                # Sample data files
├── config.py           # Configuration settings
├── requirements.txt    # Dependencies
└── STRATEGY_GUIDE.md  # Detailed strategy documentation
```

## 🎭 Available Strategies

### Session-Based Strategies
1. **SILVER_BULLET** - NY Killzone liquidity sweeps with displacement confirmation
2. **PRE_MARKET_BREAKOUT** - Range breakouts at market open with volatility filters
3. **MARKET_OPEN_REVERSAL** - False breakout reversals in first 30 minutes
4. **POWER_HOUR** - Liquidity sweeps during 2-3 PM EST session

### Pattern-Based Strategies
5. **FVG_SNIPER** - Fair Value Gap retracement entries with HTF bias
6. **ORDER_BLOCK** - Institutional order block retests with structure confirmation
7. **BREAKER_BLOCK** - Failed order blocks turned support/resistance
8. **REJECTION_BLOCK** - Multiple rejection order blocks

### Advanced Strategies
9. **SMT_DIVERGENCE** - Smart Money Tool divergence between correlated assets
10. **TURTLE_SOUP** - Failed breakout reversal pattern
11. **POWER_OF_3** - Daily accumulation-manipulation-distribution cycle
12. **DAILY_BIAS_LIQUIDITY** - Higher timeframe bias with liquidity targets

### Time-Based Strategies
13. **MORNING_SESSION** - London-NY overlap liquidity hunts
14. **AFTERNOON_REVERSAL** - Trend exhaustion and liquidity sweeps
15. **OPTIMAL_TRADE_ENTRY** - Fibonacci retracement entries (62-79%)

## 📈 Example Usage

### Basic Strategy Backtest
```python
from src.main import quick_backtest

# Quick single strategy test
results = quick_backtest(
    strategy_name='SILVER_BULLET',
    symbol='EURUSD=X',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

### Strategy Comparison
```python
from src.main import compare_strategies

# Compare multiple strategies
comparison = compare_strategies(
    strategy_names=['SILVER_BULLET', 'FVG_SNIPER', 'ORDER_BLOCK'],
    symbol='EURUSD=X',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
print(comparison)
```

### Parameter Optimization
```python
framework = ICTFramework()

# Optimize strategy parameters
optimization = framework.optimize_strategy(
    strategy_name='FVG_SNIPER',
    parameter_grid={
        'min_fvg_size_pips': [3, 5, 8, 10],
        'risk_reward_ratio': [1.5, 2.0, 2.5, 3.0]
    },
    symbol='EURUSD=X',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

### Custom Strategy Development
```python
from src.strategies.base_strategy import BaseICTStrategy

class MyCustomStrategy(BaseICTStrategy):
    def _set_default_parameters(self):
        self.parameters.setdefault('my_param', 10)
    
    def check_entry_conditions(self, data, index):
        # Implement your custom logic
        pass
    
    def generate_signals(self, data):
        # Generate all signals
        pass
```

## 📊 Performance Metrics

The framework provides comprehensive performance analysis:

- **Basic Metrics**: Total trades, win rate, profit factor, average win/loss
- **Returns**: Total return, annual return, monthly breakdown
- **Risk Metrics**: Maximum drawdown, Sharpe ratio, Calmar ratio
- **Trade Analysis**: Duration analysis, consecutive wins/losses
- **Visualizations**: Equity curves, drawdown charts, trade distributions

## 🔧 Configuration

Key configuration options in `config.py`:

```python
# Trading Sessions (EST timezone)
SESSIONS = {
    "NY_KILLZONE": {"start": "10:00", "end": "11:00"},
    "POWER_HOUR": {"start": "14:00", "end": "15:00"},
    # ... other sessions
}

# Risk Management
DEFAULT_RISK_PERCENT = 0.02  # 2% risk per trade
DEFAULT_RISK_REWARD = 2.0    # 1:2 risk-reward ratio

# ICT Pattern Parameters
FVG_MIN_PIPS = 5            # Minimum FVG size
OB_LOOKBACK = 10            # Order block lookback periods
```

## 🎓 Learning Resources

- **STRATEGY_GUIDE.md**: Detailed explanation of each strategy and ICT concepts
- **examples/basic_usage.py**: Basic framework usage examples
- **examples/advanced_usage.py**: Advanced features and custom development

## 🚨 Risk Disclaimer

This framework is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always do your own research and consider your risk tolerance before trading.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests, report bugs, or suggest new features.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inner Circle Trader (ICT) for the trading concepts and methodologies
- The Python trading community for inspiration and tools
- Contributors and testers who help improve the framework

## 📞 Support

For questions, issues, or discussions:
- Create an issue on GitHub
- Check the documentation in STRATEGY_GUIDE.md
- Review the examples in the examples/ directory

---

**Happy Trading! 📈**