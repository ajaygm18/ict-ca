# ICT Trading Framework - Strategy Implementation Guide

## Overview

This document provides detailed information about implementing and using the 15 ICT strategies in the framework.

## Implemented Strategies

### Session-Based Strategies

#### 1. SILVER_BULLET
- **Session**: NY Killzone (10:00–11:00 EST)
- **Setup**: Liquidity sweep of prior swing high/low
- **Entry**: Opposite direction after sweep + displacement
- **Filters**: FVG confirmation, displacement validation
- **Risk Management**: Stop beyond sweep wick, TP at next liquidity

#### 2. PRE_MARKET_BREAKOUT
- **Session**: Pre-market range (2:00–7:00 EST), Entry at NY open (9:30 EST)
- **Setup**: Consolidation range detection
- **Entry**: Breakout trade with confirmation
- **Filters**: ATR volatility filter, range size validation
- **Risk Management**: Stop at opposite range side, 2x risk TP

#### 3. MARKET_OPEN_REVERSAL
- **Session**: First 30 mins after NY open (9:30–10:00 EST)
- **Setup**: False breakout of pre-market levels
- **Entry**: Reversal after failed breakout + rejection candle
- **Filters**: Strong rejection confirmation, time-based
- **Risk Management**: Stop beyond fakeout wick, FVG/OB targets

#### 4. POWER_HOUR
- **Session**: 14:00–15:00 EST
- **Setup**: Liquidity sweeps into major pools
- **Entry**: After sweep + displacement
- **Filters**: Major liquidity identification, bias determination
- **Risk Management**: Stop beyond sweep, next liquidity target

### Pattern-Based Strategies

#### 5. FVG_SNIPER
- **Setup**: Fair Value Gap identification (3-candle pattern)
- **Entry**: Retracement into FVG zone
- **Filters**: HTF bias alignment, retracement confirmation
- **Risk Management**: Stop beyond FVG boundary, liquidity targets

#### 6. ORDER_BLOCK
- **Setup**: Last opposing candle before market structure shift
- **Entry**: Retest of Order Block
- **Filters**: Structure shift confirmation, retest validation
- **Risk Management**: Stop beyond OB wick, opposite OB targets

### Strategy Parameters

Each strategy supports customizable parameters:

```python
# Example: FVG Sniper parameters
parameters = {
    'min_fvg_size_pips': 5,          # Minimum FVG size
    'max_fvg_age_candles': 20,       # Maximum FVG age
    'risk_percent': 0.02,            # Risk per trade (2%)
    'risk_reward_ratio': 2.0,        # Risk-reward ratio
    'require_htf_alignment': True    # Require HTF bias alignment
}
```

## Usage Examples

### Basic Strategy Execution
```python
from src.main import ICTFramework

framework = ICTFramework(initial_capital=10000)

results = framework.run_strategy(
    strategy_name='FVG_SNIPER',
    symbol='EURUSD=X',
    start_date='2023-01-01',
    end_date='2023-12-31'
)

framework.show_performance(results)
```

### Strategy Comparison
```python
comparison = framework.run_multiple_strategies(
    strategy_names=['SILVER_BULLET', 'FVG_SNIPER', 'ORDER_BLOCK'],
    symbol='EURUSD=X',
    start_date='2023-01-01',
    end_date='2023-12-31'
)

comparison_table = framework.get_strategy_comparison(comparison)
print(comparison_table)
```

### Parameter Optimization
```python
param_grid = {
    'min_fvg_size_pips': [3, 5, 8, 10],
    'risk_reward_ratio': [1.5, 2.0, 2.5, 3.0]
}

optimization = framework.optimize_strategy(
    strategy_name='FVG_SNIPER',
    parameter_grid=param_grid,
    symbol='EURUSD=X',
    start_date='2023-01-01',
    end_date='2023-12-31'
)
```

## ICT Concepts Implementation

### Fair Value Gaps (FVG)
- **Detection**: 3-candle pattern where gap exists between candles 1 and 3
- **Types**: Bullish (low of 1st > high of 3rd), Bearish (high of 1st < low of 3rd)
- **Usage**: Entry on retracement, stop beyond gap boundaries

### Order Blocks (OB)
- **Detection**: Last opposing candle before significant move
- **Types**: Bullish (last down candle before up move), Bearish (last up candle before down move)
- **Usage**: Entry on retest, stop beyond block boundaries

### Liquidity Sweeps
- **Detection**: Price briefly breaking recent highs/lows then reversing
- **Purpose**: Institutional order triggering before real move
- **Usage**: Entry after sweep confirmation with displacement

### Market Structure
- **Higher Highs/Higher Lows**: Bullish structure
- **Lower Highs/Lower Lows**: Bearish structure
- **Structure Shift**: Break of recent swing points

### Sessions
- **NY Killzone**: 10:00-11:00 EST (high probability moves)
- **London-NY Overlap**: 8:00-11:00 EST (high volume)
- **Power Hour**: 14:00-15:00 EST (late day momentum)

## Performance Metrics

The framework calculates comprehensive metrics:

### Basic Metrics
- Total Trades
- Win Rate
- Profit Factor
- Average Win/Loss
- Total/Annual Return

### Risk Metrics
- Maximum Drawdown
- Sharpe Ratio
- Calmar Ratio
- Consecutive Wins/Losses

### Advanced Analysis
- Monthly returns
- Trade duration analysis
- Strategy breakdown
- Session performance

## Data Requirements

### Timeframes
- **Recommended**: 5-minute or 15-minute for intraday strategies
- **Minimum**: 1-minute for precise entry/exit timing
- **Higher Timeframes**: 1-hour, 4-hour for bias confirmation

### Data Sources
- **Yahoo Finance**: Free access to major forex pairs
- **CSV Files**: Custom data import capability
- **Required Columns**: OHLCV (Open, High, Low, Close, Volume)

### Data Quality
- Clean data without gaps
- Proper timezone handling (EST for ICT concepts)
- Sufficient lookback period (minimum 3 months)

## Risk Management

### Position Sizing
- Default: 2% risk per trade
- Based on distance to stop loss
- Adjustable per strategy

### Stop Loss Rules
- Beyond pattern boundaries (FVG, OB)
- Recent swing levels
- ATR-based dynamic stops

### Take Profit Targets
- Risk-reward multiples (1:2, 1:3)
- Next liquidity levels
- Opposite pattern boundaries

## Extending the Framework

### Creating Custom Strategies

```python
from src.strategies.base_strategy import BaseICTStrategy, Signal, SignalType

class MyCustomStrategy(BaseICTStrategy):
    def _set_default_parameters(self):
        self.parameters.setdefault('my_param', 10)
    
    def check_entry_conditions(self, data, index):
        # Implement your logic here
        if entry_condition_met:
            return Signal(
                timestamp=data.index[index],
                signal_type=SignalType.BUY,
                entry_price=data.iloc[index]['close'],
                stop_loss=calculate_stop_loss(),
                take_profit=calculate_take_profit(),
                confidence=0.8,
                reason="My custom signal"
            )
        return None
    
    def generate_signals(self, data):
        signals = []
        for i in range(len(data)):
            signal = self.check_entry_conditions(data, i)
            if signal:
                signals.append(signal)
        return signals
```

### Adding New Pattern Detection

```python
def detect_my_pattern(data, parameters):
    patterns = []
    # Implement pattern detection logic
    return patterns
```

## Troubleshooting

### Common Issues

1. **No Data**: Check symbol format and date range
2. **No Signals**: Verify strategy parameters and filters
3. **Import Errors**: Ensure correct Python path setup
4. **Performance Issues**: Use smaller datasets for testing

### Data Issues
- Use correct symbol format for Yahoo Finance (e.g., 'EURUSD=X')
- Ensure sufficient data history for pattern detection
- Check for data gaps or missing sessions

### Strategy Issues
- Review parameter settings for realistic values
- Check filter conditions (too restrictive filters = no signals)
- Verify session timing matches your data timezone

## Best Practices

### Backtesting
1. Use out-of-sample testing
2. Consider transaction costs
3. Test across different market conditions
4. Validate results with walk-forward analysis

### Parameter Optimization
1. Avoid overfitting with too many parameters
2. Use meaningful ranges for optimization
3. Test robustness across time periods
4. Consider parameter stability

### Strategy Development
1. Start with simple concepts
2. Add complexity gradually
3. Validate each component separately
4. Document strategy logic clearly

### Risk Management
1. Never risk more than you can afford to lose
2. Use proper position sizing
3. Consider correlation between strategies
4. Monitor maximum drawdown carefully