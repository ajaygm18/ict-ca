# ICT Trading Framework Configuration

# Data Settings
DEFAULT_TIMEFRAME = "5T"  # 5-minute timeframe
DEFAULT_SYMBOL = "EURUSD"

# Trading Sessions (EST timezone)
SESSIONS = {
    "NY_KILLZONE": {"start": "10:00", "end": "11:00"},
    "PRE_MARKET": {"start": "02:00", "end": "07:00"},
    "NY_OPEN": {"start": "09:30", "end": "10:00"},
    "POWER_HOUR": {"start": "14:00", "end": "15:00"},
    "LONDON_NY_OVERLAP": {"start": "08:00", "end": "11:00"},
    "AFTERNOON": {"start": "13:00", "end": "16:00"}
}

# Risk Management Defaults
DEFAULT_RISK_PERCENT = 0.02  # 2% risk per trade
DEFAULT_RISK_REWARD = 2.0    # 1:2 risk-reward ratio
MAX_TRADES_PER_DAY = 3

# ICT Pattern Parameters
FVG_MIN_PIPS = 5            # Minimum FVG size in pips
OB_LOOKBACK = 10            # Order block lookback periods
ATR_PERIOD = 14             # ATR calculation period
ATR_THRESHOLD = 0.0010      # Minimum ATR for volatility filter

# Fibonacci Levels for OTE
FIBONACCI_LEVELS = {
    "OTE_MIN": 0.618,  # 61.8%
    "OTE_MAX": 0.786   # 78.6%
}

# Performance Metrics
BENCHMARK_SYMBOL = "SPY"
RISK_FREE_RATE = 0.02