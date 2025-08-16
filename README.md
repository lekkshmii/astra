# Astra Trading Platform

**Advanced Monte Carlo Risk Analysis for Algorithmic Trading**

---

## Overview

Astra is a sophisticated trading platform that combines high-performance Rust Monte Carlo simulations with professional Python backtesting infrastructure. Built for quantitative researchers and algorithmic traders who need institutional-grade risk analysis.

## Key Features

### **Rust-Powered Monte Carlo Engine**
- **Flash Crash Scenarios**: Model sudden liquidity evaporation events
- **Regime Switching**: Bull/bear market transition simulations  
- **Correlation Breakdown**: Asset correlation shock modeling
- **Volatility Clustering**: GARCH-style volatility persistence
- **Portfolio Stress Testing**: Comprehensive risk assessment

### **Professional Backtesting**
- **Signal-Based Testing**: Clean separation of strategy and execution
- **Transaction Cost Modeling**: Realistic commission and slippage
- **Multiple Strategies**: Momentum, mean reversion, pairs trading, volatility breakout
- **Performance Metrics**: Sharpe, Calmar, drawdown, VaR calculations
- **VectorBT Integration**: (Optional) for advanced vectorized backtesting

### **Risk Management**
- **Circuit Breakers**: Institutional-grade safety mechanisms
- **Position Sizing**: Dynamic allocation with Monte Carlo inputs
- **Real-time Monitoring**: Portfolio risk assessment
- **Stress Testing**: Multi-scenario portfolio analysis

### **Data & Visualization**
- **Market Data Integration**: yfinance with robust error handling
- **Professional Charts**: Publication-quality visualizations
- **Comprehensive Reporting**: Detailed performance analytics

---

## Quick Start

### Prerequisites
- **Python 3.9-3.11** (required for VectorBT compatibility)
- **Rust 1.70+** (for Monte Carlo engine)
- **Poetry** (for dependency management)

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd astra-main

# Install Python dependencies
poetry install

# Build Rust Monte Carlo engine
cd rust_monte_carlo
cargo build --release
cd ..

# Build Python bindings
poetry run maturin develop --release
```

### Basic Usage

```python
from src.data.loader import DataLoader
from src.backtesting.engine import BacktestEngine
from src.backtesting.strategies import MomentumStrategy
from src.monte_carlo.scenarios import ScenarioRunner

# Load market data
loader = DataLoader()
data = loader.get_universe_data('tech', '2023-01-01', '2024-01-01')

# Create and backtest strategy
strategy = MomentumStrategy(short_window=10, long_window=30)
signals = strategy.generate_signals(data)

engine = BacktestEngine(initial_capital=100000)
result = engine.run_backtest(data, signals)

print(f"Total Return: {result.metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.3f}")

# Run Monte Carlo stress test
runner = ScenarioRunner()
stress_results = runner.portfolio_stress_test(
    portfolio_weights=[0.25, 0.25, 0.25, 0.25],
    asset_prices=[150.0, 300.0, 2500.0, 180.0]
)

for scenario, metrics in stress_results.items():
    print(f"{scenario} VaR 95%: {metrics['var_95']:.2%}")
```

---

## Architecture

### Core Components

```
astra-main/
├── src/
│   ├── data/           # Market data management
│   ├── backtesting/    # Strategy testing engine
│   ├── monte_carlo/    # Risk scenario simulations
│   ├── risk_management/# Safety systems
│   ├── strategies/     # Trading algorithms
│   ├── optimization/   # Portfolio optimization
│   └── visualization/  # Charts and dashboards
├── rust_monte_carlo/   # High-performance Rust engine
├── examples/           # Usage demonstrations
├── tests/              # Test suite
└── results/            # Output storage
```

### Technology Stack

- **Backend**: Python 3.9-3.11 + Rust 1.70+
- **Computing**: NumPy, Pandas, Rayon (parallel processing)
- **Backtesting**: Custom engine + VectorBT integration
- **Optimization**: CVXPY for portfolio optimization
- **Visualization**: Matplotlib, Plotly, Seaborn
- **Data**: yfinance, DuckDB (planned)
- **FFI**: PyO3 for Python-Rust integration

---

## Examples

### 1. Basic Momentum Strategy

```python
# Run the included momentum strategy example
poetry run python examples/basic_backtest.py
```

### 2. Monte Carlo Flash Crash

```python
from src.monte_carlo.scenarios import FlashCrashScenario

scenario = FlashCrashScenario(
    crash_probability=0.001,
    crash_magnitude=0.15,
    recovery_rate=0.1
)

result = scenario.run(
    initial_prices=[100.0],
    num_simulations=10000,
    time_steps=252
)

print(f"Crash frequency: {result.statistics['crash_frequency']:.2%}")
print(f"Mean max drawdown: {result.statistics['max_drawdown_mean']:.2%}")
```

### 3. Portfolio Stress Testing

```python
runner = ScenarioRunner()

# Test a balanced tech portfolio
stress_results = runner.run_all_scenarios(
    initial_prices=[150.0, 300.0, 2500.0, 180.0]  # AAPL, MSFT, GOOGL, AMZN
)

for scenario_name, result in stress_results.items():
    print(f"{scenario_name}:")
    print(f"  95th percentile: ${result.percentiles['p95']:.2f}")
    print(f"  5th percentile: ${result.percentiles['p5']:.2f}")
```

---

## Testing

```bash
# Run foundation tests
poetry run python test_foundation.py

# Test Monte Carlo scenarios
poetry run python examples/test_monte_carlo.py

# Run full test suite
poetry run pytest tests/ -v
```

---

## Performance Metrics

Astra provides comprehensive performance analysis:

### **Risk Metrics**
- **Value at Risk (VaR)**: 95%, 99% confidence levels
- **Expected Shortfall**: Tail risk measurement  
- **Maximum Drawdown**: Peak-to-trough losses
- **Volatility**: Annualized standard deviation

### **Return Metrics**
- **Total Return**: Cumulative performance
- **Annualized Return**: Time-adjusted performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Calmar Ratio**: Return/max drawdown ratio

### **Trading Metrics**
- **Win Rate**: Percentage of profitable trades
- **Average Trade P&L**: Mean trade profitability
- **Trade Frequency**: Number of trades per period
- **Transaction Costs**: Commission and slippage impact

---

## Configuration

### Strategy Parameters

```python
# Momentum strategy
strategy = MomentumStrategy(
    short_window=10,    # Fast moving average
    long_window=30,     # Slow moving average
)

# Mean reversion
strategy = MeanReversionStrategy(
    window=20,          # Lookback period
    threshold=2.0,      # Z-score threshold
)
```

### Monte Carlo Settings

```python
# Flash crash scenario
scenario = FlashCrashScenario(
    crash_probability=0.001,  # Daily crash probability
    crash_magnitude=0.15,     # 15% crash severity
    recovery_rate=0.1,        # Recovery speed
)
```

### Backtest Engine

```python
engine = BacktestEngine(
    initial_capital=100000,   # Starting capital
    commission=0.001,         # 10 basis points
)
```

---

## Risk Warnings

**Important Disclaimers**

1. **No Financial Advice**: This software is for educational and research purposes only
2. **Backtesting Limitations**: Past performance does not guarantee future results
3. **Model Risk**: Monte Carlo simulations are based on assumptions that may not hold
4. **Market Risk**: Real trading involves substantial risk of loss
5. **Testing Required**: Always paper trade strategies before live deployment

---

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. **Code Style**: Follow PEP 8 for Python, rustfmt for Rust
2. **Testing**: Add tests for new features
3. **Documentation**: Update docs for API changes
4. **Performance**: Profile performance-critical code

---

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

- **VectorBT**: Vectorized backtesting framework
- **PyO3**: Python-Rust integration
- **yfinance**: Market data access
- **Rust Community**: High-performance computing tools

---

**Built for the quantitative trading community**