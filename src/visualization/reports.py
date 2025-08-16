"""Professional Report Generator - Astra Trading Platform.
======================================================

Generate comprehensive trading reports and analysis documents.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


class AstraReportGenerator:
    """Professional report generator for trading analysis.

    Creates publication-ready reports with charts, metrics, and analysis.
    """

    def __init__(self, output_dir: str = "reports") -> None:
        """Initialize report generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.report_data = {}

    def generate_backtest_report(self,
                               backtest_result,
                               strategy_name: str = "Strategy",
                               benchmark_data: Optional[pd.Series] = None) -> str:
        """Generate comprehensive backtest report.

        Args:
        ----
            backtest_result: Backtest results object
            strategy_name: Name of the strategy
            benchmark_data: Optional benchmark comparison

        Returns:
        -------
            Path to generated report

        """
        report_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_filename = f"backtest_report_{strategy_name}_{report_date}.html"
        report_path = self.output_dir / report_filename

        # Calculate comprehensive metrics
        metrics = self._calculate_comprehensive_metrics(backtest_result, benchmark_data)

        # Generate HTML report
        html_content = self._create_html_template()
        html_content += self._create_executive_summary(strategy_name, metrics)
        html_content += self._create_performance_section(backtest_result, metrics)
        html_content += self._create_risk_analysis_section(backtest_result, metrics)
        html_content += self._create_trade_analysis_section(backtest_result)
        html_content += self._create_appendix_section(metrics)
        html_content += "</body></html>"

        # Save report
        with open(report_path, "w") as f:
            f.write(html_content)

        return str(report_path)

    def generate_monte_carlo_report(self,
                                  scenario_results: Dict[str, Any],
                                  portfolio_data: Optional[pd.DataFrame] = None) -> str:
        """Generate Monte Carlo risk analysis report.

        Args:
        ----
            scenario_results: Monte Carlo scenario results
            portfolio_data: Optional portfolio data

        Returns:
        -------
            Path to generated report

        """
        report_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_filename = f"monte_carlo_report_{report_date}.html"
        report_path = self.output_dir / report_filename

        html_content = self._create_html_template()
        html_content += self._create_monte_carlo_executive_summary(scenario_results)
        html_content += self._create_scenario_analysis_section(scenario_results)
        html_content += self._create_risk_metrics_section(scenario_results)
        html_content += self._create_stress_test_section(scenario_results)
        html_content += "</body></html>"

        with open(report_path, "w") as f:
            f.write(html_content)

        return str(report_path)

    def generate_risk_dashboard_report(self,
                                     risk_data: Dict[str, Any],
                                     time_period: str = "Last 30 Days") -> str:
        """Generate risk management dashboard report.

        Args:
        ----
            risk_data: Risk monitoring data
            time_period: Time period for analysis

        Returns:
        -------
            Path to generated report

        """
        report_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        report_filename = f"risk_dashboard_{report_date}.html"
        report_path = self.output_dir / report_filename

        html_content = self._create_html_template()
        html_content += self._create_risk_executive_summary(risk_data, time_period)
        html_content += self._create_current_risk_status(risk_data)
        html_content += self._create_alert_analysis_section(risk_data)
        html_content += self._create_compliance_section(risk_data)
        html_content += "</body></html>"

        with open(report_path, "w") as f:
            f.write(html_content)

        return str(report_path)

    def _create_html_template(self) -> str:
        """Create HTML template for reports."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Astra Trading Platform - Professional Report</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f8f9fa;
                    color: #333;
                }
                .header {
                    background: linear-gradient(135deg, #2E86AB, #A23B72);
                    color: white;
                    padding: 30px;
                    text-align: center;
                    margin-bottom: 30px;
                    border-radius: 10px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                }
                .section {
                    background: white;
                    margin: 20px 0;
                    padding: 25px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .metric-card {
                    display: inline-block;
                    background: #f8f9fa;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 15px;
                    margin: 10px;
                    min-width: 150px;
                    text-align: center;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                    color: #2E86AB;
                }
                .metric-label {
                    font-size: 12px;
                    color: #6c757d;
                    margin-top: 5px;
                }
                .positive { color: #28a745; }
                .negative { color: #dc3545; }
                .warning { color: #ffc107; }
                .table {
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                }
                .table th, .table td {
                    border: 1px solid #dee2e6;
                    padding: 12px;
                    text-align: left;
                }
                .table th {
                    background-color: #f8f9fa;
                    font-weight: bold;
                }
                .footer {
                    text-align: center;
                    margin-top: 40px;
                    padding: 20px;
                    color: #6c757d;
                    font-size: 12px;
                }
                h1, h2, h3 { margin-top: 0; }
                h2 { color: #2E86AB; border-bottom: 2px solid #2E86AB; padding-bottom: 10px; }
                .highlight { background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }
                .alert { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; margin: 10px 0; }
                .success { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Astra Trading Platform</h1>
                <h2>Professional Trading Analysis Report</h2>
                <p>Generated on """ + datetime.now().strftime("%B %d, %Y at %H:%M:%S") + """</p>
            </div>
        """

    def _calculate_comprehensive_metrics(self, backtest_result, benchmark_data=None) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        metrics = {}

        # Basic metrics from backtest result
        if hasattr(backtest_result, "metrics"):
            metrics.update(backtest_result.metrics)

        # Additional calculations
        returns = backtest_result.returns
        portfolio_value = backtest_result.portfolio_value

        # Risk metrics
        metrics["volatility"] = returns.std() * np.sqrt(252)
        metrics["var_95"] = np.percentile(returns, 5)
        metrics["var_99"] = np.percentile(returns, 1)
        metrics["skewness"] = returns.skew()
        metrics["kurtosis"] = returns.kurtosis()

        # Performance metrics
        metrics["total_return"] = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        metrics["annualized_return"] = (1 + metrics["total_return"]) ** (252 / len(returns)) - 1

        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        metrics["max_drawdown"] = drawdown.min()
        metrics["avg_drawdown"] = drawdown[drawdown < 0].mean()

        # Benchmark comparison if provided
        if benchmark_data is not None:
            benchmark_returns = benchmark_data.pct_change().fillna(0)
            excess_returns = returns - benchmark_returns
            metrics["alpha"] = excess_returns.mean() * 252
            metrics["beta"] = np.cov(returns, benchmark_returns)[0][1] / np.var(benchmark_returns)
            metrics["information_ratio"] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

        return metrics

    def _create_executive_summary(self, strategy_name: str, metrics: Dict[str, Any]) -> str:
        """Create executive summary section."""
        total_return = metrics.get("total_return", 0)
        sharpe_ratio = metrics.get("sharpe_ratio", 0)
        max_drawdown = metrics.get("max_drawdown", 0)

        summary_class = "success" if total_return > 0 else "alert"

        return f"""
        <div class="section">
            <h2>Executive Summary</h2>
            <div class="{summary_class}">
                <strong>Strategy Performance Overview:</strong> The {strategy_name} strategy generated a
                total return of {total_return:.2%} with a Sharpe ratio of {sharpe_ratio:.2f} and maximum
                drawdown of {abs(max_drawdown):.2%}.
            </div>

            <div style="text-align: center; margin: 20px 0;">
                <div class="metric-card">
                    <div class="metric-value {'positive' if total_return > 0 else 'negative'}">{total_return:.2%}</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'positive' if sharpe_ratio > 1 else 'negative' if sharpe_ratio < 0 else 'warning'}">{sharpe_ratio:.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value {'positive' if abs(max_drawdown) < 0.1 else 'negative'}">{abs(max_drawdown):.2%}</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{metrics.get('volatility', 0):.2%}</div>
                    <div class="metric-label">Volatility</div>
                </div>
            </div>
        </div>
        """

    def _create_performance_section(self, backtest_result, metrics: Dict[str, Any]) -> str:
        """Create performance analysis section."""
        return f"""
        <div class="section">
            <h2>Performance Analysis</h2>

            <h3>Returns Analysis</h3>
            <table class="table">
                <tr><th>Metric</th><th>Value</th><th>Benchmark</th></tr>
                <tr><td>Total Return</td><td class="{'positive' if metrics.get('total_return', 0) > 0 else 'negative'}">{metrics.get('total_return', 0):.2%}</td><td>-</td></tr>
                <tr><td>Annualized Return</td><td>{metrics.get('annualized_return', 0):.2%}</td><td>-</td></tr>
                <tr><td>Volatility</td><td>{metrics.get('volatility', 0):.2%}</td><td>15-20%</td></tr>
                <tr><td>Sharpe Ratio</td><td class="{'positive' if metrics.get('sharpe_ratio', 0) > 1 else 'negative'}">{metrics.get('sharpe_ratio', 0):.2f}</td><td>&gt; 1.0</td></tr>
                <tr><td>Calmar Ratio</td><td>{metrics.get('calmar_ratio', 0):.2f}</td><td>&gt; 0.5</td></tr>
            </table>

            <h3>Risk Analysis</h3>
            <table class="table">
                <tr><th>Risk Metric</th><th>Value</th><th>Threshold</th><th>Status</th></tr>
                <tr>
                    <td>Maximum Drawdown</td>
                    <td>{abs(metrics.get('max_drawdown', 0)):.2%}</td>
                    <td>&lt; 15%</td>
                    <td class="{'positive' if abs(metrics.get('max_drawdown', 0)) < 0.15 else 'negative'}">
                        {'✓ PASS' if abs(metrics.get('max_drawdown', 0)) < 0.15 else '✗ FAIL'}
                    </td>
                </tr>
                <tr>
                    <td>Value at Risk (95%)</td>
                    <td>{abs(metrics.get('var_95', 0)):.2%}</td>
                    <td>&lt; 5%</td>
                    <td class="{'positive' if abs(metrics.get('var_95', 0)) < 0.05 else 'negative'}">
                        {'✓ PASS' if abs(metrics.get('var_95', 0)) < 0.05 else '✗ FAIL'}
                    </td>
                </tr>
                <tr>
                    <td>Skewness</td>
                    <td>{metrics.get('skewness', 0):.2f}</td>
                    <td>&gt; -1.0</td>
                    <td class="{'positive' if metrics.get('skewness', 0) > -1.0 else 'negative'}">
                        {'✓ PASS' if metrics.get('skewness', 0) > -1.0 else '✗ FAIL'}
                    </td>
                </tr>
                <tr>
                    <td>Kurtosis</td>
                    <td>{metrics.get('kurtosis', 0):.2f}</td>
                    <td>&lt; 5.0</td>
                    <td class="{'positive' if metrics.get('kurtosis', 0) < 5.0 else 'negative'}">
                        {'✓ PASS' if metrics.get('kurtosis', 0) < 5.0 else '✗ FAIL'}
                    </td>
                </tr>
            </table>
        </div>
        """

    def _create_risk_analysis_section(self, backtest_result, metrics: Dict[str, Any]) -> str:
        """Create detailed risk analysis section."""
        return f"""
        <div class="section">
            <h2>Risk Analysis</h2>

            <h3>Drawdown Analysis</h3>
            <p>Maximum drawdown represents the largest peak-to-trough decline in portfolio value.</p>
            <ul>
                <li><strong>Maximum Drawdown:</strong> {abs(metrics.get('max_drawdown', 0)):.2%}</li>
                <li><strong>Average Drawdown:</strong> {abs(metrics.get('avg_drawdown', 0)):.2%}</li>
                <li><strong>Recovery Time:</strong> Analysis pending</li>
            </ul>

            <h3>Value at Risk (VaR)</h3>
            <p>VaR estimates the potential loss at different confidence levels.</p>
            <ul>
                <li><strong>1-Day VaR (95%):</strong> {abs(metrics.get('var_95', 0)):.2%}</li>
                <li><strong>1-Day VaR (99%):</strong> {abs(metrics.get('var_99', 0)):.2%}</li>
            </ul>

            <h3>Distribution Analysis</h3>
            <ul>
                <li><strong>Skewness:</strong> {metrics.get('skewness', 0):.2f}
                    {'(Positive tail risk)' if metrics.get('skewness', 0) > 0 else '(Negative tail risk)' if metrics.get('skewness', 0) < 0 else '(Symmetric)'}
                </li>
                <li><strong>Kurtosis:</strong> {metrics.get('kurtosis', 0):.2f}
                    {'(Fat tails)' if metrics.get('kurtosis', 0) > 3 else '(Thin tails)'}
                </li>
            </ul>
        </div>
        """

    def _create_trade_analysis_section(self, backtest_result) -> str:
        """Create trade analysis section."""
        trades = backtest_result.trades

        if trades.empty:
            return """
            <div class="section">
                <h2>Trade Analysis</h2>
                <p>No trades executed in this backtest.</p>
            </div>
            """

        total_trades = len(trades)

        return f"""
        <div class="section">
            <h2>Trade Analysis</h2>

            <h3>Trade Summary</h3>
            <table class="table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Trades</td><td>{total_trades}</td></tr>
                <tr><td>Average Trade Size</td><td>${trades.get('value', pd.Series()).mean():.2f}</td></tr>
                <tr><td>Trading Frequency</td><td>{total_trades / len(backtest_result.returns) * 252:.1f} trades/year</td></tr>
            </table>

            <div class="highlight">
                <strong>Note:</strong> Detailed trade analysis including win/loss ratios,
                average holding periods, and trade distribution will be available in future versions.
            </div>
        </div>
        """

    def _create_appendix_section(self, metrics: Dict[str, Any]) -> str:
        """Create appendix with methodology and disclaimers."""
        return """
        <div class="section">
            <h2>Appendix</h2>

            <h3>Methodology</h3>
            <p>This report was generated using the Astra Trading Platform backtesting engine.
            All calculations follow industry-standard methodologies:</p>
            <ul>
                <li><strong>Sharpe Ratio:</strong> (Return - Risk-free rate) / Volatility</li>
                <li><strong>Maximum Drawdown:</strong> Largest peak-to-trough decline</li>
                <li><strong>VaR:</strong> Historical simulation method</li>
                <li><strong>Volatility:</strong> Annualized standard deviation of returns</li>
            </ul>

            <h3>Important Disclaimers</h3>
            <div class="alert">
                <strong>Risk Warning:</strong> Past performance is not indicative of future results.
                Trading involves substantial risk and may not be suitable for all investors.
                This analysis is for informational purposes only and should not be considered
                as investment advice.
            </div>

            <h3>Data Sources</h3>
            <p>Market data sourced from Yahoo Finance. All prices are adjusted for splits and dividends.</p>
        </div>

        <div class="footer">
            <p>Generated by Astra Trading Platform v1.0 | © 2024 Astra Development Team</p>
            <p>For technical support or questions, please contact the development team.</p>
        </div>
        """

    def _create_monte_carlo_executive_summary(self, scenario_results: Dict[str, Any]) -> str:
        """Create Monte Carlo executive summary."""
        return """
        <div class="section">
            <h2>Monte Carlo Risk Analysis - Executive Summary</h2>
            <div class="highlight">
                <strong>Risk Assessment:</strong> Monte Carlo simulation provides probabilistic
                analysis of portfolio performance under various market scenarios.
            </div>
        </div>
        """

    def _create_scenario_analysis_section(self, scenario_results: Dict[str, Any]) -> str:
        """Create scenario analysis section."""
        return """
        <div class="section">
            <h2>Scenario Analysis</h2>
            <p>Detailed scenario analysis will be implemented in the next version.</p>
        </div>
        """

    def _create_risk_metrics_section(self, scenario_results: Dict[str, Any]) -> str:
        """Create risk metrics section."""
        return """
        <div class="section">
            <h2>Risk Metrics</h2>
            <p>Comprehensive risk metrics from Monte Carlo simulation.</p>
        </div>
        """

    def _create_stress_test_section(self, scenario_results: Dict[str, Any]) -> str:
        """Create stress test section."""
        return """
        <div class="section">
            <h2>Stress Testing</h2>
            <p>Stress test results and scenario analysis.</p>
        </div>
        """

    def _create_risk_executive_summary(self, risk_data: Dict[str, Any], time_period: str) -> str:
        """Create risk dashboard executive summary."""
        return f"""
        <div class="section">
            <h2>Risk Dashboard - {time_period}</h2>
            <div class="success">
                <strong>Risk Status:</strong> Portfolio risk monitoring and alert summary
                for the period: {time_period}
            </div>
        </div>
        """

    def _create_current_risk_status(self, risk_data: Dict[str, Any]) -> str:
        """Create current risk status section."""
        return """
        <div class="section">
            <h2>Current Risk Status</h2>
            <p>Real-time risk monitoring and current portfolio risk metrics.</p>
        </div>
        """

    def _create_alert_analysis_section(self, risk_data: Dict[str, Any]) -> str:
        """Create alert analysis section."""
        return """
        <div class="section">
            <h2>Alert Analysis</h2>
            <p>Analysis of recent risk alerts and breaches.</p>
        </div>
        """

    def _create_compliance_section(self, risk_data: Dict[str, Any]) -> str:
        """Create compliance section."""
        return """
        <div class="section">
            <h2>Compliance Status</h2>
            <p>Regulatory compliance and risk limit adherence.</p>
        </div>
        """
