"""Chart Builder - Astra Trading Platform.
======================================

Professional chart generation for trading analysis.
"""

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


class ChartBuilder:
    """Professional chart builder for trading platform.

    Creates publication-quality charts and visualizations.
    """

    def __init__(self, style: str = "professional") -> None:
        """Initialize chart builder."""
        self.style = style
        self._setup_style()

    def _setup_style(self) -> None:
        """Setup chart styling."""
        if self.style == "professional":
            plt.style.use("seaborn-v0_8-whitegrid")
            sns.set_palette("husl")

        # Color scheme
        self.colors = {
            "primary": "#2E86AB",
            "secondary": "#A23B72",
            "success": "#00A86B",
            "warning": "#FFB347",
            "danger": "#FF6B6B",
            "neutral": "#6C757D",
        }

    def plot_backtest_results(self,
                             backtest_result,
                             save_path: Optional[str] = None) -> plt.Figure:
        """Plot comprehensive backtest results.

        Args:
        ----
            backtest_result: BacktestResult object
            save_path: Path to save chart

        Returns:
        -------
            Matplotlib figure

        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Astra Trading Platform - Backtest Results", fontsize=16, fontweight="bold")

        # Portfolio value over time
        ax1 = axes[0, 0]
        backtest_result.portfolio_value.plot(ax=ax1, color=self.colors["primary"], linewidth=2)
        ax1.set_title("Portfolio Value", fontweight="bold")
        ax1.set_ylabel("Value ($)")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

        # Returns distribution
        ax2 = axes[0, 1]
        backtest_result.returns.hist(bins=50, ax=ax2, color=self.colors["secondary"], alpha=0.7)
        ax2.axvline(backtest_result.returns.mean(), color=self.colors["danger"],
                   linestyle="--", label=f"Mean: {backtest_result.returns.mean():.3f}")
        ax2.set_title("Returns Distribution", fontweight="bold")
        ax2.set_xlabel("Daily Returns")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Drawdown
        ax3 = axes[1, 0]
        cumulative = (1 + backtest_result.returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak

        drawdown.plot(ax=ax3, color=self.colors["danger"], linewidth=2)
        ax3.fill_between(drawdown.index, drawdown.values, alpha=0.3, color=self.colors["danger"])
        ax3.set_title("Drawdown", fontweight="bold")
        ax3.set_ylabel("Drawdown")
        ax3.grid(True, alpha=0.3)
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.1%}"))

        # Performance metrics
        ax4 = axes[1, 1]
        metrics = backtest_result.metrics
        metric_names = ["Total Return", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio"]
        metric_values = [
            metrics["total_return"],
            metrics["sharpe_ratio"],
            metrics["max_drawdown"],
            metrics["calmar_ratio"],
        ]

        bars = ax4.barh(metric_names, metric_values, color=[
            self.colors["success"] if v > 0 else self.colors["danger"]
            for v in metric_values
        ])
        ax4.set_title("Performance Metrics", fontweight="bold")
        ax4.grid(True, alpha=0.3, axis="x")

        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            width = bar.get_width()
            ax4.text(width + (0.01 if width >= 0 else -0.01), bar.get_y() + bar.get_height()/2,
                    f"{value:.3f}", ha="left" if width >= 0 else "right", va="center")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_monte_carlo_scenarios(self,
                                  scenario_results: Dict[str, Any],
                                  save_path: Optional[str] = None) -> plt.Figure:
        """Plot Monte Carlo scenario results.

        Args:
        ----
            scenario_results: Dictionary of scenario results
            save_path: Path to save chart

        Returns:
        -------
            Matplotlib figure

        """
        n_scenarios = len(scenario_results)
        fig, axes = plt.subplots(2, n_scenarios, figsize=(5*n_scenarios, 10))

        if n_scenarios == 1:
            axes = axes.reshape(2, 1)

        fig.suptitle("Monte Carlo Risk Scenarios", fontsize=16, fontweight="bold")

        colors = [self.colors["primary"], self.colors["secondary"], self.colors["success"]]

        for i, (scenario_name, result) in enumerate(scenario_results.items()):
            color = colors[i % len(colors)]

            # Plot sample paths
            ax_top = axes[0, i]
            paths = result.paths

            # Plot first 50 paths
            for j in range(min(50, len(paths))):
                ax_top.plot(paths[j], alpha=0.3, color=color, linewidth=0.5)

            ax_top.set_title(f'{scenario_name.replace("_", " ").title()}', fontweight="bold")
            ax_top.set_ylabel("Price")
            ax_top.grid(True, alpha=0.3)

            # Plot final price distribution
            ax_bottom = axes[1, i]
            if len(paths.shape) == 2:
                final_prices = paths[:, -1]
            else:
                final_prices = [path[-1] for path in paths]

            ax_bottom.hist(final_prices, bins=50, color=color, alpha=0.7, density=True)
            ax_bottom.axvline(np.mean(final_prices), color=self.colors["danger"],
                            linestyle="--", label=f"Mean: ${np.mean(final_prices):.2f}")
            ax_bottom.set_title("Final Price Distribution", fontweight="bold")
            ax_bottom.set_xlabel("Final Price")
            ax_bottom.set_ylabel("Density")
            ax_bottom.legend()
            ax_bottom.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_risk_dashboard(self,
                           risk_data: Dict[str, Any],
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot risk management dashboard.

        Args:
        ----
            risk_data: Risk monitoring data
            save_path: Path to save chart

        Returns:
        -------
            Matplotlib figure

        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Risk Management Dashboard", fontsize=16, fontweight="bold")

        metrics = risk_data.get("current_metrics")
        if not metrics:
            return fig

        # Portfolio value trend
        ax1 = axes[0, 0]
        if "value_history" in risk_data:
            value_history = risk_data["value_history"]
            value_history.plot(ax=ax1, color=self.colors["primary"], linewidth=2)
        ax1.set_title("Portfolio Value", fontweight="bold")
        ax1.set_ylabel("Value ($)")
        ax1.grid(True, alpha=0.3)

        # Risk metrics gauge
        ax2 = axes[0, 1]
        risk_metrics = [
            ("Volatility", metrics.volatility, 0.25),
            ("Max Drawdown", abs(metrics.max_drawdown), 0.15),
            ("VaR 95%", abs(metrics.var_95), 0.05),
        ]

        y_pos = np.arange(len(risk_metrics))
        values = [metric[1] for metric in risk_metrics]
        thresholds = [metric[2] for metric in risk_metrics]

        ax2.barh(y_pos, values, color=[
            self.colors["danger"] if v > t else self.colors["success"]
            for v, t in zip(values, thresholds)
        ])

        # Add threshold lines
        for i, threshold in enumerate(thresholds):
            ax2.axvline(threshold, y=i-0.4, ymax=i+0.4, color="red", linestyle="--", alpha=0.7)

        ax2.set_yticks(y_pos)
        ax2.set_yticklabels([metric[0] for metric in risk_metrics])
        ax2.set_title("Risk Metrics vs Thresholds", fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="x")

        # Position concentration
        ax3 = axes[0, 2]
        if "positions" in risk_data:
            positions = risk_data["positions"]
            symbols = list(positions.keys())
            weights = [abs(w) for w in positions.values()]

            ax3.pie(weights, labels=symbols, autopct="%1.1f%%", startangle=90)
            ax3.set_title("Position Concentration", fontweight="bold")

        # Risk score over time
        ax4 = axes[1, 0]
        if "risk_score_history" in risk_data:
            score_history = risk_data["risk_score_history"]
            score_history.plot(ax=ax4, color=self.colors["warning"], linewidth=2)
            ax4.fill_between(score_history.index, score_history.values, alpha=0.3, color=self.colors["warning"])
        ax4.set_title("Risk Score Trend", fontweight="bold")
        ax4.set_ylabel("Risk Score (0-100)")
        ax4.grid(True, alpha=0.3)

        # Alert summary
        ax5 = axes[1, 1]
        alerts = risk_data.get("recent_alerts", [])
        alert_types = {}
        for alert in alerts:
            alert_types[alert.alert_type] = alert_types.get(alert.alert_type, 0) + 1

        if alert_types:
            ax5.bar(alert_types.keys(), alert_types.values(), color=self.colors["danger"])
            ax5.set_title("Recent Alerts (24h)", fontweight="bold")
            ax5.set_ylabel("Count")
            plt.setp(ax5.get_xticklabels(), rotation=45, ha="right")
        else:
            ax5.text(0.5, 0.5, "No Recent Alerts", ha="center", va="center",
                    transform=ax5.transAxes, fontsize=14)
            ax5.set_title("Recent Alerts (24h)", fontweight="bold")

        # Circuit breaker status
        ax6 = axes[1, 2]
        cb_status = risk_data.get("circuit_breaker_status", {})
        if cb_status:
            enabled_count = sum(1 for status in cb_status.values() if status["enabled"])
            breached_count = sum(1 for status in cb_status.values() if status["breach_count"] > 0)

            labels = ["Enabled", "Breached", "Inactive"]
            sizes = [enabled_count - breached_count, breached_count, len(cb_status) - enabled_count]
            colors_pie = [self.colors["success"], self.colors["danger"], self.colors["neutral"]]

            ax6.pie(sizes, labels=labels, colors=colors_pie, autopct="%1.0f", startangle=90)
            ax6.set_title("Circuit Breaker Status", fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def create_interactive_dashboard(self, data: Dict[str, Any]) -> go.Figure:
        """Create interactive Plotly dashboard.

        Args:
        ----
            data: Dashboard data

        Returns:
        -------
            Plotly figure

        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Portfolio Performance", "Risk Metrics",
                          "Position Allocation", "Monte Carlo Results"),
            specs=[[{"secondary_y": False}, {"type": "indicator"}],
                   [{"type": "pie"}, {"secondary_y": False}]],
        )

        # Portfolio performance
        if "portfolio_values" in data:
            fig.add_trace(
                go.Scatter(
                    x=data["portfolio_values"].index,
                    y=data["portfolio_values"].values,
                    mode="lines",
                    name="Portfolio Value",
                    line={"color": self.colors["primary"], "width": 2},
                ),
                row=1, col=1,
            )

        # Risk indicator
        risk_score = data.get("risk_score", 50)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=risk_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Risk Score"},
                gauge={"axis": {"range": [None, 100]},
                      "bar": {"color": "darkblue"},
                      "steps": [
                          {"range": [0, 30], "color": "lightgreen"},
                          {"range": [30, 70], "color": "yellow"},
                          {"range": [70, 100], "color": "red"}],
                      "threshold": {"line": {"color": "red", "width": 4},
                                   "thickness": 0.75, "value": 80}},
            ),
            row=1, col=2,
        )

        # Position allocation
        if "positions" in data:
            positions = data["positions"]
            fig.add_trace(
                go.Pie(
                    labels=list(positions.keys()),
                    values=[abs(v) for v in positions.values()],
                    name="Positions",
                ),
                row=2, col=1,
            )

        # Monte Carlo results
        if "monte_carlo_paths" in data:
            paths = data["monte_carlo_paths"]
            for i, path in enumerate(paths[:10]):  # Show first 10 paths
                fig.add_trace(
                    go.Scatter(
                        y=path,
                        mode="lines",
                        name=f"Path {i+1}",
                        opacity=0.3,
                        showlegend=False,
                    ),
                    row=2, col=2,
                )

        fig.update_layout(
            title_text="Astra Trading Platform - Interactive Dashboard",
            showlegend=True,
            height=800,
        )

        return fig
