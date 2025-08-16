"""VectorBT-Inspired Visualization - Astra Trading Platform.
========================================================

Advanced visualization components inspired by VectorBT capabilities.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

class AstraVectorBTCharts:
    """Professional visualization toolkit inspired by VectorBT.

    Provides advanced charting capabilities for trading analysis.
    """

    def __init__(self, theme: str = "astra_professional") -> None:
        """Initialize chart builder with theme."""
        self.theme = theme
        self._setup_theme()

    def _setup_theme(self) -> None:
        """Setup visualization theme."""
        self.colors = {
            "primary": "#1f77b4",
            "secondary": "#ff7f0e",
            "success": "#2ca02c",
            "danger": "#d62728",
            "warning": "#ff9500",
            "info": "#17a2b8",
            "dark": "#343a40",
            "light": "#f8f9fa",
        }

        # Plotly theme
        self.plotly_template = {
            "layout": {
                "colorway": list(self.colors.values()),
                "font": {"family": "Arial, sans-serif", "size": 12},
                "plot_bgcolor": "rgba(0,0,0,0)",
                "paper_bgcolor": "rgba(0,0,0,0)",
                "grid": {"color": "rgba(128,128,128,0.2)"},
            },
        }

    def create_performance_heatmap(self,
                                 results_matrix: pd.DataFrame,
                                 title: str = "Strategy Performance Heatmap",
                                 metric: str = "Total Return") -> go.Figure:
        """Create performance heatmap for parameter optimization.

        Args:
        ----
            results_matrix: DataFrame with performance results
            title: Chart title
            metric: Metric to display

        Returns:
        -------
            Plotly heatmap figure

        """
        fig = go.Figure(data=go.Heatmap(
            z=results_matrix.values,
            x=results_matrix.columns,
            y=results_matrix.index,
            colorscale="RdYlGn",
            text=results_matrix.values,
            texttemplate="%{text:.2%}",
            textfont={"size": 10},
            colorbar={"title": metric},
        ))

        fig.update_layout(
            title=f"Astra Trading Platform - {title}",
            xaxis_title="Parameter 2",
            yaxis_title="Parameter 1",
            font={"family": "Arial, sans-serif", "size": 12},
        )

        return fig

    def create_interactive_backtest_chart(self,
                                        portfolio_value: pd.Series,
                                        benchmark: Optional[pd.Series] = None,
                                        trades: Optional[pd.DataFrame] = None,
                                        indicators: Optional[Dict[str, pd.Series]] = None) -> go.Figure:
        """Create comprehensive interactive backtest chart.

        Args:
        ----
            portfolio_value: Portfolio value time series
            benchmark: Optional benchmark comparison
            trades: Optional trade records
            indicators: Optional technical indicators

        Returns:
        -------
            Interactive Plotly figure

        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=("Portfolio Performance", "Drawdown", "Volume/Indicators"),
            row_heights=[0.5, 0.25, 0.25],
        )

        # Portfolio value
        fig.add_trace(
            go.Scatter(
                x=portfolio_value.index,
                y=portfolio_value.values,
                mode="lines",
                name="Portfolio",
                line={"color": self.colors["primary"], "width": 2},
            ),
            row=1, col=1,
        )

        # Benchmark if provided
        if benchmark is not None:
            fig.add_trace(
                go.Scatter(
                    x=benchmark.index,
                    y=benchmark.values,
                    mode="lines",
                    name="Benchmark",
                    line={"color": self.colors["secondary"], "width": 2, "dash": "dash"},
                ),
                row=1, col=1,
            )

        # Add trade markers if provided
        if trades is not None and not trades.empty:
            buy_trades = trades[trades["side"] == "buy"] if "side" in trades.columns else trades[trades["quantity"] > 0]
            sell_trades = trades[trades["side"] == "sell"] if "side" in trades.columns else trades[trades["quantity"] < 0]

            if not buy_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_trades.index,
                        y=[portfolio_value.loc[idx] for idx in buy_trades.index if idx in portfolio_value.index],
                        mode="markers",
                        name="Buy",
                        marker={"symbol": "triangle-up", "size": 10, "color": self.colors["success"]},
                    ),
                    row=1, col=1,
                )

            if not sell_trades.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_trades.index,
                        y=[portfolio_value.loc[idx] for idx in sell_trades.index if idx in portfolio_value.index],
                        mode="markers",
                        name="Sell",
                        marker={"symbol": "triangle-down", "size": 10, "color": self.colors["danger"]},
                    ),
                    row=1, col=1,
                )

        # Drawdown
        returns = portfolio_value.pct_change().fillna(0)
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak

        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode="lines",
                name="Drawdown",
                fill="tonexty",
                line={"color": self.colors["danger"], "width": 1},
                fillcolor="rgba(214, 39, 40, 0.3)",
            ),
            row=2, col=1,
        )

        # Indicators
        if indicators:
            for name, series in indicators.items():
                fig.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode="lines",
                        name=name,
                        line={"width": 1},
                    ),
                    row=3, col=1,
                )

        # Update layout
        fig.update_layout(
            title="Astra Trading Platform - Interactive Backtest Analysis",
            height=800,
            showlegend=True,
            legend={"x": 0, "y": 1, "bgcolor": "rgba(255,255,255,0.8)"},
            hovermode="x unified",
        )

        # Update axes
        fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1, tickformat=".1%")
        fig.update_yaxes(title_text="Indicators", row=3, col=1)
        fig.update_xaxes(title_text="Date", row=3, col=1)

        return fig

    def create_monte_carlo_visualization(self,
                                       paths: np.ndarray,
                                       percentiles: Optional[List[float]] = None,
                                       title: str = "Monte Carlo Simulation") -> go.Figure:
        """Create Monte Carlo paths visualization.

        Args:
        ----
            paths: Array of simulation paths
            percentiles: Percentiles to highlight
            title: Chart title

        Returns:
        -------
            Plotly figure

        """
        if percentiles is None:
            percentiles = [5, 25, 50, 75, 95]

        fig = go.Figure()

        # Add sample paths (limited for performance)
        sample_size = min(100, len(paths))
        sample_indices = np.random.choice(len(paths), sample_size, replace=False)

        for i in sample_indices[:20]:  # Show only 20 paths for clarity
            fig.add_trace(
                go.Scatter(
                    y=paths[i],
                    mode="lines",
                    name=f"Path {i+1}",
                    line={"color": "rgba(128,128,128,0.3)", "width": 1},
                    showlegend=False,
                    hovertemplate="Step: %{x}<br>Value: %{y:.2f}<extra></extra>",
                ),
            )

        # Add percentile bands
        percentile_values = np.percentile(paths, percentiles, axis=0)
        colors = ["rgba(255,0,0,0.8)", "rgba(255,165,0,0.6)", "rgba(0,128,0,0.8)",
                 "rgba(255,165,0,0.6)", "rgba(255,0,0,0.8)"]

        for i, (p, color) in enumerate(zip(percentiles, colors)):
            fig.add_trace(
                go.Scatter(
                    y=percentile_values[i],
                    mode="lines",
                    name=f"{p}th Percentile",
                    line={"color": color, "width": 2},
                    hovertemplate=f"{p}th Percentile<br>Step: %{{x}}<br>Value: %{{y:.2f}}<extra></extra>",
                ),
            )

        # Fill between percentiles for better visualization
        fig.add_trace(
            go.Scatter(
                y=percentile_values[0],
                mode="lines",
                line={"color": "rgba(0,0,0,0)"},
                showlegend=False,
                hoverinfo="skip",
            ),
        )

        fig.add_trace(
            go.Scatter(
                y=percentile_values[-1],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(128,128,128,0.2)",
                line={"color": "rgba(0,0,0,0)"},
                name="Confidence Band",
                hoverinfo="skip",
            ),
        )

        fig.update_layout(
            title=f"Astra Trading Platform - {title}",
            xaxis_title="Time Steps",
            yaxis_title="Price",
            height=600,
            hovermode="x unified",
        )

        return fig

    def create_risk_metrics_dashboard(self,
                                    metrics_history: pd.DataFrame,
                                    current_metrics: Dict[str, float]) -> go.Figure:
        """Create comprehensive risk metrics dashboard.

        Args:
        ----
            metrics_history: Historical risk metrics
            current_metrics: Current risk values

        Returns:
        -------
            Interactive dashboard figure

        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=("VaR Timeline", "Volatility Trend", "Current Risk Score",
                          "Correlation Matrix", "Risk Distribution", "Alert Frequency"),
            specs=[[{}, {}, {"type": "indicator"}],
                   [{"type": "heatmap"}, {}, {}]],
        )

        # VaR Timeline
        if "var_95" in metrics_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_history.index,
                    y=metrics_history["var_95"],
                    mode="lines+markers",
                    name="VaR 95%",
                    line={"color": self.colors["danger"]},
                ),
                row=1, col=1,
            )

        # Volatility Trend
        if "volatility" in metrics_history.columns:
            fig.add_trace(
                go.Scatter(
                    x=metrics_history.index,
                    y=metrics_history["volatility"],
                    mode="lines",
                    name="Volatility",
                    fill="tonexty",
                    line={"color": self.colors["warning"]},
                ),
                row=1, col=2,
            )

        # Current Risk Score
        risk_score = current_metrics.get("risk_score", 50)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                delta={"reference": 50},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": self.colors["primary"]},
                    "steps": [
                        {"range": [0, 30], "color": self.colors["success"]},
                        {"range": [30, 70], "color": self.colors["warning"]},
                        {"range": [70, 100], "color": self.colors["danger"]},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 80,
                    },
                },
                title={"text": "Risk Score"},
            ),
            row=1, col=3,
        )

        # Risk Distribution
        if "returns" in metrics_history.columns:
            fig.add_trace(
                go.Histogram(
                    x=metrics_history["returns"],
                    nbinsx=50,
                    name="Returns Distribution",
                    marker_color=self.colors["info"],
                ),
                row=2, col=2,
            )

        fig.update_layout(
            title="Astra Trading Platform - Risk Analytics Dashboard",
            height=800,
            showlegend=False,
        )

        return fig

    def create_correlation_matrix(self,
                                returns_data: pd.DataFrame,
                                method: str = "pearson") -> go.Figure:
        """Create interactive correlation matrix.

        Args:
        ----
            returns_data: Returns data for correlation calculation
            method: Correlation method

        Returns:
        -------
            Correlation heatmap figure

        """
        corr_matrix = returns_data.corr(method=method)

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        corr_matrix_masked = corr_matrix.mask(mask)

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix_masked.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale="RdBu",
            zmid=0,
            text=corr_matrix_masked.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar={"title": "Correlation"},
        ))

        fig.update_layout(
            title="Astra Trading Platform - Asset Correlation Matrix",
            xaxis_title="Assets",
            yaxis_title="Assets",
            height=600,
        )

        return fig

    def create_strategy_comparison(self,
                                 strategy_results: Dict[str, Dict[str, float]]) -> go.Figure:
        """Create strategy comparison radar chart.

        Args:
        ----
            strategy_results: Dictionary of strategy metrics

        Returns:
        -------
            Radar chart figure

        """
        metrics = ["Total Return", "Sharpe Ratio", "Max Drawdown", "Win Rate", "Profit Factor"]

        fig = go.Figure()

        for strategy_name, results in strategy_results.items():
            values = [results.get(metric, 0) for metric in metrics]
            values += [values[0]]  # Close the radar chart

            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=[*metrics, metrics[0]],
                fill="toself",
                name=strategy_name,
                line={"width": 2},
            ))

        fig.update_layout(
            polar={
                "radialaxis": {"visible": True, "range": [0, max(
                    max(results.values()) for results in strategy_results.values()
                )]},
            },
            title="Astra Trading Platform - Strategy Performance Comparison",
            showlegend=True,
        )

        return fig

    def export_dashboard(self,
                        figures: List[go.Figure],
                        filename: str = "astra_dashboard.html") -> str:
        """Export multiple figures to HTML dashboard.

        Args:
        ----
            figures: List of Plotly figures
            filename: Output filename

        Returns:
        -------
            File path of exported dashboard

        """
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Astra Trading Platform - Analytics Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .chart-container { margin: 20px 0; border: 1px solid #ddd; padding: 10px; }
                .header { text-align: center; margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Astra Trading Platform</h1>
                <h2>Professional Analytics Dashboard</h2>
            </div>
        """

        for i, fig in enumerate(figures):
            div_id = f"chart_{i}"
            html_content += f'<div class="chart-container"><div id="{div_id}"></div></div>'

        html_content += """
        <script>
        """

        for i, fig in enumerate(figures):
            div_id = f"chart_{i}"
            fig_json = fig.to_json()
            html_content += f"""
            Plotly.newPlot('{div_id}', {fig_json});
            """

        html_content += """
        </script>
        </body>
        </html>
        """

        with open(filename, "w") as f:
            f.write(html_content)

        return filename
