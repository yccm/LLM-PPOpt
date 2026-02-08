"""Training metrics collection and matplotlib-based visualization."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
import json
import time
import logging

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric measurement."""
    step: int
    value: float
    timestamp: float = field(default_factory=time.time)


class MetricsTracker:
    """
    Collects step-by-step training metrics and generates matplotlib plots.

    Designed to be lightweight and follow existing InteractionLogger patterns.
    """

    # Plot styling constants
    COLORS = {
        "loss": "#1f77b4",       # Blue
        "grpo_loss": "#d62728",  # Red
        "mean_reward": "#2ca02c", # Green
        "kl_mean": "#ff7f0e",    # Orange
    }
    DEFAULT_COLOR = "#1f77b4"
    FIGURE_SIZE = (10, 6)
    COMBINED_FIGURE_SIZE = (12, 10)
    DPI = 150

    def __init__(
        self,
        output_dir: str | Path,
        prefix: str = "training",
        plot_interval: int = 50,
        metrics_file: str = "metrics.jsonl",
    ) -> None:
        """
        Initialize metrics tracker.

        Args:
            output_dir: Directory to save plots and metrics file
            prefix: Prefix for plot filenames (e.g., "sft" or "rl")
            plot_interval: How often to save intermediate plots (0 = only at end)
            metrics_file: Name of JSONL file to persist raw metrics
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.plot_interval = plot_interval
        self.metrics_file = self.output_dir / metrics_file

        self._metrics: Dict[str, List[MetricPoint]] = {}
        self._last_plot_step = 0
        self._closed = False

        # Open metrics file for streaming writes
        self._file = open(self.metrics_file, "w", encoding="utf-8")
        logger.info("Metrics tracker initialized: %s", self.metrics_file)

    def record(self, step: int, **metrics: float) -> None:
        """
        Record one or more metrics for a given step.

        Args:
            step: Training step number
            **metrics: Metric name-value pairs (e.g., loss=0.5, reward=0.8)

        Example:
            tracker.record(step=100, loss=0.5)
            tracker.record(step=100, grpo_loss=0.3, kl_mean=0.01, mean_reward=0.7)
        """
        if self._closed:
            return

        timestamp = time.time()
        record = {"step": step, "timestamp": timestamp}

        for name, value in metrics.items():
            if name not in self._metrics:
                self._metrics[name] = []
            self._metrics[name].append(MetricPoint(step=step, value=value, timestamp=timestamp))
            record[name] = value

        # Stream to file
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        # Check if we should save intermediate plots
        if self.plot_interval > 0:
            self.maybe_plot(step)

    def maybe_plot(self, step: int) -> None:
        """Save plots if at interval boundary."""
        if step > 0 and step % self.plot_interval == 0 and step > self._last_plot_step:
            self.save_plots(suffix=f"_step{step}")
            self._last_plot_step = step

    def save_plots(self, suffix: str = "") -> Dict[str, str]:
        """
        Generate and save all metric plots.

        Args:
            suffix: Optional suffix for filenames (e.g., "_final")

        Returns:
            Dict mapping metric names to saved file paths
        """
        saved_paths = {}

        if not self._metrics:
            return saved_paths

        # Individual metric plots
        for metric_name, points in self._metrics.items():
            if not points:
                continue
            filename = f"{self.prefix}_{metric_name}{suffix}.png"
            output_path = self.output_dir / filename
            self._plot_single_metric(metric_name, points, output_path)
            saved_paths[metric_name] = str(output_path)

        # Combined plot for RL metrics (if we have the expected metrics)
        rl_metrics = {"grpo_loss", "mean_reward", "kl_mean"}
        if rl_metrics.issubset(self._metrics.keys()):
            combined_path = self.output_dir / f"{self.prefix}_metrics_combined{suffix}.png"
            self._plot_combined_rl_metrics(combined_path)
            saved_paths["combined"] = str(combined_path)

        return saved_paths

    def _plot_single_metric(
        self,
        metric_name: str,
        points: List[MetricPoint],
        output_path: Path,
    ) -> None:
        """Generate and save a single metric plot."""
        steps = [p.step for p in points]
        values = [p.value for p in points]

        fig, ax = plt.subplots(figsize=self.FIGURE_SIZE)
        color = self.COLORS.get(metric_name, self.DEFAULT_COLOR)

        # Plot with markers at reasonable intervals
        markevery = max(1, len(steps) // 20)
        ax.plot(steps, values, color=color, linewidth=1.5, marker='o', markersize=3, markevery=markevery)

        # Formatting
        title_name = metric_name.replace("_", " ").title()
        ax.set_title(f"{self.prefix.upper()} Training: {title_name}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Training Step", fontsize=12)
        ax.set_ylabel(title_name, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)

        plt.tight_layout()
        fig.savefig(output_path, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
        logger.debug("Saved plot: %s", output_path)

    def _plot_combined_rl_metrics(self, output_path: Path) -> None:
        """Generate combined RL metrics plot with 3 subplots."""
        fig, axes = plt.subplots(3, 1, figsize=self.COMBINED_FIGURE_SIZE, sharex=True)

        metric_configs = [
            ("grpo_loss", "GRPO Loss", self.COLORS["grpo_loss"]),
            ("mean_reward", "Mean Reward", self.COLORS["mean_reward"]),
            ("kl_mean", "KL Divergence", self.COLORS["kl_mean"]),
        ]

        for ax, (metric_name, ylabel, color) in zip(axes, metric_configs):
            points = self._metrics.get(metric_name, [])
            if points:
                steps = [p.step for p in points]
                values = [p.value for p in points]
                ax.plot(steps, values, color=color, linewidth=1.5)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(left=0)

        axes[-1].set_xlabel("Training Step", fontsize=12)
        fig.suptitle("RL Training Metrics", fontsize=14, fontweight='bold')

        plt.tight_layout()
        fig.savefig(output_path, dpi=self.DPI, bbox_inches='tight')
        plt.close(fig)
        logger.debug("Saved combined plot: %s", output_path)

    def save_metrics_file(self) -> str:
        """Flush and return path to metrics file."""
        self._file.flush()
        return str(self.metrics_file)

    def close(self) -> None:
        """Finalize: save final plots and close file."""
        if self._closed:
            return

        # Save final plots (without step suffix)
        self.save_plots(suffix="")

        # Close metrics file
        self._file.close()
        self._closed = True
        logger.info("Metrics tracker closed. Final plots and data saved to: %s", self.output_dir)


def create_sft_tracker(output_dir: str | Path, plot_interval: int = 50) -> MetricsTracker:
    """Factory function for SFT training metrics tracker."""
    return MetricsTracker(
        output_dir=output_dir,
        prefix="sft",
        plot_interval=plot_interval,
        metrics_file="sft_metrics.jsonl",
    )


def create_rl_tracker(output_dir: str | Path, plot_interval: int = 50) -> MetricsTracker:
    """Factory function for RL training metrics tracker."""
    return MetricsTracker(
        output_dir=output_dir,
        prefix="rl",
        plot_interval=plot_interval,
        metrics_file="rl_metrics.jsonl",
    )
