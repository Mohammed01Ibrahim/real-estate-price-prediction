"""Original-dollar metrics and aggregate figures without listing identifiers."""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "real-estate-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted) -> dict[str, float]:
    return {
        "mae_usd": float(mean_absolute_error(actual, predicted)),
        "rmse_usd": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def write_figures(actual, predicted, directory: Path, importances: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    positive = (actual > 0) & (predicted > 0)
    density = ax.hexbin(actual[positive], predicted[positive], gridsize=30, mincnt=1,
                        xscale="log", yscale="log", cmap="Blues")
    limits = (min(actual[positive].min(), predicted[positive].min()),
              max(actual[positive].max(), predicted[positive].max()))
    ax.plot(limits, limits, color="#b45309", linestyle="--", label="Perfect estimate")
    ax.set(xlabel="Actual asking price (USD, log axis)", ylabel="Estimated asking price (USD, log axis)",
           title="Held-out test: price estimates")
    ax.legend()
    fig.colorbar(density, ax=ax, label="Listings per bin")
    fig.savefig(directory / "prediction_vs_actual.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    residual = actual - predicted
    axes[0].hist(residual, bins=60, color="#2563eb")
    axes[0].set(xlabel="Actual minus estimated price (USD)", ylabel="Count", title="All residuals; no trimming")
    axes[1].hexbin(predicted, residual, gridsize=30, mincnt=1, cmap="Blues", xscale="log")
    axes[1].axhline(0, color="#b45309", linestyle="--")
    axes[1].set(xlabel="Estimate (USD, log axis)", ylabel="Residual (USD)", title="Residual pattern")
    fig.savefig(directory / "residual_analysis.png", dpi=150)
    plt.close(fig)
    ordered = sorted(importances.items(), key=lambda item: item[1]["mean_mae_increase_usd"])
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    ax.barh([name for name, _ in ordered], [v["mean_mae_increase_usd"] for _, v in ordered],
            xerr=[v["std_mae_increase_usd"] for _, v in ordered], color="#2563eb")
    ax.set(xlabel="Increase in validation MAE after permutation (USD)",
           title="Raw-feature permutation importance (3 repeats)")
    fig.savefig(directory / "permutation_importance.png", dpi=150)
    plt.close(fig)
