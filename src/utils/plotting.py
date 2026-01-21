"""
Plotting utilities for reliability estimation experiments.

Contains functions for creating estimation error plots with confidence intervals.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_bootstrap_cis(results_df, n_bootstrap=1000):
    """
    Compute 95% bootstrap CI for estimation errors by method and budget.

    Args:
        results_df: DataFrame with columns: method, budget, estimation_error
        n_bootstrap: Number of bootstrap samples (default: 1000)

    Returns:
        DataFrame with columns: method, budget, mean, ci_lower, ci_upper, ci_half_width
    """
    ci_data = []

    for method in results_df["method"].unique():
        for budget in results_df["budget"].unique():
            subset = results_df[
                (results_df["method"] == method) &
                (results_df["budget"] == budget)
            ]

            if len(subset) > 0:
                errors = subset["estimation_error"].values
                mean_error = errors.mean()

                if len(errors) > 1:
                    bootstrap_means = []
                    for _ in range(n_bootstrap):
                        bootstrap_sample = np.random.choice(
                            errors, size=len(errors), replace=True
                        )
                        bootstrap_means.append(bootstrap_sample.mean())

                    ci_lower = np.percentile(bootstrap_means, 2.5)
                    ci_upper = np.percentile(bootstrap_means, 97.5)
                    ci_half_width = (ci_upper - ci_lower) / 2
                else:
                    ci_lower = mean_error
                    ci_upper = mean_error
                    ci_half_width = 0

                ci_data.append({
                    "method": method,
                    "budget": budget,
                    "mean": mean_error,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "ci_half_width": ci_half_width
                })

    return pd.DataFrame(ci_data)


def create_estimation_error_plot(avg_results, title, ylabel, filename, legend_text=""):
    """
    Create a single estimation error plot with confidence intervals.

    Args:
        avg_results: DataFrame from compute_bootstrap_cis with method, budget, mean, ci_half_width
        title: Plot title
        ylabel: Y-axis label
        filename: Full path to save the plot
        legend_text: Additional text to append to title (e.g., metric values)

    Returns:
        Path to saved file
    """
    plt.figure(figsize=(10, 6))

    for method in avg_results["method"].unique():
        method_data = avg_results[avg_results["method"] == method]
        plt.errorbar(
            method_data["budget"],
            method_data["mean"],
            yerr=method_data["ci_half_width"],
            marker='o',
            linewidth=2.5,
            capsize=5,
            capthick=2,
            label=method,
            alpha=0.8
        )

    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Human Annotation Budget", fontsize=12)
    plt.title(f"{title}{legend_text}", fontsize=11)
    plt.legend(title="Method", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    return filename


def plot_metric_results(results, results_by_axis, metadata_all, metadata_by_axis,
                        dataset, plots_dir, comparison_mode,
                        metric_name="ICC", hm_key="true_hm_icc", im_key="im_icc"):
    """
    Generate plots and summary tables for a reliability metric.

    Creates three sets of plots:
    1. Averaged across both axis and model (1 plot)
    2. Averaged across axis only (k plots, where k = number of models)
    3. Averaged across model only (m plots, where m = number of evaluation axes)

    Args:
        results: DataFrame with estimation errors (model, budget, method, estimation_error)
        results_by_axis: Dict mapping axis -> DataFrame with estimation errors
        metadata_all: Dict mapping model -> metadata dict (with hm_key, im_key values)
        metadata_by_axis: Dict mapping axis -> model -> metadata dict
        dataset: Dataset name for plot titles
        plots_dir: Directory to save plots
        comparison_mode: "pairwise" or "aggregate" for legend text
        metric_name: Name of the metric (e.g., "ICC" or "Alpha")
        hm_key: Key for human-model metric in metadata
        im_key: Key for inter-model metric in metadata
    """
    if results is None or len(results) == 0:
        print(f"\nNo {metric_name} results to plot.")
        return

    metric_lower = metric_name.lower()
    mode_label = 'Pairwise' if comparison_mode == 'pairwise' else 'Aggregate'

    # =====================================
    # SET 1: Averaged across axis AND model
    # =====================================
    avg_results = compute_bootstrap_cis(results)

    legend_text = ""
    if metadata_all is not None:
        hm_vals = [v[hm_key] for v in metadata_all.values() if np.isfinite(v[hm_key])]
        im_vals = [v[im_key] for v in metadata_all.values() if np.isfinite(v[im_key])]
        if hm_vals and im_vals:
            avg_true_hm = np.mean(hm_vals)
            avg_im = np.mean(im_vals)
            legend_text = f"\nAxis: All | Avg HM-{metric_name}: {avg_true_hm:.3f} | Avg {mode_label} IM-{metric_name}: {avg_im:.3f}"

    filename = create_estimation_error_plot(
        avg_results,
        title=f"{dataset} {metric_name} Estimation: Random vs Variance-Matched",
        ylabel=f"Absolute {metric_name} Error (avg across models & axes)",
        filename=os.path.join(plots_dir, f"{dataset}_{metric_lower}_estimation_comparison_avg_all.jpg"),
        legend_text=f"{legend_text}\n(Averaged over all models & axes with 95% CI)"
    )
    print(f"\n[{metric_name} SET 1: Avg across axis AND model]")
    print(f"  Saved: {filename}")

    # =====================================
    # SET 2: Averaged across axis only (per-model plots)
    # =====================================
    if results_by_axis is not None and len(results_by_axis) > 0:
        print(f"\n[{metric_name} SET 2: Avg across axis only - {len(results['model'].unique())} model plots]")

        for model in results["model"].unique():
            model_data_across_axes = []
            for axis, axis_results in results_by_axis.items():
                if axis_results is not None and len(axis_results) > 0:
                    model_axis_data = axis_results[axis_results["model"] == model]
                    if len(model_axis_data) > 0:
                        model_data_across_axes.append(model_axis_data)

            if len(model_data_across_axes) == 0:
                continue

            model_results = pd.concat(model_data_across_axes, ignore_index=True)
            avg_model_results = compute_bootstrap_cis(model_results)

            legend_text = ""
            if metadata_by_axis is not None:
                model_hm_vals = [
                    metadata_by_axis[ax][model][hm_key]
                    for ax in metadata_by_axis
                    if model in metadata_by_axis[ax] and np.isfinite(metadata_by_axis[ax][model][hm_key])
                ]
                model_im_vals = [
                    metadata_by_axis[ax][model][im_key]
                    for ax in metadata_by_axis
                    if model in metadata_by_axis[ax] and np.isfinite(metadata_by_axis[ax][model][im_key])
                ]
                if model_hm_vals and model_im_vals:
                    avg_hm = np.mean(model_hm_vals)
                    avg_im = np.mean(model_im_vals)
                    legend_text = f"\nAxis: All | Avg HM-{metric_name}: {avg_hm:.3f} | Avg {mode_label} IM-{metric_name}: {avg_im:.3f}"

            safe_model_name = model.replace("/", "-").replace("\\", "-")
            filename = create_estimation_error_plot(
                avg_model_results,
                title=f"{dataset} {metric_name} Estimation: {model}",
                ylabel=f"Absolute {metric_name} Error (avg across axes)",
                filename=os.path.join(plots_dir, f"{dataset}_{metric_lower}_estimation_by_model_{safe_model_name}.jpg"),
                legend_text=f"{legend_text}\n(Averaged across axes with 95% CI)"
            )
            print(f"  Saved: {filename}")

    # =====================================
    # SET 3: Averaged across model only (per-axis plots)
    # =====================================
    if results_by_axis is not None and len(results_by_axis) > 0:
        print(f"\n[{metric_name} SET 3: Avg across model only - {len(results_by_axis)} axis plots]")

        for axis, axis_results in results_by_axis.items():
            if axis_results is None or len(axis_results) == 0:
                continue

            avg_axis_results = compute_bootstrap_cis(axis_results)

            legend_text = ""
            if metadata_by_axis is not None and axis in metadata_by_axis:
                axis_hm_vals = [
                    metadata_by_axis[axis][model][hm_key]
                    for model in metadata_by_axis[axis]
                    if np.isfinite(metadata_by_axis[axis][model][hm_key])
                ]
                axis_im_vals = [
                    metadata_by_axis[axis][model][im_key]
                    for model in metadata_by_axis[axis]
                    if np.isfinite(metadata_by_axis[axis][model][im_key])
                ]
                if axis_hm_vals and axis_im_vals:
                    avg_hm = np.mean(axis_hm_vals)
                    avg_im = np.mean(axis_im_vals)
                    legend_text = f"\nAxis: {axis} | Avg HM-{metric_name}: {avg_hm:.3f} | Avg {mode_label} IM-{metric_name}: {avg_im:.3f}"

            safe_axis_name = axis.replace("/", "-").replace("\\", "-").replace(" ", "_")
            filename = create_estimation_error_plot(
                avg_axis_results,
                title=f"{dataset} {metric_name} Estimation",
                ylabel=f"Absolute {metric_name} Error (avg across models)",
                filename=os.path.join(plots_dir, f"{dataset}_{metric_lower}_estimation_by_axis_{safe_axis_name}.jpg"),
                legend_text=f"{legend_text}\n(Averaged across models with 95% CI)"
            )
            print(f"  Saved: {filename}")

    # =====================================
    # Summary table
    # =====================================
    print("\n" + "=" * 50)
    print(f"Average {metric_name} Estimation Error by Method and Budget (Mean with 95% CI)")
    print("(Averaged across all models and axes)")
    print("=" * 50)

    for method in avg_results["method"].unique():
        method_data = avg_results[avg_results["method"] == method].copy()
        method_data["formatted"] = method_data.apply(
            lambda row: f"{row['mean']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]",
            axis=1
        )
        print(f"\n{method}:")
        for _, row in method_data.iterrows():
            print(f"  Budget {int(row['budget']):2d}: {row['formatted']}")


def plot_all_results(icc_results, alpha_results, icc_results_by_axis, alpha_results_by_axis,
                     reliability_metadata_all, reliability_metadata_by_axis,
                     dataset, plots_dir, comparison_mode):
    """
    Generate plots for both ICC and Krippendorff's Alpha estimation errors.

    Args:
        icc_results: DataFrame with ICC estimation errors
        alpha_results: DataFrame with Alpha estimation errors
        icc_results_by_axis: Dict mapping axis -> ICC results DataFrame
        alpha_results_by_axis: Dict mapping axis -> Alpha results DataFrame
        reliability_metadata_all: Dict mapping model -> aggregated metadata
        reliability_metadata_by_axis: Dict mapping axis -> model -> metadata
        dataset: Dataset name
        plots_dir: Directory to save plots
        comparison_mode: "pairwise" or "aggregate"
    """
    print("\n" + "=" * 60)
    print("PLOTTING ICC ESTIMATION ERROR RESULTS")
    print("=" * 60)
    plot_metric_results(
        icc_results,
        icc_results_by_axis,
        reliability_metadata_all,
        reliability_metadata_by_axis,
        dataset,
        plots_dir,
        comparison_mode,
        metric_name="ICC",
        hm_key="true_hm_icc",
        im_key="im_icc"
    )

    print("\n" + "=" * 60)
    print("PLOTTING KRIPPENDORFF'S ALPHA ESTIMATION ERROR RESULTS")
    print("=" * 60)
    plot_metric_results(
        alpha_results,
        alpha_results_by_axis,
        reliability_metadata_all,
        reliability_metadata_by_axis,
        dataset,
        plots_dir,
        comparison_mode,
        metric_name="Alpha",
        hm_key="true_hm_alpha",
        im_key="im_alpha"
    )
