"""
Plotting utilities for reliability estimation experiments.

Contains functions for creating estimation error plots with confidence intervals.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results_dataframes(plots_dir, icc_results, alpha_results, mse_results,
                             icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis,
                             reliability_metadata_all, reliability_metadata_by_axis,
                             dataset=None):
    """
    Save all result DataFrames and metadata dicts to disk for later reloading.

    Saves into a '{dataset}/dataframes/' subdirectory within plots_dir so that
    concurrent runs on different datasets do not overwrite each other's files.

    Args:
        plots_dir: Directory where plots are saved (dataframes go in plots_dir/{dataset}/dataframes/)
        icc_results: Combined ICC results DataFrame
        alpha_results: Combined Alpha results DataFrame
        mse_results: Combined MSE results DataFrame
        icc_results_by_axis: Dict mapping axis -> ICC results DataFrame
        alpha_results_by_axis: Dict mapping axis -> Alpha results DataFrame
        mse_results_by_axis: Dict mapping axis -> MSE results DataFrame
        reliability_metadata_all: Dict mapping model -> aggregated metadata
        reliability_metadata_by_axis: Dict mapping axis -> model -> metadata
        dataset: Dataset name used to namespace the output subdirectory (e.g. "hanna")
    """
    df_dir = os.path.join(plots_dir, dataset, "dataframes", dataset) if dataset else os.path.join(plots_dir, "dataframes")
    os.makedirs(df_dir, exist_ok=True)

    icc_results.to_csv(os.path.join(df_dir, "icc_results.csv"), index=False)
    alpha_results.to_csv(os.path.join(df_dir, "alpha_results.csv"), index=False)
    mse_results.to_csv(os.path.join(df_dir, "mse_results.csv"), index=False)

    axes = list(icc_results_by_axis.keys())
    with open(os.path.join(df_dir, "axes.json"), "w") as f:
        json.dump(axes, f)

    for axis, df in icc_results_by_axis.items():
        safe_axis = axis.replace("/", "-").replace("\\", "-").replace(" ", "_")
        if df is not None and len(df) > 0:
            df.to_csv(os.path.join(df_dir, f"icc_by_axis_{safe_axis}.csv"), index=False)
    for axis, df in alpha_results_by_axis.items():
        safe_axis = axis.replace("/", "-").replace("\\", "-").replace(" ", "_")
        if df is not None and len(df) > 0:
            df.to_csv(os.path.join(df_dir, f"alpha_by_axis_{safe_axis}.csv"), index=False)
    for axis, df in mse_results_by_axis.items():
        safe_axis = axis.replace("/", "-").replace("\\", "-").replace(" ", "_")
        if df is not None and len(df) > 0:
            df.to_csv(os.path.join(df_dir, f"mse_by_axis_{safe_axis}.csv"), index=False)

    with open(os.path.join(df_dir, "reliability_metadata_all.json"), "w") as f:
        json.dump(reliability_metadata_all, f, cls=_NumpyEncoder)
    with open(os.path.join(df_dir, "reliability_metadata_by_axis.json"), "w") as f:
        json.dump(reliability_metadata_by_axis, f, cls=_NumpyEncoder)

    print(f"\nDataframes saved to: {df_dir}")


def load_results_dataframes(results_dir, dataset=None):
    """
    Load previously saved result DataFrames and metadata from disk.

    Expects data in a '{dataset}/dataframes/' subdirectory within results_dir
    (i.e., the same directory that was passed as plots_dir when the results were
    saved, with the same dataset name).

    Args:
        results_dir: Directory containing the '{dataset}/dataframes/' subdirectory
        dataset: Dataset name used when saving (e.g. "hanna"); must match the
                 value passed to save_results_dataframes

    Returns:
        Tuple of (icc_results, alpha_results, mse_results,
                  icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis,
                  reliability_metadata_all, reliability_metadata_by_axis)
    """
    df_dir = os.path.join(results_dir, dataset, "dataframes", dataset) if dataset else os.path.join(results_dir, "dataframes")

    icc_results = pd.read_csv(os.path.join(df_dir, "icc_results.csv"))
    alpha_results = pd.read_csv(os.path.join(df_dir, "alpha_results.csv"))
    mse_results = pd.read_csv(os.path.join(df_dir, "mse_results.csv"))

    with open(os.path.join(df_dir, "axes.json")) as f:
        axes = json.load(f)

    icc_results_by_axis = {}
    alpha_results_by_axis = {}
    mse_results_by_axis = {}
    for axis in axes:
        safe_axis = axis.replace("/", "-").replace("\\", "-").replace(" ", "_")
        icc_path = os.path.join(df_dir, f"icc_by_axis_{safe_axis}.csv")
        alpha_path = os.path.join(df_dir, f"alpha_by_axis_{safe_axis}.csv")
        mse_path = os.path.join(df_dir, f"mse_by_axis_{safe_axis}.csv")
        icc_results_by_axis[axis] = pd.read_csv(icc_path) if os.path.exists(icc_path) else pd.DataFrame()
        alpha_results_by_axis[axis] = pd.read_csv(alpha_path) if os.path.exists(alpha_path) else pd.DataFrame()
        mse_results_by_axis[axis] = pd.read_csv(mse_path) if os.path.exists(mse_path) else pd.DataFrame()

    with open(os.path.join(df_dir, "reliability_metadata_all.json")) as f:
        reliability_metadata_all = json.load(f)
    with open(os.path.join(df_dir, "reliability_metadata_by_axis.json")) as f:
        reliability_metadata_by_axis = json.load(f)

    print(f"Dataframes loaded from: {df_dir}")
    return (icc_results, alpha_results, mse_results,
            icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis,
            reliability_metadata_all, reliability_metadata_by_axis)


def compute_variance_by_method(results_df):
    """
    Compute variance of estimation errors by method and budget.

    Args:
        results_df: DataFrame with columns: method, budget, estimation_error

    Returns:
        DataFrame with columns: method, budget, variance
    """
    var_data = []
    for method in results_df["method"].unique():
        for budget in sorted(results_df["budget"].unique()):
            subset = results_df[
                (results_df["method"] == method) &
                (results_df["budget"] == budget)
            ]
            if len(subset) > 0:
                var_data.append({
                    "method": method,
                    "budget": budget,
                    "variance": subset["estimation_error"].var(),
                })
    return pd.DataFrame(var_data)


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


def create_variance_plot(var_results, title, ylabel, filename):
    """
    Create a variance-vs-budget plot, one line per method.

    Args:
        var_results: DataFrame from compute_variance_by_method with method, budget, variance
        title: Plot title
        ylabel: Y-axis label
        filename: Full path to save the plot

    Returns:
        Path to saved file
    """
    plt.figure(figsize=(10, 6))

    for method in var_results["method"].unique():
        method_data = var_results[var_results["method"] == method].sort_values("budget")
        plt.plot(
            method_data["budget"],
            method_data["variance"],
            marker='o',
            linewidth=2.5,
            label=method,
            alpha=0.8
        )

    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Human Annotation Budget", fontsize=12)
    plt.title(title, fontsize=11)
    plt.legend(title="Method", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

    return filename


def create_log_abs_error_plot(avg_results, title, ylabel, filename):
    """
    Create a log-scale absolute error vs budget plot with confidence intervals.

    Identical to create_estimation_error_plot but with a log y-axis so small
    differences between methods are easier to see.

    Args:
        avg_results: DataFrame from compute_bootstrap_cis with method, budget,
                     mean, ci_lower, ci_upper, ci_half_width
        title: Plot title
        ylabel: Y-axis label
        filename: Full path to save the plot

    Returns:
        Path to saved file
    """
    plt.figure(figsize=(10, 6))

    for method in avg_results["method"].unique():
        method_data = avg_results[avg_results["method"] == method].sort_values("budget")
        means = method_data["mean"].values
        # Clip CI bounds to be non-negative so log scale doesn't break
        yerr_lower = np.clip(means - method_data["ci_lower"].values, 0, None)
        yerr_upper = np.clip(method_data["ci_upper"].values - means, 0, None)
        plt.errorbar(
            method_data["budget"],
            means,
            yerr=[yerr_lower, yerr_upper],
            marker='o',
            linewidth=2.5,
            capsize=5,
            capthick=2,
            label=method,
            alpha=0.8
        )

    plt.yscale('log')
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel("Human Annotation Budget", fontsize=12)
    plt.title(title, fontsize=11)
    plt.legend(title="Method", fontsize=11)
    plt.grid(alpha=0.3, which='both')
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

    var_results_all = compute_variance_by_method(results)
    create_variance_plot(
        var_results_all,
        title=f"{dataset} {metric_name} Estimation Error Variance (avg across models & axes)",
        ylabel=f"Variance of {metric_name} Estimation Error",
        filename=os.path.join(plots_dir, f"{dataset}_{metric_lower}_variance_avg_all.jpg")
    )
    create_log_abs_error_plot(
        avg_results,
        title=f"{dataset} {metric_name} Absolute Error - Log Scale (avg across models & axes)",
        ylabel=f"Absolute {metric_name} Error - log scale",
        filename=os.path.join(plots_dir, f"{dataset}_{metric_lower}_log_abs_error_avg_all.jpg")
    )

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

            var_model_results = compute_variance_by_method(model_results)
            create_variance_plot(
                var_model_results,
                title=f"{dataset} {metric_name} Error Variance: {model} (avg across axes)",
                ylabel=f"Variance of {metric_name} Estimation Error",
                filename=os.path.join(plots_dir, f"{dataset}_{metric_lower}_variance_by_model_{safe_model_name}.jpg")
            )
            create_log_abs_error_plot(
                avg_model_results,
                title=f"{dataset} {metric_name} Abs Error - Log Scale: {model} (avg across axes)",
                ylabel=f"Absolute {metric_name} Error - log scale",
                filename=os.path.join(plots_dir, f"{dataset}_{metric_lower}_log_abs_error_by_model_{safe_model_name}.jpg")
            )

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

            var_axis_results = compute_variance_by_method(axis_results)
            create_variance_plot(
                var_axis_results,
                title=f"{dataset} {metric_name} Error Variance: {axis} (avg across models)",
                ylabel=f"Variance of {metric_name} Estimation Error",
                filename=os.path.join(plots_dir, f"{dataset}_{metric_lower}_variance_by_axis_{safe_axis_name}.jpg")
            )
            create_log_abs_error_plot(
                avg_axis_results,
                title=f"{dataset} {metric_name} Abs Error - Log Scale: {axis} (avg across models)",
                ylabel=f"Absolute {metric_name} Error - log scale",
                filename=os.path.join(plots_dir, f"{dataset}_{metric_lower}_log_abs_error_by_axis_{safe_axis_name}.jpg")
            )

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


def plot_all_results(icc_results, alpha_results, mse_results,
                     icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis,
                     reliability_metadata_all, reliability_metadata_by_axis,
                     dataset, plots_dir, comparison_mode):
    """
    Generate plots for ICC, Krippendorff's Alpha, and MSE estimation errors.
    Also saves all DataFrames and metadata to plots_dir/dataframes/ for later reuse.

    Args:
        icc_results: DataFrame with ICC estimation errors
        alpha_results: DataFrame with Alpha estimation errors
        mse_results: DataFrame with MSE estimation errors
        icc_results_by_axis: Dict mapping axis -> ICC results DataFrame
        alpha_results_by_axis: Dict mapping axis -> Alpha results DataFrame
        mse_results_by_axis: Dict mapping axis -> MSE results DataFrame
        reliability_metadata_all: Dict mapping model -> aggregated metadata
        reliability_metadata_by_axis: Dict mapping axis -> model -> metadata
        dataset: Dataset name
        plots_dir: Directory to save plots
        comparison_mode: "pairwise" or "aggregate"
    """
    save_results_dataframes(
        plots_dir, icc_results, alpha_results, mse_results,
        icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis,
        reliability_metadata_all, reliability_metadata_by_axis,
        dataset=dataset
    )
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

    print("\n" + "=" * 60)
    print("PLOTTING MSE ESTIMATION ERROR RESULTS")
    print("=" * 60)
    plot_metric_results(
        mse_results,
        mse_results_by_axis,
        reliability_metadata_all,
        reliability_metadata_by_axis,
        dataset,
        plots_dir,
        comparison_mode,
        metric_name="MSE",
        hm_key="true_hm_mse",
        im_key="im_mse"
    )


def save_predictor_inputs(plots_dir, dataset, axis_jobs,
                          per_model_variance_by_axis, comparison_mode, ensemble_models):
    """
    Save all data needed to run predictor scatter analysis post-hoc.

    Writes into plots_dir/{dataset}/dataframes/predictor_inputs/:
        predictor_config.json     – comparison_mode and ensemble_models
        per_model_variance.json   – variance components per axis per model
        axis_data_{safe_axis}.csv – raw scores DataFrame per axis

    Args:
        plots_dir: Base results directory (same value passed to save_results_dataframes).
        dataset: Dataset name used to namespace the subdirectory.
        axis_jobs: List of (axis, axis_df) from the experiment run.
        per_model_variance_by_axis: Dict axis -> model -> {im_msb, im_mse, hm_msb, hm_mse}.
        comparison_mode: Comparison mode string used in the run.
        ensemble_models: List of ensemble model names used in the run.
    """
    pred_dir = os.path.join(plots_dir, dataset, "dataframes", dataset, "predictor_inputs")
    os.makedirs(pred_dir, exist_ok=True)

    config = {"comparison_mode": comparison_mode, "ensemble_models": list(ensemble_models)}
    with open(os.path.join(pred_dir, "predictor_config.json"), "w") as f:
        json.dump(config, f)

    with open(os.path.join(pred_dir, "per_model_variance.json"), "w") as f:
        json.dump(per_model_variance_by_axis, f, cls=_NumpyEncoder)

    for axis, axis_df in axis_jobs:
        safe_axis = axis.replace("/", "-").replace("\\", "-").replace(" ", "_")
        axis_df.to_csv(os.path.join(pred_dir, f"axis_data_{safe_axis}.csv"), index=False)

    axes = [axis for axis, _ in axis_jobs]
    with open(os.path.join(pred_dir, "axes.json"), "w") as f:
        json.dump(axes, f)

    print(f"\nPredictor inputs saved to: {pred_dir}")


def load_predictor_inputs(results_dir, dataset):
    """
    Load predictor inputs saved by save_predictor_inputs.

    Args:
        results_dir: Base results directory (same as plots_dir used when saving).
        dataset: Dataset name.

    Returns:
        Tuple of (axis_jobs, per_model_variance_by_axis, config) where:
            axis_jobs: List of (axis, axis_df).
            per_model_variance_by_axis: Dict axis -> model -> variance components.
            config: Dict with keys 'comparison_mode' and 'ensemble_models'.
    """
    pred_dir = os.path.join(results_dir, dataset, "dataframes", dataset, "predictor_inputs")

    with open(os.path.join(pred_dir, "predictor_config.json")) as f:
        config = json.load(f)

    with open(os.path.join(pred_dir, "per_model_variance.json")) as f:
        per_model_variance_by_axis = json.load(f)

    with open(os.path.join(pred_dir, "axes.json")) as f:
        axes = json.load(f)

    axis_jobs = []
    for axis in axes:
        safe_axis = axis.replace("/", "-").replace("\\", "-").replace(" ", "_")
        csv_path = os.path.join(pred_dir, f"axis_data_{safe_axis}.csv")
        axis_df = pd.read_csv(csv_path)
        axis_jobs.append((axis, axis_df))

    print(f"Predictor inputs loaded from: {pred_dir}")
    return axis_jobs, per_model_variance_by_axis, config


def plot_predictor_scatter(predictor_records, dataset, plots_dir):
    """
    Create 6 scatter plots: 2 predictors × 3 selection methods vs random.

    Each point represents one (axis, model) pair.
        x-axis: mean_ICC_error(method) − mean_ICC_error(random), averaged across
                all budgets/trials.  Negative = method beats random.
        y-axis (plot type A): mean_shift  = (im_msb+im_mse) − (hm_msb+hm_mse)
                               on the full dataset.
        y-axis (plot type B): correlation = Pearson r between im_ms and hm_ms
                               across 500 bootstrap samples of 10 text_ids.

    The 3 comparison methods are:
        variance_matched_combined
        variance_matched_combined_imc
        variance_matched_combined_tc_imc

    Args:
        predictor_records: List of dicts as returned by compute_predictor_records.
            Each dict has keys: axis, model, mean_shift, correlation,
            icc_gap_<method> for each PREDICTOR_COMPARISON_METHODS entry.
        dataset: Dataset name used in plot titles and filenames.
        plots_dir: Directory to save the plots.
    """
    from src.utils.predictor_analysis import PREDICTOR_COMPARISON_METHODS

    if not predictor_records:
        print("No predictor records to plot.")
        return

    df = pd.DataFrame(predictor_records)

    # Short display names for x-axis method labels
    method_labels = {
        "variance_matched_combined":        "VM",
        "variance_matched_combined_imc":    "VM+IMC",
        "variance_matched_combined_tc_imc": "VM+TC+IMC",
    }

    predictors = [
        ("mean_shift",   "Mean Shift\n(im_msb+im_mse) − (hm_msb+hm_mse)"),
        ("correlation",  "Correlation\nr(im_ms, hm_ms) across bootstrap samples"),
    ]

    print(f"\n[Predictor scatter plots] {len(df)} (axis, model) records")

    for predictor_col, predictor_label in predictors:
        fig, axes = plt.subplots(1, len(PREDICTOR_COMPARISON_METHODS),
                                 figsize=(5 * len(PREDICTOR_COMPARISON_METHODS), 4),
                                 sharey=True)
        if len(PREDICTOR_COMPARISON_METHODS) == 1:
            axes = [axes]

        for ax, method in zip(axes, PREDICTOR_COMPARISON_METHODS):
            gap_col = f"icc_gap_{method}"
            plot_df = df[[predictor_col, gap_col]].dropna()

            if len(plot_df) == 0:
                ax.set_title(method_labels.get(method, method))
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            ax.scatter(plot_df[gap_col], plot_df[predictor_col],
                       alpha=0.7, edgecolors="k", linewidths=0.5, s=60)
            ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.6)
            ax.set_xlabel("ICC error gap vs random\n(method − random; negative = better)",
                          fontsize=9)
            ax.set_title(method_labels.get(method, method), fontsize=10)

            # Pearson r annotation
            if len(plot_df) >= 3:
                r = np.corrcoef(plot_df[gap_col].values,
                                plot_df[predictor_col].values)[0, 1]
                ax.text(0.05, 0.95, f"r={r:.2f}", transform=ax.transAxes,
                        fontsize=8, va="top")

            ax.grid(alpha=0.3)

        axes[0].set_ylabel(predictor_label, fontsize=9)
        fig.suptitle(f"{dataset}: predictor vs ICC error gap ({predictor_col})",
                     fontsize=11)
        fig.tight_layout()

        safe_pred = predictor_col.replace(" ", "_")
        filename = os.path.join(plots_dir,
                                f"{dataset}_predictor_scatter_{safe_pred}.jpg")
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        print(f"  Saved: {filename}")
