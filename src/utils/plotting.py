"""
Plotting utilities for reliability estimation experiments.

Contains functions for creating estimation error plots with confidence intervals.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _get_colors_for_methods(methods):
    """Return a dict mapping each method name to a unique color from tab20."""
    import matplotlib.cm as cm
    n = len(methods)
    if n == 0:
        return {}
    cmap = cm.get_cmap("tab20" if n <= 20 else "hsv")
    return {m: cmap(i / max(n, 1)) for i, m in enumerate(methods)}


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
                             dataset=None,
                             rho_results=None, tau_results=None,
                             rho_results_by_axis=None, tau_results_by_axis=None):
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
    df_dir = os.path.join(plots_dir, dataset, "dataframes") if dataset else os.path.join(plots_dir, "dataframes")
    os.makedirs(df_dir, exist_ok=True)

    icc_results.to_csv(os.path.join(df_dir, "icc_results.csv"), index=False)
    alpha_results.to_csv(os.path.join(df_dir, "alpha_results.csv"), index=False)
    mse_results.to_csv(os.path.join(df_dir, "mse_results.csv"), index=False)
    if rho_results is not None and len(rho_results) > 0:
        rho_results.to_csv(os.path.join(df_dir, "rho_results.csv"), index=False)
    if tau_results is not None and len(tau_results) > 0:
        tau_results.to_csv(os.path.join(df_dir, "tau_results.csv"), index=False)

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
    if rho_results_by_axis:
        for axis, df in rho_results_by_axis.items():
            safe_axis = axis.replace("/", "-").replace("\\", "-").replace(" ", "_")
            if df is not None and len(df) > 0:
                df.to_csv(os.path.join(df_dir, f"rho_by_axis_{safe_axis}.csv"), index=False)
    if tau_results_by_axis:
        for axis, df in tau_results_by_axis.items():
            safe_axis = axis.replace("/", "-").replace("\\", "-").replace(" ", "_")
            if df is not None and len(df) > 0:
                df.to_csv(os.path.join(df_dir, f"tau_by_axis_{safe_axis}.csv"), index=False)

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
    df_dir = os.path.join(results_dir, dataset, "dataframes") if dataset else os.path.join(results_dir, "dataframes")
    # Handle legacy nested paths: dataset/dataset/dataframes/ (repeated dataset subfolder).
    if dataset and not os.path.exists(os.path.join(df_dir, "alpha_results.csv")):
        double_nested = os.path.join(results_dir, dataset, dataset, "dataframes")
        if os.path.exists(os.path.join(double_nested, "alpha_results.csv")):
            df_dir = double_nested
    # Legacy results may have most files nested one level deeper (dataset/dataframes/dataset/)
    # while icc_results.csv stays at the flat level. Detect and handle this split.
    meta_dir = df_dir
    if dataset and not os.path.exists(os.path.join(df_dir, "alpha_results.csv")):
        nested = os.path.join(df_dir, dataset)
        if os.path.exists(os.path.join(nested, "alpha_results.csv")):
            meta_dir = nested

    icc_results = pd.read_csv(os.path.join(df_dir, "icc_results.csv"))
    alpha_results = pd.read_csv(os.path.join(meta_dir, "alpha_results.csv"))
    mse_results = pd.read_csv(os.path.join(meta_dir, "mse_results.csv"))

    rho_path = os.path.join(meta_dir, "rho_results.csv")
    tau_path = os.path.join(meta_dir, "tau_results.csv")
    rho_results = pd.read_csv(rho_path) if os.path.exists(rho_path) else pd.DataFrame()
    tau_results = pd.read_csv(tau_path) if os.path.exists(tau_path) else pd.DataFrame()

    with open(os.path.join(meta_dir, "axes.json")) as f:
        axes = json.load(f)

    icc_results_by_axis = {}
    alpha_results_by_axis = {}
    mse_results_by_axis = {}
    rho_results_by_axis = {}
    tau_results_by_axis = {}
    for axis in axes:
        safe_axis = axis.replace("/", "-").replace("\\", "-").replace(" ", "_")
        icc_path = os.path.join(meta_dir, f"icc_by_axis_{safe_axis}.csv")
        alpha_path = os.path.join(meta_dir, f"alpha_by_axis_{safe_axis}.csv")
        mse_path = os.path.join(meta_dir, f"mse_by_axis_{safe_axis}.csv")
        rho_ax_path = os.path.join(meta_dir, f"rho_by_axis_{safe_axis}.csv")
        tau_ax_path = os.path.join(meta_dir, f"tau_by_axis_{safe_axis}.csv")
        icc_results_by_axis[axis] = pd.read_csv(icc_path) if os.path.exists(icc_path) else pd.DataFrame()
        alpha_results_by_axis[axis] = pd.read_csv(alpha_path) if os.path.exists(alpha_path) else pd.DataFrame()
        mse_results_by_axis[axis] = pd.read_csv(mse_path) if os.path.exists(mse_path) else pd.DataFrame()
        rho_results_by_axis[axis] = pd.read_csv(rho_ax_path) if os.path.exists(rho_ax_path) else pd.DataFrame()
        tau_results_by_axis[axis] = pd.read_csv(tau_ax_path) if os.path.exists(tau_ax_path) else pd.DataFrame()

    with open(os.path.join(meta_dir, "reliability_metadata_all.json")) as f:
        reliability_metadata_all = json.load(f)
    with open(os.path.join(meta_dir, "reliability_metadata_by_axis.json")) as f:
        reliability_metadata_by_axis = json.load(f)

    print(f"Dataframes loaded from: {meta_dir}")
    return (icc_results, alpha_results, mse_results,
            icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis,
            reliability_metadata_all, reliability_metadata_by_axis,
            rho_results_by_axis, tau_results_by_axis)


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
    Create a side-by-side estimation error plot: left panel shows non-oracle methods,
    right panel shows oracle methods. Both panels share a y-axis.

    Args:
        avg_results: DataFrame from compute_bootstrap_cis with method, budget, mean, ci_half_width
        title: Plot title
        ylabel: Y-axis label
        filename: Full path to save the plot
        legend_text: Additional text to append to title (e.g., metric values)

    Returns:
        Path to saved file
    """
    all_methods = list(avg_results["method"].unique())
    oracle_methods = [m for m in all_methods if m.startswith("oracle_")]
    non_oracle_methods = [m for m in all_methods if not m.startswith("oracle_")]

    colors = _get_colors_for_methods(all_methods)

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

    def _plot_methods(ax, methods, panel_title):
        for method in methods:
            method_data = avg_results[avg_results["method"] == method]
            ax.errorbar(
                method_data["budget"],
                method_data["mean"],
                yerr=method_data["ci_half_width"],
                marker='o',
                linewidth=2.5,
                capsize=5,
                capthick=2,
                label=method,
                color=colors[method],
                alpha=0.8
            )
        ax.set_xlabel("Human Annotation Budget", fontsize=12)
        ax.set_title(panel_title, fontsize=11)
        ax.legend(title="Method", fontsize=9, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        ax.grid(alpha=0.3)

    _plot_methods(ax_left,  non_oracle_methods, f"{title} — Our Methods{legend_text}")
    _plot_methods(ax_right, oracle_methods,     f"{title} — Oracle Methods{legend_text}")

    ax_left.set_ylabel(ylabel, fontsize=12)

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

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
    fig, ax = plt.subplots(figsize=(11, 6))

    methods = list(var_results["method"].unique())
    colors = _get_colors_for_methods(methods)
    for method in methods:
        method_data = var_results[var_results["method"] == method].sort_values("budget")
        ax.plot(
            method_data["budget"],
            method_data["variance"],
            marker='o',
            linewidth=2.5,
            label=method,
            color=colors[method],
            alpha=0.8
        )

    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Human Annotation Budget", fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(title="Method", fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

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
    fig, ax = plt.subplots(figsize=(11, 6))

    methods = list(avg_results["method"].unique())
    colors = _get_colors_for_methods(methods)
    for method in methods:
        method_data = avg_results[avg_results["method"] == method].sort_values("budget")
        means = method_data["mean"].values
        # Clip CI bounds to be non-negative so log scale doesn't break
        yerr_lower = np.clip(means - method_data["ci_lower"].values, 0, None)
        yerr_upper = np.clip(method_data["ci_upper"].values - means, 0, None)
        ax.errorbar(
            method_data["budget"],
            means,
            yerr=[yerr_lower, yerr_upper],
            marker='o',
            linewidth=2.5,
            capsize=5,
            capthick=2,
            label=method,
            color=colors[method],
            alpha=0.8
        )

    ax.set_yscale('log')
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Human Annotation Budget", fontsize=12)
    ax.set_title(title, fontsize=11)
    ax.legend(title="Method", fontsize=10, bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

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


def plot_all_results(icc_results, alpha_results, mse_results, rho_results, tau_results,
                     icc_results_by_axis, alpha_results_by_axis, mse_results_by_axis,
                     rho_results_by_axis, tau_results_by_axis,
                     reliability_metadata_all, reliability_metadata_by_axis,
                     dataset, plots_dir, comparison_mode):
    """
    Generate plots for ICC, Krippendorff's Alpha, MSE, Spearman's Rho, and Kendall's Tau
    estimation errors. Also saves all DataFrames and metadata to plots_dir/dataframes/.

    Args:
        icc_results: DataFrame with ICC estimation errors
        alpha_results: DataFrame with Alpha estimation errors
        mse_results: DataFrame with MSE estimation errors
        rho_results: DataFrame with Spearman's rho estimation errors
        tau_results: DataFrame with Kendall's tau estimation errors
        icc_results_by_axis: Dict mapping axis -> ICC results DataFrame
        alpha_results_by_axis: Dict mapping axis -> Alpha results DataFrame
        mse_results_by_axis: Dict mapping axis -> MSE results DataFrame
        rho_results_by_axis: Dict mapping axis -> Rho results DataFrame
        tau_results_by_axis: Dict mapping axis -> Tau results DataFrame
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
        dataset=dataset,
        rho_results=rho_results, tau_results=tau_results,
        rho_results_by_axis=rho_results_by_axis, tau_results_by_axis=tau_results_by_axis,
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
        metric_name="MSRE",
        hm_key="true_hm_msre",
        im_key="im_mse"
    )

    if rho_results is not None and len(rho_results) > 0:
        print("\n" + "=" * 60)
        print("PLOTTING SPEARMAN'S RHO ESTIMATION ERROR RESULTS")
        print("=" * 60)
        plot_metric_results(
            rho_results,
            rho_results_by_axis,
            reliability_metadata_all,
            reliability_metadata_by_axis,
            dataset,
            plots_dir,
            comparison_mode,
            metric_name="Rho",
            hm_key="true_hm_rho",
            im_key="im_rho"
        )

    if tau_results is not None and len(tau_results) > 0:
        print("\n" + "=" * 60)
        print("PLOTTING KENDALL'S TAU ESTIMATION ERROR RESULTS")
        print("=" * 60)
        plot_metric_results(
            tau_results,
            tau_results_by_axis,
            reliability_metadata_all,
            reliability_metadata_by_axis,
            dataset,
            plots_dir,
            comparison_mode,
            metric_name="Tau",
            hm_key="true_hm_tau",
            im_key="im_tau"
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
    pred_dir = os.path.join(plots_dir, dataset, "dataframes", "predictor_inputs")
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
    pred_dir = os.path.join(results_dir, dataset, "dataframes", "predictor_inputs")
    if not os.path.exists(pred_dir):
        nested_pred_dir = os.path.join(results_dir, dataset, dataset, "dataframes", "predictor_inputs")
        if os.path.exists(nested_pred_dir):
            pred_dir = nested_pred_dir

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
    Create scatter plots: 3 predictors × N methods × 3 metrics (ICC, Alpha, MSE).

    Each point represents one (axis, model) pair.
        x-axis: mean_error(method) − mean_error(random) for the given metric.
                Negative = method beats random.
        y-axis (plot type A): mean_shift      = (im_msb+im_mse) − (hm_msb+hm_mse)
        y-axis (plot type B): correlation     = Pearson r(im_ms, hm_ms)
        y-axis (plot type C): correlation_msb = Pearson r(im_msb, hm_msb)

    One figure is saved per (metric, predictor) combination. Methods wrap at 4
    per row.

    Args:
        predictor_records: List of dicts as returned by compute_predictor_records.
        dataset: Dataset name used in plot titles and filenames.
        plots_dir: Directory to save the plots.
    """
    if not predictor_records:
        print("No predictor records to plot.")
        return

    df = pd.DataFrame(predictor_records)

    # Discover which metrics have gap columns in the records
    metric_info = [
        ("icc",   "ICC",   "ICC error gap vs random\n(method − random; negative = better)"),
        ("alpha", "Alpha", "Alpha error gap vs random\n(method − random; negative = better)"),
        ("mse",   "MSE",   "MSE error gap vs random\n(method − random; negative = better)"),
    ]
    available_metrics = [
        (key, label, xlabel) for key, label, xlabel in metric_info
        if any(col.startswith(f"{key}_gap_") for col in df.columns)
    ]

    # Short display names for subplot titles (fallback to raw name)
    method_labels = {
        "variance_matched_combined":        "VM (combined)",
        "variance_matched_combined_imc":    "VM+IMC",
        "variance_matched_combined_tc":     "VM+TC",
        "variance_matched_combined_tc_imc": "VM+TC+IMC",
        "variance_matched_msb":             "VM (MSB)",
        "variance_matched_msb_imc":         "VM MSB+IMC",
        "variance_matched_msb_tc":          "VM MSB+TC",
        "variance_matched_msb_tc_imc":      "VM MSB+TC+IMC",
        "proxy_oracle":                     "Proxy Oracle",
        "proxy_oracle_imc":                 "Proxy Oracle+IMC",
        "oracle":                           "Oracle",
        "oracle_imc":                       "Oracle+IMC",
        "random_imc":                       "Random+IMC",
        "stratified":                       "Stratified",
        "metric_matched_icc":               "Metric (ICC)",
        "metric_matched_alpha":             "Metric (Alpha)",
        "metric_matched_mse":               "Metric (MSE)",
        "variance_matched_weighted_.5":     "VM Weighted (.5/.5)",
        "variance_matched_weighted_.5_imc": "VM Weighted (.5/.5)+IMC",
        "variance_matched_weighted_.7":     "VM Weighted (.7/.3)",
        "variance_matched_weighted_.7_imc": "VM Weighted (.7/.3)+IMC",
    }

    predictors = [
        ("mean_shift",       "Mean Shift\n(im_msb+im_mse) − (hm_msb+hm_mse)"),
        ("correlation",      "Correlation\nr(im_msb+im_mse, hm_msb+hm_mse) across bootstrap samples"),
        ("correlation_msb",  "MSB Correlation\nr(im_msb, hm_msb) across bootstrap samples"),
    ]

    print(f"\n[Predictor scatter plots] {len(df)} (axis, model) records, "
          f"{len(available_metrics)} metric(s)")

    for metric_key, metric_label, x_label in available_metrics:
        comparison_methods = sorted(
            col[len(f"{metric_key}_gap_"):] for col in df.columns
            if col.startswith(f"{metric_key}_gap_")
        )
        n_methods = len(comparison_methods)
        n_cols = min(n_methods, 4)
        n_rows = int(np.ceil(n_methods / n_cols))

        for predictor_col, predictor_label in predictors:
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(5 * n_cols, 4 * n_rows),
                                     sharey=True, squeeze=False)
            ax_flat = [axes[r][c] for r in range(n_rows) for c in range(n_cols)]
            for ax in ax_flat[n_methods:]:
                ax.set_visible(False)

            scatter_colors = _get_colors_for_methods(comparison_methods)
            for ax, method in zip(ax_flat, comparison_methods):
                gap_col = f"{metric_key}_gap_{method}"
                plot_df = df[[predictor_col, gap_col]].dropna()

                if len(plot_df) == 0:
                    ax.set_title(method_labels.get(method, method))
                    ax.text(0.5, 0.5, "no data", ha="center", va="center",
                            transform=ax.transAxes)
                    continue

                ax.scatter(plot_df[gap_col], plot_df[predictor_col],
                           alpha=0.7, edgecolors="k", linewidths=0.5, s=60,
                           color=scatter_colors[method])
                ax.axvline(0, color="red", linestyle="--", linewidth=1, alpha=0.6)
                ax.set_xlabel(x_label, fontsize=9)
                ax.set_title(method_labels.get(method, method), fontsize=10)

                if len(plot_df) >= 3:
                    r = np.corrcoef(plot_df[gap_col].values,
                                    plot_df[predictor_col].values)[0, 1]
                    ax.text(0.05, 0.95, f"r={r:.2f}", transform=ax.transAxes,
                            fontsize=8, va="top")

                ax.grid(alpha=0.3)

            for r in range(n_rows):
                axes[r][0].set_ylabel(predictor_label, fontsize=9)

            fig.suptitle(
                f"{dataset}: predictor vs {metric_label} error gap ({predictor_col})",
                fontsize=11
            )
            fig.tight_layout()

            safe_pred = predictor_col.replace(" ", "_")
            filename = os.path.join(
                plots_dir,
                f"{dataset}_predictor_scatter_{metric_key}_{safe_pred}.jpg"
            )
            fig.savefig(filename, dpi=300)
            plt.close(fig)
            print(f"  Saved: {filename}")


def plot_ms_budget_scatter(axis_jobs, im_df_builder, dataset, plots_dir,
                            budgets=(10, 20, 30, 40, 50), n_samples=200, seed=123):
    """
    For each (axis, model) pair, scatter-plot im_msb+im_mse (x) vs hm_msb+hm_mse (y)
    across random samples of text_ids, with one subplot per budget.

    Each point represents one random draw of `budget` text_ids; the axes show the
    mean-square components of model-model (IM) vs human-model (HM) annotations on
    that draw.  A y=x reference line is drawn in red.

    One figure is saved per (axis, model), with subplots for budgets [10,20,30,40,50].
    Figures are saved into plots_dir/ms_budget_scatter/.

    Args:
        axis_jobs: List of (axis, axis_df) pairs.
        im_df_builder: Callable(axis_df, model) -> im_full_df.
        dataset: Dataset name for titles and filenames.
        plots_dir: Base directory to save plots.
        budgets: Sequence of sample sizes (default: (10, 20, 30, 40, 50)).
        n_samples: Random draws per budget (default: 200).
        seed: Random seed.
    """
    from src.utils.predictor_analysis import compute_ms_budget_samples

    budgets = list(budgets)
    n_budgets = len(budgets)

    subdir = os.path.join(plots_dir, "ms_budget_scatter")
    os.makedirs(subdir, exist_ok=True)

    for axis, axis_df in axis_jobs:
        models = [m for m in axis_df["model_name"].unique() if m != "original"]

        for model in models:
            im_full_df = im_df_builder(axis_df, model)
            hm_full_df = axis_df[axis_df["model_name"].isin([model, "original"])]

            print(f"  MS budget scatter: axis={axis}, model={model} ...")
            budget_samples = compute_ms_budget_samples(
                im_full_df, hm_full_df,
                budgets=budgets, n_samples=n_samples, seed=seed,
            )

            safe_axis = str(axis).replace(" ", "_").replace("/", "-")
            safe_model = model.replace(" ", "_").replace("/", "-")

            component_configs = [
                ("combined", "IM MSB+MSE\n(model-model)", "HM MSB+MSE\n(human-model)",
                 "IM MSB+MSE vs HM MSB+MSE across random samples", "ms_budget_scatter"),
                ("msb",      "IM MSB\n(model-model)",     "HM MSB\n(human-model)",
                 "IM MSB vs HM MSB across random samples",         "msb_budget_scatter"),
                ("mse",      "IM MSE\n(model-model)",     "HM MSE\n(human-model)",
                 "IM MSE vs HM MSE across random samples",         "mse_budget_scatter"),
                ("icc",      "IM ICC\n(model-model)",     "HM ICC\n(human-model)",
                 "IM ICC vs HM ICC across random samples",         "icc_budget_scatter"),
            ]

            budget_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                             "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

            for comp_key, x_label, y_label, suptitle_suffix, file_suffix in component_configs:
                fig, axes = plt.subplots(1, n_budgets,
                                         figsize=(4 * n_budgets, 4),
                                         squeeze=False)
                ax_row = axes[0]

                for i, (ax, budget) in enumerate(zip(ax_row, budgets)):
                    pairs = budget_samples[budget][comp_key]
                    color = budget_colors[i % len(budget_colors)]
                    ax.set_title(f"Budget = {budget}", fontsize=10)
                    ax.set_xlabel(x_label, fontsize=9)
                    if i == 0:
                        ax.set_ylabel(y_label, fontsize=9)

                    if not pairs:
                        ax.text(0.5, 0.5, "insufficient data",
                                ha="center", va="center", transform=ax.transAxes)
                        ax.grid(alpha=0.3)
                        continue

                    xs = [p[0] for p in pairs]
                    ys = [p[1] for p in pairs]
                    ax.scatter(xs, ys, alpha=0.4, edgecolors="none", s=20, color=color)

                    lim_min = min(min(xs), min(ys))
                    lim_max = max(max(xs), max(ys))
                    ax.plot([lim_min, lim_max], [lim_min, lim_max],
                            color="red", linestyle="--", linewidth=1, alpha=0.6)

                    m, b = np.polyfit(xs, ys, 1)
                    x_line = np.linspace(min(xs), max(xs), 100)
                    ax.plot(x_line, m * x_line + b, color="black", linewidth=1.2, alpha=0.7)

                    if len(pairs) >= 3:
                        r = float(np.corrcoef(xs, ys)[0, 1])
                        ax.text(0.05, 0.95, f"r={r:.2f}, slope={m:.2f}",
                                transform=ax.transAxes, fontsize=8, va="top")

                    ax.grid(alpha=0.3)

                fig.suptitle(
                    f"{dataset} | axis={axis} | model={model}\n{suptitle_suffix}",
                    fontsize=10
                )
                fig.tight_layout()

                filename = os.path.join(
                    subdir, f"{dataset}_{safe_axis}_{safe_model}_{file_suffix}.jpg"
                )
                fig.savefig(filename, dpi=300)
                plt.close(fig)
                print(f"    Saved: {filename}")
