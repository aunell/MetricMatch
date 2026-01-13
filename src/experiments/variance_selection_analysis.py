import os
import json
import numpy as np
import pandas as pd
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.stats import pearsonr

# Set random seed for reproducibility
np.random.seed(42)

# -------------------------
# CONFIG
# -------------------------
# Number of bootstrap samples for confidence interval estimation
N_BOOTSTRAP_SAMPLES = 10

datasets = ["hanna", "medval", "mslr", "summeval"]
model_names = ["claude-3.5-sonnet", "gpt-4.1", "gpt-4o-mini", "meta-llama-Llama-3.1-8B-Instruct", "gpt-5", "google-gemma-3-1b-it", "Qwen-Qwen2.5-7B-Instruct"]
# model_names = ["gpt-4o-mini", "meta-llama-Llama-3.1-8B-Instruct", "google-gemma-3-1b-it", "Qwen-Qwen2.5-7B-Instruct"]
# model_names = ["claude-3.5-sonnet", "gpt-4.1", "gpt-5"]

evaluation_axes = {
    "hanna": ["Coherence", "Complexity", "Empathy", "Engagement", "Relevance", "Surprise"],
    "medval": ["Risk"],
    "mslr": ["fluency", "intervention", "outcome", "population"],
    "summeval": ["coherence", "consistency", "fluency", "relevance"]
}

dataset =  "medval" #"mslr" #"hanna" #"medval" #summeval
DATA_DIR = "data/judge_scores"

# Plots output directory - change this to specify where plots should be saved
PLOTS_DIR = "results/01_13_test"  # Default: results/plots

# Comparison mode: "pairwise" or "aggregate"
# "pairwise": Compare each model vs average of other models (k=2 for both HM and IM)
# "aggregate": Use all models for inter-model (k=4), each model+human for HM (k=2)
COMPARISON_MODE = "pairwise"  # Toggle this between "pairwise" and "aggregate"

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)

# -------------------------
# LOAD DATA
# -------------------------
dfs = []

for model_name in model_names:
    for ev_ax in evaluation_axes[dataset]:
        path = os.path.join(
            DATA_DIR,
            dataset,
            f"results_{dataset}_{model_name}_{ev_ax}.json"
        )
        if not os.path.exists(path):
            continue

        with open(path) as f:
            results = json.load(f)["detailed_results"]

        rows = []
        for r in results:
            try:
                score = r["evaluation"]["evaluation"]["score"]
            except:
                score = r["evaluation"]["score"]
            r = {k: v for k, v in r.items() if k != "evaluation"}
            r["evaluation_score"] = score
            r["model_name"] = model_name
            r["evaluation_axis"] = ev_ax  # Add axis information
            rows.append(r)

        dfs.append(pd.DataFrame(rows))

df = pd.concat(dfs, ignore_index=True)

# Add human ("original") scores
text_info = (
    df[["text_id", "input_text", "source_text", "original_score", "evaluation_axis"]]
    .drop_duplicates()
)

human_df = (
    text_info[["text_id", "original_score", "evaluation_axis"]]
    .rename(columns={"original_score": "evaluation_score"})
)
human_df["model_name"] = "original"

df = pd.concat(
    [df[["text_id", "model_name", "evaluation_score", "evaluation_axis"]], human_df],
    ignore_index=True
)

# -------------------------
# VARIANCE COMPONENTS
# -------------------------
def compute_ms_components(data):
    """Compute MSB and MSE components for ICC calculation."""
    k = data["model_name"].nunique()
    n = data["text_id"].nunique()

    if n <= 1 or k <= 1:
        return np.nan, np.nan

    # Cache groupby results to avoid repeated computation
    grouped_by_text = data.groupby("text_id")["evaluation_score"]
    grouped_by_model = data.groupby("model_name")["evaluation_score"]

    s = grouped_by_text.mean()
    m = grouped_by_model.mean()
    xbar = data["evaluation_score"].mean()

    msb = (k / (n - 1)) * ((s - xbar) ** 2).sum()

    # MSE: within-text variance - vectorized approach
    # Merge model means back to original data for vectorized computation
    data_with_means = data.merge(m.rename("model_mean"), left_on="model_name", right_index=True, how="left")
    mse_vals = (data_with_means["evaluation_score"] - data_with_means["model_mean"]) ** 2
    mse = mse_vals.mean()

    return msb, mse

def icc_from_ms(msb, mse):
    """Calculate ICC(3,k) from MSB and MSE components."""
    if not np.isfinite(msb) or not np.isfinite(mse):
        return np.nan
    if msb <= 0:
        return np.nan
    # ICC(3,k) = (MSB - MSE) / MSB = 1 - MSE/MSB
    icc = (msb - mse) / msb



    return icc

# -------------------------
# VARIANCE ALIGNMENT CHECK
# -------------------------
def compute_variance_alignment(df, model_names, mode="aggregate"):
    """
    Compute inter-model and human-model variance components.

    Args:
        df: DataFrame with text_id, model_name, evaluation_score
        model_names: List of model names (excluding "original")
        mode: "aggregate" (all models vs each model+human) or
              "pairwise" (model vs avg(other models) for both IM and HM)
    
    Returns:
        per_model_variance: dict mapping model_name to {im_msb, im_mse, hm_msb, hm_mse}
        aggregate_stats: dict with overall statistics
    """
    per_model_variance = {}
    
    if mode == "aggregate":
        # Original behavior: all models for IM, each model+human for HM
        im_df = df[df["model_name"] != "original"]
        im_msb_global, im_mse_global = compute_ms_components(im_df)

        for m in model_names:
            hm_msb, hm_mse = compute_ms_components(
                df[df["model_name"].isin([m, "original"])]
            )
            per_model_variance[m] = {
                "im_msb": im_msb_global,
                "im_mse": im_mse_global,
                "hm_msb": hm_msb,
                "hm_mse": hm_mse
            }

        print(f"\nMode: AGGREGATE (k=4 for IM, k=2 for HM)")
        print(f"Inter-model (all 4 models): MSB={im_msb_global:.4f}, MSE={im_mse_global:.4f}")

        aggregate_stats = {
            "im_msb": im_msb_global,
            "im_mse": im_mse_global,
            "hm_msb_list": [per_model_variance[m]["hm_msb"] for m in model_names],
            "hm_mse_list": [per_model_variance[m]["hm_mse"] for m in model_names]
        }

    elif mode == "pairwise":
        # Fair comparison: each model vs avg(other models) for both IM and HM
        im_msb_list, im_mse_list = [], []
        hm_msb_list, hm_mse_list = [], []

        # Pre-filter to only model data (avoid repeated filtering in loop)
        im_subset = df[df["model_name"].isin(model_names)].copy()
        # Group by text_id once for efficiency
        im_grouped = im_subset.groupby("text_id")

        for m in model_names:
            other_models = [x for x in model_names if x != m]

            # Inter-model: this model vs avg of other models
            # Create "avg_other" pseudo-rater using vectorized operations
            im_avg_df = []
            for text_id, text_data in im_grouped:
                # Current model score
                model_score = text_data.loc[text_data["model_name"] == m, "evaluation_score"]
                if len(model_score) > 0:
                    im_avg_df.append({"text_id": text_id, "model_name": m, "evaluation_score": model_score.iloc[0]})
                # Avg of other models
                other_scores = text_data.loc[text_data["model_name"].isin(other_models), "evaluation_score"]
                if len(other_scores) > 0:
                    im_avg_df.append({"text_id": text_id, "model_name": "avg_other", "evaluation_score": other_scores.mean()})

            im_pair_df = pd.DataFrame(im_avg_df)
            if len(im_pair_df) > 0:
                im_msb, im_mse = compute_ms_components(im_pair_df)
                im_msb_list.append(im_msb)
                im_mse_list.append(im_mse)
            else:
                im_msb, im_mse = np.nan, np.nan
                im_msb_list.append(im_msb)
                im_mse_list.append(im_mse)

            # Human-model: this model vs human
            hm_msb, hm_mse = compute_ms_components(
                df[df["model_name"].isin([m, "original"])]
            )
            hm_msb_list.append(hm_msb)
            hm_mse_list.append(hm_mse)

            # Store per-model variance
            per_model_variance[m] = {
                "im_msb": im_msb,
                "im_mse": im_mse,
                "hm_msb": hm_msb,
                "hm_mse": hm_mse
            }

        print(f"\nMode: PAIRWISE (k=2 for both IM and HM)")
        print(f"Inter-model MSBs (each model vs avg of others): {[f'{x:.4f}' for x in im_msb_list]}")
        print(f"Inter-model MSEs: {[f'{x:.4f}' for x in im_mse_list]}")
        im_msb_mean = np.mean(im_msb_list)
        im_mse_mean = np.mean(im_mse_list)
        print(f"Inter-model mean: MSB={im_msb_mean:.4f}, MSE={im_mse_mean:.4f}")
        
        aggregate_stats = {
            "im_msb": im_msb_mean,
            "im_mse": im_mse_mean,
            "hm_msb_list": hm_msb_list,
            "hm_mse_list": hm_mse_list
        }

    print(f"Human-model MSBs: {[f'{x:.4f}' for x in aggregate_stats['hm_msb_list']]}")
    print(f"Human-model MSEs: {[f'{x:.4f}' for x in aggregate_stats['hm_mse_list']]}")

    if len(aggregate_stats['hm_msb_list']) > 1:
        print(f"\nMSB: IM={aggregate_stats['im_msb']:.4f}, HM mean={np.mean(aggregate_stats['hm_msb_list']):.4f}")
        print(f"MSE: IM={aggregate_stats['im_mse']:.4f}, HM mean={np.mean(aggregate_stats['hm_mse_list']):.4f}")

    return per_model_variance, aggregate_stats

per_model_variance, aggregate_stats = compute_variance_alignment(df, model_names, mode=COMPARISON_MODE)

# -------------------------
# ICC ESTIMATION EXPERIMENT
# -------------------------
def evaluate_icc_estimators(
    df,
    model_names,
    per_model_variance,
    budgets=range(5, 55, 5),
    n_trials=None,  # Will default to N_BOOTSTRAP_SAMPLES
    evaluation_axis=None  # Optional: filter by evaluation axis
):
    results = []
    icc_metadata = {}  # Store ICC values for each model

    # Use N_BOOTSTRAP_SAMPLES if n_trials not specified
    if n_trials is None:
        n_trials = N_BOOTSTRAP_SAMPLES

    # Filter by axis if specified
    if evaluation_axis is not None:
        df = df[df["evaluation_axis"] == evaluation_axis].copy()

    for model in model_names:
        # Get this model's specific inter-model variance components
        im_msb_target = per_model_variance[model]["im_msb"]
        im_mse_target = per_model_variance[model]["im_mse"]

        hm_full_df = df[df["model_name"].isin([model, "original"])]
        true_msb, true_mse = compute_ms_components(hm_full_df)
        true_icc = icc_from_ms(true_msb, true_mse)

        # Store ICC metadata for this model
        im_icc = icc_from_ms(im_msb_target, im_mse_target)
        icc_metadata[model] = {
            "true_hm_icc": true_icc,
            "im_icc": im_icc
        }

        text_ids = hm_full_df["text_id"].unique()
        
        # Get inter-model data for variance matching
        if COMPARISON_MODE == "pairwise":
            # For pairwise mode, recreate the model vs avg(others) dataset
            other_models = [x for x in model_names if x != model]
            im_subset = df[df["model_name"].isin(model_names)].copy()

            # Group once outside the loop for efficiency
            im_grouped = im_subset.groupby("text_id")
            im_full_df = []
            for text_id, text_data in im_grouped:
                model_score = text_data.loc[text_data["model_name"] == model, "evaluation_score"]
                if len(model_score) > 0:
                    im_full_df.append({"text_id": text_id, "model_name": model, "evaluation_score": model_score.iloc[0]})
                other_scores = text_data.loc[text_data["model_name"].isin(other_models), "evaluation_score"]
                if len(other_scores) > 0:
                    im_full_df.append({"text_id": text_id, "model_name": "avg_other", "evaluation_score": other_scores.mean()})
            im_full_df = pd.DataFrame(im_full_df)
        else:
            # For aggregate mode, use all models
            im_full_df = df[df["model_name"] != "original"]

        for k in budgets:
            # Random baseline - each trial uses a different seed
            random_errors = []
            for trial_idx in range(n_trials):
                # Set a unique seed for each trial for reproducibility
                np.random.seed(42 + trial_idx)

                sampled_ids = np.random.choice(
                    text_ids, size=min(k, len(text_ids)), replace=False
                )
                hm_sample = hm_full_df[hm_full_df["text_id"].isin(sampled_ids)]

                try:
                    hm_msb, hm_mse = compute_ms_components(hm_sample)
                    est_icc = icc_from_ms(hm_msb, hm_mse)

                    if np.isfinite(est_icc):
                        random_errors.append(abs(est_icc - true_icc))
                except Exception:
                    pass

            # Store each trial separately for proper bootstrapping
            for error in random_errors:
                results.append({
                    "model": model,
                    "budget": k,
                    "method": "random",
                    "estimation_error": error
                })

            # Variance-matched: select subset that best matches THIS MODEL's IM variance
            matched_errors = []
            for trial_idx in range(n_trials):
                # Set a unique seed for each trial for reproducibility
                np.random.seed(42 + trial_idx)

                # Try multiple random subsets and pick the one with variance closest to this model's IM variance
                best_ids = None
                best_score = float('inf')

                # Try 20 candidate subsets
                for _ in range(20):
                    candidate_ids = np.random.choice(
                        text_ids, size=min(k, len(text_ids)), replace=False
                    )

                    # Compute IM variance for this subset
                    im_candidate = im_full_df[im_full_df["text_id"].isin(candidate_ids)]

                    if len(im_candidate) > 0:
                        cand_msb, cand_mse = compute_ms_components(im_candidate)

                        if np.isfinite(cand_msb) and np.isfinite(cand_mse):
                            # Score: how well does this subset match THIS MODEL's population variance?
                            msb_diff = abs(cand_msb - im_msb_target)
                            mse_diff = abs(cand_mse - im_mse_target)
                            score = msb_diff + mse_diff  # Could also use weighted combination

                            if score < best_score:
                                best_score = score
                                best_ids = candidate_ids

                if best_ids is not None:
                    hm_sample = hm_full_df[hm_full_df["text_id"].isin(best_ids)]

                    try:
                        hm_msb, hm_mse = compute_ms_components(hm_sample)
                        est_icc = icc_from_ms(hm_msb, hm_mse)

                        if np.isfinite(est_icc):
                            matched_errors.append(abs(est_icc - true_icc))
                    except Exception:
                        pass

            # Store each trial separately for proper bootstrapping
            for error in matched_errors:
                results.append({
                    "model": model,
                    "budget": k,
                    "method": "variance_matched",
                    "estimation_error": error
                })

    return pd.DataFrame(results), icc_metadata

def main():
    """Main execution function."""
    print("\n" + "="*50)
    print("Running ICC estimation experiment...")
    print("="*50)

    # Run experiment for all axes combined
    results, icc_metadata_all = evaluate_icc_estimators(df, model_names, per_model_variance)

    print(f"\nTotal results collected: {len(results)}")
    print(f"Results by method:")
    for method in results["method"].unique():
        count = len(results[results["method"] == method])
        print(f"  {method}: {count}")

    # Run experiment for each axis separately
    results_by_axis = {}
    icc_metadata_by_axis = {}
    for axis in evaluation_axes[dataset]:
        print(f"\n{'='*50}")
        print(f"Running for axis: {axis}")
        print(f"{'='*50}")

        # Filter df to this axis
        axis_df = df[df["evaluation_axis"] == axis]

        # Recompute variance components for THIS axis only
        axis_per_model_variance, axis_aggregate_stats = compute_variance_alignment(
            axis_df, model_names, mode=COMPARISON_MODE
        )

        # Now run evaluation with axis-specific variance targets
        axis_results, axis_icc_metadata = evaluate_icc_estimators(
            axis_df, model_names, axis_per_model_variance
        )
        results_by_axis[axis] = axis_results
        icc_metadata_by_axis[axis] = axis_icc_metadata
        print(f"Collected {len(axis_results)} results for {axis}")

    return results, results_by_axis, icc_metadata_all, icc_metadata_by_axis

if __name__ == "__main__":
    results, results_by_axis, icc_metadata_all, icc_metadata_by_axis = main()
else:
    # When imported, just compute variance alignment
    results = None
    results_by_axis = None
    icc_metadata_all = None
    icc_metadata_by_axis = None

# -------------------------
# PLOTS
# -------------------------

def compute_model_cis(results_df, n_bootstrap=1000):
    """Compute 95% bootstrap CI for each model separately, then average."""
    ci_data = []

    for method in results_df["method"].unique():
        for budget in results_df["budget"].unique():
            # Get all model estimates for this method-budget combo
            subset = results_df[
                (results_df["method"] == method) &
                (results_df["budget"] == budget)
            ]

            if len(subset) > 0:
                errors = subset["estimation_error"].values
                mean_error = errors.mean()

                # 95% CI using bootstrap resampling
                if len(errors) > 1:
                    bootstrap_means = []
                    for _ in range(n_bootstrap):
                        # Resample with replacement
                        bootstrap_sample = np.random.choice(errors, size=len(errors), replace=True)
                        bootstrap_means.append(bootstrap_sample.mean())

                    # Compute 95% CI using percentiles
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

def plot_results(results, results_by_axis=None, icc_metadata_all=None, icc_metadata_by_axis=None):
    """Generate plots and summary tables.

    Creates three sets of plots:
    1. Averaged across both axis and model (1 plot)
    2. Averaged across axis only (k plots, where k = number of models)
    3. Averaged across model only (m plots, where m = number of evaluation axes)
    """
    if results is None or len(results) == 0:
        print("\nNo results to plot.")
        return

    # =====================================
    # SET 1: Averaged across axis AND model (1 plot)
    # =====================================
    avg_results = compute_model_cis(results)

    # Compute average ICC values across all models
    legend_text = ""
    if icc_metadata_all is not None:
        avg_true_hm_icc = np.mean([v["true_hm_icc"] for v in icc_metadata_all.values() if np.isfinite(v["true_hm_icc"])])
        avg_im_icc = np.mean([v["im_icc"] for v in icc_metadata_all.values() if np.isfinite(v["im_icc"])])
        legend_text = f"\nAxis: All | Avg HM-ICC: {avg_true_hm_icc:.3f} | Avg {'Pairwise' if COMPARISON_MODE == 'pairwise' else 'Aggregate'} IM-ICC: {avg_im_icc:.3f}"

    plt.figure(figsize=(10, 6))

    # Plot each method separately with 95% CI error bars
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

    plt.ylabel("Absolute ICC Error (avg across models & axes)", fontsize=12)
    plt.xlabel("Human Annotation Budget", fontsize=12)
    plt.title(f"{dataset} ICC Estimation: Random vs Variance-Matched{legend_text}\n(Averaged over all models & axes with 95% CI)", fontsize=11)
    plt.legend(title="Method", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{dataset}_icc_estimation_comparison_avg_all.jpg"), dpi=300)
    plt.close()
    print(f"\n[SET 1: Avg across axis AND model]")
    print(f"  Saved: {os.path.join(PLOTS_DIR, f'{dataset}_icc_estimation_comparison_avg_all.jpg')}")

    # =====================================
    # SET 2: Averaged across axis only (k model plots)
    # =====================================
    if results_by_axis is not None and len(results_by_axis) > 0:
        print(f"\n[SET 2: Avg across axis only - {len(results['model'].unique())} model plots]")
        for model in results["model"].unique():
            # Collect data across all axes for this model
            model_data_across_axes = []
            for axis, axis_results in results_by_axis.items():
                if axis_results is not None and len(axis_results) > 0:
                    model_axis_data = axis_results[axis_results["model"] == model]
                    if len(model_axis_data) > 0:
                        model_data_across_axes.append(model_axis_data)

            if len(model_data_across_axes) == 0:
                continue

            # Concatenate all axis data for this model
            model_results = pd.concat(model_data_across_axes, ignore_index=True)

            # Compute bootstrap CIs using the same function as SET 1 and SET 3
            avg_model_results = compute_model_cis(model_results)

            # Compute average ICC across axes for this model
            legend_text = ""
            if icc_metadata_by_axis is not None:
                model_hm_iccs = [icc_metadata_by_axis[ax][model]["true_hm_icc"]
                                for ax in icc_metadata_by_axis if model in icc_metadata_by_axis[ax]
                                and np.isfinite(icc_metadata_by_axis[ax][model]["true_hm_icc"])]
                model_im_iccs = [icc_metadata_by_axis[ax][model]["im_icc"]
                                for ax in icc_metadata_by_axis if model in icc_metadata_by_axis[ax]
                                and np.isfinite(icc_metadata_by_axis[ax][model]["im_icc"])]
                if model_hm_iccs and model_im_iccs:
                    avg_hm_icc = np.mean(model_hm_iccs)
                    avg_im_icc = np.mean(model_im_iccs)
                    legend_text = f"\nAxis: All | Avg HM-ICC: {avg_hm_icc:.3f} | Avg {'Pairwise' if COMPARISON_MODE == 'pairwise' else 'Aggregate'} IM-ICC: {avg_im_icc:.3f}"

            plt.figure(figsize=(10, 6))

            # Plot each method separately with 95% CI error bars
            for method in avg_model_results["method"].unique():
                method_data = avg_model_results[avg_model_results["method"] == method]
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

            plt.ylabel("Absolute ICC Error (avg across axes)", fontsize=12)
            plt.xlabel("Human Annotation Budget", fontsize=12)
            plt.title(f"{dataset} ICC Estimation: {model}{legend_text}\n(Averaged across axes with 95% CI)", fontsize=11)
            plt.legend(title="Method", fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()

            # Clean model name for filename
            safe_model_name = model.replace("/", "-").replace("\\", "-")
            plt.savefig(os.path.join(PLOTS_DIR, f"{dataset}_icc_estimation_by_model_{safe_model_name}.jpg"), dpi=300)
            plt.close()

            print(f"  Saved: {os.path.join(PLOTS_DIR, f'{dataset}_icc_estimation_by_model_{safe_model_name}.jpg')}")

    # =====================================
    # SET 3: Averaged across model only (m axis plots)
    # =====================================
    if results_by_axis is not None and len(results_by_axis) > 0:
        print(f"\n[SET 3: Avg across model only - {len(results_by_axis)} axis plots]")
        for axis, axis_results in results_by_axis.items():
            if axis_results is None or len(axis_results) == 0:
                continue

            avg_axis_results = compute_model_cis(axis_results)

            # Compute average ICC across models for this axis
            legend_text = ""
            if icc_metadata_by_axis is not None and axis in icc_metadata_by_axis:
                axis_hm_iccs = [icc_metadata_by_axis[axis][model]["true_hm_icc"]
                                for model in icc_metadata_by_axis[axis]
                                if np.isfinite(icc_metadata_by_axis[axis][model]["true_hm_icc"])]
                axis_im_iccs = [icc_metadata_by_axis[axis][model]["im_icc"]
                                for model in icc_metadata_by_axis[axis]
                                if np.isfinite(icc_metadata_by_axis[axis][model]["im_icc"])]
                if axis_hm_iccs and axis_im_iccs:
                    avg_hm_icc = np.mean(axis_hm_iccs)
                    avg_im_icc = np.mean(axis_im_iccs)
                    legend_text = f"\nAxis: {axis} | Avg HM-ICC: {avg_hm_icc:.3f} | Avg {'Pairwise' if COMPARISON_MODE == 'pairwise' else 'Aggregate'} IM-ICC: {avg_im_icc:.3f}"

            plt.figure(figsize=(10, 6))

            # Plot each method separately with 95% CI error bars
            for method in avg_axis_results["method"].unique():
                method_data = avg_axis_results[avg_axis_results["method"] == method]
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

            plt.ylabel("Absolute ICC Error (avg across models)", fontsize=12)
            plt.xlabel("Human Annotation Budget", fontsize=12)
            plt.title(f"{dataset} ICC Estimation{legend_text}\n(Averaged across models with 95% CI)", fontsize=11)
            plt.legend(title="Method", fontsize=11)
            plt.grid(alpha=0.3)
            plt.tight_layout()

            # Clean axis name for filename
            safe_axis_name = axis.replace("/", "-").replace("\\", "-").replace(" ", "_")
            plt.savefig(os.path.join(PLOTS_DIR, f"{dataset}_icc_estimation_by_axis_{safe_axis_name}.jpg"), dpi=300)
            plt.close()

            print(f"  Saved: {os.path.join(PLOTS_DIR, f'{dataset}_icc_estimation_by_axis_{safe_axis_name}.jpg')}")

    # =====================================
    # Summary table
    # =====================================
    print("\n" + "="*50)
    print("Average ICC Estimation Error by Method and Budget (Mean with 95% CI)")
    print("(Averaged across all models and axes)")
    print("="*50)

    # Create a nicer formatted table
    for method in avg_results["method"].unique():
        method_data = avg_results[avg_results["method"] == method].copy()
        method_data["formatted"] = method_data.apply(
            lambda row: f"{row['mean']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]", axis=1
        )
        print(f"\n{method}:")
        for _, row in method_data.iterrows():
            print(f"  Budget {int(row['budget']):2d}: {row['formatted']}")

if __name__ == "__main__":
    plot_results(results, results_by_axis, icc_metadata_all, icc_metadata_by_axis)