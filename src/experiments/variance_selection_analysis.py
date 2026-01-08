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
datasets = ["hanna", "medval", "mslr", "summeval"]
model_names = ["claude-3.5-sonnet", "gpt-4.1", "gpt-4o-mini", "meta-llama-Llama-3.1-8B-Instruct", "gpt-5", "google-gemma-3-1b-it", "Qwen-Qwen2.5-7B-Instruct"]
evaluation_axes = {
    "hanna": ["Coherence", "Complexity", "Empathy", "Engagement", "Relevance", "Surprise"],
    "medval": ["Risk"],
    "mslr": ["fluency", "intervention", "outcome", "population"],
    "summeval": ["coherence", "consistency", "fluency", "relevance"]
}

dataset =  "medval" #"summeval" #"hanna" #"medval" #
DATA_DIR = "data/judge_scores"

# Comparison mode: "pairwise" or "aggregate"
# "pairwise": Compare each model vs average of other models (k=2 for both HM and IM)
# "aggregate": Use all models for inter-model (k=4), each model+human for HM (k=2)
COMPARISON_MODE = "pairwise"  # Toggle this between "pairwise" and "aggregate"

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
            rows.append(r)

        dfs.append(pd.DataFrame(rows))

df = pd.concat(dfs, ignore_index=True)

# Add human ("original") scores
text_info = (
    df[["text_id", "input_text", "source_text", "original_score"]]
    .drop_duplicates()
)

human_df = (
    text_info[["text_id", "original_score"]]
    .rename(columns={"original_score": "evaluation_score"})
)
human_df["model_name"] = "original"

df = pd.concat(
    [df[["text_id", "model_name", "evaluation_score"]], human_df],
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

    s = data.groupby("text_id")["evaluation_score"].mean()
    m = data.groupby("model_name")["evaluation_score"].mean()
    xbar = data["evaluation_score"].mean()

    msb = (k / (n - 1)) * ((s - xbar) ** 2).sum()

    # MSE: within-text variance
    mse_vals = []
    for text_id, group in data.groupby("text_id"):
        scores = group.set_index("model_name")["evaluation_score"]
        # For each rater in this text, compute squared deviation from rater's overall mean
        for rater, score in scores.items():
            if rater in m.index:
                mse_vals.append((score - m[rater]) ** 2)

    mse = np.mean(mse_vals) if mse_vals else np.nan

    return msb, mse

def icc_from_ms(msb, mse):
    """Calculate ICC(3,k) from MSB and MSE components."""
    if not np.isfinite(msb) or not np.isfinite(mse):
        return np.nan
    if msb <= 0:
        return np.nan
    # ICC(3,k) = (MSB - MSE) / MSB = 1 - MSE/MSB
    icc = (msb - mse) / msb

    # Clip ICC to be between 0 and 1
    # if not np.isnan(icc):
    #     icc = np.clip(icc, 0, 1)

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

        for m in model_names:
            other_models = [x for x in model_names if x != m]

            # Inter-model: this model vs avg of other models
            im_subset = df[df["model_name"].isin(model_names)].copy()
            # Create "avg_other" pseudo-rater
            im_avg_df = []
            for text_id in im_subset["text_id"].unique():
                text_data = im_subset[im_subset["text_id"] == text_id]
                # Current model score
                model_score = text_data[text_data["model_name"] == m]["evaluation_score"]
                if len(model_score) > 0:
                    im_avg_df.append({"text_id": text_id, "model_name": m, "evaluation_score": model_score.iloc[0]})
                # Avg of other models
                other_scores = text_data[text_data["model_name"].isin(other_models)]["evaluation_score"]
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
    n_trials=100
):
    results = []

    for model in model_names:
        # Get this model's specific inter-model variance components
        im_msb_target = per_model_variance[model]["im_msb"]
        im_mse_target = per_model_variance[model]["im_mse"]
        
        hm_full_df = df[df["model_name"].isin([model, "original"])]
        true_msb, true_mse = compute_ms_components(hm_full_df)
        true_icc = icc_from_ms(true_msb, true_mse)

        text_ids = hm_full_df["text_id"].unique()
        
        # Get inter-model data for variance matching
        if COMPARISON_MODE == "pairwise":
            # For pairwise mode, recreate the model vs avg(others) dataset
            other_models = [x for x in model_names if x != model]
            im_subset = df[df["model_name"].isin(model_names)].copy()
            
            im_full_df = []
            for text_id in im_subset["text_id"].unique():
                text_data = im_subset[im_subset["text_id"] == text_id]
                model_score = text_data[text_data["model_name"] == model]["evaluation_score"]
                if len(model_score) > 0:
                    im_full_df.append({"text_id": text_id, "model_name": model, "evaluation_score": model_score.iloc[0]})
                other_scores = text_data[text_data["model_name"].isin(other_models)]["evaluation_score"]
                if len(other_scores) > 0:
                    im_full_df.append({"text_id": text_id, "model_name": "avg_other", "evaluation_score": other_scores.mean()})
            im_full_df = pd.DataFrame(im_full_df)
        else:
            # For aggregate mode, use all models
            im_full_df = df[df["model_name"] != "original"]

        for k in budgets:
            # Random baseline
            random_errors = []
            for _ in range(n_trials):
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

            if random_errors:
                results.append({
                    "model": model,
                    "budget": k,
                    "method": "random",
                    "estimation_error": np.mean(random_errors)
                })

            # Variance-matched: select subset that best matches THIS MODEL's IM variance
            matched_errors = []
            for _ in range(n_trials):
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

            if matched_errors:
                results.append({
                    "model": model,
                    "budget": k,
                    "method": "variance_matched",
                    "estimation_error": np.mean(matched_errors)
                })

    return pd.DataFrame(results)

def main():
    """Main execution function."""
    print("\n" + "="*50)
    print("Running ICC estimation experiment...")
    print("="*50)
    results = evaluate_icc_estimators(df, model_names, per_model_variance)

    print(f"\nTotal results collected: {len(results)}")
    print(f"Results by method:")
    for method in results["method"].unique():
        count = len(results[results["method"] == method])
        print(f"  {method}: {count}")

    return results

if __name__ == "__main__":
    results = main()
else:
    # When imported, just compute variance alignment
    results = None

# -------------------------
# PLOTS
# -------------------------
# Compute 95% CI for each model, then average the CIs
from scipy import stats

def compute_model_cis(results_df):
    """Compute 95% CI for each model separately, then average."""
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

                # 95% CI using t-distribution
                if len(errors) > 1:
                    ci_half_width = stats.sem(errors) * stats.t.ppf(0.975, len(errors) - 1)
                else:
                    ci_half_width = 0

                ci_data.append({
                    "method": method,
                    "budget": budget,
                    "mean": mean_error,
                    "ci_lower": mean_error - ci_half_width,
                    "ci_upper": mean_error + ci_half_width,
                    "ci_half_width": ci_half_width
                })

    return pd.DataFrame(ci_data)

def plot_results(results):
    """Generate plots and summary tables."""
    if results is None or len(results) == 0:
        print("\nNo results to plot.")
        return

    avg_results = compute_model_cis(results)

    # Plot 1: Averaged across all models
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

    plt.ylabel("Absolute ICC Error (avg across models)", fontsize=12)
    plt.xlabel("Human Annotation Budget", fontsize=12)
    plt.title(f"{dataset} ICC Estimation: Random vs Variance-Matched\n(Averaged over all models with 95% CI)", fontsize=13)
    plt.legend(title="Method", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{dataset}_icc_estimation_comparison.jpg", dpi=300)
    plt.close()
    print(f"Saved averaged plot: {dataset}_icc_estimation_comparison.jpg")

    # Plot 2-N: Individual plots for each model
    for model in results["model"].unique():
        model_results = results[results["model"] == model]
        
        plt.figure(figsize=(10, 6))
        
        for method in model_results["method"].unique():
            method_data = model_results[model_results["method"] == method]
            
            # Group by budget and compute mean and CI
            budget_stats = []
            for budget in sorted(method_data["budget"].unique()):
                budget_data = method_data[method_data["budget"] == budget]
                errors = budget_data["estimation_error"].values
                
                if len(errors) > 0:
                    mean_error = errors.mean()
                    if len(errors) > 1:
                        ci_half_width = stats.sem(errors) * stats.t.ppf(0.975, len(errors) - 1)
                    else:
                        ci_half_width = 0
                    
                    budget_stats.append({
                        "budget": budget,
                        "mean": mean_error,
                        "ci_half_width": ci_half_width
                    })
            
            if budget_stats:
                budget_df = pd.DataFrame(budget_stats)
                plt.errorbar(
                    budget_df["budget"],
                    budget_df["mean"],
                    yerr=budget_df["ci_half_width"],
                    marker='o',
                    linewidth=2.5,
                    capsize=5,
                    capthick=2,
                    label=method,
                    alpha=0.8
                )
        
        plt.ylabel("Absolute ICC Error", fontsize=12)
        plt.xlabel("Human Annotation Budget", fontsize=12)
        plt.title(f"{dataset} ICC Estimation: {model}\n(Random vs Variance-Matched with 95% CI)", fontsize=13)
        plt.legend(title="Method", fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Clean model name for filename
        safe_model_name = model.replace("/", "-").replace("\\", "-")
        plt.savefig(f"{dataset}_icc_estimation_{safe_model_name}.jpg", dpi=300)
        plt.close()
        
        print(f"Saved plot for model: {model} -> {dataset}_icc_estimation_{safe_model_name}.jpg")

    print("\n" + "="*50)
    print("Average ICC Estimation Error by Method and Budget (Mean with 95% CI)")
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
    plot_results(results)