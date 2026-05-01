"""
Compare metric_matched_selection vs dev_metric_matched_selection.

Test if both functions return the same selected IDs when given identical inputs,
accounting for their different function signatures.
"""
import numpy as np
import pandas as pd
from src.utils.data_loading import load_judge_scores
from src.utils.match_metrics import (
    compute_icc_pingouin,
    compute_krippendorff_alpha,
    compute_ms_components,
)
from src.utils.selection_strategies import (
    metric_matched_selection,
    dev_metric_matched_selection
)

# Configuration
DATASET = "hanna"
MODEL_NAMES = ["claude-3.5-sonnet", "gpt-4.1", "gpt-5", "deepseek-r1", "gemini-2.5-pro"]
DATA_DIR = "data/judge_scores"
AXIS = "Coherence"
BUDGET = 50
N_TRIALS = 5
N_CANDIDATES = 20
SEED = 42

def _build_im_pairwise_df(df, model, model_names):
    """Build inter-model DataFrame for pairwise mode (model vs avg of others)."""
    other_models = [x for x in model_names if x != model]

    # Filter to only text_ids that have ALL models
    im_subset = df[df["model_name"].isin(model_names)]
    texts_per_model = im_subset.groupby("text_id")["model_name"].nunique()
    shared_text_ids = texts_per_model[texts_per_model == len(model_names)].index
    im_subset = im_subset[im_subset["text_id"].isin(shared_text_ids)]

    model_rows = (
        im_subset[im_subset["model_name"] == model][["text_id", "evaluation_score"]]
        .copy()
    )
    model_rows["model_name"] = model

    avg_rows = (
        im_subset[im_subset["model_name"].isin(other_models)]
        .groupby("text_id")["evaluation_score"]
        .mean()
        .reset_index()
    )
    avg_rows["model_name"] = "avg_other"
    ret = pd.concat(
        [model_rows[["text_id", "model_name", "evaluation_score"]],
         avg_rows[["text_id", "model_name", "evaluation_score"]]],
        ignore_index=True
    )
    return ret


def main():
    print("=" * 60)
    print("COMPARING metric_matched_selection vs dev_metric_matched_selection")
    print("=" * 60)

    # Load data
    print(f"\nLoading {DATASET} data...")
    EVALUATION_AXES = {
        "hanna": ["Coherence"],
    }
    df = load_judge_scores(DATASET, MODEL_NAMES, DATA_DIR, EVALUATION_AXES)

    # Filter to axis
    axis_df = df[df["evaluation_axis"] == AXIS]

    # Filter to shared text_ids (same as preprocessing in both scripts)
    num_models = axis_df["model_name"].nunique()
    texts_per_model = axis_df.groupby("text_id")["model_name"].nunique()
    shared_text_ids = texts_per_model[texts_per_model == num_models].index[:300]
    axis_df = axis_df[axis_df["text_id"].isin(shared_text_ids)]

    print(f"Axis: {AXIS}")
    print(f"Models: {MODEL_NAMES}")
    print(f"Shared text_ids: {len(shared_text_ids)}")
    print(f"Total rows: {len(axis_df)}")

    # Test for first model
    test_model = MODEL_NAMES[0]
    print(f"\nTesting with model: {test_model}")

    # Build pairwise IM dataframe
    im_full_df = _build_im_pairwise_df(axis_df, test_model, MODEL_NAMES)
    im_models = [test_model, "avg_other"]

    # Compute target ICC and Alpha
    im_icc = compute_icc_pingouin(im_full_df, models=im_models)
    im_alpha = compute_krippendorff_alpha(im_full_df, models=im_models)

    print(f"IM ICC target: {im_icc:.4f}")
    print(f"IM Alpha target: {im_alpha:.4f}")

    # Get available text_ids
    text_ids = shared_text_ids.values  # Keep as numpy array

    print(f"\n{'=' * 60}")
    print("Running comparison trials...")
    print(f"{'=' * 60}")

    all_match = True
    results = []

    # Test ICC
    print(f"\n--- Testing ICC (target={im_icc:.4f}) ---")
    for trial_idx in range(N_TRIALS):
        # Call dev version (takes array of targets/functions)
        ids_dev = dev_metric_matched_selection(
            text_ids=text_ids,
            k=BUDGET,
            im_full_df=im_full_df.copy(),
            target_metric_values=[im_icc],
            compute_metric_fns=[lambda df: compute_icc_pingouin(df, models=im_models)],
            alpha_weight=[1.0],
            seed=SEED + trial_idx,
            n_candidates=N_CANDIDATES
        )

        # Call regular version (takes single target/metric name)
        ids_regular = metric_matched_selection(
            text_ids=text_ids,
            k=BUDGET,
            im_full_df=im_full_df.copy(),
            target_value=im_icc,
            target_metric="icc",
            compute_ms_fn=compute_ms_components,
            compute_icc_fn=compute_icc_pingouin,
            compute_alpha_fn=compute_krippendorff_alpha,
            seed=SEED + trial_idx,
            n_candidates=N_CANDIDATES,
            im_models=im_models,
            forced_ids=None  # Batch mode (no forced IDs)
        )

        # Convert to sets for comparison
        set_regular = set(ids_regular) if ids_regular is not None else set()
        set_dev = set(ids_dev) if ids_dev is not None else set()

        # Check if identical
        ids_match = set_regular == set_dev
        overlap = len(set_regular & set_dev)
        overlap_pct = (overlap / BUDGET) * 100 if BUDGET > 0 else 0

        result = {
            "metric": "ICC",
            "trial": trial_idx,
            "match": ids_match,
            "overlap": overlap,
            "overlap_pct": overlap_pct,
            "regular_count": len(ids_regular) if ids_regular is not None else 0,
            "dev_count": len(ids_dev) if ids_dev is not None else 0
        }
        results.append(result)

        status = "✓ MATCH" if ids_match else f"✗ DIFFER (overlap: {overlap}/{BUDGET} = {overlap_pct:.1f}%)"
        print(f"  Trial {trial_idx}: {status}")

        if not ids_match:
            all_match = False
            # Show first few differences
            only_regular = set_regular - set_dev
            only_dev = set_dev - set_regular
            if only_regular:
                print(f"    Only in regular (first 3): {list(only_regular)[:3]}")
            if only_dev:
                print(f"    Only in dev (first 3): {list(only_dev)[:3]}")

    # Test Alpha
    print(f"\n--- Testing Alpha (target={im_alpha:.4f}) ---")
    for trial_idx in range(N_TRIALS):
        # Call dev version (takes array of targets/functions)
        ids_dev = dev_metric_matched_selection(
            text_ids=text_ids,
            k=BUDGET,
            im_full_df=im_full_df.copy(),
            target_metric_values=[im_alpha],
            compute_metric_fns=[lambda df: compute_krippendorff_alpha(df, models=im_models)],
            alpha_weight=[1.0],
            seed=SEED + trial_idx,
            n_candidates=N_CANDIDATES
        )

        # Call regular version (takes single target/metric name)
        ids_regular = metric_matched_selection(
            text_ids=text_ids,
            k=BUDGET,
            im_full_df=im_full_df.copy(),
            target_value=im_alpha,
            target_metric="alpha",
            compute_ms_fn=compute_ms_components,
            compute_icc_fn=compute_icc_pingouin,
            compute_alpha_fn=compute_krippendorff_alpha,
            seed=SEED + trial_idx,
            n_candidates=N_CANDIDATES,
            im_models=im_models,
            forced_ids=None  # Batch mode (no forced IDs)
        )

        # Convert to sets for comparison
        set_regular = set(ids_regular) if ids_regular is not None else set()
        set_dev = set(ids_dev) if ids_dev is not None else set()

        # Check if identical
        ids_match = set_regular == set_dev
        overlap = len(set_regular & set_dev)
        overlap_pct = (overlap / BUDGET) * 100 if BUDGET > 0 else 0

        result = {
            "metric": "Alpha",
            "trial": trial_idx,
            "match": ids_match,
            "overlap": overlap,
            "overlap_pct": overlap_pct,
            "regular_count": len(ids_regular) if ids_regular is not None else 0,
            "dev_count": len(ids_dev) if ids_dev is not None else 0
        }
        results.append(result)

        status = "✓ MATCH" if ids_match else f"✗ DIFFER (overlap: {overlap}/{BUDGET} = {overlap_pct:.1f}%)"
        print(f"  Trial {trial_idx}: {status}")

        if not ids_match:
            all_match = False
            # Show first few differences
            only_regular = set_regular - set_dev
            only_dev = set_dev - set_regular
            if only_regular:
                print(f"    Only in regular (first 3): {list(only_regular)[:3]}")
            if only_dev:
                print(f"    Only in dev (first 3): {list(only_dev)[:3]}")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    results_df = pd.DataFrame(results)

    for metric_name in ["ICC", "Alpha"]:
        metric_results = results_df[results_df["metric"] == metric_name]
        n_match = metric_results["match"].sum()
        avg_overlap = metric_results["overlap_pct"].mean()

        print(f"\n{metric_name}:")
        print(f"  Trials with exact match: {n_match}/{N_TRIALS}")
        print(f"  Average overlap: {avg_overlap:.1f}%")

    if all_match:
        print("\n✓ Both functions return IDENTICAL IDs across all trials!")
    else:
        print("\n✗ Functions return DIFFERENT IDs!")
        print("\nThis could explain the performance difference between the two scripts.")


if __name__ == "__main__":
    main()
