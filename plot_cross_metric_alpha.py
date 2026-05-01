# """
# Plot cross-metric alpha match results comparing random vs metric-matched selection.
# """
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
# from src.utils.plotting import compute_bootstrap_cis, create_estimation_error_plot

# # Load the results
# results = pd.read_csv('/Users/alyssaunell/code/SmartSample_local/results/04_16/cross_metric_alpha_match_results.csv')

# # Filter for alpha estimation with:
# # 1. Random method with est_metric=alpha
# # 2. Metric-match method with match_metric=alpha and est_metric=alpha
# alpha_results = results[
#     ((results['method'] == 'random') & (results['est_metric'] == 'alpha')) |
#     ((results['method'] == 'metric_match') &
#      (results['match_metric'] == 'alpha') &
#      (results['est_metric'] == 'alpha'))
# ].copy()

# # Rename for consistency with plotting functions
# alpha_results.rename(columns={'est_error': 'estimation_error'}, inplace=True)

# # Compute bootstrap CIs
# avg_results = compute_bootstrap_cis(alpha_results, n_bootstrap=1000)

# # Create the plot
# output_path = create_estimation_error_plot(
#     avg_results,
#     title="Alpha Estimation: Random vs Metric-Matched (Alpha)",
#     ylabel="Absolute Alpha Error (avg across models & axes)",
#     filename='/Users/alyssaunell/code/SmartSample_local/results/04_16/alpha_comparison_plot.jpg',
#     legend_text="\n(Averaged over all models & axes with 95% CI)"
# )

# print(f"Plot saved to: {output_path}")

# # Print summary statistics
# print("\n" + "=" * 50)
# print("Average Alpha Estimation Error by Method and Budget")
# print("=" * 50)
# for method in avg_results["method"].unique():
#     method_data = avg_results[avg_results["method"] == method].copy()
#     print(f"\n{method}:")
#     for _, row in method_data.iterrows():
#         print(f"  Budget {int(row['budget']):2d}: {row['mean']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")


import os
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

# Load predetermined CSV file
csv_file_path = '/Users/alyssaunell/code/SmartSample_local/results/04_16/cross_metric_alpha_match_results.csv'
plotpath = "./plots/alyssa_metric_match"

# Original implementation (commented out):
# results_path = "/Users/alyssaunell/code/SmartSample_local/results/04_16/" # # "./results/metric_matched_subsets" # "./results/train_test_split_metric_matched_subsets"
# # updated_results_path = "./results/alyssa_metric_matched_subsets"
# dirs = os.listdir(results_path)

# metrics to use in the full graph
all_match_metric_names = [
    "icc",
    "alpha",
    "msb+msre",
    "spearman",
    "kendalltau"
]

# Modified: Load single predetermined file instead of iterating through directories
dataset = "cross_metric_alpha"  # Fixed dataset name
matched_metric_name = "alpha"  # Extract from filename
df = pd.read_csv(csv_file_path)

# Original loop (commented out):
# for dataset in dirs:
#     print(dataset)
#     dataset_path = os.path.join(results_path, dataset)
#     files = os.listdir(dataset_path)

print(dataset)
os.makedirs(os.path.join(plotpath, dataset, "metric_matched"), exist_ok=True)

# Original nested loop (commented out):
# random_added = False
# df_list = []
# for file in files:
#     extract = re.match("^cross_metric_(.*?)_match_results.csv$", file)
#     if not extract:
#         continue
#     matched_metric_name = extract.group(1)
#     df = pd.read_csv(os.path.join(dataset_path, file))
#     if os.path.exists(os.path.join(updated_results_path, dataset, f"cross_metric_{matched_metric_name}_match_results.csv")):
#         add_df = pd.read_csv(os.path.join(updated_results_path, dataset, f"cross_metric_{matched_metric_name}_match_results.csv"))
#         df = pd.concat([df, add_df.loc[add_df["est_metric"] == "kendalltau"]], ignore_index=True)
#     else:
#         pass

df["est_error"] = np.where(df["est_metric"].isin(["icc", "pearson", "alpha", "spearman", "kendalltau"]), df["est_error"].clip(0, 2), df["est_error"])

df["method"] = np.where(df["method"]!="random", df["method"] + "_" + df["match_metric"].replace({np.nan: ""}), df["method"])

# Original random_added logic (commented out - not needed for single file):
# if not random_added:
#     df_list.append(df)
#     random_added = True
# else:
#     df_list.append(df.loc[df["method"] != "random"])
# df.drop("match_metric", axis=1, inplace=True)

est_metrics = df["est_metric"].unique()
model_names = df["model"].unique()

for est_metric in est_metrics:
    for model_name in model_names:
        df_metric_model = df.loc[np.logical_and(df["est_metric"] == est_metric, df["model"] == model_name)]

        ev_ax_list = df_metric_model["axis"].unique().tolist()
        g = sb.FacetGrid(data=df_metric_model, col="axis", col_wrap=min(3, len(ev_ax_list)), hue="method", palette="Set1")

        g.map_dataframe(sb.lineplot,
                        x="budget",
                        y="est_error",
                        markers=True,
                        marker="o",
                        estimator="mean",
                        err_style="bars",
                        errorbar=("ci", 95),
                        err_kws={'capsize': 3})
        g.add_legend()
        g.figure.suptitle(f"{matched_metric_name} metric matching:\n {model_name} {est_metric} estimation error")
        g.tight_layout()

        plt.savefig(
            os.path.join(plotpath, dataset, "metric_matched", f"{model_name}_{matched_metric_name}_matched_{est_metric}_estimation_error.png"), dpi=300
        )
        plt.close()

    ev_ax_list = df["axis"].unique().tolist()

    g = sb.lineplot(data=df.loc[df["est_metric"] == est_metric],
                x="budget",
                y="est_error",
                hue="method",
                markers=True,
                marker="o",
                palette="Set1",
                estimator="mean",
                err_style="bars",
                errorbar=("ci", 95),
                err_kws={'capsize': 3},
                legend=True)
    g.set_title(f"{matched_metric_name} metric matching: {est_metric} average estimation error")
    plt.savefig(
        os.path.join(plotpath, dataset, "metric_matched", f"average_{matched_metric_name}_matched_{est_metric}_estimation_error.png"), dpi=300
    )
    plt.close()

# Original "ALL MATCHED ON ONE GRAPH" section (commented out - not needed for single file):
# ## ALL MATCHED ON ONE GRAPH
# full_df = pd.concat(df_list, ignore_index=True)
# for est_metric in est_metrics:
#     for model_name in model_names:
#         full_df_metric_model = full_df.loc[np.logical_and(full_df["est_metric"] == est_metric,
#                                                           full_df["model"] == model_name)]
#
#         ev_ax_list = full_df_metric_model["axis"].unique().tolist()
#         g = sb.FacetGrid(data=full_df_metric_model.loc[full_df_metric_model["match_metric"].isin([*all_match_metric_names, np.nan])], col="axis", col_wrap=min(3, len(ev_ax_list)), hue="method", palette="Set1")
#
#         g.map_dataframe(sb.lineplot,
#                         x="budget",
#                         y="est_error",
#                         markers=True,
#                         marker="o",
#                         estimator="mean",
#                         err_style="bars",
#                         errorbar=("ci", 95),
#                         err_kws={'capsize': 3})
#         g.add_legend()
#         g.figure.suptitle(f"{model_name} {est_metric} estimation error")
#         g.tight_layout()
#
#         plt.savefig(
#             os.path.join(plotpath, dataset, "metric_matched", f"{model_name}_all_match_metrics_{est_metric}_estimation_error.png"), dpi=300
#         )
#         plt.close()
#
#     ev_ax_list = full_df["axis"].unique().tolist()
#
#     g = sb.lineplot(data=full_df.loc[np.logical_and(full_df["est_metric"] == est_metric,
#                             full_df["match_metric"].isin([*all_match_metric_names, np.nan]))],
#                 x="budget",
#                 y="est_error",
#                 hue="method",
#                 markers=True,
#                 marker="o",
#                 palette="Set1",
#                 estimator="mean",
#                 err_style="bars",
#                 errorbar=("ci", 95),
#                 err_kws={'capsize': 3},
#                 legend=True)
#     g.set_title(f"{est_metric} average estimation error")
#     plt.savefig(
#         os.path.join(plotpath, dataset, "metric_matched", f"average_all_match_metrics_matched_{est_metric}_estimation_error.png"), dpi=300
#     )
#     plt.close()

print(f"\nPlots saved to: {os.path.join(plotpath, dataset, 'metric_matched')}")