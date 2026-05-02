import os
import re

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sb

plotpath = "./results/05_01_natalie/plots/cross_metric_matched"
results_path = "./results/05_01_natalie/metric_matched_subsets" 
# updated_results_path = "./results/alyssa_metric_matched_subsets"
dirs = os.listdir(results_path)

# metrics to use in the full graph
all_match_metric_names = [
    "icc",
    "alpha",
    "msb+msre",
    "spearman",
    "kendalltau"
]

for dataset in dirs:
    print(dataset)
    dataset_path = os.path.join(results_path, dataset)
    files = os.listdir(dataset_path)
    os.makedirs(os.path.join(plotpath, dataset, "metric_matched"), exist_ok=True)
    random_added = False
    df_list = []
    for file in files:
        extract = re.match("^cross_metric_(.*?)_match_results.csv$", file)
        if not extract:
            continue 
        matched_metric_name = extract.group(1)
        df = pd.read_csv(os.path.join(dataset_path, file))
        # if os.path.exists(os.path.join(updated_results_path, dataset, f"cross_metric_{matched_metric_name}_match_results.csv")):
        #     add_df = pd.read_csv(os.path.join(updated_results_path, dataset, f"cross_metric_{matched_metric_name}_match_results.csv"))
        #     df = pd.concat([df, add_df.loc[add_df["est_metric"] == "kendalltau"]], ignore_index=True)
        # else:
        #     pass

        df["est_error"] = np.where(df["est_metric"].isin(["icc", "pearson", "alpha", "spearman", "kendalltau"]), df["est_error"].clip(0, 2), df["est_error"]) 

        df["method"] = np.where(df["method"]!="random", df["method"] + "_" + df["match_metric"].replace({np.nan: ""}), df["method"])
        if not random_added:
            df_list.append(df)
            random_added = True
        else:
            df_list.append(df.loc[df["method"] != "random"])
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
    
    ## ALL MATCHED ON ONE GRAPH
    full_df = pd.concat(df_list, ignore_index=True)
    for est_metric in est_metrics:
        for model_name in model_names:
            full_df_metric_model = full_df.loc[np.logical_and(full_df["est_metric"] == est_metric, 
                                                              full_df["model"] == model_name)]

            ev_ax_list = full_df_metric_model["axis"].unique().tolist()
            g = sb.FacetGrid(data=full_df_metric_model.loc[full_df_metric_model["match_metric"].isin([*all_match_metric_names, np.nan])], col="axis", col_wrap=min(3, len(ev_ax_list)), hue="method", palette="Set1")

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
            g.figure.suptitle(f"{model_name} {est_metric} estimation error")
            g.tight_layout()

            plt.savefig(
                os.path.join(plotpath, dataset, "metric_matched", f"{model_name}_all_match_metrics_{est_metric}_estimation_error.png"), dpi=300
            )
            plt.close()

        ev_ax_list = full_df["axis"].unique().tolist()

        g = sb.lineplot(data=full_df.loc[np.logical_and(full_df["est_metric"] == est_metric,
                                full_df["match_metric"].isin([*all_match_metric_names, np.nan]))], 
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
        g.set_title(f"{est_metric} average estimation error")
        plt.savefig(
            os.path.join(plotpath, dataset, "metric_matched", f"average_all_match_metrics_matched_{est_metric}_estimation_error.png"), dpi=300
        )
        plt.close()