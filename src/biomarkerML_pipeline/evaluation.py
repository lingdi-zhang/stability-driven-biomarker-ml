import pandas as pd
from functools import reduce
from upsetplot import from_contents, plot,UpSet
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
import seaborn as sns
import  numpy as np

# =========================
# Metrics
# =========================
def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Metrics require both classes encoded as 0 and 1")
    if y_prob.shape != y_true.shape or not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError("Probabilities must match labels and be finite values between 0 and 1")
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }




def create_metric_table(model_output_dict):
    """
    Create formatted metric summary table.

    Parameters
    ----------
    model_output_dict : dict
        Example:
        {
            "EN": {
                "auc_mean": ...,
                "auc_sd": ...,
                ...
            }
        }

    Returns
    -------
    pd.DataFrame
    """
    df = pd.DataFrame.from_dict(
    model_output_dict,
    orient="index",
    columns=[
        "auc_mean", "auc_sd",
        "pr_mean", "pr_sd",
        "f1_mean", "f1_sd"
    ]).reset_index()

    df = df.rename(columns={"index": "model"})
    df["AUC"] = df["auc_mean"].round(3).map("{:.3f}".format) + " ± " + df["auc_sd"].round(3).map("{:.3f}".format)
    df["PR"]  = df["pr_mean"].round(3).map("{:.3f}".format)  + " ± " + df["pr_sd"].round(3).map("{:.3f}".format)
    df["F1"]  = df["f1_mean"].round(3).map("{:.3f}".format)  + " ± " + df["f1_sd"].round(3).map("{:.3f}".format)
    summary = df[["model", "AUC", "PR", "F1"]]
    return summary

# ===================================================
# plot feature stability and importance for each model
# ===================================================

def summarize_feature_importance(feature_importance, nfold):
    """Summarize measured scores; selection frequency uses all outer folds.

    Actual zero scores remain measurements. Missing scores are excluded from
    mean and SD. SD is undefined for a single measured fold.
    """
    agg = (feature_importance.groupby("gene").agg(
          mean_importance=("abs_coef", "mean"),
          sd_importance=("abs_coef", "std"),
          measured_folds=("abs_coef", "count"))
      .reset_index())
    appearances = feature_importance.loc[feature_importance['selected']].groupby('gene')['fold'].nunique()
    agg['appearances'] = agg['gene'].map(appearances).fillna(0).astype(int)
    agg['percent_appearance']=agg['appearances']/nfold
    agg["stability_weighted_importance"] = (
    agg["mean_importance"] * agg["percent_appearance"])
    
    return agg


def analyze_feature_stability_importance_each_model(model_name,feature_importance,nfold,cross_validation_selection,output_dir=None):
    agg = summarize_feature_importance(feature_importance, nfold)
    agg.to_csv(output_dir / f"Feature_stability_and_importance_summary_{model_name}.csv", index=False)

    fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 6),
    gridspec_kw={"width_ratios": [1.0, 1.9]})

    #####stability vs importance all features
    fold_cutoff = cross_validation_selection
#    importance_cutoff = agg["mean_importance"].quantile(0.75)

    axes[0].scatter(
    agg["mean_importance"],
    agg["percent_appearance"]* 100,
    alpha=0.75,
    s=45)

    axes[0].axhline(fold_cutoff* 100, linestyle="--", linewidth=2)
#    axes[0].axvline(importance_cutoff, linestyle="--", linewidth=1)

    axes[0].set_xlabel("Mean |importance|\nacross measured folds",fontsize=14,fontweight="bold")
    axes[0].set_ylabel("Fold appearance (%)",fontsize=14,fontweight="bold")
    axes[0].tick_params(axis="both", labelsize=12)
    axes[0].set_title("All features",fontsize=16, fontweight="bold",pad=12)

    ####with the selected features
    selected = agg[
    (agg["percent_appearance"] >= fold_cutoff) & (agg['appearances'] > 0)]
#    (agg["mean_importance"] >= importance_cutoff)]

#    selected = selected.sort_values(["percent_appearance", "mean_importance"],ascending=[False, False])
    selected = selected.sort_values(
    "percent_appearance",
    ascending=False)

    y_pos = np.arange(len(selected))

    axes[1].barh(
    y_pos,
    selected["percent_appearance"] * 100,
    height=0.65)
    
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_ylim(-0.7, len(selected) - 0.2)

    axes[1].set_yticks([])
    axes[1].set_xlabel("Fold appearance (%)",fontsize=14,fontweight="bold")
    axes[1].tick_params(axis="x", labelsize=12)
    axes[1].set_title(
    "Stability prioritized biomarkers",
    fontsize=16,
    fontweight="bold",
    pad=12)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    feature_x = -5
    importance_x = 104
    header_y = len(selected) - 0.35

    for i, (_, row) in enumerate(selected.iterrows()):
        axes[1].text(
        feature_x,
        i,
        row["gene"],
        ha="right",
        va="center",
        fontsize=12,
        clip_on=False)

        axes[1].text(
        importance_x,
        i,
        (f'{row["mean_importance"]:.3f} ± '
         f'{row["sd_importance"]:.3f}' if pd.notna(row['sd_importance'])
         else f'{row["mean_importance"]:.3f} ± N/A') + f' (n={int(row["measured_folds"])})',
        ha="left",
        va="center",
        fontsize=12,
        clip_on=False)

    axes[1].text(
    feature_x,
    header_y,
    "Feature",
    ha="right",
    va="bottom",
    fontsize=12,
    fontweight="bold",
    clip_on=False)

#    axes[1].text(
#    50,
#    header_y,
#    "Fold appearance (%)",
#    ha="center",
#    va="bottom",
#    fontsize=12,
#    fontweight="bold")

    axes[1].text(
    importance_x,
    header_y,
    "Measured-fold mean\n|importance| ± SD (n)",
    ha="left",
    va="bottom",
    fontsize=12,
    fontweight="bold",
    clip_on=False)

    axes[1].grid(
    axis="x",
    linestyle="--",
    alpha=0.25)

    axes[1].set_axisbelow(True)
    
    fig.suptitle(
    f"Feature Stability and Biomarker Prioritization ({model_name})",
    fontsize=22,
    fontweight="bold",
    y=1.03)

    plt.subplots_adjust(left=0.08, right=0.82, wspace=0.55)
    plt.savefig(output_dir / f"Feature_stability_and_importance_summary_{model_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    return selected

# ===================================
# plot features selected across models
# ===================================
def aggregate_biomarkers(model_biomarkers, appearance_threshold=0.5):
    """Build aggregate groups without modifying model-specific lists."""
    from collections import Counter
    lists = list(model_biomarkers.values())
    if not lists:
        raise ValueError("At least one model is required for consensus analysis")
    counts = Counter(gene for genes in lists for gene in set(genes))
    return {
        "union": sorted(counts),
        "intersection": sorted(reduce(set.intersection, map(set, lists))),
        "consensus_features_for_appearance_threshold": sorted(
            gene for gene, count in counts.items()
            if count / len(lists) >= appearance_threshold),
    }


def analyze_features_acorss_models(gene_list,output_dir=None):
    aggregates = aggregate_biomarkers(gene_list)
    union = aggregates["union"]
    rows = [{"group_type": "model", "model": model, "genes": ", ".join(genes)}
            for model, genes in gene_list.items()]
    rows.extend({"group_type": "aggregate", "model": group, "genes": ", ".join(aggregates[group])}
                for group in ("union", "intersection"))
    gene_df = pd.DataFrame(rows)
    if not union:
        (output_dir / "feature_selection_consensus_heatmap.png").unlink(missing_ok=True)
        return gene_df

    mat = pd.DataFrame(0, index=union, columns=gene_list.keys())
    for model, features in gene_list.items():
        mat.loc[features, model] = 1
    # Use a separate series so a model named n_models remains a model column.
    order = mat.sum(axis=1).sort_values(ascending=False, kind="stable").index
    plot_mat = mat.loc[order]

    plt.figure(figsize=(7, max(3, 0.35 * plot_mat.shape[0])))
    sns.heatmap(
        plot_mat,
        annot=plot_mat.replace({1: "✓", 0: ""}),
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=0.5,
        linecolor="white")

    plt.title("Feature selection consensus across models")
    plt.xlabel("Model")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_selection_consensus_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()
    return gene_df
