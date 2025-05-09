import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import numpy as np

score_fields = ["summary_score", "experience_score", "education_score", "skills_score", "languages_score", "other_score"]

# --- Score Normalization & Ranking ---

def normalize_scores(df, method_name):
    normalized = df.copy()
    for field in score_fields:
        if field in normalized.columns:
            min_val = normalized[field].min()
            max_val = normalized[field].max()
            if max_val != min_val:
                normalized[f"{method_name}_{field}"] = (
                    normalized[field] - min_val
                ) / (max_val - min_val)
            else:
                normalized[f"{method_name}_{field}"] = 0.5
    return normalized[["id"] + [f"{method_name}_{f}" for f in score_fields]]


def load_and_rank(file_path, method_name):

    df = pd.read_excel(file_path)

    # Verify required column
    if "summary_score" not in df.columns:
        raise ValueError(f"{file_path} is missing 'summary_score' column")

    # Sort by summary_score, then composite_score if available
    if "composite_score" in df.columns:
        df = df.sort_values(by=["summary_score", "composite_score"], ascending=[False, False])
    else:
        df = df.sort_values(by="summary_score", ascending=False)

    df = df.reset_index(drop=True)
    df[f"{method_name}_rank"] = df.index + 1

    return df[["id", f"{method_name}_rank"]]

def sort_project_results(df):
    df_sorted = df.sort_values(
        by=["summary_score", "composite_score"], 
        ascending=[False, False]
    ).reset_index(drop=True)
    
    df_sorted["project_rank"] = df_sorted.index + 1
    return df_sorted

def load_website_results(path):
    df = pd.read_excel(path)
    
    df_sorted = df.sort_values(
        by=["summary_score", "experience_score"], 
        ascending=[False, False]
    ).reset_index(drop=True)
    
    df_sorted["website_rank"] = df_sorted.index + 1
    return df_sorted[["id", "website_rank"]]

def compare_ranks(project_df, website_df):
    """
    Merge project results and website ATS rankings by 'id'.
    Adds a 'match' column indicating if the ranks are equal.
    """
    merged = pd.merge(project_df, website_df, on="id")
    merged["ranks_match"] = merged["project_rank"] == merged["website_rank"]
    return merged

def print_ranking_comparison(df):
    print("\n Ranking Comparison (All Methods):")
    display_cols = ["id", "website_rank", "tot_rank", "oneshot_rank"]
    print(df[display_cols])


def merge_all_ranks(website_df, tot_df, oneshot_df, resume_count=10):
    """
    Merges rank outputs from three methods into a single comparison DataFrame.

    Parameters:
        website_df (pd.DataFrame): Contains 'id' and 'website_rank'
        tot_df (pd.DataFrame): Contains 'id' and 'tot_rank'
        oneshot_df (pd.DataFrame): Contains 'id' and 'oneshot_rank'
        resume_count (int): Total number of resumes evaluated

    Returns:
        pd.DataFrame: Merged DataFrame of all ranks (with NaNs for missing entries)
    """
    # Initialize full ID list as base
    merged_df = pd.DataFrame({"id": range(1, resume_count + 1)})

    # Merge each method's ranks (left join to preserve all IDs)
    merged_df = merged_df.merge(website_df, on="id", how="left")
    merged_df = merged_df.merge(tot_df, on="id", how="left")
    merged_df = merged_df.merge(oneshot_df, on="id", how="left")

    # Optional: Add agreement flags
    merged_df["website_vs_tot"] = merged_df["website_rank"] == merged_df["tot_rank"]
    merged_df["tot_vs_oneshot"] = merged_df["tot_rank"] == merged_df["oneshot_rank"]
    merged_df["website_vs_oneshot"] = merged_df["website_rank"] == merged_df["oneshot_rank"]

    return merged_df


# --- Visualization Functions ---

def plot_rank_scatter(df):
    """
    Scatter plot comparing project_rank vs website_rank.
    Diagonal line indicates perfect agreement.
    """
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x="website_rank", y="tot_rank", s=100)

    plt.plot([1, len(df)], [1, len(df)], 'r--', label="Perfect match")
    plt.title("Scatter Plot: Project vs Website Rankings")
    plt.xlabel("Website ATS Rank")
    plt.ylabel("Project (ToT) Rank")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_normalized_score_comparison_all(website_path, tot_path, oneshot_path):
    """
    Reads all three result files, normalizes key summary/composite scores, and plots them side-by-side.
    """
    website_df = pd.read_excel(website_path)
    tot_df = pd.read_excel(tot_path)
    oneshot_df = pd.read_excel(oneshot_path)

    combined = website_df[["id", "summary_score"]].rename(columns={"summary_score": "website_score"})
    combined = combined.merge(
        tot_df[["id", "composite_score"]].rename(columns={"composite_score": "tot_score"}), on="id"
    ).merge(
        oneshot_df[["id", "summary_score"]].rename(columns={"summary_score": "oneshot_score"}), on="id"
    )

    for col in ["website_score", "tot_score", "oneshot_score"]:
        min_val = combined[col].min()
        max_val = combined[col].max()
        combined[f"norm_{col}"] = (combined[col] - min_val) / (max_val - min_val + 1e-9)

    combined = combined.sort_values("id", ascending=True)

    plt.figure(figsize=(10, 5))
    sns.lineplot(x="id", y="norm_website_score", data=combined, label="Website ATS (Normalized)", marker="o")
    sns.lineplot(x="id", y="norm_tot_score", data=combined, label="ToT Project (Normalized)", marker="o")
    sns.lineplot(x="id", y="norm_oneshot_score", data=combined, label="One-Shot (Normalized)", marker="o")

    plt.title("Normalized Score Comparison Across Methods")
    plt.xlabel("Resume ID")
    plt.ylabel("Normalized Score [0-1]")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ranks_by_resume_scatter(website_path, tot_path, oneshot_path):

    def rank_dataframe(path, method):
        df = pd.read_excel(path)
        if "composite_score" in df.columns:
            df = df.sort_values(by=["summary_score", "composite_score"], ascending=[False, False])
        else:
            df = df.sort_values(by="summary_score", ascending=False)
        df[f"{method}_rank"] = range(1, len(df) + 1)
        return df[["id", f"{method}_rank"]]

    website = rank_dataframe(website_path, "website")
    tot = rank_dataframe(tot_path, "tot")
    oneshot = rank_dataframe(oneshot_path, "oneshot")
    merged = website.merge(tot, on="id").merge(oneshot, on="id")

    plt.figure(figsize=(10, 6))

    # Slight X-axis offsets for overlapping points
    plt.scatter(merged["id"] - 0.15, merged["website_rank"], color="red", label="Website", s=80, marker='o')
    plt.scatter(merged["id"],        merged["tot_rank"],     color="blue", label="ToT", s=80, marker='o')
    plt.scatter(merged["id"] + 0.15, merged["oneshot_rank"], color="green", label="One-Shot", s=80, marker='o')

    plt.gca().invert_yaxis()
    plt.xlabel("Resume ID")
    plt.ylabel("Rank (lower is better)")
    plt.title("Resume Rankings by Method")
    plt.xticks(merged["id"])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_featurewise_correlation(website_path, tot_path, oneshot_path):
    # Load and normalize
    website_df = pd.read_excel(website_path)
    tot_df = pd.read_excel(tot_path)
    oneshot_df = pd.read_excel(oneshot_path)

    website_norm = normalize_scores(website_df, "website")
    tot_norm = normalize_scores(tot_df, "tot")
    oneshot_norm = normalize_scores(oneshot_df, "oneshot")

    # Merge all methods on resume ID
    merged = website_norm.merge(tot_norm, on="id").merge(oneshot_norm, on="id")

    correlations = []
    for field in score_fields:
        w = merged[f"website_{field}"]
        t = merged[f"tot_{field}"]
        o = merged[f"oneshot_{field}"]

        # Compute and safely handle NaNs in correlations
        wt_corr = spearmanr(w, t).correlation
        wo_corr = spearmanr(w, o).correlation
        to_corr = spearmanr(t, o).correlation

        correlations.append({
            "Feature": field.replace("_score", "").capitalize(),
            "Website–ToT": wt_corr if not np.isnan(wt_corr) else 0.0,
            "Website–OneShot": wo_corr if not np.isnan(wo_corr) else 0.0,
            "ToT–OneShot": to_corr if not np.isnan(to_corr) else 0.0
        })

    # Format as DataFrame
    corr_df = pd.DataFrame(correlations).set_index("Feature")

    # Plot annotated heatmap with proper spacing
    plt.figure(figsize=(8, 6), constrained_layout=True)
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5, linecolor='white')
    plt.title("Spearman Correlation by Feature Across Methods")
    plt.ylabel("Feature")
    plt.xlabel("Model Pair")
    plt.show()