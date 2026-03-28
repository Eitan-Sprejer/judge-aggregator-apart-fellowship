import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import sys

def main():
    results_dir = Path("rebutal_augusto_pt2/results")
    plots_dir = results_dir / "phrasing_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    stats_csv = results_dir / "phrasing_stats.csv"
    if not stats_csv.exists():
        print("phrasing_stats.csv not found.")
        return
        
    df_stats = pd.read_csv(stats_csv)

    print("Generating extra plots...")
    
    # --- 1. Exact Match Heatmap ---
    match_pivot = df_stats.pivot(index="persona", columns="variant", values="exact_match_pct")
    personas = match_pivot.index.values
    variant_names = match_pivot.columns.values
    data = match_pivot.values
    
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(data, cmap="YlGnBu", aspect="auto")
    fig.colorbar(cax, label="Exact Match Percentage (%) vs Baseline")
    
    ax.set_xticks(np.arange(len(variant_names)))
    ax.set_yticks(np.arange(len(personas)))
    ax.set_xticklabels(variant_names)
    ax.set_yticklabels(personas)
    ax.set_title("Exact Match Percentage Across Personas and Variants", pad=20)
    
    # Annotate heatmap
    for idx_y in range(len(personas)):
        for idx_x in range(len(variant_names)):
            val = data[idx_y, idx_x]
            color = "white" if val > (data.max() + data.min()) / 2 else "black"
            ax.text(idx_x, idx_y, f"{val:.1f}%", ha="center", va="center", color=color)
            
    fig.tight_layout()
    fig.savefig(plots_dir / "phrasing_exact_matches.png", dpi=150)
    plt.close(fig)
    print("  Saved phrasing_exact_matches.png")
    
    # --- 2. Score Variance Boxplots ---
    variants = {}
    for v in ["base", "v1", "v2", "v3", "v4"]:
        pkl_path = results_dir / f"{v}.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as fh:
                variants[v] = pickle.load(fh)
                
    records = []
    for v_name, v_data in variants.items():
        for persona, scores in v_data["scores"]["per_persona"].items():
            for s in scores:
                if s is not None and not pd.isna(s):
                    try:
                        val = float(s)
                        records.append({"Variant": v_name, "Persona": persona, "Score": val})
                    except (ValueError, TypeError):
                        pass
                    
    df_scores = pd.DataFrame(records)
    
    # Try seaborn if available, otherwise fallback to matplotlib
    try:
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(16, 8))
        sns.boxplot(data=df_scores, x="Persona", y="Score", hue="Variant", ax=ax, palette="Set2")
        ax.set_title("Distribution of Raw Scores Given by Each Persona Across Phrasings")
        plt.xticks(rotation=45, ha='right')
        plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
        fig.tight_layout()
        fig.savefig(plots_dir / "phrasing_score_distributions.png", dpi=150)
        plt.close(fig)
        print("  Saved phrasing_score_distributions.png (High Quality)")
    except ImportError:
        print("  Seaborn not installed, skipping advanced boxplot...")
        
    # --- 3. Overall strictness density/histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for v_name, df_v in df_scores.groupby("Variant"):
        # Plot kde over average persona scores per variant
        variant_means = []
        for p, df_p in df_v.groupby("Persona"):
            variant_means.append(df_p["Score"].mean())
        # Let's plot histograms of all scores across all personas for the variant
        ax.hist(df_v["Score"].values, bins=20, alpha=0.4, label=v_name, density=True)
        
    ax.legend(title="Variant")
    ax.set_xlabel("Score (1 to 10)")
    ax.set_ylabel("Density / Frequency")
    ax.set_title("Overall Score Distributions Across All Personas by Variant")
    fig.tight_layout()
    fig.savefig(plots_dir / "phrasing_overall_density.png", dpi=150)
    plt.close(fig)
    print("  Saved phrasing_overall_density.png")
    
if __name__ == "__main__":
    main()
