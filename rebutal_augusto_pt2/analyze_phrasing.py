#!/usr/bin/env python3
"""
Analyze Persona Phrasing Robustness

Loads results from the wording variations (base, v1, v2, v3, v4)
and computes how sensitive scores are to persona phrasing changes.

Usage:
    python rebutal_augusto_pt2/analyze_phrasing.py
    python rebutal_augusto_pt2/analyze_phrasing.py --results-dir rebutal_augusto_pt2/results
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

def load_variants(results_dir):
    results_dir = Path(results_dir)
    variants = {}
    
    for v in ["base", "v1", "v2", "v3", "v4"]:
        path = results_dir / f"{v}.pkl"
        if not path.exists():
            print(f"Warning: Missing results for variant {v} at {path}")
            continue
        with open(path, "rb") as fh:
            variants[v] = pickle.load(fh)
            
    if "base" not in variants:
        print("ERROR: baseline results ('base.pkl') required for comparison.")
        sys.exit(1)
        
    print(f"Loaded {len(variants)} wording variants: {list(variants.keys())}")
    return variants

def compute_mean_shifts(variants):
    base_scores = variants["base"]["scores"]["per_persona"]
    persona_names = list(base_scores.keys())
    
    results = []
    
    for v_name, v_data in variants.items():
        if v_name == "base": continue
        
        v_scores = v_data["scores"]["per_persona"]
        
        for persona in persona_names:
            base_arr = np.array(base_scores[persona], dtype=float)
            var_arr = np.array(v_scores[persona], dtype=float)
            
            valid = ~np.isnan(base_arr) & ~np.isnan(var_arr)
            if np.sum(valid) > 0:
                base_mean = np.mean(base_arr[valid])
                var_mean = np.mean(var_arr[valid])
                mean_shift = var_mean - base_mean
                
                # Correlation
                if np.sum(valid) > 2:
                    r_pearson, _ = stats.pearsonr(base_arr[valid], var_arr[valid])
                    r_spearman, _ = stats.spearmanr(base_arr[valid], var_arr[valid])
                else:
                    r_pearson = np.nan
                    r_spearman = np.nan
                    
                # Exact match %
                exact_match = np.sum(base_arr[valid] == var_arr[valid]) / np.sum(valid) * 100
                
                results.append({
                    "persona": persona,
                    "variant": v_name,
                    "base_mean": base_mean,
                    "variant_mean": var_mean,
                    "mean_shift": mean_shift,
                    "pearson": r_pearson,
                    "spearman": r_spearman,
                    "exact_match_pct": exact_match
                })
                
    return pd.DataFrame(results)

def plot_mean_shifts(df, output_dir):
    if df.empty: return
    
    # Pivot for easier plotting
    shift_pivot = df.pivot(index="persona", columns="variant", values="mean_shift")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    shift_pivot.plot(kind="barh", ax=ax, width=0.8, colormap="coolwarm")
    
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_xlabel("Mean Score Shift (Variant - Base)", fontsize=12)
    ax.set_ylabel("Persona", fontsize=12)
    ax.set_title("Impact of Phrasing on Persona Strictness", fontsize=14)
    ax.legend(title="Variant vs Baseline")
    
    fig.tight_layout()
    fig.savefig(output_dir / "phrasing_mean_shifts.png", dpi=150)
    plt.close(fig)
    print("  Saved phrasing_mean_shifts.png")


def plot_correlations(df, output_dir):
    if df.empty: return
    
    # Pivot for Spearman correlation to see how well ranking is preserved
    corr_pivot = df.pivot(index="persona", columns="variant", values="spearman")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_pivot.plot(kind="barh", ax=ax, width=0.8, colormap="viridis")
    
    ax.set_xlim(0, 1)
    ax.set_xlabel("Spearman Correlation vs Baseline", fontsize=12)
    ax.set_ylabel("Persona", fontsize=12)
    ax.set_title("How well phrasing variants preserve relative ranking", fontsize=14)
    ax.legend(title="Variant", loc="lower left")
    
    fig.tight_layout()
    fig.savefig(output_dir / "phrasing_correlations.png", dpi=150)
    plt.close(fig)
    print("  Saved phrasing_correlations.png")


def main():
    parser = argparse.ArgumentParser(description="Analyze persona phrasing sensitivity")
    parser.add_argument("--results-dir", type=str, default="rebutal_augusto_pt2/results")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    plots_dir = results_dir / "phrasing_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    variants = load_variants(results_dir)
    
    if len(variants) < 2:
        print("Need at least baseline and 1 variant to compare. Exiting.")
        sys.exit(0)
        
    print(f"\n📊 Analyzing {len(variants)} phrasing variations\n")

    # 1. Compute Shifts & Correlations per persona
    df_stats = compute_mean_shifts(variants)
    
    print("Aggregate metrics across all Personas & Variants (comparing each variant to baseline):")
    print(f"  Average Absolute Mean Shift: {np.abs(df_stats['mean_shift']).mean():.3f} points")
    print(f"  Average Pearson Correlation: {df_stats['pearson'].mean():.3f}")
    print(f"  Average Spearman Rank Corr:  {df_stats['spearman'].mean():.3f}")
    print(f"  Average Exact Score Match:   {df_stats['exact_match_pct'].mean():.1f}%")
    
    print("\nMost strict-shifting personas (Variant - Base < 0):")
    strict = df_stats.sort_values("mean_shift").head(5)
    for _, row in strict.iterrows():
        print(f"  {row['persona']} ({row['variant']}): {row['mean_shift']:.3f}")
        
    print("\nMost lenient-shifting personas (Variant - Base > 0):")
    lenient = df_stats.sort_values("mean_shift", ascending=False).head(5)
    for _, row in lenient.iterrows():
        print(f"  {row['persona']} ({row['variant']}): {row['mean_shift']:.3f}")
        
    print("\nPersonas most sensitive to phrasing (lowest rank correlation vs base):")
    sensitive = df_stats.sort_values("spearman").head(5)
    for _, row in sensitive.iterrows():
        print(f"  {row['persona']} ({row['variant']}): r_spearman = {row['spearman']:.3f}")

    print("\nGenerating visual plots...")
    plot_mean_shifts(df_stats, plots_dir)
    plot_correlations(df_stats, plots_dir)
    
    # Save statistics
    df_stats.to_csv(results_dir / "phrasing_stats.csv", index=False)
    
    summary = {
        "n_variants": len(variants),
        "aggregate_mean_shift_abs": float(np.abs(df_stats['mean_shift']).mean()),
        "aggregate_spearman": float(df_stats['spearman'].mean()),
        "aggregate_exact_match": float(df_stats['exact_match_pct'].mean()),
        "stats_by_persona_and_variant": df_stats.to_dict(orient="records")
    }
    
    with open(results_dir / "phrasing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Analysis complete!")
    print(f"   Summary: {results_dir}/phrasing_summary.json")
    print(f"   Table:   {results_dir}/phrasing_stats.csv")
    print(f"   Plots:   {plots_dir}/")

if __name__ == "__main__":
    main()
