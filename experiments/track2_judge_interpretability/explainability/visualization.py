"""Visualization utilities for model interpretability."""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def box_plot_for_subset(data, subset_to_analyze, judges_names, attrib_column='attributions', output_path=None):
    os.makedirs(output_path, exist_ok=True)
    data_subset = data[data['category'] == subset_to_analyze]
    judge_attribution_labelled = {name: [] for name in judges_names}
    for i, row in data_subset.iterrows():
        attrib  = row[attrib_column]

        for j, attr in enumerate(attrib):
            judge_name = judges_names[j]
            judge_attribution_labelled[judge_name].append(attr)
    # violin plot of attributions
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=pd.DataFrame(judge_attribution_labelled))
    plt.title(f'Judge Attributions for {subset_to_analyze} Applications')
    plt.xticks(rotation=45)
    plt.ylabel('Attribution Value')
    if output_path: 
        plt.savefig(os.path.join(output_path, f"attrib_{subset_to_analyze}.png"))
    else: 
        plt.show()
def bar_plot_feature_importance(judge_names, importance_scores, output_path=None, title="Feature Importance based on Variance of Attribution Contributions"):
    plt.bar(judge_names, importance_scores)

    # tilt labels on x axis
    plt.xticks(rotation=45, ha='right')
    plt.show()
    if output_path:
        plt.savefig(os.path.join(output_path, "judge_importance_barplot.png"))
    else: 
        plt.show()
    return

def plot_feature_attributions(attributions, feature_names, output_path=None, title="Feature Attributions"):
    """
    Create a boxplot of feature attributions across different categories.
    
    Parameters:
        attributions (list): Attribution values list
        feature_names (list): List of feature names
        title (str): Plot title
    """
    attributions_dict = [{ feature: j for j,feature in zip(i, feature_names)} for i in attributions]
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=pd.DataFrame(attributions_dict))
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylabel('Attribution Value')
    plt.grid(True)
    if output_path: 
        plt.savefig(os.path.join(output_path, "judge_attributions_boxplot.png"))
    else: 
        plt.show()
    return 
