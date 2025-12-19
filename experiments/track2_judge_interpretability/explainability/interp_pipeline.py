import argparse
import os
import pickle
import torch
import sys
# Set base path as one level above
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "....")))
from pipeline.core.aggregator_training import MLPTrainer, SingleLayerMLP
import pandas as pd
from fetch_attributions import compute_input_x_gradient_batch, contribution_based_importance, gam_interp
from ice_analysis import plot_h_statistics
from visualization import bar_plot_feature_importance, box_plot_for_subset
def run_interpretability_pipeline(model_path, model_type,importance_analysis,  data_path, output_path, logging):
    print(f"Running interpretability pipeline with model: {model_path}")
    # Load data
    print(f"Loading data from: {data_path}")
    data = pd.read_pickle(data_path)
    # Load from the text file
    with open("judge_names.txt", "r") as f:
        judge_names = [line.strip() for line in f]

    if model_type == "GAM":
        print("Using Generalized Additive Model (GAM) for interpretation.")
        with open(model_path, "rb") as f:
            model_pickle = pickle.load(f) 
        model = model_pickle["model"]
        attribs = gam_interp(model, data, config=model_pickle['config'])
    elif model_type == "MLP":
        os.makedirs(os.path.join(output_path, model_type), exist_ok=True)
        print("Using Multi-Layer Perceptron (MLP) for interpretation.")
        # model = MLPTrainer().load_model(model_path)
        checkpoint = torch.load(model_path)
        model = SingleLayerMLP(
            n_judges=checkpoint['n_judges'],
            hidden_dim=checkpoint['hidden_dim']
        )
        model.load_state_dict(checkpoint['model_state_dict'])

        if logging !=0: 
            plot_h_statistics(data, model, judge_names, output_path=os.path.join(output_path, model_type))
        attribs = compute_input_x_gradient_batch(model, data)
    else:
        print("Unknown model type. Please specify either 'GAM' or 'MLP'.")
        return
    data["attributions"] = attribs
    if importance_analysis == "True":
        print("Calculating importance scores based on variance of attribution contributions.")
        importance_scores = contribution_based_importance(attribs)
        
        importance_output_path = os.path.join(output_path, model_type, "importance_scores.csv")
        bar_plot_feature_importance(judge_names, importance_scores, output_path=os.path.join(output_path, model_type), title="Feature Importance based on Variance of Attribution Contributions")
        print(f"Importance scores saved to: {importance_output_path}")
    # Run interpretation
    print(f"Saving results to: {output_path}")
    for i, group in data.groupby("category"):
        box_plot_for_subset(data, i, judge_names, output_path=os.path.join(output_path, model_type, "attribs_cateogrized"))

    return

if __name__  == "__main__":
    argparser = argparse.ArgumentParser(description="Run interpretability pipeline.")
    argparser.add_argument("--model", type=str, required=True, help="Path to the trained model.")
    argparser.add_argument("--model_type", type=str, required=True, help="Type of the model GAM or MLP")
    argparser.add_argument("--data", type=str, required=True, help="Path to the input data pkl file.")
    argparser.add_argument("--importance_plot", type=str, required=True, help="Calulate and return importance score based on variance of attribution contribution")
    argparser.add_argument("--logging", type=int, default=1, help="Set 0 for simply getting attributions, Set 1 for intermediate analyis plots")
    argparser.add_argument("--output", type=str, required=True, help="Path to the output directory.")
    args = argparser.parse_args()
    run_interpretability_pipeline(model_path=args.model, model_type=args.model_type, importance_analysis=args.importance_plot,
                                  data_path=args.data, 
                                  output_path=args.output, 
                                  logging=args.logging)
    