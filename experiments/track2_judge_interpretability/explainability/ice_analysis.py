"""Functions for ICE (Individual Conditional Expectation) plot analysis."""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt

def generate_ice_plot(index, dataset, model, feature_names=None):
    """
    Generate an ICE plot for a specific index in the input array.

    Parameters:
        index (int): The index of the feature to vary.
        dataset (np.ndarray): The dataset containing input arrays.
        model (callable): The model to generate predictions.
        feature_names (list): Optional list of feature names for plotting.

    Returns:
        np.ndarray: ICE curves for further analysis.
    """
    feature_values = dataset[:, index]
    min_value, max_value = feature_values.min(), feature_values.max()
    grid = np.linspace(min_value, max_value, num=50)
    ice_curves = []

    for sample in dataset:
        modified_sample = sample.copy()
        predictions = []

        for value in grid:
            modified_sample[index] = value
            model.eval()
            with torch.no_grad():
                prediction = model(torch.tensor(modified_sample.reshape(1, -1), dtype=torch.float32))
            predictions.append(prediction.item())

        ice_curves.append(predictions)

    ice_curves = np.array(ice_curves)

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # for curve in ice_curves:
    #     plt.plot(grid, curve, alpha=0.5, color='blue')
    
    # title = f'ICE Plot for Feature {index}'
    # if feature_names and index < len(feature_names):
    #     title = f'ICE Plot for {feature_names[index]}'
    
    # plt.title(title)
    # plt.xlabel('Feature Value')
    # plt.ylabel('Model Prediction')
    # plt.grid(True)
    # plt.show()
    
    return ice_curves

def compute_h_statistic_from_ice(ice_curves):
    """
    Approximate Friedman's H-statistic using ICE curves.
    
    Parameters:
        ice_curves (np.ndarray): Array of shape (n_samples, n_points) containing ICE curves
    
    Returns:
        float: H-statistic value
    """
    ice_curves = np.array(ice_curves)
    pdp = np.mean(ice_curves, axis=0)
    
    # Center each ICE curve
    ice_centered = ice_curves - np.mean(ice_curves, axis=1, keepdims=True)
    
    # Compute residuals after removing both main effects
    residuals = ice_centered - (pdp - np.mean(pdp))
    
    # Calculate H-statistic
    numerator = np.sum(residuals ** 2)
    denominator = np.sum((ice_curves - np.mean(ice_curves)) ** 2)
    
    return np.sqrt(numerator / denominator)

def plot_h_statistics(data, model, feature_names, output_path=None):
    """
    Create a bar plot of H-statistics for each feature.
    
    Parameters:
        h_statistics (list): List of H-statistic values
        feature_names (list): List of feature names
    """
    h_statistics = [] 
    for i in range(len(data['judge_scores'].iloc[0])):
        ice_curves = (generate_ice_plot(i, np.array(data['judge_scores'].values.tolist()), model))
        h_stat = compute_h_statistic_from_ice(ice_curves)
        h_statistics.append(h_stat)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.barh(feature_names, h_statistics, color='skyblue')
    plt.xlabel('H-statistic')
    plt.title('H-statistics for Each Feature')
    plt.grid(axis='x')

    if output_path:
        plt.savefig(os.path.join(output_path, "h_statistics.png"))
    else: 
        plt.show()
    return