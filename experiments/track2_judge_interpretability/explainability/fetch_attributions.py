"""Functions for gradient-based model interpretability."""

import numpy as np
import torch

def compute_input_x_gradient_batch(model, dataset):
    """
    Compute Input X Gradient attributions for a batch of inputs.
    
    Parameters:
        model (torch.nn.Module): The trained model
        dataset (list): List of input arrays
        
    Returns:
        list: List of attribution arrays
    """
    model.eval()
    attributions = []
    
    for i, inp in dataset.iterrows():
        X_tensor = torch.tensor(inp['judge_scores'], dtype=torch.float32, requires_grad=True)
        outputs = model(X_tensor).squeeze()
        
        # Compute gradients
        outputs.mean().backward()
        attribution = X_tensor.grad * X_tensor
        attributions.append(attribution.detach().numpy())
        
    return attributions

def gam_interp(model, data, config): 
    attributions = []
    # taken from config -> risk ------------------> 
    spline_count = config['n_splines']
    for i, row in data.iterrows():
        X = row['judge_scores']
        modelmat = model._modelmat([X])
        b = model.coef_[model.terms.get_coef_indices(-1)]
        # get individual basis function contributions   
        resids = []
        for i in range(0, spline_count*len(X), spline_count):
            # print(f'Basis functions {i} to {i+5}:')
            contrib = modelmat[:, i:i+spline_count].dot(b[i:i+spline_count])
            # print(contrib)
            resids.append(contrib[0])
        attributions.append(resids)
    return attributions

def contribution_based_importance(attributions): 
    """
    Compute contribution-based feature importance from attributions.
    
    Parameters:
        attributions (list): List of attribution arrays
        
    Returns:
        list: List of feature importance scores
    """
    
    for attrib in attributions:
        # Sum absolute contributions for each feature
        contribution = []
        for i in attrib: 
            contribution.append(abs(i)/sum(abs(np.array(attrib))))
    variance_importance = importance_scores(contribution)
        
    return variance_importance

def importance_scores(contributions): 
    # df= df[df['category']=="Code Generation"]
    var_scores = []
    for i in range(len(contributions[0])): 
        var = np.array([x[i] for x in contributions]).var()
        var_scores.append(var)

    return var_scores


def analyze_feature_importance(attributions, feature_names=None):
    """
    Analyze feature importance based on attributions.
    
    Parameters:
        attributions (list): List of attribution arrays
        feature_names (list): Optional list of feature names
        
    Returns:
        dict: Dictionary containing feature importance statistics
    """
    import numpy as np
    
    # Convert to numpy array for easier analysis
    attr_array = np.array(attributions)
    
    # Compute mean absolute attribution for each feature
    mean_importance = np.abs(attr_array).mean(axis=0)
    
    # Create feature importance dictionary
    importance_dict = {}
    for idx, imp in enumerate(mean_importance):
        feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
        importance_dict[feature_name] = imp
        
    return importance_dict