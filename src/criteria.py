"""
Metrics for evaluating splits in decision stumps (C5.0 style).
Includes entropy, information gain, split information, and gain ratio.
"""

import numpy as np
from typing import Optional

def entropy(y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    """Calcule l'entropie de Shannon pondérée."""
    if len(y) == 0: return 0.0
    
    if sample_weight is None:
        sample_weight = np.ones(len(y))
    
    total_weight = np.sum(sample_weight)
    if total_weight == 0: return 0.0
    
    # Normalisation
    sample_weight = sample_weight / total_weight
    
    classes = np.unique(y)
    probs = np.zeros(len(classes))
    
    for i, cls in enumerate(classes):
        mask = (y == cls)
        probs[i] = np.sum(sample_weight[mask])
    
    probs = probs[probs > 0]
    if len(probs) == 0: return 0.0
    
    return -np.sum(probs * np.log2(probs))

def information_gain(X_feature: np.ndarray, y: np.ndarray, split_value: Optional[float] = None, feature_type: str = 'categorical', sample_weight: Optional[np.ndarray] = None) -> float:
    """Calcule le Gain d'Information (ID3)."""
    if sample_weight is None: sample_weight = np.ones(len(y))
    
    entropy_before = entropy(y, sample_weight)
    
    # Gestion valeurs manquantes (weighted)
    mask_not_nan = ~np.isnan(X_feature)
    if not np.any(mask_not_nan): return 0.0
    
    weight_not_nan = np.sum(sample_weight[mask_not_nan])
    total_weight = np.sum(sample_weight)
    fraction_known = weight_not_nan / total_weight if total_weight > 0 else 0.0
    
    weighted_entropies = 0.0
    
    if feature_type == 'categorical':
        categories = np.unique(X_feature[mask_not_nan])
        for category in categories:
            mask = (X_feature == category) & mask_not_nan
            if np.any(mask):
                w = np.sum(sample_weight[mask])
                e = entropy(y[mask], sample_weight[mask])
                weighted_entropies += (w / weight_not_nan) * e
    else: # numerical
        if split_value is None: return 0.0
        mask_left = (X_feature <= split_value) & mask_not_nan
        mask_right = (X_feature > split_value) & mask_not_nan
        
        for mask in [mask_left, mask_right]:
            if np.any(mask):
                w = np.sum(sample_weight[mask])
                e = entropy(y[mask], sample_weight[mask])
                weighted_entropies += (w / weight_not_nan) * e
                
    return fraction_known * (entropy_before - weighted_entropies)

def split_info(X_feature: np.ndarray, split_value: Optional[float] = None, feature_type: str = 'categorical', sample_weight: Optional[np.ndarray] = None) -> float:
    """Calcule le Split Info (C4.5/C5.0)."""
    if sample_weight is None: sample_weight = np.ones(len(X_feature))
    
    mask_not_nan = ~np.isnan(X_feature)
    if not np.any(mask_not_nan): return 0.0
    
    weight_not_nan = np.sum(sample_weight[mask_not_nan])
    total_weight = np.sum(sample_weight)
    fraction_known = weight_not_nan / total_weight if total_weight > 0 else 0.0
    
    weighted_probs = []
    
    if feature_type == 'categorical':
        categories = np.unique(X_feature[mask_not_nan])
        for category in categories:
            mask = (X_feature == category) & mask_not_nan
            weighted_probs.append(np.sum(sample_weight[mask]) / weight_not_nan)
    else: # numerical
        if split_value is None: return 0.0
        mask_left = (X_feature <= split_value) & mask_not_nan
        mask_right = (X_feature > split_value) & mask_not_nan
        for mask in [mask_left, mask_right]:
            weighted_probs.append(np.sum(sample_weight[mask]) / weight_not_nan)
            
    weighted_probs = np.array(weighted_probs)
    weighted_probs = weighted_probs[weighted_probs > 0]
    
    if len(weighted_probs) == 0: return 0.0
    
    split_entropy = -np.sum(weighted_probs * np.log2(weighted_probs))
    return fraction_known * split_entropy

def gain_ratio(X_feature: np.ndarray, y: np.ndarray, split_value: Optional[float] = None, feature_type: str = 'categorical', sample_weight: Optional[np.ndarray] = None, min_split_info: float = 1e-10) -> float:
    """Calcule le Gain Ratio (C5.0)."""
    ig = information_gain(X_feature, y, split_value, feature_type, sample_weight)
    si = split_info(X_feature, split_value, feature_type, sample_weight)
    if si < min_split_info: return 0.0
    return ig / si