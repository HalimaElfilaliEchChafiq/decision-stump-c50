"""
Functions for finding the best split for decision stumps.
"""
import numpy as np
from typing import Dict, Any, Optional
# On importe depuis le fichier criteria.py qui est dans le même dossier
from .criteria import gain_ratio

def best_split_numeric(X_feature, y, sample_weight=None, n_thresholds=50):
    if sample_weight is None: sample_weight = np.ones(len(y))
    mask_not_nan = ~np.isnan(X_feature)
    
    if not np.any(mask_not_nan): return {'score': -np.inf, 'split_value': None}
    
    X_valid = X_feature[mask_not_nan]
    unique_values = np.unique(X_valid)
    
    if len(unique_values) <= 1: return {'score': -np.inf, 'split_value': None}
    
    # Stratégie rapide pour les seuils
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
    if len(thresholds) > n_thresholds:
        thresholds = np.percentile(thresholds, np.linspace(0, 100, n_thresholds))
    
    best_score = -np.inf
    best_threshold = None
    
    for t in thresholds:
        score = gain_ratio(X_feature, y, split_value=t, feature_type='numerical', sample_weight=sample_weight)
        if score > best_score:
            best_score = score
            best_threshold = t
            
    return {
        'score': best_score,
        'feature_type': 'numerical',
        'split_value': best_threshold
    }

def best_split_categorical(X_feature, y, sample_weight=None):
    if sample_weight is None: sample_weight = np.ones(len(y))
    mask_not_nan = ~np.isnan(X_feature)
    if not np.any(mask_not_nan): return {'score': -np.inf}
    
    score = gain_ratio(X_feature, y, feature_type='categorical', sample_weight=sample_weight)
    
    categories = np.unique(X_feature[mask_not_nan]).tolist()
    
    return {
        'score': score,
        'feature_type': 'categorical',
        'categories': categories,
        'split_value': None
    }

def find_best_split(X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Trouve la meilleure feature et le meilleur split."""
    n_features = X.shape[1]
    best_overall = None
    best_score = -np.inf
    
    for feature_idx in range(n_features):
        col_data = X[:, feature_idx]
        
        # Détection de type simple
        is_numeric = np.issubdtype(col_data.dtype, np.number)
        
        if is_numeric:
            res = best_split_numeric(col_data, y, sample_weight)
        else:
            res = best_split_categorical(col_data, y, sample_weight)
            
        if res['score'] > best_score:
            best_score = res['score']
            best_overall = res
            best_overall['feature_idx'] = feature_idx
            
    return best_overall