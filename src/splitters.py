"""
Module de recherche de splits (Decision Stump C5.0).
Supporte les arguments variables (**kwargs) pour éviter les crashs.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, Union
import numpy as np

# Import relatif depuis src.criteria
from .criteria import gain_ratio, gini_impurity, information_gain

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _score_by_criterion(
    y: np.ndarray, 
    mask: np.ndarray, 
    weights: Optional[np.ndarray], 
    criterion: str
) -> float:
    """
    Calcule le score d'un split donné selon le critère choisi.
    """
    if criterion == "gain_ratio":
        return gain_ratio(y, mask, weights)
    
    elif criterion == "information_gain":
        return information_gain(y, mask, weights)
    
    elif criterion == "gini":
        parent_gini = gini_impurity(y, weights)
        
        y = np.asarray(y)
        mask = np.asarray(mask, dtype=bool)
        weights = weights if weights is not None else np.ones(len(y))
        
        # Split
        w_left = weights[mask]
        y_left = y[mask]
        w_right = weights[~mask]
        y_right = y[~mask]
        
        n_total = np.sum(weights)
        if n_total == 0: return 0.0
        
        n_left = np.sum(w_left)
        n_right = np.sum(w_right)
        
        g_left = gini_impurity(y_left, w_left)
        g_right = gini_impurity(y_right, w_right)
        
        weighted_child_gini = (n_left * g_left + n_right * g_right) / n_total
        
        return float(parent_gini - weighted_child_gini)
        
    else:
        raise ValueError(f"Critère inconnu : {criterion}")


def pd_isnan(x: np.ndarray) -> np.ndarray:
    """Détecte les NaNs de manière robuste."""
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.number):
        return np.isnan(x)
    
    return np.array([
        el is None or (isinstance(el, float) and np.isnan(el)) 
        for el in x
    ], dtype=bool)

# =============================================================================
# SPLITTERS
# =============================================================================

def best_split_numeric(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    criterion: str = "gain_ratio",
    n_thresholds: int = 50,
    **kwargs  # Sécurité pour absorber d'autres args
) -> Optional[Dict[str, Any]]:
    """
    Trouve le meilleur split pour une feature numérique.
    """
    X = np.asarray(X)
    y = np.asarray(y)

    finite_mask = ~pd_isnan(X) & np.isfinite(X.astype(float))
    
    if np.sum(finite_mask) < 2:
        return None

    X_valid = X[finite_mask]
    unique_values = np.unique(X_valid)
    if len(unique_values) < 2:
        return None

    midpoints = (unique_values[:-1] + unique_values[1:]) / 2.0
    
    if len(midpoints) > n_thresholds:
        midpoints = np.percentile(midpoints, np.linspace(0, 100, n_thresholds))
        midpoints = np.unique(midpoints)

    best_split = None
    best_score = -float("inf")

    for threshold in midpoints:
        left_mask = np.zeros(X.shape[0], dtype=bool)
        left_mask[finite_mask] = X_valid <= threshold
        
        try:
            score = _score_by_criterion(y, left_mask, sample_weight, criterion)
        except Exception:
            score = -float("inf")

        if score > best_score:
            best_score = score
            best_split = {
                "feature_type": "numerical",
                "threshold": float(threshold),
                "score": float(score),
                "left_indices": left_mask,
                "right_indices": ~left_mask
            }

    return best_split


def best_split_categorical(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    criterion: str = "gain_ratio",
    **kwargs # <--- CORRECTION ICI : Absorbe n_thresholds sans planter
) -> Optional[Dict[str, Any]]:
    """
    Trouve le meilleur split pour une feature catégorielle (One-vs-Rest).
    """
    X = np.asarray(X)
    y = np.asarray(y)

    finite_mask = ~pd_isnan(X)
    X_valid = X[finite_mask]
    
    if X_valid.size == 0:
        return None

    unique_categories = np.unique(X_valid)
    if len(unique_categories) < 1:
        return None
    
    best_split = None
    best_score = -float("inf")

    for category in unique_categories:
        left_mask = np.zeros(X.shape[0], dtype=bool)
        left_mask[finite_mask] = (X_valid == category)
        
        try:
            score = _score_by_criterion(y, left_mask, sample_weight, criterion)
        except Exception:
            score = -float("inf")

        if score > best_score:
            best_score = score
            best_split = {
                "feature_type": "categorical",
                "categories": [category],
                "score": float(score),
                "left_indices": left_mask,
                "right_indices": ~left_mask
            }

    return best_split


def find_best_split(
    X_feature: np.ndarray, 
    y: np.ndarray, 
    feature_type: str = "numerical", 
    **kwargs
) -> Optional[Dict[str, Any]]:
    """
    Fonction dispatcher pour trouver le split sur UNE feature donnée.
    """
    if feature_type == "numerical":
        return best_split_numeric(X_feature, y, **kwargs)
    elif feature_type == "categorical":
        return best_split_categorical(X_feature, y, **kwargs)
    else:
        raise ValueError("feature_type doit être 'numerical' ou 'categorical'")