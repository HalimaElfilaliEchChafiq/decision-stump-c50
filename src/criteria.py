"""
Module de critères pour l'évaluation des splits (Decision Stump C5.0).
Combine la robustesse technique  avec la logique C5.0 .

Implémente:
- Entropy (Shannon)
- Gini Impurity (pour comparaison)
- Information Gain (avec pénalité C5.0 pour valeurs manquantes)
- Split Info
- Gain Ratio
"""

from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np


def _validate_inputs(
    y: np.ndarray, 
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Valide les dimensions et les valeurs des entrées."""
    y = np.asarray(y)
    
    if y.size == 0:
        # Retourne des tableaux vides valides
        return y, np.array([])

    if weights is None:
        weights = np.ones(y.shape[0], dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)
        if weights.shape[0] != y.shape[0]:
            raise ValueError(f"La taille des poids ({weights.shape[0]}) ne correspond pas à y ({y.shape[0]})")
        if np.any(weights < 0):
            raise ValueError("Les poids ne peuvent pas être négatifs")

    return y, weights


def entropy(y: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Calcule l'entropie de Shannon pondérée.
    H(S) = - sum(p_i * log2(p_i))
    """
    y, weights = _validate_inputs(y, weights)
    
    if y.size == 0:
        return 0.0

    total_weight = np.sum(weights)
    if total_weight <= 0:
        return 0.0

    # Calcul des probabilités pondérées de chaque classe
    classes, inverse_indices = np.unique(y, return_inverse=True)
    class_weights = np.zeros(len(classes))
    
    for i in range(len(classes)):
        class_weights[i] = np.sum(weights[inverse_indices == i])

    probs = class_weights / total_weight
    
    # Masquage des probabilités nulles pour éviter log2(0)
    probs = probs[probs > 0]
    
    return float(-np.sum(probs * np.log2(probs)))


def gini_impurity(y: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """
    Calcule l'impureté de Gini pondérée.
    Gini(S) = 1 - sum(p_i^2)
    """
    y, weights = _validate_inputs(y, weights)
    
    if y.size == 0:
        return 0.0

    total_weight = np.sum(weights)
    if total_weight <= 0:
        return 0.0

    classes, inverse_indices = np.unique(y, return_inverse=True)
    class_weights = np.zeros(len(classes))
    
    for i in range(len(classes)):
        class_weights[i] = np.sum(weights[inverse_indices == i])

    probs = class_weights / total_weight
    
    return float(1.0 - np.sum(probs ** 2))


def information_gain(
    y: np.ndarray, 
    mask: np.ndarray, 
    weights: Optional[np.ndarray] = None,
    correction_factor: float = 1.0
) -> float:
    """
    Calcule le Gain d'Information pour un split binaire (défini par le masque).
    
    IG = H(Parent) - Sum( (Ni/N) * H(Child_i) )
    
    Args:
        y: Labels cibles
        mask: Masque booléen (True = Branche Gauche, False = Branche Droite)
        weights: Poids des échantillons
        correction_factor: Facteur C5.0 pour valeurs manquantes (Fraction des données connues).
                          1.0 signifie pas de valeurs manquantes.
    """
    y, weights = _validate_inputs(y, weights)
    mask = np.asarray(mask, dtype=bool)
    
    if y.size == 0 or mask.size != y.size:
        return 0.0
        
    # Si tous les exemples vont d'un seul côté, le gain est nul
    if mask.all() or (~mask).all():
        return 0.0

    # Entropie Parent
    parent_entropy = entropy(y, weights)

    # Séparation Gauche / Droite
    w_left = weights[mask]
    y_left = y[mask]
    
    w_right = weights[~mask]
    y_right = y[~mask]

    n_left = np.sum(w_left)
    n_right = np.sum(w_right)
    n_total = n_left + n_right

    if n_total == 0:
        return 0.0

    # Entropie Enfants
    e_left = entropy(y_left, w_left)
    e_right = entropy(y_right, w_right)

    weighted_child_entropy = (n_left * e_left + n_right * e_right) / n_total
    
    ig = parent_entropy - weighted_child_entropy
    
    # Correction C5.0 (pénalité si données manquantes)
    return float(ig * correction_factor)


def split_info(
    mask: np.ndarray, 
    weights: Optional[np.ndarray] = None
) -> float:
    """
    Calcule l'Information de Split (Split Info) pour normaliser le Gain.
    IV = - sum( (|Si|/|S|) * log2(|Si|/|S|) )
    """
    if weights is None:
        weights = np.ones(len(mask))
        
    n_left = np.sum(weights[mask])
    n_right = np.sum(weights[~mask])
    n_total = n_left + n_right
    
    if n_total == 0:
        return 0.0
        
    p_left = n_left / n_total
    p_right = n_right / n_total
    
    si = 0.0
    if p_left > 0:
        si -= p_left * np.log2(p_left)
    if p_right > 0:
        si -= p_right * np.log2(p_right)
        
    return float(si)


def gain_ratio(
    y: np.ndarray, 
    mask: np.ndarray, 
    weights: Optional[np.ndarray] = None,
    correction_factor: float = 1.0,
    epsilon: float = 1e-10
) -> float:
    """
    Calcule le Gain Ratio.
    GR = InformationGain / (SplitInfo + epsilon)
    """
    ig = information_gain(y, mask, weights, correction_factor)
    si = split_info(mask, weights)
    
    # C5.0 ajoute parfois une petite valeur pour éviter la division par zéro
    # ou vérifie si le SplitInfo est trop petit.
    if si < epsilon:
        return 0.0
        
    return ig / si