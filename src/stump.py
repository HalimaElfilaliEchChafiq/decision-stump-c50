"""
Implémentation du Decision Stump (C5.0).
Version Compatible Scikit-Learn (BaseEstimator, ClassifierMixin).
"""

from __future__ import annotations
from typing import Optional, Union, List, Dict, Any
import numpy as np
import warnings

# Gestion robuste de l'import pandas
try:
    import pandas as pd
except ImportError:
    pd = None

# --- AJOUT INDISPENSABLE POUR SKLEARN ---
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

# Imports internes
from .splitters import find_best_split

class DecisionStump(BaseEstimator, ClassifierMixin):
    """
    Un Decision Stump (arbre de profondeur 1) implémentant la logique C5.0.
    Compatible avec les outils Scikit-Learn (cross_val_score, GridSearchCV).
    """

    def __init__(
        self,
        criterion: str = "gain_ratio",
        missing_strategy: str = "weighted",
        min_samples_split: int = 2,
        random_state: Optional[int] = None,
    ):
        self.criterion = criterion
        self.missing_strategy = missing_strategy
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def fit(
        self,
        X: Union[np.ndarray, "pd.DataFrame"],
        y: Union[np.ndarray, list],
        sample_weight: Optional[Union[np.ndarray, list]] = None,
        feature_names: Optional[List[str]] = None,
    ) -> "DecisionStump":
        """Entraîne le modèle sur les données."""
        
        # Réinitialisation des attributs
        self.feature_index_ = None
        self.feature_name_ = None
        self.feature_type_ = None
        self.threshold_ = None
        self.categories_ = None
        self.class_distributions_ = None
        self.root_distribution_ = None
        self.branch_proportions_ = None
        
        # 1. Validation Sklearn (convertit tout en numpy)
        # accept_sparse=False car notre stump ne gère pas les matrices creuses pour l'instant
        # force_all_finite='allow-nan' car on gère les NaN (C5.0)
        X_array, y_array = check_X_y(X, y, force_all_finite='allow-nan', dtype=None)
        
        self.classes_ = unique_labels(y_array)
        self.n_classes_ = len(self.classes_)
        
        # Gestion des poids
        if sample_weight is None:
            sample_weight_array = np.ones(len(y_array))
        else:
            sample_weight_array = np.asarray(sample_weight).ravel()
        
        # Normalisation des poids
        total_weight = np.sum(sample_weight_array)
        if total_weight > 0:
            sample_weight_array /= total_weight

        # Noms de features
        self.feature_names_ = self._get_feature_names(X, feature_names, X_array.shape[1])

        # 2. Calcul de la distribution racine
        self.root_distribution_ = self._compute_class_distribution(y_array, sample_weight_array)

        # 3. Détection des types
        feature_types = self._detect_feature_types(X_array)

        # 4. Recherche du meilleur split
        best_score = -np.inf
        best_split_info = None

        for feature_idx in range(X_array.shape[1]):
            feature_col = X_array[:, feature_idx]
            ftype = feature_types[feature_idx]

            split_result = find_best_split(
                X_feature=feature_col,
                y=y_array,
                feature_type=ftype,
                sample_weight=sample_weight_array,
                criterion=self.criterion,
                n_thresholds=50
            )

            if split_result and split_result["score"] > best_score:
                best_score = split_result["score"]
                best_split_info = split_result
                best_split_info["feature_index"] = feature_idx

        # 5. Stockage
        if best_split_info:
            self._store_split_info(best_split_info, X_array, y_array, sample_weight_array)
        else:
            # Mode constant
            self._store_no_split_info()

        # Marqueur officiel sklearn pour dire "je suis entraîné"
        self.is_fitted_ = True
        return self

    def predict(self, X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        """Prédit les classes."""
        check_is_fitted(self)
        X_array = check_array(X, force_all_finite='allow-nan', dtype=None)
        
        n_samples = X_array.shape[0]
        predictions = np.empty(n_samples, dtype=self.classes_.dtype)

        for i in range(n_samples):
            predictions[i] = self._predict_single(X_array[i])

        return predictions
    
    def predict_proba(self, X: Union[np.ndarray, "pd.DataFrame"]) -> np.ndarray:
        """Prédit les probabilités."""
        check_is_fitted(self)
        X_array = check_array(X, force_all_finite='allow-nan', dtype=None)
        
        n_samples = X_array.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))
        
        for i in range(n_samples):
            proba[i] = self._predict_proba_single(X_array[i])
            
        return proba

    # =========================================================================
    # Helpers Internes
    # =========================================================================

    def _get_feature_names(self, X_origin, feature_names_arg, n_cols):
        if feature_names_arg is not None:
            if len(feature_names_arg) != n_cols:
                return [f"feature_{i}" for i in range(n_cols)]
            return list(feature_names_arg)
        elif hasattr(X_origin, "columns"):
            return X_origin.columns.tolist()
        else:
            return [f"feature_{i}" for i in range(n_cols)]

    def _store_split_info(self, split_info, X, y, sample_weight):
        self.feature_index_ = split_info["feature_index"]
        self.feature_name_ = self.feature_names_[self.feature_index_]
        self.feature_type_ = split_info["feature_type"]
        
        if self.feature_type_ == "numerical":
            self.threshold_ = split_info["threshold"]
        else:
            self.categories_ = split_info["categories"]

        left_mask = split_info["left_indices"]
        right_mask = split_info["right_indices"]

        self.class_distributions_ = {
            "left": self._compute_class_distribution(y[left_mask], sample_weight[left_mask]),
            "right": self._compute_class_distribution(y[right_mask], sample_weight[right_mask]),
            "root": self.root_distribution_
        }

        n_total = np.sum(sample_weight)
        n_left = np.sum(sample_weight[left_mask]) if np.any(left_mask) else 0
        n_right = np.sum(sample_weight[right_mask]) if np.any(right_mask) else 0
        
        self.branch_proportions_ = {
            "left": n_left / n_total if n_total > 0 else 0,
            "right": n_right / n_total if n_total > 0 else 0
        }

    def _store_no_split_info(self):
        self.feature_index_ = None
        self.feature_name_ = None
        self.feature_type_ = "constant"
        self.class_distributions_ = {
            "left": self.root_distribution_,
            "right": self.root_distribution_,
            "root": self.root_distribution_
        }
        self.branch_proportions_ = {"left": 0.5, "right": 0.5}

    def _predict_single(self, x):
        if self.feature_index_ is None:
            return self.classes_[np.argmax(self.root_distribution_)]
            
        feature_val = x[self.feature_index_]

        if self._is_nan(feature_val):
            return self._handle_missing_prediction()

        branch = self._get_branch(feature_val)
        return self.classes_[np.argmax(self.class_distributions_[branch])]

    def _predict_proba_single(self, x):
        if self.feature_index_ is None:
            return self.root_distribution_
            
        feature_val = x[self.feature_index_]

        if self._is_nan(feature_val):
            return self._handle_missing_probability()

        branch = self._get_branch(feature_val)
        return self.class_distributions_[branch]

    def _get_branch(self, feature_val):
        if self.feature_type_ == "numerical":
            return "left" if feature_val <= self.threshold_ else "right"
        else:
            return "left" if feature_val in self.categories_ else "right"

    def _handle_missing_prediction(self):
        if self.missing_strategy == "weighted":
            weighted_probs = np.zeros(self.n_classes_)
            for branch, prop in self.branch_proportions_.items():
                weighted_probs += prop * self.class_distributions_[branch]
            return self.classes_[np.argmax(weighted_probs)]
        elif self.missing_strategy == "majority":
            return self.classes_[np.argmax(self.root_distribution_)]
        else: # ignore or random
            return np.random.choice(self.classes_)

    def _handle_missing_probability(self):
        if self.missing_strategy == "weighted":
            weighted_probs = np.zeros(self.n_classes_)
            for branch, prop in self.branch_proportions_.items():
                weighted_probs += prop * self.class_distributions_[branch]
            return weighted_probs
        else:
            return self.root_distribution_

    def _compute_class_distribution(self, y, sample_weight):
        # Utilisation de self.classes_ pour garantir l'ordre et la taille
        dist = np.zeros(self.n_classes_)
        for i, cls in enumerate(self.classes_):
            mask = (y == cls)
            if np.any(mask):
                dist[i] = np.sum(sample_weight[mask])
        
        total = np.sum(dist)
        if total > 0:
            dist /= total
        return dist

    def _detect_feature_types(self, X):
        types = []
        for i in range(X.shape[1]):
            col = X[:, i]
            # On ignore les NaNs pour détecter le type
            valid = col[~self._is_nan_array(col)]
            
            if len(valid) == 0:
                types.append("categorical")
                continue
                
            is_num = np.issubdtype(np.array(valid).dtype, np.number)
            types.append("numerical" if is_num else "categorical")
        return types
        
    def _is_nan(self, val):
        return val is None or (isinstance(val, float) and np.isnan(val))

    def _is_nan_array(self, arr):
        return pd.isna(arr) if pd is not None else np.isnan(arr.astype(float))