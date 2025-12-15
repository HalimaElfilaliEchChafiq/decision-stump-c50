"""
Decision Stump C5.0 Implementation.
Utilise le Gain Ratio et la gestion des valeurs manquantes.
"""
import numpy as np
import pandas as pd
from collections import Counter
from .splitters import find_best_split

class DecisionStump:
    def __init__(self, criterion="gain_ratio"):
        # Le criterion est forcé à gain_ratio via nos imports, mais on le garde pour l'API
        self.criterion = criterion
        self.feature_index_ = None
        self.threshold_ = None
        self.split_type_ = None
        self.distributions_ = None # Pour stocker les probas des feuilles
        self.root_majority_ = None

    def fit(self, X, y, sample_weight=None):
        X = np.array(X)
        y = np.array(y)
        if sample_weight is None: sample_weight = np.ones(len(y))
        
        # 1. Calculer la majorité globale (Fallback)
        self.classes_ = np.unique(y)
        self.root_majority_ = Counter(y).most_common(1)[0][0]
        
        # 2. Trouver le meilleur split via splitters.py
        split_result = find_best_split(X, y, sample_weight)
        
        if split_result is None or split_result.get('score', -np.inf) <= 0:
            # Pas de bon split trouvé
            self.feature_index_ = None
            return self
            
        # 3. Sauvegarder les paramètres du modèle
        self.feature_index_ = split_result['feature_idx']
        self.split_type_ = split_result['feature_type']
        self.threshold_ = split_result.get('split_value')
        
        # 4. Calculer les distributions des feuilles (Pour la prédiction)
        self._compute_leaf_distributions(X, y, sample_weight)
        
        return self

    def _compute_leaf_distributions(self, X, y, weights):
        """Calcule la classe majoritaire pour chaque branche."""
        self.distributions_ = {}
        col = X[:, self.feature_index_]
        
        if self.split_type_ == 'numerical':
            mask_left = col <= self.threshold_
            mask_right = ~mask_left
            
            self.distributions_['left'] = self._get_majority(y[mask_left])
            self.distributions_['right'] = self._get_majority(y[mask_right])
        else:
            # Categorical
            unique_vals = np.unique(col[~np.isnan(col)])
            for val in unique_vals:
                mask = (col == val)
                self.distributions_[val] = self._get_majority(y[mask])

    def _get_majority(self, y_subset):
        if len(y_subset) == 0: return self.root_majority_
        return Counter(y_subset).most_common(1)[0][0]

    def predict(self, X):
        X = np.array(X)
        n_samples = len(X)
        preds = np.empty(n_samples, dtype=self.classes_.dtype)
        
        if self.feature_index_ is None:
            return np.full(n_samples, self.root_majority_)
            
        col = X[:, self.feature_index_]
        
        for i in range(n_samples):
            val = col[i]
            
            # Gestion basique NaN -> Majorité racine
            if pd.isna(val):
                preds[i] = self.root_majority_
                continue
                
            prediction = None
            if self.split_type_ == 'numerical':
                direction = 'left' if val <= self.threshold_ else 'right'
                prediction = self.distributions_.get(direction)
            else:
                prediction = self.distributions_.get(val)
            
            # Si branche inconnue -> Majorité racine
            if prediction is None:
                preds[i] = self.root_majority_
            else:
                preds[i] = prediction
                
        return preds
        
    def score(self, X, y):
        return np.mean(self.predict(X) == y)