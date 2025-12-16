import os
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union
from sklearn.datasets import load_iris, load_wine

# PARTIE 1 : Robustesse & Calculs (Engineering )

def validate_input_shapes(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Valide les dimensions des tableaux d'entrée.

    Parameters
    ----------
    X : Feature matrix (2D)
    y : Target vector (1D)
    sample_weight : Poids optionnels

    Returns
    -------
    X, y, sample_weight validés
    """
    # Check X
    if X.ndim != 2:
        raise ValueError(f"X doit être un tableau 2D, reçu : {X.shape}")

    # Check y
    y = np.asarray(y).ravel()
    if X.shape[0] != len(y):
        raise ValueError(
            f"X et y ont un nombre d'échantillons différent: {X.shape[0]} vs {len(y)}"
        )

    # Check sample_weight
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight).ravel()
        if len(sample_weight) != len(y):
            raise ValueError(
                f"sample_weight a une taille différente de y: "
                f"{len(sample_weight)} vs {len(y)}"
            )
        if np.any(sample_weight < 0):
            raise ValueError("sample_weight ne peut pas contenir de valeurs négatives")
        if np.sum(sample_weight) == 0:
            raise ValueError("La somme de sample_weight ne peut pas être nulle")

    return X, y, sample_weight


def detect_feature_types(
    X: np.ndarray,
    categorical_threshold: int = 10,
) -> List[str]:
    """
    Détecte si les features sont catégorielles ou numériques.
    Utilisé pour choisir la stratégie de split.
    """
    n_features = X.shape[1]
    feature_types = []

    for col in range(n_features):
        feature = X[:, col]
        finite_mask = ~np.isnan(feature)

        if not np.any(finite_mask):
            feature_types.append("categorical")
            continue

        finite_values = feature[finite_mask]

        # Check dtype first
        if not np.issubdtype(feature.dtype, np.number):
            feature_types.append("categorical")
            continue

        # Check number of unique values
        unique_values = np.unique(finite_values)

        if len(unique_values) <= categorical_threshold:
            feature_types.append("categorical")
        else:
            feature_types.append("numerical")

    return feature_types


def check_constant_features(X: np.ndarray) -> List[bool]:
    """Retourne True pour les colonnes qui n'ont qu'une seule valeur unique."""
    is_constant = []
    for col in range(X.shape[1]):
        feature = X[:, col]
        finite_mask = ~np.isnan(feature)

        if not np.any(finite_mask):
            is_constant.append(True)
        else:
            finite_values = feature[finite_mask]
            if len(np.unique(finite_values)) <= 1:
                is_constant.append(True)
            else:
                is_constant.append(False)
    return is_constant


def safe_divide(
    numerator: Union[float, np.ndarray],
    denominator: Union[float, np.ndarray],
    default: float = 0.0,
) -> Union[float, np.ndarray]:
    """
    Division sécurisée pour éviter les erreurs de division par zéro (utile pour Gain Ratio).
    """
    if isinstance(denominator, np.ndarray):
        result = np.full_like(denominator, default, dtype=float)
        mask = denominator != 0
        result[mask] = numerator[mask] / denominator[mask]
        return result
    else:
        if denominator == 0:
            return default
        return numerator / denominator

# PARTIE 2 : Gestion des Données & I/O (Expériences )

def load_dataset(name: str = 'iris', return_X_y: bool = True) -> Union[Tuple, pd.DataFrame]:
    """
    Charge un dataset standard (Iris ou Wine).
    
    Args:
        name: 'iris' ou 'wine'
        return_X_y: Si True, retourne numpy arrays. Si False, retourne DataFrame.
    """
    if name.lower() == 'iris':
        data = load_iris()
    elif name.lower() == 'wine':
        data = load_wine()
    else:
        raise ValueError(f"Dataset '{name}' inconnu. Choix: 'iris', 'wine'.")
    
    if return_X_y:
        return data.data, data.target, data.feature_names, data.target_names
    else:
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df


def save_results(df_results: pd.DataFrame, filename: str, output_dir: str = 'reports/tables'):
    """
    Sauvegarde un DataFrame de résultats en CSV ou Excel.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    
    try:
        if filename.endswith('.csv'):
            df_results.to_csv(path, index=False)
        elif filename.endswith('.xlsx'):
            df_results.to_excel(path, index=False)
        else:
            # Par défaut CSV
            path += '.csv'
            df_results.to_csv(path, index=False)
            
        print(f"💾 Résultats sauvegardés : {path}")
    except Exception as e:
        print(f"❌ Erreur lors de la sauvegarde : {e}")