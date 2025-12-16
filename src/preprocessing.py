"""
Module de pré-traitement des données (Preprocessing).
Validation, Nettoyage et Encodage avant l'entraînement.
"""

import numpy as np
try:
    import pandas as pd
except ImportError:
    pd = None

def check_X_y(X, y):
    """
    Valide et convertit X et y en tableaux Numpy.
    Vérifie la cohérence des dimensions.

    Returns
    -------
    X_arr, y_arr : numpy.ndarray
    """
    # 1. Conversion de X
    if pd is not None and isinstance(X, (pd.DataFrame, pd.Series)):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)

    # 2. Conversion de y
    if pd is not None and isinstance(y, (pd.DataFrame, pd.Series)):
        y_arr = y.values
    else:
        y_arr = np.asarray(y)
    
    # Aplatir y si c'est un vecteur colonne
    y_arr = y_arr.ravel()

    # 3. Vérifications
    if X_arr.ndim != 2:
        # Si X est 1D (une seule feature), on le reshape en (N, 1)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        else:
            raise ValueError(f"X doit être 2D (n_samples, n_features), reçu: {X_arr.shape}")

    if len(y_arr) != X_arr.shape[0]:
        raise ValueError(f"Incohérence dimensions : X a {X_arr.shape[0]} lignes, y a {len(y_arr)} labels.")

    if len(y_arr) == 0:
        raise ValueError("Les données d'entraînement sont vides.")

    return X_arr, y_arr

def encode_target(y):
    """
    Encode les classes cibles (ex: ['chat', 'chien']) en entiers (ex: [0, 1]).
    
    Returns
    -------
    y_encoded : np.ndarray (int)
    classes : np.ndarray (les labels originaux)
    """
    y = np.asarray(y).ravel()
    classes, y_encoded = np.unique(y, return_inverse=True)
    return y_encoded, classes

def clean_data(X):
    """
    Remplace les valeurs infinies par NaN (pour être gérées par la stratégie C5.0).
    Ne supprime PAS les lignes (on veut garder l'information).
    """
    X_arr = np.array(X, copy=True)
    
    # Si c'est numérique, on nettoie les infinis
    if np.issubdtype(X_arr.dtype, np.number):
        X_arr[~np.isfinite(X_arr)] = np.nan
        
    return X_arr