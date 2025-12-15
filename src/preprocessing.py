import numpy as np
import pandas as pd

def check_X_y(X, y):
    """
    Vérifie et nettoie X et y avant l'entraînement.
    Convertit en numpy array si nécessaire et vérifie les dimensions.
    """
    # 1. Conversion sécurisée
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
        
    X = np.array(X)
    y = np.array(y)
    
    # 2. Vérification des dimensions
    if len(X) != len(y):
        raise ValueError(f"Erreur de dimension : X({len(X)}) et y({len(y)}) doivent avoir la même taille.")
        
    if len(X) == 0:
        raise ValueError("Erreur : Les données d'entraînement sont vides.")
        
    # 3. Vérification des valeurs infinies
    if np.any(np.isinf(X)):
        raise ValueError("Erreur : X contient des valeurs infinies.")

    return X, y

def encode_labels(y):
    """
    Encode les labels textuels en entiers si nécessaire (ex: 'setosa' -> 0).
    Retourne les labels encodés et le dictionnaire de mapping.
    """
    unique_classes = np.unique(y)
    mapping = {cls: i for i, cls in enumerate(unique_classes)}
    y_encoded = np.array([mapping[val] for val in y])
    
    return y_encoded, mapping