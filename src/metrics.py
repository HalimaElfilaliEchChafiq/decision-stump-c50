import numpy as np
import pandas as pd

def accuracy(y_true, y_pred):
    """
    Calcule la précision globale (Taux de succès).
    Formule: (TP + TN) / Total
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred):
    """
    Génère la matrice de confusion brute.
    Retourne un DataFrame Pandas pour un affichage lisible (Lignes=Vrai, Colonnes=Prédit).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Trouver toutes les classes uniques
    classes = np.unique(np.concatenate([y_true, y_pred]))
    
    # Créer un DataFrame vide
    matrix = pd.DataFrame(0, index=classes, columns=classes)
    matrix.index.name = 'Vérité'
    matrix.columns.name = 'Prédiction'
    
    # Remplir la matrice
    for t, p in zip(y_true, y_pred):
        matrix.loc[t, p] += 1
        
    return matrix

def f1_score_macro(y_true, y_pred):
    """
    Calcule le F1-Score Macro (Moyenne des F1 de chaque classe).
    Essentiel si les données sont déséquilibrées.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    classes = np.unique(y_true)
    f1_scores = []
    
    for c in classes:
        # On traite 'c' comme la classe Positive, et toutes les autres comme Négatives
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        
        # Calcul Précision et Rappel
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        rappel = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calcul F1 pour cette classe
        if (precision + rappel) > 0:
            f1 = 2 * (precision * rappel) / (precision + rappel)
        else:
            f1 = 0
        f1_scores.append(f1)
        
    return np.mean(f1_scores)