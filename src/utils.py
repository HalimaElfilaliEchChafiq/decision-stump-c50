import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_iris, load_wine

def load_dataset(name='iris', return_X_y=True):
    """
    Charge un dataset (Iris ou Wine) et retourne les formats attendus.
    
    Args:
        name (str): 'iris' ou 'wine'
        return_X_y (bool): Si True, retourne (X, y, feature_names, target_names)
                           Si False, retourne un DataFrame Pandas complet
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

def save_results(df_results, filename, output_dir='reports/tables'):
    """
    Sauvegarde propre des résultats (CSV ou Excel) dans le dossier du projet.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    
    if filename.endswith('.csv'):
        df_results.to_csv(path, index=False)
    elif filename.endswith('.xlsx'):
        df_results.to_excel(path, index=False)
    
    print(f"💾 Résultats sauvegardés : {path}")