import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Ajout du path pour src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.stump import DecisionStump
from src.metrics import confusion_matrix

def load_raw_data(filename='iris.csv'):
    path = os.path.join(parent_dir, 'data', 'raw', filename)
    if not os.path.exists(path):
        print(f"❌ Erreur : Fichier introuvable à {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y, df.columns[:-1], np.unique(y)

def plot_decision_boundary(X, y, feature_names, target_names, filename):
    """Affiche la frontière de décision sur les 2 meilleures features."""
    # On entraîne un Stump juste sur les 2 premières colonnes pour la visualisation 2D
    # (Ou on cherche les 2 meilleures, ici on simplifie : Longueur vs Largeur Pétale souvent)
    feature_indices = [2, 3] # Petal Length, Petal Width (souvent les meilleures)
    
    # Sécurité si dataset différent d'Iris
    if X.shape[1] < 4: feature_indices = [0, 1]
        
    X_pair = X[:, feature_indices]
    
    # Encodage y pour l'affichage couleur
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Entraînement spécifique pour la visu
    clf = DecisionStump()
    clf.fit(X_pair, y) # Attention : on fit sur y original (texte), le stump gère
    
    # Création de la grille
    x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
    y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # Prédiction sur la grille
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = le.transform(Z) # On remet en entiers pour le contourf
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    
    # Plot des points
    scatter = plt.scatter(X_pair[:, 0], X_pair[:, 1], c=y_enc, s=40, edgecolor='k', cmap='viridis')
    plt.xlabel(feature_names[feature_indices[0]])
    plt.ylabel(feature_names[feature_indices[1]])
    plt.title(f"Decision Boundary (C5.0 Stump)\nSplit sur: {feature_names[feature_indices[clf.feature_index_]]} <= {clf.threshold_}")
    
    # Sauvegarde
    save_path = os.path.join(parent_dir, 'reports', 'figures', filename)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"🖼️  Frontière sauvegardée : {save_path}")
    plt.close()

if __name__ == "__main__":
    print("--- Génération des Figures ---")
    
    # 1. Charger Iris depuis RAW
    X, y, feats, targets = load_raw_data('iris.csv')
    
    # 2. Générer Frontière de Décision
    plot_decision_boundary(X, y, feats, targets, 'decision_boundary_iris.png')
    
    print("✅ Terminé.")