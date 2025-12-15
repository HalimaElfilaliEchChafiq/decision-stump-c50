import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Setup des chemins pour importer src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.stump import DecisionStump
from src.metrics import accuracy, f1_score_macro, confusion_matrix

def generate_figures():
    # Création du dossier cible s'il n'existe pas
    output_dir = os.path.join(parent_dir, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Dossier cible : {output_dir}")

    # --- 1. Chargement & Entraînement ---
    data_path = os.path.join(parent_dir, 'data', 'raw', 'iris.csv')
    df = pd.read_csv(data_path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Modèle
    model = DecisionStump(criterion='gain_ratio')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métriques
    acc = accuracy(y_test, y_pred)
    f1 = f1_score_macro(y_test, y_pred)
    
    # Baseline (Hasard / Majorité)
    majority_class = pd.Series(y_train).mode()[0]
    y_baseline = [majority_class] * len(y_test)
    acc_baseline = accuracy(y_test, y_baseline)

    # --- FIGURE 1 : CONFUSION MATRIX ---
    print("🎨 Génération de confusion_matrix.png...")
    cm_df = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matrice de Confusion - Iris C5.0 Stump\nAccuracy: {acc:.2%}')
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    
    save_path_cm = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(save_path_cm, bbox_inches='tight', dpi=150)
    plt.close()

    # --- FIGURE 2 : RESULTS TABLE ---
    print("🎨 Génération de results_table.png...")
    
    # Données du tableau
    data = [
        ["Baseline (Majorité)", f"{acc_baseline:.2%}", "N/A"],
        ["Decision Stump (C5.0)", f"{acc:.2%}", f"{f1:.2%}"]
    ]
    columns = ["Modèle", "Accuracy", "F1-Score (Macro)"]
    
    # Création de la figure "Tableau"
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    # Dessin du tableau
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5) # Ajuster la taille
    
    # Styliser l'en-tête
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif row == 2: # Notre modèle
            cell.set_facecolor('#e6ffe6') # Vert clair
            
    plt.title("Comparaison des Performances (Dataset Iris)", pad=20, weight='bold')
    
    save_path_table = os.path.join(output_dir, 'results_table.png')
    plt.savefig(save_path_table, bbox_inches='tight', dpi=150)
    plt.close()
    
    print("✅ Terminé. Vérifie le dossier 'figures/'.")

if __name__ == "__main__":
    generate_figures()