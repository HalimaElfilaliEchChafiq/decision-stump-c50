import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder # Nécessaire pour la frontière

# --- 1. CONFIGURATION DES CHEMINS ---
# Permet de lancer le script depuis la racine OU depuis scripts/
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir) if os.path.basename(script_dir) == 'scripts' else script_dir

if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.stump import DecisionStump
    print("✅ Module 'src' importé.")
except ImportError as e:
    print(f"❌ Erreur import src: {e}")
    sys.exit(1)

def generate_figures():
    # Dossier de sortie
    output_dir = os.path.join(project_root, 'figures')
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Les images seront dans : {output_dir}")

    # --- 2. CHARGEMENT DONNÉES ---
    data_path = os.path.join(project_root, 'data', 'raw', 'iris.csv')
    try:
        df = pd.read_csv(data_path)
        # On suppose que la target est la dernière colonne
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        feature_names = df.columns[:-1]
        print(f"✅ Données chargées ({len(df)} lignes).")
    except Exception:
        print("⚠️ Chargement depuis Sklearn (Fallback).")
        from sklearn.datasets import load_iris
        data = load_iris()
        X, y = data.data, data.target
        feature_names = data.feature_names

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- 3. ENTRAÎNEMENT GLOBAL ---
    print("🔄 Entraînement du modèle principal...")
    model = DecisionStump(criterion='gain_ratio')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculs métriques
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    # Baseline
    maj_class = pd.Series(y_train).mode()[0]
    acc_base = accuracy_score(y_test, [maj_class]*len(y_test))

    # =========================================================================
    # IMAGE 1 : MATRICE DE CONFUSION
    # =========================================================================
    print("🎨 Génération 1/3 : Matrice de Confusion...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matrice de Confusion (Accuracy: {acc:.2%})', fontsize=14)
    plt.ylabel('Vraie Classe')
    plt.xlabel('Classe Prédite')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), bbox_inches='tight', dpi=150)
    plt.close()

    # =========================================================================
    # IMAGE 2 : TABLEAU DE RÉSULTATS
    # =========================================================================
    print("🎨 Génération 2/3 : Tableau de Résultats...")
    cell_text = [
        ["Baseline (Majorité)", f"{acc_base:.2%}", "N/A"],
        ["Decision Stump (C5.0)", f"{acc:.2%}", f"{f1:.2%}"]
    ]
    columns = ["Modèle", "Accuracy", "F1-Score (Macro)"]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight'); ax.axis('off')
    table = ax.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False); table.set_fontsize(12); table.scale(1.2, 1.8)
    
    # Couleurs
    for (row, col), cell in table.get_celld().items():
        if row == 0: 
            cell.set_facecolor('#40466e'); cell.set_text_props(color='white', weight='bold')
        elif row == 2:
            cell.set_facecolor('#d4edda'); cell.set_text_props(weight='bold')
            
    plt.title("Performances Comparées", pad=20, weight='bold', fontsize=14)
    plt.savefig(os.path.join(output_dir, 'results_table.png'), bbox_inches='tight', dpi=150)
    plt.close()

    # =========================================================================
    # IMAGE 3 : FRONTIÈRE DE DÉCISION (Le visuel "Wow")
    # =========================================================================
    print("🎨 Génération 3/3 : Frontière de Décision (2D)...")
    
    # On prend les features 2 et 3 (Pétales) pour Iris, ou 0 et 1 sinon
    f_idx = [2, 3] if X.shape[1] >= 4 else [0, 1]
    X_pair = X[:, f_idx]
    
    # On ré-entraîne un petit Stump juste pour la visu 2D
    # Encodage des labels en entiers pour la couleur (si c'est du texte)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    clf_visu = DecisionStump(criterion='gain_ratio')
    clf_visu.fit(X_pair, y) # Le stump gère y, mais on utilisera y_enc pour le scatter plot
    
    # Grille
    x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
    y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))
    
    # Prédiction grille
    Z = clf_visu.predict(np.c_[xx.ravel(), yy.ravel()])
    # Conversion inverse si le modèle prédit des strings, pour que contourf comprenne
    if Z.dtype.kind in {'U', 'S', 'O'}: # Si c'est du texte/objet
        # On refit un label encoder sur les prédictions possibles pour être sûr
        Z = le.transform(Z)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='viridis')
    plt.scatter(X_pair[:, 0], X_pair[:, 1], c=y_enc, s=50, edgecolor='k', cmap='viridis')
    
    # Noms des axes
    name_x = feature_names[f_idx[0]] if hasattr(feature_names, '__getitem__') else f"Feat {f_idx[0]}"
    name_y = feature_names[f_idx[1]] if hasattr(feature_names, '__getitem__') else f"Feat {f_idx[1]}"
    
    plt.xlabel(name_x)
    plt.ylabel(name_y)
    plt.title(f"Frontière de Décision C5.0\nSplit: {clf_visu.feature_name_} <= {clf_visu.threshold_:.2f}", fontsize=12)
    
    plt.savefig(os.path.join(output_dir, 'decision_boundary.png'), bbox_inches='tight', dpi=150)
    plt.close()

    print("✅ TERMINÉ ! 3 Images générées dans le dossier 'figures/'.")

if __name__ == "__main__":
    generate_figures()