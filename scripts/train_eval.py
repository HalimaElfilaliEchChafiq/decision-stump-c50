import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Configuration du chemin
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Imports
from src.stump import DecisionStump
from src.metrics import accuracy, confusion_matrix, f1_score_macro

def load_raw_csv(filename):
    """Charge un CSV depuis data/raw/."""
    path = os.path.join(parent_dir, 'data', 'raw', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Le fichier {path} n'existe pas. Lancez download_data.py d'abord.")
    
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y, df.columns[:-1]

def run_experiment(filename='iris.csv'):
    dataset_name = filename.split('.')[0]
    print(f"\n{'='*40}")
    print(f"🚀 Lancement Expérience : {dataset_name.upper()}")
    print(f"{'='*40}")

    # 1. Chargement Data (RAW)
    try:
        X, y, feature_names = load_raw_csv(filename)
    except Exception as e:
        print(f"❌ Erreur : {e}")
        return

    # 2. Split (70% train, 30% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 3. Entraînement
    print("🔹 Entraînement du Decision Stump (C5.0)...")
    model = DecisionStump(criterion="gain_ratio")
    model.fit(X_train, y_train)

    if model.feature_index_ is not None:
        feat_name = feature_names[model.feature_index_]
        threshold = model.threshold_
        print(f"✅ Règle trouvée : Si '{feat_name}' <= {threshold:.2f}")
    else:
        print("⚠️ Modèle constant.")

    # 4. Évaluation
    y_pred = model.predict(X_test)
    
    acc = accuracy(y_test, y_pred)
    f1 = f1_score_macro(y_test, y_pred)
    
    print(f"\n🏆 RÉSULTATS :")
    print(f"   Accuracy : {acc:.2%}")
    print(f"   F1-Score : {f1:.2%} (Macro)")

    # 5. Matrice de Confusion
    cm_df = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de Confusion - {dataset_name.capitalize()}\nAcc: {acc:.2%}')
    plt.ylabel('Vérité')
    plt.xlabel('Prédiction')
    
    # Sauvegarde
    os.makedirs(os.path.join(parent_dir, 'reports', 'figures'), exist_ok=True)
    save_path = os.path.join(parent_dir, 'reports', 'figures', f'cm_{dataset_name}.png')
    plt.savefig(save_path)
    print(f"🖼️  Image sauvegardée : {save_path}")
    plt.close()

if __name__ == "__main__":
    run_experiment('iris.csv')
    run_experiment('wine.csv') 