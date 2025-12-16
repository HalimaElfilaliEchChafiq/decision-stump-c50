#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'entraînement et d'évaluation pour DecisionStump (C5.0).
Combine l'architecture robuste  avec la visualisation .

Usage:
    python scripts/train_eval.py --dataset iris
    python scripts/train_eval.py --dataset wine --criterion entropy
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split

# --- Configuration du chemin pour importer src ---
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# Imports du module
from src.stump import DecisionStump
from src.metrics import accuracy, confusion_matrix, f1_score_macro

def load_data(dataset_name):
    """Charge un CSV depuis data/raw/."""
    # Gestion flexible : soit un nom de fichier, soit un chemin complet
    if dataset_name.endswith('.csv'):
        filename = dataset_name
    else:
        filename = f"{dataset_name}.csv"
        
    path = parent_dir / 'data' / 'raw' / filename
    
    if not path.exists():
        # Essayer chemin direct
        path = Path(dataset_name)
        
    if not path.exists():
        raise FileNotFoundError(f"❌ Impossible de trouver {filename} dans data/raw/ ou le chemin spécifié.")
    
    print(f"📂 Chargement de : {path}")
    df = pd.read_csv(path)
    
    # On suppose que la cible est la dernière colonne
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    feature_names = df.columns[:-1]
    
    return X, y, feature_names

def save_confusion_matrix_plot(y_true, y_pred, dataset_name, acc):
    """Génère et sauvegarde la matrice de confusion."""
    cm_df = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matrice de Confusion - {dataset_name.capitalize()}\nAccuracy: {acc:.2%}')
    plt.ylabel('Vérité Terrain')
    plt.xlabel('Prédiction')
    
    # Création du dossier figures s'il n'existe pas
    output_dir = parent_dir / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    save_path = output_dir / 'confusion_matrix.png' # Nom générique ou f'cm_{dataset_name}.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"🖼️  Matrice de confusion sauvegardée : {save_path}")

def main():
    # --- 1. Gestion des Arguments (CLI) ---
    parser = argparse.ArgumentParser(description="Entraînement Decision Stump C5.0")
    parser.add_argument("--dataset", type=str, default="iris", help="Nom du dataset (ex: iris, wine)")
    parser.add_argument("--criterion", type=str, default="gain_ratio", choices=["gain_ratio", "entropy", "gini"], help="Critère de split")
    parser.add_argument("--strategy", type=str, default="weighted", choices=["weighted", "ignore"], help="Stratégie valeurs manquantes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"🚀 DÉMARRAGE EXPÉRIENCE : {args.dataset.upper()}")
    print(f"   Config: Critère={args.criterion}, Missing={args.strategy}")
    print(f"{'='*60}")

    # --- 2. Chargement ---
    try:
        X, y, feature_names = load_data(args.dataset)
    except Exception as e:
        print(f"❌ Erreur critique : {e}")
        return

    # --- 3. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )
    print(f"📊 Données : {len(X_train)} Train, {len(X_test)} Test")

    # --- 4. Entraînement ---
    print("\n🔹 Entraînement du modèle...")
    model = DecisionStump(
        criterion=args.criterion,
        missing_strategy=args.strategy,
        random_state=args.seed
    )
    model.fit(X_train, y_train)

    # Affichage de la règle découverte
    if model.feature_index_ is not None:
        feat_name = feature_names[model.feature_index_]
        threshold = model.threshold_
        print(f"✅ MEILLEURE RÈGLE TROUVÉE :")
        print(f"   SI {feat_name} <= {threshold:.3f}")
        print(f"   ALORS Classe Gauche (Distribution): {model.left_dist_}")
        print(f"   SINON Classe Droite (Distribution): {model.right_dist_}")
        print(f"   (Gain Ratio: {model.split_score_:.4f})")
    else:
        print("⚠️ Modèle constant (Aucun split trouvé).")

    # --- 5. Évaluation ---
    print("\n🔹 Évaluation sur le Test Set...")
    y_pred = model.predict(X_test)
    
    acc = accuracy(y_test, y_pred)
    f1 = f1_score_macro(y_test, y_pred)
    
    print(f"🏆 RÉSULTATS FINAUX :")
    print(f"   Accuracy : {acc:.2%}")
    print(f"   F1-Score : {f1:.2%} (Macro)")
    
    # --- 6. Visualisation & Sauvegarde ---
    save_confusion_matrix_plot(y_test, y_pred, args.dataset, acc)
    
    # Sauvegarde des métriques texte
    results_path = parent_dir / 'figures' / 'metrics.txt'
    with open(results_path, "w") as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")
        f.write(f"Rule: {feat_name if model.feature_index_ is not None else 'None'} <= {threshold if model.feature_index_ is not None else 0}\n")
    print(f"📝 Métriques sauvegardées dans : {results_path}")

if __name__ == "__main__":
    main()