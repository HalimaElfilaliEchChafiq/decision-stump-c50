import nbformat as nbf
import os

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    # --- CELLULE 1 : Titre ---
    text_1 = """\
# 🏆 Résultats Finaux : Decision Stump C5.0
**Projet Knowledge Data Discovery (2025-2026)**
*Auteurs : Groupe M*

Ce notebook présente l'évaluation finale de notre implémentation du **Decision Stump** basé sur l'algorithme **C5.0**.
Nous utilisons le critère du **Gain Ratio** et une gestion pondérée des valeurs manquantes.

### Objectifs :
1. Charger les données brutes (Iris).
2. Entraîner le modèle C5.0 (Profondeur 1).
3. Évaluer les performances (Accuracy, F1-Score).
4. Visualiser la frontière de décision et la matrice de confusion."""
    
    # --- CELLULE 2 : Imports ---
    code_2 = """\
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configuration du chemin pour accéder à 'src'
sys.path.append(os.path.abspath('..'))

# Importation de notre package
from src.stump import DecisionStump
from src.metrics import accuracy, f1_score_macro, confusion_matrix

%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
print("✅ Environnement chargé.")"""

    # --- CELLULE 3 : Chargement des Données ---
    text_3 = """## 1. Chargement et Préparation des Données"""
    code_3 = """\
# Chargement direct depuis le CSV Raw
data_path = '../data/raw/iris.csv'
df = pd.read_csv(data_path)

print(f"Dataset chargé : {df.shape[0]} exemples, {df.shape[1]} colonnes")
display(df.head())

# Séparation Features / Target
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
feature_names = df.columns[:-1]

# Split Train/Test (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")"""

    # --- CELLULE 4 : Entraînement ---
    text_4 = """## 2. Entraînement du Modèle (C5.0)
Nous utilisons ici le critère **'gain_ratio'**, spécifique à C5.0 (contrairement à ID3 qui utilise le Gain d'Information simple)."""
    code_4 = """\
# Initialisation du modèle Stump C5.0
model = DecisionStump(criterion='gain_ratio')

# Entraînement
print("🔄 Entraînement en cours...")
model.fit(X_train, y_train)

# Affichage de la règle découverte
if model.feature_index_ is not None:
    feat = feature_names[model.feature_index_]
    seuil = model.threshold_
    print(f"\\n✅ Règle Optimale Trouvée :")
    print(f"   SI {feat} <= {seuil:.2f} ALORS Classe Gauche")
    print(f"   SINON Classe Droite")
    print(f"   (Gain Ratio: {model.split_type_})")
else:
    print("Modèle constant.")"""

    # --- CELLULE 5 : Évaluation ---
    text_5 = """## 3. Évaluation des Performances
Nous utilisons l'**Accuracy** (précision globale) et le **F1-Score Macro** (moyenne harmonique, utile si déséquilibre)."""
    code_5 = """\
# Prédiction sur le test set
y_pred = model.predict(X_test)

# Calcul des métriques
acc = accuracy(y_test, y_pred)
f1 = f1_score_macro(y_test, y_pred)

print(f"🏆 RÉSULTATS SUR TEST :")
print(f"   Accuracy : {acc:.2%}")
print(f"   F1-Score : {f1:.2%} (Macro)")"""

    # --- CELLULE 6 : Visualisation Matrice Confusion ---
    text_6 = """## 4. Visualisations
### 4.1 Matrice de Confusion"""
    code_6 = """\
# Génération de la matrice
cm_df = confusion_matrix(y_test, y_pred)

# Affichage
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Matrice de Confusion (Iris Test)')
plt.ylabel('Vraie Classe')
plt.xlabel('Classe Prédite')
plt.show()"""

    # --- CELLULE 7 : Frontière de Décision ---
    text_7 = """### 4.2 Frontière de Décision (2D)
Pour visualiser la coupure, nous ré-entraînons un Stump sur les deux meilleures features uniquement (généralement Longueur/Largeur Pétale)."""
    code_7 = """\
def plot_boundary(X, y, model_class):
    # On garde seulement les features 2 et 3 (Petal Length/Width) pour la visu
    X_pair = X[:, [2, 3]]
    
    # Encodage numérique pour l'affichage
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    
    # Entraînement visu
    clf = model_class(criterion='gain_ratio')
    clf.fit(X_pair, y) # Le modèle gère les labels texte, mais le plot veut des nombres
    
    # Grille
    x_min, x_max = X_pair[:, 0].min() - 1, X_pair[:, 0].max() + 1
    y_min, y_max = X_pair[:, 1].min() - 1, X_pair[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    # Pred
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = le.transform(Z).reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X_pair[:, 0], X_pair[:, 1], c=y_enc, s=50, edgecolor='k', cmap='viridis')
    plt.xlabel('Petal Length')
    plt.ylabel('Petal Width')
    plt.title(f"Frontière de Décision C5.0 Stump\\n(Split: {clf.threshold_:.2f})")
    plt.show()

plot_boundary(X, y, DecisionStump)"""

    # --- Construction du Notebook ---
    nb['cells'] = [
        nbf.v4.new_markdown_cell(text_1),
        nbf.v4.new_code_cell(code_2),
        nbf.v4.new_markdown_cell(text_3),
        nbf.v4.new_code_cell(code_3),
        nbf.v4.new_markdown_cell(text_4),
        nbf.v4.new_code_cell(code_4),
        nbf.v4.new_markdown_cell(text_5),
        nbf.v4.new_code_cell(code_5),
        nbf.v4.new_markdown_cell(text_6),
        nbf.v4.new_code_cell(code_6),
        nbf.v4.new_markdown_cell(text_7),
        nbf.v4.new_code_cell(code_7)
    ]

    # Sauvegarde
    os.makedirs('notebooks', exist_ok=True)
    with open('notebooks/02_final_results.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print("✅ Notebook 'notebooks/02_final_results.ipynb' généré avec succès !")

if __name__ == "__main__":
    create_notebook()