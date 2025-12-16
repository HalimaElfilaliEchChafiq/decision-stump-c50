import nbformat as nbf
import os

def get_header_cell(title, description):
    """Génère une cellule de titre standardisée."""
    return nbf.v4.new_markdown_cell(f"""\
# {title}
**Projet KDD - Decision Stump C5.0**

{description}
---""")

def get_import_cell():
    """Génère la cellule d'importations avec le fix de chemin."""
    return nbf.v4.new_code_cell("""\
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ajout du dossier parent au chemin pour importer 'src'
sys.path.append(os.path.abspath('..'))

# Configuration graphique
%matplotlib inline
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

print("✅ Environnement chargé.")""")

# =============================================================================
# NOTEBOOK 1 : 00_data_check.ipynb
# =============================================================================
def create_nb_00():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # 1. Header
    cells.append(get_header_cell(
        "00. Exploration des Données (Data Check)",
        "Ce notebook a pour but de vérifier l'intégrité du dataset (Iris), d'analyser la distribution des variables et d'identifier les features les plus prometteuses pour un Decision Stump (profondeur 1)."
    ))
    cells.append(get_import_cell())
    
    # 2. Chargement
    cells.append(nbf.v4.new_markdown_cell("## 1. Chargement des Données"))
    cells.append(nbf.v4.new_code_cell("""\
# Chargement
try:
    df = pd.read_csv('../data/raw/iris.csv')
    print(f"Dataset chargé : {df.shape}")
except FileNotFoundError:
    print("⚠️ Chargement depuis Sklearn (fichier local absent)")
    from sklearn.datasets import load_iris
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

display(df.head())
print(df.info())"""))

    # 3. Analyse des Valeurs Manquantes
    cells.append(nbf.v4.new_markdown_cell("## 2. Vérification des Valeurs Manquantes\nL'algorithme C5.0 gère les NaN, mais il est bon de savoir s'il y en a."))
    cells.append(nbf.v4.new_code_cell("""\
missing = df.isnull().sum()
if missing.sum() == 0:
    print("✅ Aucune valeur manquante détectée (Dataset propre).")
else:
    print("⚠️ Valeurs manquantes détectées :")
    print(missing[missing > 0])"""))

    # 4. Visualisation (Pairplot)
    cells.append(nbf.v4.new_markdown_cell("## 3. Visualisation des Relations (Pairplot)\nNous cherchons une feature capable de séparer au moins une classe avec une seule ligne droite (Stump)."))
    cells.append(nbf.v4.new_code_cell("""\
# On suppose que la dernière colonne est la cible
target_col = df.columns[-1]

sns.pairplot(df, hue=target_col, height=2.5)
plt.suptitle("Relations entre Features", y=1.02)
plt.show()"""))
    
    cells.append(nbf.v4.new_markdown_cell("**Observation :** Regardez les distributions sur la diagonale. Si une classe est isolée (comme les points bleus souvent pour *Setosa*), le Decision Stump aura une excellente performance."))

    # 5. Corrélation
    cells.append(nbf.v4.new_markdown_cell("## 4. Matrice de Corrélation"))
    cells.append(nbf.v4.new_code_cell("""\
plt.figure(figsize=(8, 6))
# On retire la cible pour la corrélation
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Corrélation entre les Features")
plt.show()"""))

    nb['cells'] = cells
    return nb

# =============================================================================
# NOTEBOOK 2 : 01_sanity_baselines.ipynb
# =============================================================================
def create_nb_01():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(get_header_cell(
        "01. Sanity Check & Baselines",
        "Avant de valider notre modèle, nous devons vérifier qu'il fait mieux que le hasard (Dummy Classifier) et comparer ses performances avec une référence établie (Scikit-Learn Decision Tree, depth=1)."
    ))
    cells.append(get_import_cell())
    
    # Préparation Data
    cells.append(nbf.v4.new_code_cell("""\
from sklearn.model_selection import train_test_split
from src.preprocessing import check_X_y

# Chargement rapide
try:
    df = pd.read_csv('../data/raw/iris.csv')
except:
    from sklearn.datasets import load_iris
    d = load_iris(); df = pd.DataFrame(d.data, columns=d.feature_names); df['target'] = d.target

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"Data split: {len(X_train)} train, {len(X_test)} test")"""))

    # Baseline 1 : Dummy
    cells.append(nbf.v4.new_markdown_cell("## 1. Baseline Naïve (Dummy Classifier)\nSi notre modèle fait moins bien que ça, il est inutile."))
    cells.append(nbf.v4.new_code_cell("""\
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_dummy = dummy.predict(X_test)
acc_dummy = accuracy_score(y_test, y_dummy)

print(f"🎯 Accuracy Dummy (Majorité): {acc_dummy:.2%}")"""))

    # Baseline 2 : Sklearn
    cells.append(nbf.v4.new_markdown_cell("## 2. Baseline Scikit-Learn (Tree depth=1)\nComparaison avec l'implémentation standard."))
    cells.append(nbf.v4.new_code_cell("""\
from sklearn.tree import DecisionTreeClassifier

# Profondeur 1 = Decision Stump
clf_sk = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=42)
clf_sk.fit(X_train, y_train)
acc_sk = clf_sk.score(X_test, y_test)

print(f"🌲 Accuracy Sklearn (Stump): {acc_sk:.2%}")"""))

    # Notre Modèle
    cells.append(nbf.v4.new_markdown_cell("## 3. Notre Modèle (Decision Stump C5.0)"))
    cells.append(nbf.v4.new_code_cell("""\
from src.stump import DecisionStump

# On utilise Gain Ratio
clf_custom = DecisionStump(criterion='gain_ratio')
clf_custom.fit(X_train, y_train)
y_pred = clf_custom.predict(X_test)
acc_custom = np.mean(y_pred == y_test)

print(f"🚀 Accuracy Notre Modèle: {acc_custom:.2%}")"""))

    # Conclusion
    cells.append(nbf.v4.new_markdown_cell("### 📝 Conclusion du Benchmark\nComparons les scores :"))
    cells.append(nbf.v4.new_code_cell("""\
results = pd.DataFrame({
    'Modèle': ['Dummy', 'Sklearn Tree', 'Notre Stump'],
    'Accuracy': [acc_dummy, acc_sk, acc_custom]
})
sns.barplot(data=results, x='Modèle', y='Accuracy')
plt.ylim(0, 1.0)
plt.title("Comparaison des Performances")
plt.show()"""))

    nb['cells'] = cells
    return nb

# =============================================================================
# NOTEBOOK 3 : 02_final_results.ipynb
# =============================================================================
def create_nb_02():
    nb = nbf.v4.new_notebook()
    cells = []
    
    cells.append(get_header_cell(
        "02. Résultats Finaux & Interprétabilité",
        "Présentation détaillée du modèle final : règle découverte, matrice de confusion et frontière de décision."
    ))
    cells.append(get_import_cell())
    
    # Setup
    cells.append(nbf.v4.new_code_cell("""\
from src.stump import DecisionStump
from src.metrics import accuracy, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Data setup
try:
    df = pd.read_csv('../data/raw/iris.csv')
except:
    from sklearn.datasets import load_iris
    d = load_iris(); df = pd.DataFrame(d.data, columns=d.feature_names); df['target'] = d.target

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
features = df.columns[:-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)"""))

    # Règle
    cells.append(nbf.v4.new_markdown_cell("## 1. La Règle d'Or (The Golden Rule)\nQuel est l'attribut le plus discriminant selon le Gain Ratio ?"))
    cells.append(nbf.v4.new_code_cell("""\
model = DecisionStump(criterion='gain_ratio')
model.fit(X_train, y_train, feature_names=features)

if model.feature_index_ is not None:
    print(f"✨ Feature Séparatrice : {model.feature_name_}")
    print(f"✨ Seuil de coupure    : <= {model.threshold_:.3f}")
    print(f"✨ Gain de la règle    : {model.feature_type_}") # feature_type stocke 'numerical' ici
else:
    print("Modèle constant.")"""))

    # Confusion Matrix
    cells.append(nbf.v4.new_markdown_cell("## 2. Analyse des Erreurs (Matrice de Confusion)"))
    cells.append(nbf.v4.new_code_cell("""\
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.ylabel('Vrai Label')
plt.xlabel('Label Prédit')
plt.title(f"Accuracy: {accuracy(y_test, y_pred):.2%}")
plt.show()"""))

    # Frontière de décision
    cells.append(nbf.v4.new_markdown_cell("## 3. Visualisation de la Frontière de Décision\nProjection 2D sur les deux meilleures features."))
    cells.append(nbf.v4.new_code_cell("""\
def plot_boundary(X, y, feat_idx_1, feat_idx_2, features):
    # Subset 2 features
    X_sub = X[:, [feat_idx_1, feat_idx_2]]
    
    # Re-train simple stump
    clf = DecisionStump(criterion='gain_ratio')
    clf.fit(X_sub, y)
    
    # Grid
    x_min, x_max = X_sub[:, 0].min() - 1, X_sub[:, 0].max() + 1
    y_min, y_max = X_sub[:, 1].min() - 1, X_sub[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    plt.scatter(X_sub[:, 0], X_sub[:, 1], c=y, s=50, edgecolor='k', cmap='viridis')
    plt.xlabel(features[feat_idx_1])
    plt.ylabel(features[feat_idx_2])
    plt.title("Frontière de Décision (Decision Stump)")
    plt.show()

# On prend généralement Longueur et Largeur Pétale (indices 2 et 3)
plot_boundary(X, y, 2, 3, features)"""))

    nb['cells'] = cells
    return nb

# =============================================================================
# MAIN execution
# =============================================================================
if __name__ == "__main__":
    os.makedirs('notebooks', exist_ok=True)
    
    nb0 = create_nb_00()
    with open('notebooks/00_data_check.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb0, f)
    print("✅ Généré : notebooks/00_data_check.ipynb")
    
    nb1 = create_nb_01()
    with open('notebooks/01_sanity_baselines.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb1, f)
    print("✅ Généré : notebooks/01_sanity_baselines.ipynb")
    
    nb2 = create_nb_02()
    with open('notebooks/02_final_results.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb2, f)
    print("✅ Généré : notebooks/02_final_results.ipynb")