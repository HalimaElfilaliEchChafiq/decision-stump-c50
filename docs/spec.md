# Spécifications Techniques : Decision Stump & C5.0

## 1. Définitions du Modèle

### 1.1 Decision Stump (Souche de Décision)
Conformément à la définition formelle, le modèle implémenté est un arbre de décision de profondeur 1, composé d'un nœud racine unique et de deux feuilles.
- **Formule :** $h(x) = c_1$ si $x_j \le \theta$, sinon $c_2$.
- **Complexité Temporelle :** $O(d \cdot n \log n)$ (dominée par le tri des valeurs pour la recherche de seuil).
- **Complexité Spatiale :** $O(n + d)$.

### 1.2 Spécificités C5.0 Intégrées
L'implémentation intègre les améliorations majeures de l'algorithme C5.0 par rapport à ID3/C4.5 :
1.  **Critère de Division (Gain Ratio) :** Utilisation du ratio de gain pour corriger le biais envers les attributs multi-valués.
    - Inclut : Entropie, Gain d'Information, Split Information.
    - Garde-fou : Gestion des cas où `split_info ≈ 0`.
2.  **Gestion des Valeurs Manquantes :**
    - *Entraînement :* Pondération des exemples (stratégie "Weighted") ou exclusion, avec ajustement du score par la fraction d'exemples connus.
    - *Prédiction :* Gestion probabiliste (vote pondéré selon les proportions des branches) ou fallback vers la racine.
3.  **Types de Données :** Support natif des attributs numériques (recherche de seuil) et catégoriels (sous-ensembles).

## 2. Architecture et API

### 2.1 Structure de la Classe
Le modèle suit une interface compatible avec scikit-learn.

```python
class DecisionStump:
    def __init__(self, criterion='gain_ratio', missing_strategy='weighted', 
                 min_samples_split=2, random_state=None):
        """
        Initialise le modèle Decision Stump C5.0.
        Args:
            criterion (str): 'gain_ratio', 'entropy', 'gini'.
            missing_strategy (str): Stratégie pour les NaN ('weighted', 'ignore').
        """
        pass
    
    def fit(self, X, y, sample_weight=None):
        """
        Entraîne le modèle sur les données X et les étiquettes y.
        Gère les features numériques et catégorielles automatiquement.
        """
        pass
    
    def predict(self, X):
        """
        Prédit les classes pour X.
        Gère les valeurs manquantes via la stratégie définie (fallback ou pondération).
        """
        pass
    
    def predict_proba(self, X):
        """
        Retourne les probabilités des classes (utile pour l'évaluation ROC/AUC).
        """
        pass
```
### 2.2 Stratégie de Recherche de Split
- **Features Numériques :** Tri des valeurs uniques et évaluation des points médians comme seuils candidats.
- **Features Catégorielles :** Évaluation des partitions basées sur les catégories uniques.
- **Support Multi-classes :** L'algorithme doit supporter nativement plus de 2 classes cibles.

## 3. Comparatif Technique (Objectifs)

Le projet vise à reproduire les caractéristiques suivantes (basé sur le Tableau 4.1 du rapport) :

| Caractéristique | Decision Stump (Implémenté) | C5.0 Complet (Référence) |
| :--- | :--- | :--- |
| **Profondeur** | 1 (Fixe) | Variable (élagué) |
| **Biais** | Élevé (Sous-apprentissage) | Faible |
| **Variance** | Faible [cite: 689] | Élevée (réduite par élagage) |
| **Valeurs Manquantes** | Géré (Stratégie C5.0 Pondérée)  | Oui (Probabiliste) |
| **Boosting** | Prêt pour AdaBoost (Weak Learner)| Intégré |

## 4. Contraintes d'Implémentation et Stack

- **Langage :** Python 3.x
- **Dépendances Principales :**
  - `numpy` (calcul matriciel)
  - `pandas` (manipulation de DataFrames)
  - `matplotlib` / `seaborn` (visualisation et reporting)
- **Architecture du Code :** Modulaire 
  - `src/` : Code source (`criteria.py`, `splitters.py`, `stump.py`).
  - `scripts/` : Scripts d'exécution (`train_eval.py`, `make_figures.py`).
  - `tests/` : Tests unitaires (`pytest`).