# Rapport Technique : Implémentation du Decision Stump C5.0
**Module :** Knowledge Data Discovery (2025)
**Auteurs :** Groupe M

---

## 1. Introduction

Les arbres de décision sont des modèles fondamentaux en apprentissage automatique. Ce projet vise à implémenter "from scratch" un **Decision Stump** (souche de décision), en utilisant les critères d'optimisation de l'algorithme **C5.0**. L'objectif est de comprendre comment un "apprenant faible" (*Weak Learner*) sélectionne la caractéristique la plus discriminante.

## 2. Fondements Mathématiques

### 2.1 Définition du Decision Stump
Un Decision Stump est un arbre de décision de profondeur 1. Il effectue une classification basée sur une règle unique composée d'une feature $j$ et d'un seuil $\theta$ :

$$h(x) = \begin{cases} c_{gauche} & \text{si } x_j \le \theta \\ c_{droite} & \text{si } x_j > \theta \end{cases}$$

### 2.2 Le Critère C5.0 : Gain Ratio
Contrairement à ID3 (Gain d'Information) ou CART (Gini), C5.0 utilise le **Gain Ratio** pour sélectionner la meilleure division. Cela corrige le biais qui favorise naturellement les attributs ayant beaucoup de valeurs distinctes.

1.  **Entropie de Shannon :** Mesure l'impureté du nœud.
    $$H(S) = - \sum_{k=1}^{K} p_k \log_2(p_k)$$

2.  **Gain d'Information (IG) :** Réduction d'entropie après division.
    $$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|}H(S_v)$$

3.  **Gain Ratio :** Normalisation par l'entropie de division.
    $$GainRatio(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$$

## 3. Architecture Logicielle

Le projet est structuré sous forme de package Python modulaire, respectant les standards de **Scikit-Learn**.

### 3.1 Compatibilité (Scikit-Learn Interface)
La classe `DecisionStump` hérite de `BaseEstimator` et `ClassifierMixin`. Cela permet :
* L'utilisation de `cross_val_score` pour la validation croisée.
* L'intégration dans des Pipelines (`make_pipeline`).
* La compatibilité avec `GridSearchCV`.

### 3.2 Gestion des Valeurs Manquantes (Weighted Strategy)
Inspirée de C5.0, notre implémentation ne supprime pas les données manquantes (`NaN`). Si une valeur est absente lors de la prédiction, le modèle :
1.  Calcule la probabilité d'appartenir à la branche gauche ou droite (basée sur l'entraînement).
2.  Combine les prédictions des deux branches pondérées par ces probabilités.

## 4. Conclusion Théorique
Ce modèle constitue la brique élémentaire idéale pour des algorithmes de Boosting (comme AdaBoost). Bien que limité par sa profondeur de 1 (biais élevé), il offre une variance très faible et une interprétabilité maximale.