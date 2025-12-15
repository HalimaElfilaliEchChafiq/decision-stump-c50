
# Rapport de Projet : Decision Stumps et Algorithme C5.0

**Année académique :** 2025-2026

**Module :** Knowledge Data Discovery

---

## 1. Introduction

Les arbres de décision constituent une famille d'algorithmes d'apprentissage supervisé particulièrement intuitifs et puissants [Hastie et al., 2009]. Ce rapport examine en profondeur deux concepts fondamentaux : les **Decision Stumps** (souches de décision) et l'algorithme **C5.0**, successeur moderne de C4.5 [Quinlan, 1993].

### Contexte Historique
* **1986** : ID3 par Ross Quinlan.
* **1993** : C4.5, amélioration majeure d'ID3.
* **1998** : C5.0, version optimisée (plus rapide, moins de mémoire).
* **2000s** : Popularisation des Decision Stumps via le Boosting (AdaBoost).

---

## 2. Fondements Mathématiques

### 2.1 Définition du Decision Stump
Un Decision Stump est un arbre de décision de profondeur 1, composé d'un nœud racine unique et de deux feuilles[cite: 345]. Il effectue une seule décision basée sur une caractéristique et un seuil.

Formellement, pour une donnée $x$, le modèle prédit :
$$h(x) = \begin{cases} c_1 & \text{si } x_j \le \theta \\ c_2 & \text{si } x_j > \theta \end{cases}$$

### 2.2 Critères de Division (C5.0)

Contrairement aux approches basiques utilisant l'indice de Gini, notre implémentation suit la logique de C5.0 basée sur la théorie de l'information.

#### Entropie de Shannon
L'incertitude d'un ensemble $S$ est mesurée par:
$$H(S) = - \sum_{k=1}^{K} p_k \log_2(p_k)$$

#### Gain d'Information (IG)
La réduction d'entropie obtenue par une division:
$$IG(S, j, \theta) = H(S) - \left( \frac{|S_L|}{|S|}H(S_L) + \frac{|S_R|}{|S|}H(S_R) \right)$$

#### Gain Ratio (Spécificité C4.5/C5.0)
Pour éviter le biais envers les attributs ayant beaucoup de valeurs (ex: ID), C5.0 utilise le Gain Ratio :
$$GainRatio(S, A) = \frac{IG(S, A)}{SplitInfo(S, A)}$$
$$SplitInfo(S, A) = - \sum_{i=1}^{v} \frac{|S_i|}{|S|} \log_2 \left( \frac{|S_i|}{|S|} \right)$$

---

## 3. Architecture de l'Implémentation

Notre projet combine la simplicité structurelle du Stump avec la robustesse mathématique de C5.0.

### 3.1 Gestion des Types de Données
* **Numérique** : Recherche dichotomique du meilleur seuil $\theta$ par tri des valeurs.
* **Catégoriel** : Support des divisions multi-branches ou binaires selon la configuration.

### 3.2 Gestion des Valeurs Manquantes
Inspirée de C5.0, nous traitons les `NaN` de manière probabiliste plutôt que de les supprimer. Les exemples manquants sont distribués dans les branches avec un poids proportionnel.

---

## 4. Résultats Expérimentaux

*(Cette section sera complétée une fois le modèle final entraîné)*

### 4.1 Protocole de Test
* **Datasets** : Iris, Wine.
* **Métriques** : Accuracy, F1-Score, Matrice de Confusion.
* **Baseline** : Comparaison avec `sklearn.tree.DecisionTreeClassifier(max_depth=1)`.

### 4.2 Résultats Préliminaires (Baselines)
Sur le dataset Iris (test set), les scores à battre sont :
* **Modèle Majoritaire** : 30.00%
* **Sklearn Stump (Entropie)** : 63.33%

---

## 5. Conclusion et Limites

*(À rédiger en fin de projet)*

## Références
* [Quinlan, 1993] Quinlan, J.R. (1993). *C4.5: Programs for Machine Learning*. Morgan Kaufmann.
* [Hastie et al., 2009] Hastie, T., Tibshirani, R., Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.