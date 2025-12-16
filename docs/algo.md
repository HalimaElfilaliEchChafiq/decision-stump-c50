# Algorithme Decision Stump (Style C5.0)

Ce document détaille les fondements mathématiques et la logique algorithmique utilisée pour l'implémentation du Decision Stump avec les spécificités de C5.0.

## 1. Définition
Un **Decision Stump** (ou "souche de décision") est un arbre de décision de profondeur 1. Il effectue un test unique sur une seule caractéristique pour diviser les données en deux groupes (ou plus pour les variables catégorielles).

---

## 2. Fondements Mathématiques

Nous utilisons le **Gain Ratio** comme critère de division principal, conformément à l'algorithme C5.0, pour éviter le biais envers les attributs ayant beaucoup de valeurs uniques.

### 2.1 Entropie de Shannon
L'entropie mesure l'impureté d'un ensemble d'échantillons $S$. Plus l'entropie est élevée, plus le mélange de classes est hétérogène.

$$
H(S) = -\sum_{i=1}^{c} p_i \log_2 p_i
$$

*Où $p_i$ est la probabilité de la classe $i$ dans $S$.*

### 2.2 Gain d'Information (IG)
Le gain d'information mesure la réduction d'entropie obtenue en divisant l'ensemble $S$ selon l'attribut $A$.

$$
IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)
$$

### 2.3 Information de Split (Intrinsic Value)
C'est la pénalité appliquée aux attributs qui divisent les données en trop nombreux petits morceaux.

$$
IV(S, A) = -\sum_{v \in Values(A)} \frac{|S_v|}{|S|} \log_2 \left( \frac{|S_v|}{|S|} \right)
$$

### 2.4 Gain Ratio (Critère Final)
C'est le critère utilisé pour choisir la meilleure division.

$$
GR(S, A) = \frac{IG(S, A)}{IV(S, A) + \epsilon}
$$

*Note : Un $\epsilon$ (epsilon) est ajouté au dénominateur pour éviter la division par zéro.*

---

## 3. Gestion des Valeurs Manquantes (Spécificité C5.0)

### Pendant l'apprentissage
1.  **Exclusion :** Les échantillons avec `NaN` sont exclus du calcul des scores d'entropie immédiats.
2.  **Pénalité :** Le gain final est multiplié par la fraction $F$ d'échantillons connus : $Gain_{final} = F \times Gain_{calculé}$.
3.  **Fallback :** La distribution des classes pour les valeurs manquantes est assimilée à celle de la racine (ou du nœud parent).

### Pendant la prédiction
Si une valeur est manquante pour un nouvel échantillon :
* Utiliser la distribution de probabilité globale du nœud (Weighted Fallback).
* Alternative : Distribution pondérée selon les proportions des branches (méthode C4.5/C5.0 standard).

---

## 4. Pseudocode Détaillé

L'algorithme est divisé en trois fonctions principales pour la clarté : le contrôleur principal, l'évaluateur catégoriel et l'évaluateur numérique.

### Fonction Principale : Recherche du Meilleur Split

```pseudo
FUNCTION find_best_decision_stump(X, y, weights):
    best_score = -INFINITY
    best_split = NULL
    
    FOR each feature_index IN features:
        feature_data = X[:, feature_index]
        
        IF is_categorical(feature_data):
            current_split = evaluate_categorical_split(feature_data, y, weights)
        ELSE:
            current_split = evaluate_numerical_split(feature_data, y, weights)
        
        # Mise à jour si meilleur score trouvé
        IF current_split.score > best_score:
            best_score = current_split.score
            best_split = current_split
            best_split.feature_index = feature_index
    
    RETURN best_split
```

### Fonction : Évaluation Variable Catégorielle
Pour une variable catégorielle, chaque catégorie unique devient une branche.

```pseudo
FUNCTION evaluate_categorical_split(feature, y, weights):
    # 1. Ignorer les NaN
    mask_valid = NOT is_nan(feature)
    valid_feature = feature[mask_valid]
    
    # 2. Identifier les branches possibles
    categories = unique(valid_feature)
    
    # 3. Calculer les distributions pour chaque branche
    distributions = []
    FOR category IN categories:
        mask_cat = (feature == category) AND mask_valid
        dist = calculate_weighted_distribution(y[mask_cat], weights[mask_cat])
        distributions.append(dist)
    
    # 4. Calculer le Gain Ratio
    score = calculate_gain_ratio(y[mask_valid], distributions)
    
    RETURN SplitObject(type='categorical', score=score, categories=categories)
```


### Fonction : Évaluation Variable Numérique
Pour une variable numérique, on cherche le meilleur seuil de coupure binaire (≤ vs >).

```pseudo
FUNCTION evaluate_numerical_split(feature, y, weights):
    # 1. Ignorer les NaN et Trier
    mask_valid = NOT is_nan(feature)
    values = sort(unique(feature[mask_valid]))
    
    # 2. Générer les seuils candidats (points médians)
    thresholds = compute_midpoints(values)
    
    # Optimisation : Limiter le nombre de seuils si trop grand
    IF length(thresholds) > 50:
        thresholds = select_percentiles(thresholds, n=50)
    
    best_thresh_score = -INFINITY
    best_threshold = NULL
    
    # 3. Tester chaque seuil
    FOR threshold IN thresholds:
        mask_left = (feature <= threshold) AND mask_valid
        mask_right = (feature > threshold) AND mask_valid
        
        # Calcul des distributions gauche/droite
        dist_left = calculate_weighted_distribution(y[mask_left], weights[mask_left])
        dist_right = calculate_weighted_distribution(y[mask_right], weights[mask_right])
        
        # Calcul du score
        score = calculate_gain_ratio(y[mask_valid], [dist_left, dist_right])
        
        IF score > best_thresh_score:
            best_thresh_score = score
            best_threshold = threshold
            
    RETURN SplitObject(type='numerical', score=best_thresh_score, threshold=best_threshold)
```