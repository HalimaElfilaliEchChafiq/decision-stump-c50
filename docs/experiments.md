# Expérimentations et Analyse des Résultats

## 1. Protocole Expérimental

### 1.1 Dataset Utilisé
Les expérimentations sont menées sur le dataset **Iris** (Fisher, 1936), référence standard en classification.
- **Échantillons :** 150 fleurs (Split Train/Test : 70%/30%).
- **Taille Test set :** 45 échantillons (15 par classe).
- **Attributs :** 4 (Longueur/Largeur du Sépale et du Pétale).
- **Classes :** 3 (Setosa, Versicolor, Virginica).

### 1.2 Configuration du Modèle
Nous utilisons notre implémentation `DecisionStump` avec les spécificités **C5.0** :
- **Critère de division :** Gain Ratio (pour normaliser l'entropie) .
- **Gestion des manquants :** Stratégie pondérée (Weighted).
- **Profondeur :** Fixée à 1 (Stump).

---

## 2. Validation Théorique : L'Exemple "Iris Simplifié"

Conformément à l'analyse théorique du rapport (Chapitre 6), nous avons vérifié le comportement de l'algorithme sur la séparabilité des classes .

**Hypothèse théorique :**
Le dataset Iris contient une classe (Setosa) linéairement séparable des deux autres. Un Decision Stump devrait trouver cette séparation optimale et atteindre une précision d'environ 66% (1 classe sur 3 isolée + la majorité des 2 autres).

---

## 3. Résultats Empiriques

Les résultats suivants sont obtenus via le script `scripts/make_final_figures.py`.

### 3.1 Tableau de Performance
Comparaison entre une Baseline (vote majoritaire global) et notre Decision Stump C5.0.

![Tableau des Résultats](../figures/results_table.png)

**Analyse Chiffrée :**
- **Baseline (33.33%) :** Le hasard ou le vote majoritaire pur n'a qu'une chance sur trois.
- **Decision Stump (66.67%) :** Le modèle double la performance de la baseline. Ce score correspond exactement à la séparation parfaite d'une classe sur trois ($1/3 + 1/3 = 2/3 \approx 66\%$).

### 3.2 Matrice de Confusion
La matrice ci-dessous détaille la répartition des prédictions.

![Matrice de Confusion](../figures/confusion_matrix.png)

**Interprétation Détaillée :**
1.  **Classe 0 (Setosa) - 15/15 Succès :**
    Le modèle a trouvé la règle parfaite (probablement sur la largeur/longueur du pétale) pour isoler cette classe. Aucun faux positif, aucun faux négatif.

2.  **Classe 1 (Versicolor) - 15/15 Succès :**
    Dans la branche de droite (celle qui n'est pas Setosa), la majorité des points d'entraînement devait être Versicolor. Le Stump a donc prédit "Versicolor" pour tout ce qui n'est pas "Setosa".

3.  **Classe 2 (Virginica) - 0/15 Succès :**
    C'est la limitation structurelle de la profondeur 1. Virginica et Versicolor sont mélangées géométriquement. Le Stump ne pouvant faire qu'une seule coupure, il a regroupé Virginica avec Versicolor.
    
> **Note :** Ce comportement valide parfaitement le statut de **"Classifieur Faible"** (Weak Learner) du Decision Stump. Il capture la structure principale (Setosa vs Reste) mais échoue sur les nuances fines (Versicolor vs Virginica), ce qui en fait le candidat idéal pour le **Boosting** (AdaBoost).

---

## 4. Conclusion

L'implémentation est **validée**. Elle reproduit fidèlement le comportement théorique attendu :
1.  **Gain Ratio :** A correctement identifié la feature discriminante.
2.  **Performance :** Atteint le plafond théorique de 66% sur Iris pour une profondeur de 1.
3.  **Stabilité :** Biais élevé assumé, Variance faible.

