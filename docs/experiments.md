# Analyse des Expérimentations et Résultats

## 1. Protocole Expérimental

Les tests ont été menés principalement sur le dataset **Iris** (Fisher, 1936), qui permet de valider visuellement la capacité du modèle à séparer linéairement des classes.

* **Dataset :** Iris (150 échantillons, 4 features).
* **Split :** 70% Train / 30% Test (Stratifié).
* **Modèle :** Decision Stump (Critère : Gain Ratio).
* **Comparaison :** Baseline (Vote Majoritaire) vs Notre Modèle.

---

## 2. Analyse des Résultats

### 2.1 Performance Globale
Le tableau ci-dessous compare notre modèle à une approche naïve.

![Tableau de Résultats](../figures/results_table.png)

**Analyse :**
* Le modèle atteint une **Accuracy de 66.67%**.
* Ce score est théoriquement cohérent pour un Stump sur Iris : il parvient à isoler parfaitement une classe (Setosa, 33% du dataset) et échoue à séparer les deux autres (Versicolor/Virginica), regroupant la majorité restante (33%). Le total donne bien $\approx 66\%$.

### 2.2 Matrice de Confusion
La matrice nous permet de comprendre exactement quelles classes sont bien prédites.

![Matrice de Confusion](../figures/confusion_matrix.png)

**Observations Clés :**
1.  **Setosa (Classe 0) :** 100% de succès. Le modèle a trouvé la règle parfaite pour l'isoler.
2.  **Versicolor (Classe 1) :** Le modèle prédit "Versicolor" pour tout ce qui n'est pas "Setosa". Cela capture bien les vrais Versicolor.
3.  **Virginica (Classe 2) :** Confusion totale avec Versicolor. C'est la limite attendue d'une coupure unique (profondeur 1).

### 2.3 Visualisation de la Règle (Frontière de Décision)
L'image suivante montre la coupe effectuée par l'algorithme dans l'espace 2D (Pétales).

![Frontière de Décision](../figures/decision_boundary.png)

* On observe une **ligne verticale unique** (le seuil $\theta$).
* À gauche de la ligne : Zone pure (Setosa).
* À droite de la ligne : Zone mixte (Versicolor et Virginica mélangés).

---

## 3. Étude de Robustesse (Benchmark Iris vs Wine)

En plus de l'analyse visuelle sur Iris, nous avons soumis le modèle à des tests de stress sur le dataset **Wine** (plus complexe) :

1.  **Robustesse au Bruit :** L'accuracy diminue progressivement mais ne s'effondre pas, prouvant que le Gain Ratio sélectionne des features stables.
2.  **Données Manquantes :** La stratégie *Weighted* de C5.0 maintient une performance supérieure à la suppression pure ou au remplacement par la moyenne lorsque le taux de données manquantes dépasse 20%.

## 4. Conclusion des Expérimentations
L'implémentation est **validée**. Elle reproduit le comportement exact attendu d'un *Weak Learner* : capable de capturer la tendance principale (séparabilité linéaire simple) mais nécessitant du Boosting pour résoudre les cas complexes (chevauchement de classes).