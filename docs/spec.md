# Spécifications Techniques : Decision Stump & C5.0

## 1. Définitions du Modèle

### 1.1 Decision Stump (Souche de Décision)
Conformément à la Définition 2.1 du rapport, le modèle implémenté est un arbre de décision de profondeur 1, composé d'un nœud racine unique et de deux feuilles.
- Formule :** $h(x) = c_1$ si $x_j \le \theta$, sinon $c_2$.
- **Complexité Temporelle :** $O(d \cdot n \log n)$ pour le tri et la recherche de seuil.
- **Complexité Spatiale :** $O(n + d)$.

### 1.2 Spécificités C5.0
L'implémentation intègre les améliorations majeures de l'algorithme C5.0 par rapport à ID3/C4.5 :
1.  **Critère de Division :** Utilisation du **Gain Ratio** pour corriger le biais envers les attributs multi-valués.
2.  **Valeurs Manquantes :** Gestion par propagation fractionnelle (pondération des exemples) .
3.  **Types de Données :** Support natif des attributs numériques et catégoriels.

## 2. Comparatif Technique (Objectifs)

Le projet vise à reproduire les caractéristiques suivantes (Tableau 4.1)[cite: 681]:

| Caractéristique | Decision Stump (Implémenté) | C5.0 Complet (Référence) |
| :--- | :--- | :--- |
| **Profondeur** | 1 (Fixe) | Variable (élagué) |
| **Biais** | Élevé (Sous-apprentissage) [cite: 689] | Faible |
| **Variance** | Faible [cite: 689] | Élevée (réduite par élagage) |
| **Valeurs Manquantes** | Géré (Stratégie C5.0) | Oui (Probabiliste) |
| **Boosting** | Prêt pour AdaBoost [cite: 705] | Intégré |

## 3. Contraintes d'Implémentation
- **Langage :** Python 3.x
- **Dépendances :** NumPy, Pandas (pour la manipulation de données), Matplotlib/Seaborn (visualisation).
- **Architecture :** Modulaire (`src/` pour le code, `scripts/` pour l'exécution), séparant la logique de calcul (Criteria) de la structure de l'arbre (Stump).