"""
Tests End-to-End Complets (Iris + Synthétique).
"""

import sys
import os
import numpy as np
import pandas as pd

# --- FIX DU CHEMIN (Indispensable) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# -------------------------------------

try:
    from src.stump import DecisionStump
    # On essaye d'importer accuracy, sinon on la définit pour le test
    try:
        from src.metrics import accuracy
    except ImportError:
        def accuracy(y_true, y_pred):
            return np.mean(y_true == y_pred)
            
    # Pour le test synthétique
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    print("✅ Importation des modules réussie.")
except ImportError as e:
    print(f"❌ Erreur critique d'import : {e}")
    sys.exit(1)

# =============================================================================
# TEST 1 : Données Réelles (Iris)
# =============================================================================

def load_iris_data():
    """Charge les données Iris depuis data/raw/."""
    path = os.path.join(parent_dir, 'data', 'raw', 'iris.csv')
    if not os.path.exists(path):
        # Fallback : on essaye de charger depuis sklearn si le fichier n'est pas là
        print("⚠️  Fichier CSV non trouvé, tentative via Sklearn...")
        from sklearn.datasets import load_iris
        data = load_iris()
        return data.data, data.target
    
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def test_pipeline_iris():
    print(f"\n{'='*20} TEST 1 : IRIS DATASET {'='*20}")
    
    # 1. Chargement
    try:
        X, y = load_iris_data()
        print(f"   ✅ Chargement données: OK ({len(X)} lignes)")
    except Exception as e:
        print(f"   ❌ Échec Chargement: {e}")
        return False

    # 2. Entraînement
    try:
        clf = DecisionStump(criterion="gain_ratio")
        clf.fit(X, y)
        if clf.feature_index_ is None:
            print("   ⚠️  Warning: Modèle constant (pas de split trouvé)")
        else:
            print(f"   ✅ Entraînement: OK (Split sur feature {clf.feature_index_} <= {clf.threshold_:.2f})")
    except Exception as e:
        print(f"   ❌ Échec Entraînement: {e}")
        return False

    # 3. Prédiction
    try:
        y_pred = clf.predict(X)
        acc = accuracy(y, y_pred)
        print(f"   📊 Accuracy obtenue: {acc:.2%}")
        
        if acc > 0.50:
            print("   ✅ Performance: OK (Mieux que l'aléatoire)")
            return True
        else:
            print("   ❌ Performance: FAIBLE")
            return False
    except Exception as e:
        print(f"   ❌ Échec Prédiction: {e}")
        return False


# TEST 2 : Données Synthétiques & Robustesse


def test_pipeline_synthetic():
    print(f"\n{'='*20} TEST 2 : SYNTHETIC & ROBUSTNESS {'='*20}")
    
    # 1. Création Dataset Complexe
    try:
        X, y = make_classification(
            n_samples=200, n_features=5, n_informative=2, 
            n_redundant=1, n_classes=2, random_state=42
        )
        
        # Ajout de valeurs manquantes (NaN)
        rng = np.random.RandomState(42)
        mask = rng.rand(*X.shape) < 0.1  # 10% de NaN
        X_nan = X.copy()
        X_nan[mask] = np.nan
        
        X_train, X_test, y_train, y_test = train_test_split(X_nan, y, test_size=0.3, random_state=42)
        print("   ✅ Génération données synthétiques (avec NaNs): OK")
    except Exception as e:
        print(f"   ❌ Erreur Génération Données: {e}")
        return False

    # 2. Test Robustesse (Valeurs Manquantes)
    try:
        # On utilise la stratégie 'weighted' de C5.0
        model = DecisionStump(criterion='gain_ratio', missing_strategy='weighted')
        model.fit(X_train, y_train)
        print("   ✅ Entraînement sur données incomplètes: OK")
        
        # Prédiction
        y_pred = model.predict(X_test)
        acc = accuracy(y_test, y_pred)
        print(f"   📊 Accuracy sur Test (avec NaNs): {acc:.2%}")
        
        # Vérification Proba
        y_proba = model.predict_proba(X_test)
        if y_proba.shape == (len(y_test), 2):
            print("   ✅ Predict Proba: Format OK")
        else:
            print(f"   ❌ Predict Proba: Mauvais format {y_proba.shape}")
            return False

        return True

    except Exception as e:
        print(f"   ❌ Échec Test Synthétique: {e}")
        # Affiche la trace complète pour le debug si besoin
        import traceback
        traceback.print_exc()
        return False

# MAIN


if __name__ == "__main__":
    print(f"{'='*60}")
    print("🚀 DÉBUT DU TEST END-TO-END (COMPLET)")
    print(f"{'='*60}")
    
    pass_iris = test_pipeline_iris()
    pass_synth = test_pipeline_synthetic()
    
    print(f"\n{'='*60}")
    if pass_iris and pass_synth:
        print("🎉 SUCCÈS TOTAL : LE SYSTÈME EST ROBUSTE ET FONCTIONNEL.")
        sys.exit(0)
    else:
        print("💥 ÉCHEC : AU MOINS UN TEST N'A PAS PASSÉ.")
        sys.exit(1)