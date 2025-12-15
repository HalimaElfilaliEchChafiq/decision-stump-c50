import sys
import os
import numpy as np
import pandas as pd

# --- CORRECTION DU CHEMIN (Le Fix) ---
# On récupère le dossier où se trouve ce fichier
current_dir = os.path.dirname(os.path.abspath(__file__))
# On récupère le dossier parent (la racine du projet)
parent_dir = os.path.dirname(current_dir)
# On ajoute les deux au "Path" de Python pour qu'il trouve 'src'
sys.path.append(current_dir)
sys.path.append(parent_dir)
# -------------------------------------

try:
    from src.stump import DecisionStump
    from src.metrics import accuracy
    print("✅ Importation de src réussie.")
except ImportError as e:
    print(f"❌ Erreur critique d'import : {e}")
    print(f"   Chemins testés : {sys.path}")
    sys.exit(1)

def load_data():
    """Charge les données Iris depuis raw pour le test."""
    # On construit le chemin de manière robuste
    path = os.path.join(parent_dir, 'data', 'raw', 'iris.csv')
    
    if not os.path.exists(path):
        # Fallback : Si on est à la racine
        path = os.path.join('data', 'raw', 'iris.csv')
        
    if not os.path.exists(path):
        print(f"❌ CRITICAL: Fichier de données introuvable à {path}")
        sys.exit(1)
    
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def test_pipeline_iris():
    print("🔵 Test 1: Pipeline Complet sur Iris...")
    
    # 1. Chargement
    try:
        X, y = load_data()
        print(f"   ✅ Chargement données: OK ({len(X)} lignes)")
    except Exception as e:
        print(f"   ❌ Échec Chargement: {e}")
        return False

    # 2. Initialisation
    try:
        clf = DecisionStump(criterion="gain_ratio")
        print("   ✅ Initialisation Modèle: OK")
    except Exception as e:
        print(f"   ❌ Échec Initialisation: {e}")
        return False

    # 3. Entraînement
    try:
        clf.fit(X, y)
        if clf.feature_index_ is None:
            print("   ⚠️  Warning: Modèle constant (pas de split trouvé)")
        else:
            print(f"   ✅ Entraînement: OK (Split sur feature {clf.feature_index_} <= {clf.threshold_:.2f})")
    except Exception as e:
        print(f"   ❌ Échec Entraînement: {e}")
        return False

    # 4. Prédiction & Performance
    try:
        y_pred = clf.predict(X)
        acc = accuracy(y, y_pred)
        
        print(f"   📊 Accuracy obtenue: {acc:.2%}")
        
        if acc > 0.50:
            print("   ✅ Performance: OK (Mieux que l'aléatoire)")
        else:
            print("   ❌ Performance: FAIBLE (Vérifier l'algo)")
            return False
            
    except Exception as e:
        print(f"   ❌ Échec Prédiction: {e}")
        return False

    return True

def test_api_compliance():
    print("\n🔵 Test 2: Conformité API (Scikit-Learn style)...")
    clf = DecisionStump()
    
    has_fit = hasattr(clf, 'fit')
    has_predict = hasattr(clf, 'predict')
    has_score = hasattr(clf, 'score')
    
    if has_fit and has_predict and has_score:
        print("   ✅ Méthodes fit/predict/score présentes: OK")
        return True
    else:
        print(f"   ❌ API Incomplète (Fit: {has_fit}, Predict: {has_predict})")
        return False

if __name__ == "__main__":
    print(f"{'='*40}")
    print("🚀 DÉBUT DU TEST END-TO-END (E2E)")
    print(f"{'='*40}\n")
    
    passed_1 = test_pipeline_iris()
    passed_2 = test_api_compliance()
    
    print(f"\n{'='*40}")
    if passed_1 and passed_2:
        print("🎉 SUCCÈS : TOUS LES TESTS SONT PASSÉS.")
        sys.exit(0)
    else:
        print("💥 ÉCHEC : CERTAINS TESTS ONT ÉCHOUÉ.")
        sys.exit(1)