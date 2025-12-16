import sys
import os
import numpy as np
import pytest

# --- FIX DU CHEMIN (Pour trouver src/) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# -----------------------------------------

from src.stump import DecisionStump

class TestDecisionStumpAPI:
    """Tests unitaires pour l'API du modèle DecisionStump."""

    def setup_method(self):
        """Données simples pour les tests."""
        # Cas binaire simple (Seuil évident à 2.5)
        self.X = np.array([[1.0], [2.0], [3.0], [4.0]])
        self.y = np.array([0, 0, 1, 1])

    def test_initialization(self):
        """Vérifie que les paramètres par défaut sont bons."""
        clf = DecisionStump()
        assert clf.criterion == 'gain_ratio'
        assert clf.missing_strategy == 'weighted'

    def test_fit_predict_basic(self):
        """Vérifie que fit et predict fonctionnent sans erreur."""
        clf = DecisionStump()
        clf.fit(self.X, self.y)
        
        assert clf.is_fitted_ is True
        assert clf.feature_index_ is not None
        
        preds = clf.predict(self.X)
        assert len(preds) == len(self.y)
        # Il doit réussir ce cas trivial (100% accuracy)
        assert np.array_equal(preds, self.y)

    def test_constant_target(self):
        """Si la cible est constante, le modèle doit prédire la constante."""
        y_const = np.array([1, 1, 1, 1])
        clf = DecisionStump()
        clf.fit(self.X, y_const)
        
        preds = clf.predict(self.X)
        assert np.all(preds == 1)

    def test_dimension_mismatch(self):
        """Doit lever une erreur si X et y n'ont pas la même taille."""
        y_bad = np.array([0, 1]) # Trop court
        clf = DecisionStump()
        
        try:
            clf.fit(self.X, y_bad)
            assert False, "Devrait lever une ValueError"
        except ValueError:
            assert True

if __name__ == "__main__":
    # Permet de lancer le test directement avec python
    print("🚀 Lancement des tests unitaires API...")
    t = TestDecisionStumpAPI()
    t.setup_method()
    t.test_fit_predict_basic()
    print("✅ test_fit_predict_basic: OK")
    t.test_constant_target()
    print("✅ test_constant_target: OK")
    t.test_dimension_mismatch()
    print("✅ test_dimension_mismatch: OK")
    print("🎉 Tous les tests unitaires API sont passés.")