import sys
import os
import numpy as np
import pytest

# --- FIX DU CHEMIN (Indispensable pour trouver src/) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# -------------------------------------------------------

from src.criteria import entropy, gain_ratio, gini_impurity, information_gain
# Correction : Import de DecisionStump (et non DecisionStumpC50)
from src.stump import DecisionStump  

class TestCriteriaFunctions:
    """Test criteria calculation functions."""

    def test_entropy_basic(self):
        """Test entropy calculation with basic cases."""
        # Pure node (all same class)
        y_pure = np.array([0, 0, 0, 0])
        assert entropy(y_pure) == 0.0

        # Balanced binary classification
        y_balanced = np.array([0, 0, 1, 1])
        entropy_val = entropy(y_balanced)
        assert entropy_val > 0
        assert abs(entropy_val - 1.0) < 0.001  # Should be 1.0 for perfect balance

        # With sample weights
        y = np.array([0, 0, 1, 1])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        entropy_weighted = entropy(y, weights)
        assert abs(entropy_weighted - entropy(y)) < 1e-10

    def test_entropy_edge_cases(self):
        """Test entropy with edge cases."""
        # Single sample
        y_single = np.array([0])
        assert entropy(y_single) == 0.0

        # Empty array
        y_empty = np.array([])
        assert entropy(y_empty) == 0.0

    def test_information_gain(self):
        """Test information gain calculation."""
        y = np.array([0, 0, 1, 1])
        mask = np.array([True, True, False, False])  # First half goes left

        ig = information_gain(y, mask)
        assert ig >= 0

        # Pure split should have maximum gain
        y_pure = np.array([0, 0, 1, 1])
        mask_pure = np.array([True, True, False, False])
        ig_pure = information_gain(y_pure, mask_pure)
        assert ig_pure > 0

        # No split (all in one branch)
        mask_no_split = np.array([True, True, True, True])
        ig_no_split = information_gain(y, mask_no_split)
        assert ig_no_split == 0.0

    def test_gain_ratio(self):
        """Test gain ratio calculation."""
        y = np.array([0, 0, 1, 1, 2, 2])
        mask = np.array([True, True, True, False, False, False])

        gr = gain_ratio(y, mask)
        assert gr >= 0

        # Test with split_info = 0 (should handle gracefully)
        y_single = np.array([0, 0, 0, 0])
        mask_single = np.array([True, True, True, True])
        gr_single = gain_ratio(y_single, mask_single)
        assert gr_single == 0.0 

    def test_gini_impurity(self):
        """Test Gini impurity calculation."""
        # Pure node
        y_pure = np.array([0, 0, 0, 0])
        assert gini_impurity(y_pure) == 0.0

        # Balanced binary
        y_balanced = np.array([0, 0, 1, 1])
        gini = gini_impurity(y_balanced)
        assert abs(gini - 0.5) < 0.001

        # With weights
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        gini_weighted = gini_impurity(y_balanced, weights)
        assert 0 <= gini_weighted <= 1

    def test_criteria_with_stump_integration(self):
        """Test that criteria functions work with DecisionStump."""
        # Test each criterion with the stump
        for criterion in ["gain_ratio", "information_gain", "gini"]:
            # Utilisation de la bonne classe DecisionStump
            stump = DecisionStump(criterion=criterion)
            X = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
            y = np.array([0, 0, 1, 1])

            # Should not raise errors
            stump.fit(X, y)
            predictions = stump.predict(X)

            assert len(predictions) == len(y)
            assert set(predictions).issubset(set(y))

if __name__ == "__main__":
    print("🚀 Lancement des tests Criteria...")
    t = TestCriteriaFunctions()
    t.test_entropy_basic()
    t.test_entropy_edge_cases()
    t.test_information_gain()
    t.test_gain_ratio()
    t.test_gini_impurity()
    t.test_criteria_with_stump_integration()
    print("✅ Tous les tests Criteria sont passés.")