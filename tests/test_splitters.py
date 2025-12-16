"""
Tests for splitter functions (best_split_categorical, best_split_numeric)
Implémentation des algorithmes de split
 Tests d'intégration avec le modèle et API
"""

import sys
import os
import numpy as np

# --- FIX DU CHEMIN (Indispensable) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# -------------------------------------

from src.splitters import best_split_categorical, best_split_numeric, find_best_split
# Correction Import : DecisionStump au lieu de DecisionStumpC50
from src.stump import DecisionStump

class TestSplitters:
    """Test split finding functions."""

    def test_best_split_numeric_basic(self):
        """Test basic numeric split finding."""
        X = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([0, 0, 1, 1, 1])

        result = best_split_numeric(X, y, criterion="information_gain")

        assert result is not None
        assert "threshold" in result
        assert "score" in result
        assert "left_indices" in result
        assert "right_indices" in result

        # Threshold should be between min and max
        assert result["threshold"] >= np.min(X)
        assert result["threshold"] <= np.max(X)

        # Score should be positive for good split
        assert result["score"] > 0

    def test_best_split_categorical_basic(self):
        """Test basic categorical split finding."""
        X = np.array(["A", "A", "B", "B", "C"])
        y = np.array([0, 0, 1, 1, 1])

        result = best_split_categorical(X, y, criterion="gain_ratio")

        assert result is not None
        assert "categories" in result
        assert "score" in result
        assert "left_indices" in result
        assert "right_indices" in result

        # Categories should be a subset of unique values
        unique_values = np.unique(X)
        assert all(cat in unique_values for cat in result["categories"])

    def test_best_split_with_missing_values(self):
        """Test split finding with missing values."""
        X = np.array([1.0, np.nan, 3.0, 4.0, np.nan])
        y = np.array([0, 0, 1, 1, 1])

        result = best_split_numeric(X, y, criterion="information_gain")

        # Should handle NaN values gracefully
        assert result is not None
        if "threshold" in result:
            assert not np.isnan(result["threshold"])

    def test_best_split_constant_feature(self):
        """Test split finding with constant feature."""
        X = np.array([1.0, 1.0, 1.0, 1.0])
        y = np.array([0, 0, 1, 1])

        result = best_split_numeric(X, y)

        # Constant feature should return None or have score 0
        if result is None:
            assert True  # Acceptable behavior
        else:
            assert result["score"] == 0.0

    def test_find_best_split_function(self):
        """Test the generic find_best_split function."""
        X_numeric = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([0, 0, 1, 1])

        result_numeric = find_best_split(X_numeric, y, feature_type="numerical")
        assert result_numeric is not None

        X_categorical = np.array(["A", "B", "A", "B"])
        result_categorical = find_best_split(
            X_categorical, y, feature_type="categorical"
        )
        assert result_categorical is not None

    def test_split_with_sample_weights(self):
        """Test split finding with sample weights."""
        X = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([0, 0, 1, 1])
        sample_weight = np.array([0.1, 0.2, 0.3, 0.4])

        result = best_split_numeric(X, y, sample_weight=sample_weight)
        assert result is not None
        assert result["score"] >= 0

    def test_integration_with_decision_stump(self):
        """Test that splitters integrate correctly with DecisionStump."""
        # Create stump that uses the splitters
        stump = DecisionStump()

        # Test with mixed data types
        # Note: dtype=object is important for mixed types
        X_mixed = np.array(
            [[1.0, "A"], [2.0, "A"], [3.0, "B"], [4.0, "B"]], dtype=object
        )
        y = np.array([0, 0, 1, 1])

        # Should detect types and use appropriate splitters
        stump.fit(X_mixed, y)

        assert stump.is_fitted_
        assert stump.feature_index_ is not None
        assert stump.feature_type_ in ["numerical", "categorical"]

if __name__ == "__main__":
    print("🚀 Lancement des tests Splitters...")
    t = TestSplitters()
    t.test_best_split_numeric_basic()
    t.test_best_split_categorical_basic()
    t.test_best_split_with_missing_values()
    t.test_best_split_constant_feature()
    t.test_find_best_split_function()
    t.test_split_with_sample_weights()
    t.test_integration_with_decision_stump()
    print("✅ Tous les tests Splitters sont passés.")