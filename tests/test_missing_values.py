"""
Tests for missing value handling
 Stratégie de gestion des valeurs manquantes 
 Implémentation dans le modèle et tests
"""

import sys
import os
import numpy as np
import pytest

# --- FIX DU CHEMIN (Indispensable) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# -------------------------------------

# Correction Import : DecisionStump au lieu de DecisionStumpC50
from src.stump import DecisionStump

class TestMissingValues:
    """Test missing value handling in DecisionStump."""

    def test_missing_values_in_training(self):
        """Test training with missing values."""
        X = np.array(
            [
                [1.0, 2.0],
                [np.nan, 3.0],  # Missing value
                [3.0, np.nan],  # Missing value
                [4.0, 5.0],
            ]
        )
        y = np.array([0, 0, 1, 1])

        for strategy in ["weighted", "majority", "ignore"]:
            stump = DecisionStump(missing_strategy=strategy, random_state=42)

            # Should not crash
            stump.fit(X, y)
            assert stump.is_fitted_

            # Should be able to predict
            predictions = stump.predict(X)
            assert len(predictions) == len(y)

    def test_missing_values_in_prediction(self):
        """Test prediction with missing values."""
        # Train on complete data
        X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y_train = np.array([0, 0, 1, 1])

        stump = DecisionStump(missing_strategy="weighted", random_state=42)
        stump.fit(X_train, y_train)

        # Predict with missing values
        X_test_missing = np.array(
            [
                [np.nan, 2.0],  # Missing in feature 0
                [3.0, np.nan],  # Missing in feature 1
                [np.nan, np.nan],  # All missing
            ]
        )

        predictions = stump.predict(X_test_missing)
        assert len(predictions) == 3
        assert all(pred in [0, 1] for pred in predictions)

    def test_different_missing_strategies(self):
        """Test different missing value strategies."""
        X_train = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])
        y_train = np.array([0, 0, 1, 1])

        X_test = np.array([[np.nan, 2.0]])  # Single sample with missing value

        predictions = {}
        for strategy in ["weighted", "majority", "ignore"]:
            stump = DecisionStump(
                missing_strategy=strategy, random_state=42  # For 'ignore' strategy
            )
            stump.fit(X_train, y_train)
            predictions[strategy] = stump.predict(X_test)[0]

        # Different strategies might give different results
        # All should be valid predictions
        assert all(pred in [0, 1] for pred in predictions.values())

    def test_missing_values_all_features(self):
        """Test when all features have missing values."""
        X = np.array(
            [[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]]
        )
        y = np.array([0, 0, 1, 1])

        stump = DecisionStump(missing_strategy="majority", random_state=42)

        # Should handle or raise appropriate warning
        with pytest.warns(UserWarning):
            stump.fit(X, y)

        # Should still predict something
        predictions = stump.predict(X)
        assert len(predictions) == len(y)

    def test_missing_values_with_sample_weights(self):
        """Test missing values with sample weights."""
        X = np.array([[1.0, 2.0], [np.nan, 3.0], [3.0, 4.0], [4.0, np.nan]])
        y = np.array([0, 0, 1, 1])
        sample_weight = np.array([0.25, 0.25, 0.25, 0.25])

        stump = DecisionStump(missing_strategy="weighted")
        stump.fit(X, y, sample_weight=sample_weight)

        predictions = stump.predict(X)
        assert len(predictions) == len(y)

    def test_missing_value_handling_consistency(self):
        """Test that missing value handling is consistent."""
        X = np.array([[1.0, 2.0], [np.nan, 3.0], [3.0, np.nan], [4.0, 5.0]])
        y = np.array([0, 0, 1, 1])

        stump = DecisionStump(random_state=42)
        stump.fit(X, y)

        # Predict multiple times - should get same results
        pred1 = stump.predict(X)
        pred2 = stump.predict(X)

        np.testing.assert_array_equal(pred1, pred2)

if __name__ == "__main__":
    print("🚀 Lancement des tests Missing Values...")
    t = TestMissingValues()
    t.test_missing_values_in_training()
    t.test_missing_values_in_prediction()
    t.test_different_missing_strategies()
    t.test_missing_values_all_features()
    t.test_missing_values_with_sample_weights()
    t.test_missing_value_handling_consistency()
    print("✅ Tous les tests Missing Values sont passés.")