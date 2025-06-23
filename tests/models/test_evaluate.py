"""Test module for the evaluate model functionality."""

import os
import sys

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Set up project root for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models import evaluate  # Ensure this import is correct

def test_evaluate_model_metrics_report(capfd):
    """Test that evaluation prints expected metric headers."""
    # Load iris dataset
    features, labels = load_iris(return_X_y=True)
    x_train, x_val, y_train, y_val = train_test_split(
        features, labels, stratify=labels, random_state=0
    )

    # Train RandomForestClassifier
    model = RandomForestClassifier(random_state=0)
    model.fit(x_train, y_train)

    # Evaluate model
    evaluate.evaluate_model(model, x_val, y_val)

    # Capture printed output
    captured = capfd.readouterr()

    # Assertions to check for expected output
    assert "accuracy" in captured.out.lower()
    assert "classification report" in captured.out.lower()