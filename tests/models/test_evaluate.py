import pytest
import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.models import evaluate
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def test_evaluate_model_metrics_report(capfd):
    """Test that evaluation prints expected metric headers"""
    # Load iris dataset
    X, y = load_iris(return_X_y=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, random_state=0)

    # Train RandomForestClassifier
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)

    # Evaluate model
    evaluate.evaluate_model(model, X_val, y_val)

    # Capture printed output
    captured = capfd.readouterr()

    # Assertions to check for expected output
    assert "accuracy" in captured.out.lower()
    assert "classification report" in captured.out.lower()