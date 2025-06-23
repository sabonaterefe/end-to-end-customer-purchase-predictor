import pytest
import os   
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.models import train

def test_train_model_returns_valid_outputs():
    """Test that the training pipeline returns a model and validation data"""
    model, X_val, y_val = train.train_model(save_artifacts=False, verbose=False)
    
    assert model is not None, "Model object is None"
    assert not X_val.empty, "X_val should not be empty"
    assert not y_val.empty, "y_val should not be empty"
    assert len(X_val) == len(y_val), "Mismatched lengths between X_val and y_val"