"""Test module for the train model functionality."""

import os
import sys

# Set up project root for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models import train  # Ensure this import is correct

def test_train_model_returns_valid_outputs():
    """Test that the training pipeline returns a model and validation data."""
    model, x_val, y_val = train.train_model(save_artifacts=False, verbose=False)
    
    # Assertions to verify the outputs
    assert model is not None, "Model object is None"
    assert not x_val.empty, "x_val should not be empty"
    assert not y_val.empty, "y_val should not be empty"
    assert len(x_val) == len(y_val), "Mismatched lengths between x_val and y_val"