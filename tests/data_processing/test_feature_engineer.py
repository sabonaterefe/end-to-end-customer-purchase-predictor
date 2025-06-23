import sys
import os
import pandas as pd

# Ensure the src directory is in the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.feature_engineer import feature_engineer


def test_feature_engineer_preserves_row_count():
    """Ensure no rows are dropped during feature transformation."""
    df = pd.DataFrame({
        "Age": [25, 30],
        "Gender": ["M", "F"],
        "AnnualIncome": [50000, 60000],
        "TimeSpentOnWebsite": [10, 20],
        "ProductCategory": ["A", "B"],
        "LoyaltyProgram": ["Yes", "No"],
        "PurchaseStatus": [1, 0]
    })
    result = feature_engineer(df)
    assert len(result) == len(df), "Row count should remain unchanged"


def test_feature_engineer_preserves_target_column():
    """Ensure the target column remains in the transformed DataFrame."""
    df = pd.DataFrame({
        "Age": [45, 52],
        "Gender": ["F", "M"],
        "AnnualIncome": [70000, 85000],
        "TimeSpentOnWebsite": [15, 8],
        "ProductCategory": ["A", "B"],
        "LoyaltyProgram": ["Yes", "No"],
        "PurchaseStatus": [1, 0]
    })
    result = feature_engineer(df)
    assert "PurchaseStatus" in result.columns, "Target column missing after transformation"


def test_feature_engineer_adds_expected_columns():
    """Check that key engineered features are present after transformation."""
    df = pd.DataFrame({
        "Age": [22, 40],
        "Gender": ["F", "M"],
        "AnnualIncome": [30000, 60000],
        "TimeSpentOnWebsite": [5, 20],
        "ProductCategory": ["A", "B"],
        "LoyaltyProgram": ["Yes", "No"],
        "PurchaseStatus": [0, 1]
    })
    result = feature_engineer(df)

    # Only non-baseline dummy features expected due to drop='first'
    expected_columns = {
        "Age",
        "AnnualIncome",
        "TimeSpentOnWebsite",
        "PurchaseStatus",
        "Income_per_Minute",
        "ProductCategory_B",
        "LoyaltyProgram_Yes"
    }

    assert expected_columns.issubset(set(result.columns)), (
        f"Expected columns missing. Got: {set(result.columns)}"
    )
