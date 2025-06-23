import os
import sys
import json
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_processing.feature_engineer import feature_engineer

def train_model(input_path="data/processed/cleaned_customer_purchase_data.csv",
                model_path="models/random_forest_v1.pkl",
                feature_path="data/outputs/feature_columns.json",
                shap_path="data/outputs/shap_summary.png"):
    df = pd.read_csv(input_path)
    df = feature_engineer(df)

    X = df.drop(columns=["PurchaseStatus"])
    y = df["PurchaseStatus"]

    os.makedirs(os.path.dirname(feature_path), exist_ok=True)
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump(X.columns.tolist(), f)

    X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # SHAP summary plot
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    try:
        shap_summary = shap_values[1] if isinstance(shap_values, list) else shap_values
        shap.summary_plot(shap_summary, X_val, show=False)
        os.makedirs(os.path.dirname(shap_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(shap_path)
        plt.close()
        print(f"📊 SHAP summary plot saved to {shap_path}")
    except Exception as e:
        print(f"⚠️ SHAP visualization skipped due to: {e}")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model trained and saved to {model_path}")
    print(f"📁 Feature layout saved to {feature_path}")

    sample = X_val.iloc[:1].to_dict(orient="records")
    prob = model.predict_proba(X_val.iloc[:1])[0][1]
    print(f"\n🚦 Sample API Input:\n{sample}\n")
    print(f"🎯 Predicted Purchase Probability: {prob:.2%}")

    return model, X_val, y_val

def main():
    train_model()

if __name__ == "__main__":
    main()
