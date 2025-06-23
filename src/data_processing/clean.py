# New commit
import pandas as pd
import os

def load_and_clean_data(
    input_path="data/raw/customer_purchase_data.csv",
    output_path="data/processed/cleaned_customer_purchase_data.csv"
):
    # Load the raw data
    df = pd.read_csv(input_path)

    # Cleaning steps
    df.dropna(inplace=True)                         # Remove rows with any missing values
    df.drop_duplicates(inplace=True)                # Remove duplicate rows
    df.reset_index(drop=True, inplace=True)         # Reset index
    df.ffill(inplace=True)                          # Forward fill
    df.bfill(inplace=True)                          # Backward fill

    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"✅ Cleaned data saved to: {output_path}")

    return df
if __name__ == "__main__":
    load_and_clean_data()
    print("📊 Data cleaning completed successfully!")