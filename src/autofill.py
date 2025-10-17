"""
autofill.py
------------
Handles automatic filling of missing or incomplete LCA input data.

Uses:
- Trained iterative imputation model (Random Forest + MICE)
- Stored label encoders for categorical columns
- Simple rule-based autofill for specific domain logic
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# ---------------------------------------------
# Define paths
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load saved model & encoders
IMPUTER_PATH = os.path.join(MODEL_DIR, "rf_imputer.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")

# Optional: fallback dataset schema
DATASET_PATH = os.path.join(DATA_DIR, "lca_dataset.csv")

imputer = joblib.load(IMPUTER_PATH) if os.path.exists(IMPUTER_PATH) else None
label_encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else {}

# Fallback to schema if dataset exists
if os.path.exists(DATASET_PATH):
    df_training = pd.read_csv(DATASET_PATH)
else:
    df_training = pd.DataFrame()

categorical_cols = list(label_encoders.keys()) if label_encoders else df_training.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df_training.select_dtypes(include=[np.number]).columns.tolist() if not df_training.empty else []


# ---------------------------------------------
# Core Autofill Function
# ---------------------------------------------
def autofill_missing_values(input_dict: dict) -> dict:
    """
    Autofills missing numeric/categorical values using imputation models + rules.
    Args:
        input_dict (dict): Input data from frontend (may have nulls)
    Returns:
        dict: Fully filled dictionary (ready for prediction)
    """
    try:
        # Step 1️ — Convert input to DataFrame
        df_user = pd.DataFrame([input_dict])

        # Step 2️ — Rule-based autofill examples
        if 'Process Stage' in df_user.columns and df_user['Process Stage'].iloc[0] == 'End-of-Life':
            if 'End-of-Life Treatment' not in df_user or pd.isna(df_user['End-of-Life Treatment']).any():
                df_user['End-of-Life Treatment'] = 'Recycling'

        # Step 3️ — Align with training columns (if known)
        if not df_training.empty:
            df_user = df_user.reindex(columns=df_training.columns, fill_value=np.nan)

        # Step 4️ — Encode categorical columns
        for col in categorical_cols:
            if col in df_user.columns and col in label_encoders:
                le = label_encoders[col]
                df_user[col] = df_user[col].map(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

        # Step 5️ — Impute missing values using trained imputer
        if imputer:
            imputed_array = imputer.transform(df_user)
            df_imputed = pd.DataFrame(imputed_array, columns=df_user.columns)
        else:
            # Fallback: fill numeric with mean, categorical with mode
            df_imputed = df_user.copy()
            df_imputed[numeric_cols] = df_imputed[numeric_cols].apply(lambda c: c.fillna(c.mean() if c.mean() else 0))
            df_imputed[categorical_cols] = df_imputed[categorical_cols].apply(
                lambda c: c.fillna(c.mode().iloc[0] if not c.mode().empty else "Unknown")
            )

        # Step 6 — Decode categorical columns
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                classes = np.append(le.classes_, "Unknown")
                df_imputed[col] = df_imputed[col].round().astype(int)
                df_imputed[col] = df_imputed[col].map(
                    lambda x: classes[x] if x < len(classes) - 1 else "Unknown"
                )

        # Step 7 — Domain rule for GHG if missing but CO2 present
        if (
            "Greenhouse Gas Emissions (kg CO2-eq)" in df_imputed.columns
            and pd.isna(df_imputed.at[0, "Greenhouse Gas Emissions (kg CO2-eq)"])
            and not pd.isna(df_imputed.at[0, "Emissions to Air CO2 (kg)"])
        ):
            df_imputed.at[0, "Greenhouse Gas Emissions (kg CO2-eq)"] = (
                df_imputed.at[0, "Emissions to Air CO2 (kg)"] * 1.05
            )

        # Step — Return as dict
        return df_imputed.iloc[0].to_dict()

    except Exception as e:
        raise RuntimeError(f"Error during autofill: {str(e)}")


# ---------------------------------------------
# Test block (can remove in production)
# ---------------------------------------------
if __name__ == "__main__":
    sample_input={
        'Process Stage': 'Manufacturing',
        'Technology': 'Emerging',
        'Time Period': '2020-2025',
        'Location': 'Asia',
        'Functional Unit': '1 kg Aluminium Sheet',
        'Raw Material Type': 'Aluminium Scrap',
        'Raw Material Quantity (kg or unit)': 100.0,
        'Energy Input Type': 'Electricity',
        'Energy Input Quantity (MJ)': 250.0,
        'Processing Method': 'Advanced',
        'Transport Mode': None,
        'Transport Distance (km)': 300.0,
        'Fuel Type': 'Diesel',
        'Metal Quality Grade': 'High',
        'Material Scarcity Level': 'Medium',
        'Material Cost (USD)': 500.0,
        'Processing Cost (USD)': 200.0,
        'End-of-Life Treatment': 'Recycling',
        'Emissions to Air CO2 (kg)': None,
        'Emissions to Air SOx (kg)': None,
        'Emissions to Air NOx (kg)': None,
        'Emissions to Air Particulate Matter (kg)': None,
        'Greenhouse Gas Emissions (kg CO2-eq)': None,
        'Scope 1 Emissions (kg CO2-eq)': None,
        'Scope 2 Emissions (kg CO2-eq)': None,
        'Scope 3 Emissions (kg CO2-eq)': None,
        'Environmental Impact Score': None,
        'Metal Recyclability Factor': None
    }

    result = autofill_missing_values(sample_input)
    print("\nAutofilled Result:")
    print(json.dumps(result, indent=2))