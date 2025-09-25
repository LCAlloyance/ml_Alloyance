import pandas as pd
import json
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# --- Load your real dataset (replace with the actual path to your CSV) ---
df_training = pd.read_csv("your_dataset.csv")

# 1. Separate categorical and numeric columns
categorical_cols = df_training.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df_training.select_dtypes(include=[np.number]).columns.tolist()

# 2. Encode all categorical columns using LabelEncoder (store encoders for each)
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_training[col] = le.fit_transform(df_training[col].astype(str))
    label_encoders[col] = le

# 3. Prepare IterativeImputer
imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=50, random_state=42),
    max_iter=10,
    random_state=0
)

# Fit the imputer on the entire encoded dataset
imputer.fit(df_training)

def autofill_lca_data(json_input):
    """
    Takes a JSON input from a user, applies hybrid autofill logic,
    and returns a fully imputed pandas DataFrame.
    """
    # 1. Load the JSON input into a DataFrame
    user_data = json.loads(json_input)
    df_user = pd.DataFrame([user_data])

    # 2. Apply Rule-Based Imputation (Optional – you can add your domain logic here)
    # Example: If "End-of-Life Treatment" is missing but Process Stage is "End-of-Life"
    if 'Process Stage' in df_user.columns and df_user['Process Stage'].iloc[0] == 'End-of-Life':
        if 'End-of-Life Treatment' not in df_user or pd.isna(df_user['End-of-Life Treatment']).any():
            df_user['End-of-Life Treatment'] = 'Recycling'  # Default assumption

    # 3. Ensure all columns match training set, fill missing columns with NaN
    df_user = df_user.reindex(columns=df_training.columns, fill_value=np.nan)

    # 4. Encode categorical columns using fitted label encoders
    for col in categorical_cols:
        if col in df_user.columns:
            # If new category appears that wasn't seen in training, map it to -1
            df_user[col] = df_user[col].map(
                lambda x: label_encoders[col].transform([x])[0] if x in label_encoders[col].classes_ else -1
            )

    # 5. Apply the MICE-based Imputation
    imputed_array = imputer.transform(df_user)
    df_imputed = pd.DataFrame(imputed_array, columns=df_user.columns)

    # 6. Decode categorical columns back to original values
    for col in categorical_cols:
        # Handle -1 (unknown categories) as "Unknown"
        df_imputed[col] = df_imputed[col].round().astype(int)
        classes = np.append(label_encoders[col].classes_, "Unknown")
        df_imputed[col] = df_imputed[col].map(lambda x: classes[x] if x < len(classes)-1 else "Unknown")

    return df_imputed

# --- Example usage ---
user_json = '''
{
  "Process Stage": "Use",
  "Technology": "Advanced",
  "Location": "Europe",
  "Raw Material Quantity (kg or unit)": null,
  "Energy Input Quantity (MJ)": null,
  "Transport Distance (km)": 500,
  "Emissions to Air CO2 (kg)": null
}
'''
final_df = autofill_lca_data(user_json)
print("Final Autofilled DataFrame:")
print(final_df)
