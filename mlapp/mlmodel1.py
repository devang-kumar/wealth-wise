import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

import os
from django.conf import settings

# Load and preprocess dataset
csv_path = os.path.join(settings.BASE_DIR, 'investment_survey.csv')
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip().str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
df.dropna(axis=1, how='all', inplace=True)
df.drop([col for col in ['Motivation_cause', 'Resources_used'] if col in df.columns], axis=1, inplace=True)

missing_values = ["Nil", "nil", "NIL", "", "NA", "na", "Na", "None", "none", "NONE", "null", "NULL"]
df.replace(missing_values, np.nan, inplace=True)

for col in df.columns:
    if df[col].isna().any():
        if df[col].dtype == 'object':
            mode_values = df[col].mode()
            df[col] = df[col].fillna(mode_values[0] if not mode_values.empty else 'Unknown')
        else:
            median_value = df[col].median() if not df[col].isna().all() else 0
            df[col] = df[col].fillna(median_value)

le_dict = {}
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le

X = df.drop("Mode_of_investment", axis=1)
y = df["Mode_of_investment"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

def predict_investment(user_input):
    input_df = pd.DataFrame([user_input])
    input_df.columns = input_df.columns.str.strip().str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)

    for col in input_df.columns:
        if col in le_dict:
            try:
                input_df[col] = le_dict[col].transform(input_df[col].astype(str))
            except ValueError:
                input_df[col] = le_dict[col].transform([le_dict[col].classes_[0]])[0]

    missing_cols = set(X.columns) - set(input_df.columns)
    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df[X.columns]
    prediction = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[0]
    predicted_label = le_dict['Mode_of_investment'].inverse_transform(prediction)[0]

    prob_dict = {
        le_dict['Mode_of_investment'].classes_[i]: f"{prob:.1%}"
        for i, prob in enumerate(probabilities)
    }

    return predicted_label, prob_dict
