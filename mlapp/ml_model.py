import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from django.conf import settings

# Initialize model components
encoder = None
best_model = None

def initialize_model():
    global encoder, best_model
    
    model_path = os.path.join(settings.BASE_DIR, 'mlapp', 'ml_model.joblib')
    encoder_path = os.path.join(settings.BASE_DIR, 'mlapp', 'encoder.joblib')
    
    # Try to load saved model if exists
    if os.path.exists(model_path) and os.path.exists(encoder_path):
        try:
            best_model = joblib.load(model_path)
            encoder = joblib.load(encoder_path)
            return
        except Exception as e:
            print(f"Error loading saved model: {e}")
            # Continue to train new model if loading fails
    
    try:
        # Load and prepare data
        data_path = os.path.join(settings.BASE_DIR, 'income_expenditure.csv')
        data = pd.read_csv(data_path)
        
        # Encode categorical features
        encoder = LabelEncoder()
        data["Highest_Qualified_Member"] = encoder.fit_transform(data["Highest_Qualified_Member"])
        
        # Feature Engineering
        data["Savings_Ratio"] = (data["Mthly_HH_Income"] - data["Mthly_HH_Expense"]) / data["Mthly_HH_Income"]
        
        # Prepare features and target
        X = data.drop(columns=["Mthly_HH_Expense"])
        y = data["Mthly_HH_Expense"]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        
        # Save model and encoder
        joblib.dump(best_model, model_path)
        joblib.dump(encoder, encoder_path)
        
    except Exception as e:
        raise RuntimeError(f"Model initialization failed: {e}")

# Initialize the model when module loads
initialize_model()

def predict_finances(monthly_income, family_members, emi_rent, annual_income, qualification, earning_members):
    try:
        # Validate inputs
        if not all(isinstance(x, (int, float)) for x in [monthly_income, family_members, emi_rent, annual_income, earning_members]):
            raise ValueError("All numerical inputs must be numbers")
        
        if monthly_income <= 0:
            raise ValueError("Monthly income must be positive")
        
        if family_members <= 0 or earning_members <= 0:
            raise ValueError("Member counts must be positive")
        
        # Handle qualification
        if qualification not in encoder.classes_:
            qualification = "Graduate"
        
        qualification_encoded = encoder.transform([qualification])[0]
        
        # Calculate savings ratio (temporary placeholder)
        savings_ratio = (monthly_income - 0) / monthly_income
        
        # Prepare input array
        data_input = np.array([
            monthly_income,
            family_members,
            emi_rent,
            annual_income,
            qualification_encoded,
            earning_members,
            savings_ratio
        ]).reshape(1, -1)
        
        # Make prediction
        predicted_spending = float(best_model.predict(data_input)[0])
        predicted_savings = monthly_income - predicted_spending
        savings_percentage = (predicted_savings / monthly_income) * 100
        
        # Validate predictions
        if predicted_spending < 0 or predicted_savings < 0:
            raise ValueError("Model predicted negative values")
        
        return {
            "status": "success",
            "predicted_spending": round(predicted_spending, 2),
            "predicted_savings": round(predicted_savings, 2),
            "savings_percentage": round(savings_percentage, 2)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }