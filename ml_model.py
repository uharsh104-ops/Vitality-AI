import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from database import get_all_logs

MODEL_PATH = "base_health_model.pkl"

def generate_synthetic_data(n_samples=500):
    np.random.seed(42)
    sleep = np.random.uniform(3, 10, n_samples)
    exercise = np.random.uniform(0, 120, n_samples)
    meal = np.random.choice([0, 1, 2], n_samples) # 0: Junk, 1: Average, 2: Healthy
    mood = np.random.choice([0, 1, 2, 3], n_samples)
    water = np.random.uniform(0.5, 4.0, n_samples) # 0.5 to 4 liters
    
    diabetes = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    obesity = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    hyper = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    
    score = np.full(n_samples, 40.0)
    score += np.where(sleep < 5, -10, np.where(sleep > 9, -5, 10))
    score += (exercise / 120) * 15
    score += meal * 10
    score += mood * 5
    score += (water / 4.0) * 10 # Water factor
    score -= diabetes * 5
    score -= obesity * 5
    score -= hyper * 5
    score += np.random.normal(0, 3, n_samples)
    score = np.clip(score, 0, 100)
    
    return pd.DataFrame({
        'sleep_hours': sleep,
        'exercise_minutes': exercise,
        'meal_type': meal,
        'mood': mood,
        'water_intake': water,
        'diabetes': diabetes,
        'obesity': obesity,
        'hypertension': hyper,
        'health_score': score
    })

def train_base_model():
    """Trains a base model strictly on synthetic data."""
    df = generate_synthetic_data()
    X = df.drop(columns=['health_score'])
    y = df['health_score']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    return model

def get_user_model(user_id):
    """Loads a personalized model if the user has enough data, else the base model."""
    if not os.path.exists(MODEL_PATH):
        base_model = train_base_model()
    else:
        base_model = joblib.load(MODEL_PATH)
        
    user_logs = get_all_logs(user_id)
    
    # Need at least 5 logs to "retrain/personalize"
    if len(user_logs) >= 5:
        # Map categorical to numeric
        meal_map = {'Junk': 0, 'Average': 1, 'Healthy': 2}
        mood_map = {'Poor': 0, 'Okay': 1, 'Good': 2, 'Great': 3}
        
        udf = user_logs.copy()
        udf['meal_type'] = udf['meal_type'].map(meal_map)
        udf['mood'] = udf['mood'].map(mood_map)
        
        # Prepare mixed data (Synthetic + User Data)
        synthetic_df = generate_synthetic_data(n_samples=300) 
        
        # Focus on these columns
        columns = ['sleep_hours', 'exercise_minutes', 'meal_type', 'mood', 'water_intake', 'diabetes', 'obesity', 'hypertension', 'health_score']
        
        udf = udf[columns].dropna()
        
        if len(udf) > 0:
            # Duplicate user records to give them higher weight in the random forest
            boosted_user_data = pd.concat([udf]*10, ignore_index=True)
            combined_df = pd.concat([synthetic_df, boosted_user_data], ignore_index=True)
            
            X = combined_df.drop(columns=['health_score'])
            y = combined_df['health_score']
            
            personalized_model = RandomForestRegressor(n_estimators=100, random_state=42)
            personalized_model.fit(X, y)
            return personalized_model
            
    return base_model

def predict_health_score(user_id, sleep, exercise, meal_str, mood_str, water, diab, obesi, hyper):
    model = get_user_model(user_id)
    
    meal_map = {'Junk': 0, 'Average': 1, 'Healthy': 2}
    mood_map = {'Poor': 0, 'Okay': 1, 'Good': 2, 'Great': 3}
    
    meal_val = meal_map.get(meal_str, 1)
    mood_val = mood_map.get(mood_str, 1)
    
    X_new = pd.DataFrame({
        'sleep_hours': [sleep],
        'exercise_minutes': [exercise],
        'meal_type': [meal_val],
        'mood': [mood_val],
        'water_intake': [water],
        'diabetes': [int(diab)],
        'obesity': [int(obesi)],
        'hypertension': [int(hyper)]
    })
    
    pred = model.predict(X_new)[0]
    return max(0, min(100, pred))
