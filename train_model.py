import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

print("Loading and preprocessing data...")

# Load data
df = pd.read_csv('insurance.csv')

print("Preparing features...")
# Define features
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

print("Building model pipeline...")
# Create model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

print("Training model...")
# Train model
X = df.drop('expenses', axis=1)
y = df['expenses']
model.fit(X, y)

print("Saving model...")
# Save model
joblib.dump(model, 'insurance_model.pkl')

print("Model trained and saved successfully!")