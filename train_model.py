import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle  # Changed from joblib

# Load data
df = pd.read_csv('insurance.csv')

# Separate features and target
X = df.drop('expenses', axis=1)
y = df['expenses']

# Define preprocessing
categorical_features = ['sex', 'smoker', 'region']
numeric_features = ['age', 'bmi', 'children']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Create pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train model
model.fit(X, y)

# Save model using pickle
with open('health_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("Model trained and saved successfully with pickle!")