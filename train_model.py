import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import mean_absolute_error

def train_and_save_model(data_path='insurance.csv', model_path='health_model.pkl'):
    """
    Train and save a health insurance cost prediction model.
    
    Args:
        data_path (str): Path to the CSV data file
        model_path (str): Path to save the trained model
    """
    # Load data
    df = pd.read_csv(data_path)

    # Separate features and target
    X = df.drop('expenses', axis=1)
    y = df['expenses']

    # Split data for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Model trained with MAE: ${mae:,.2f}")

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()