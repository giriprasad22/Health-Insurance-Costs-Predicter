from flask import Flask, request, render_template
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# Global model variable (trained at startup)
model = None

def train_model():
    """Train the model in memory"""
    global model
    
    # 1. Load data
    df = pd.read_csv('insurance.csv')
    X = df.drop('expenses', axis=1)
    y = df['expenses']
    
    # 2. Create model pipeline
    numeric_features = ['age', 'bmi', 'children']
    categorical_features = ['sex', 'smoker', 'region']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # 3. Train model
    model.fit(X, y)
    print("Model trained in memory")

# Train model when starting the app
train_model()

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    
    if request.method == 'POST':
        try:
            # Get form data
            input_data = {
                'age': float(request.form['age']),
                'sex': request.form['sex'],
                'bmi': float(request.form['bmi']),
                'children': int(request.form['children']),
                'smoker': request.form['smoker'],
                'region': request.form['region']
            }
            
            # Create DataFrame and predict
            df = pd.DataFrame([input_data])
            prediction = round(model.predict(df)[0], 2)
            
        except Exception as e:
            prediction = f"Error: {str(e)}"
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)