from flask import Flask, request, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load model and data
try:
    model = joblib.load('insurance_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Please run train_model.py first to generate model files")
    model = None

@app.route('/')
def home():
    regions = ['southwest', 'southeast', 'northwest', 'northeast']
    return render_template('index.html', regions=regions)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', 
                            error="Model not loaded. Please try again later.",
                            regions=['southwest', 'southeast', 'northwest', 'northeast'])
    
    try:
        data = {
            'age': int(request.form['age']),
            'sex': request.form['sex'],
            'bmi': float(request.form['bmi']),
            'children': int(request.form['children']),
            'smoker': request.form['smoker'],
            'region': request.form['region']
        }
        
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        
        return render_template('index.html', 
                            prediction=f"${prediction:,.2f}",
                            form_data=data,
                            regions=['southwest', 'southeast', 'northwest', 'northeast'])
    
    except Exception as e:
        return render_template('index.html', 
                            error=str(e),
                            regions=['southwest', 'southeast', 'northwest', 'northeast'])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)