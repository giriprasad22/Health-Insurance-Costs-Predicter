from flask import Flask, request, render_template
import pandas as pd
import pickle
import os
from typing import Optional, Dict, Any

def create_app(model_path: str = 'health_model.pkl') -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, static_folder='static', template_folder='templates')
    
    # Load model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

    @app.route('/', methods=['GET', 'POST'])
    def predict() -> str:
        """Handle predictions and form rendering."""
        form_data: Optional[Dict[str, Any]] = None
        
        if request.method == 'POST':
            try:
                # Get form data
                form_data = {
                    'age': int(request.form['age']),
                    'bmi': float(request.form['bmi']),
                    'children': int(request.form['children']),
                    'sex': request.form['sex'],
                    'smoker': request.form['smoker'],
                    'region': request.form['region']
                }
                
                # Create DataFrame and predict
                df = pd.DataFrame([form_data])
                prediction = model.predict(df)[0]
                
                return render_template('index.html', 
                                    prediction=f"${prediction:,.2f}",
                                    form_data=form_data)
                
            except Exception as e:
                return render_template('index.html', 
                                    error=str(e),
                                    form_data=form_data)
        
        return render_template('index.html')

    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)