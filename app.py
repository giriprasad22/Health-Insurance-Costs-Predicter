from flask import Flask, request, render_template
import pandas as pd
import pickle
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load model
with open('health_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = {
                'age': int(request.form['age']),
                'bmi': float(request.form['bmi']),
                'children': int(request.form['children']),
                'sex': request.form['sex'],
                'smoker': request.form['smoker'],
                'region': request.form['region']
            }
            df = pd.DataFrame([data])
            prediction = model.predict(df)[0]
            return render_template('index.html', prediction=f"${prediction:,.2f}", form_data=data)
        except Exception as e:
            return render_template('index.html', error=str(e), form_data=data if 'data' in locals() else None)
    return render_template('index.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
