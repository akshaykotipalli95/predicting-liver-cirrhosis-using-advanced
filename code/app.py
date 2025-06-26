from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('rf_model.pkl')        # Random Forest model
scaler = joblib.load('scaler.pkl')         # StandardScaler

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        input_features = [
            float(request.form['Age']),
            float(request.form['Gender']),
            float(request.form['TB']),
            float(request.form['DB']),
            float(request.form['Alkphos']),
            float(request.form['Sgpt']),
            float(request.form['Sgot']),
            float(request.form['TP']),
            float(request.form['ALB']),
            float(request.form['A/G Ratio'])
        ]

        # Reshape and scale input
        input_array = np.array([input_features])
        input_scaled = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        # Interpret result
        if prediction == 1:
            result = "⚠️ Liver Disease Detected"
        else:
            result = "✅ No Liver Disease Detected"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)