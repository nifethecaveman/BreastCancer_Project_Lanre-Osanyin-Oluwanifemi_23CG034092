from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('model/breast_cancer_model.pkl')
scaler = joblib.load('model/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get values from form
    features = [float(x) for x in request.form.values()]
    final_features = np.array([features])
    
    # Scale and Predict
    scaled_features = scaler.transform(final_features)
    prediction = model.predict(scaled_features)
    
    output = 'Malignant' if prediction[0] == 1 else 'Benign'
    
    return render_template('index.html', prediction_text=f'Analysis Result: {output}')

if __name__ == "__main__":
    app.run(debug=True)