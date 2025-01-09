from flask import Flask, render_template, request
from joblib import load
import numpy as np

app = Flask(__name__)

# Load the model
try:
    model = load('decision_tree_model.pkl')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract inputs from the form
        step = float(request.form['step'])
        type_val = request.form['type']
        amount = float(request.form['amount'])
        nameOrig = request.form['nameOrig']
        oldbalanceOrig = float(request.form['oldbalanceOrig'])
        newbalanceOrig = float(request.form['newbalanceOrig'])
        nameDest = request.form['nameDest']
        oldbalanceDest = float(request.form['oldbalanceDest'])
        newbalanceDest = float(request.form['newbalanceDest'])

        # Encode the categorical 'type' feature
        type_mapping = {'CASH_OUT': 1, 'TRANSFER': 4, 'PAYMENT': 3, 'CASH_IN': 0, 'DEBIT': 2}
        if type_val not in type_mapping:
            return "Invalid transaction type!"
        type_encoded = type_mapping[type_val]

        # Prepare the input data
        input_data = np.array([[step, type_encoded, amount, oldbalanceOrig, newbalanceOrig, oldbalanceDest, newbalanceDest]])
        
        # Predict using the model
        prediction = model.predict(input_data)
        is_fraud = prediction[0]  # 0 = Not Fraud, 1 = Fraud
        
        # Return the result
        return render_template('index.html', prediction_text=f"Is Fraudulent Transaction: {is_fraud}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error during prediction: {e}")

if __name__ == '__main__':
    app.run(debug=True)
