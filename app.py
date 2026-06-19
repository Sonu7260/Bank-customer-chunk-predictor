import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

# Initialize Flask Application
app = Flask(__name__)

# Load the Scikit-Learn RandomForestClassifier model file
MODEL_PATH = "random1_pxl.pxl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    model = None

# Single monolithic HTML string containing modern CSS styles, DOM structure, and UI logic
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bank Customer Churn Predictor</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --panel-bg: rgba(30, 41, 59, 0.7);
            --accent-color: #6366f1;
            --accent-hover: #4f46e5;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --border-ui: rgba(255, 255, 255, 0.08);
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', system-ui, sans-serif; }
        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px 20px;
        }
        .container {
            background: var(--panel-bg);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-ui);
            border-radius: 24px;
            width: 100%;
            max-width: 850px;
            padding: 40px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4);
        }
        header { text-align: center; margin-bottom: 35px; }
        header h1 { font-size: 2.2rem; font-weight: 700; background: linear-gradient(to right, #fff, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }
        header p { color: var(--text-muted); font-size: 0.95rem; }
        
        .grid-form {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 24px;
        }
        @media(max-width: 650px) { .grid-form { grid-template-columns: 1fr; } }
        
        .form-group { display: flex; flex-direction: column; gap: 8px; }
        .form-group label { font-size: 0.85rem; font-weight: 600; color: #c7d2fe; text-transform: uppercase; letter-spacing: 0.5px; }
        .form-group input, .form-group select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-ui);
            border-radius: 12px;
            padding: 12px 16px;
            color: var(--text-main);
            font-size: 1rem;
            transition: all 0.3s ease;
            outline: none;
        }
        .form-group input:focus, .form-group select:focus {
            border-color: var(--accent-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        }
        .form-group select option { background: #0f172a; color: var(--text-main); }
        
        .btn-container { grid-column: 1 / -1; margin-top: 15px; }
        .submit-btn {
            width: 100%;
            background: var(--accent-color);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 16px;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
        }
        .submit-btn:hover { background: var(--accent-hover); }
        .submit-btn:active { transform: scale(0.99); }
        
        .result-panel {
            grid-column: 1 / -1;
            margin-top: 30px;
            border-radius: 16px;
            padding: 24px;
            text-align: center;
            animation: fadeIn 0.5s ease-out;
        }
        .churn-yes { background: rgba(239, 68, 68, 0.15); border: 1px solid rgba(239, 68, 68, 0.3); color: #fca5a5; }
        .churn-no { background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.3); color: #86efac; }
        .result-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 4px; }
        .result-desc { font-size: 0.95rem; opacity: 0.9; }
        
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>Customer Churn Analytics</h1>
        <p>Input account parameters below to evaluate real-time model retention risk.</p>
    </header>

    <form method="POST" action="/" class="grid-form">
        <div class="form-group">
            <label for="credit_score">Credit Score</label>
            <input type="number" id="credit_score" name="credit_score" min="300" max="850" value="650" required>
        </div>
        
        <div class="form-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="age" min="18" max="100" value="38" required>
        </div>

        <div class="form-group">
            <label for="tenure">Tenure (Years)</label>
            <input type="number" id="tenure" name="tenure" min="0" max="10" value="5" required>
        </div>

        <div class="form-group">
            <label for="balance">Account Balance ($)</label>
            <input type="number" step="0.01" id="balance" name="balance" value="50000.00" required>
        </div>

        <div class="form-group">
            <label for="num_products">Number of Products</label>
            <input type="number" id="num_products" name="num_products" min="1" max="4" value="1" required>
        </div>

        <div class="form-group">
            <label for="estimated_salary">Estimated Salary ($)</label>
            <input type="number" step="0.01" id="estimated_salary" name="estimated_salary" value="85000.00" required>
        </div>

        <div class="form-group">
            <label for="geography">Geography Region</label>
            <select id="geography" name="geography">
                <option value="France">France (Base Region)</option>
                <option value="Germany">Germany</option>
                <option value="Spain">Spain</option>
            </select>
        </div>

        <div class="form-group">
            <label for="gender">Gender</label>
            <select id="gender" name="gender">
                <option value="Male">Male</option>
                <option value="Female">Female</option>
            </select>
        </div>

        <div class="form-group">
            <label for="has_crcard">Has Credit Card?</label>
            <select id="has_crcard" name="has_crcard">
                <option value="1">Yes</option>
                <option value="0">No</option>
            </select>
        </div>

        <div class="form-group">
            <label for="is_active">Is Active Member?</label>
            <select id="is_active" name="is_active">
                <option value="1">Yes</option>
                <option value="0">No</option>
            </select>
        </div>

        <div class="btn-container">
            <button type="submit" class="submit-btn">Run Prediction Analysis</button>
        </div>
        
        {% if prediction is not none %}
            {% if prediction == 1 %}
                <div class="result-panel churn-yes">
                    <div class="result-title">High Churn Risk Detected</div>
                    <div class="result-desc">The predictive model indicates this customer is likely to leave the bank.</div>
                </div>
            {% else %}
                <div class="result-panel churn-no">
                    <div class="result-title">Customer Stable</div>
                    <div class="result-desc">The model flags this account with high retention probability.</div>
                </div>
            {% endif %}
        {% endif %}
    </form>
</div>

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        if model is None:
            return "Error: Serialized model file ('random1_pxl.pxl') not found on server.", 500

        # Extract numerical characteristics from the form
        credit_score = float(request.form.get("credit_score", 650))
        age = float(request.form.get("age", 38))
        tenure = float(request.form.get("tenure", 5))
        balance = float(request.form.get("balance", 0.0))
        num_products = float(request.form.get("num_products", 1))
        has_crcard = float(request.form.get("has_crcard", 1))
        is_active = float(request.form.get("is_active", 1))
        estimated_salary = float(request.form.get("estimated_salary", 0.0))
        
        # Capture raw text string values from the interface dropdowns
        geography = request.form.get("geography", "France")
        gender = request.form.get("gender", "Male")
        
        # Categorical Mapping Logic (One-Hot Encoding to match the model's expected 10 inputs)
        geo_germany = 1.0 if geography == "Germany" else 0.0
        geo_spain = 1.0 if geography == "Spain" else 0.0
        gender_male = 1.0 if gender == "Male" else 0.0

        # Constructing the input vector array mapping exactly 10 parameters:
        # If your model dropped Gender instead of Geography during encoding, swap the categorical elements below.
        input_vector = np.array([[
            credit_score, age, tenure, balance, num_products, 
            has_crcard, is_active, estimated_salary, 
            geo_germany, geo_spain
        ]])
        
        # Alternatively, if your pipeline includes gender_male instead of geo_spain:
        # input_vector = np.array([[credit_score, age, tenure, balance, num_products, has_crcard, is_active, estimated_salary, geo_germany, gender_male]])

        # Execute prediction mapping matrix calculation
        pred_raw = model.predict(input_vector)
        prediction = int(pred_raw[0])

    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    # Bound to wildcard host port for seamless accessibility over AWS EC2 Public IPs
    app.run(host="0.0.0.0", port=5000)
