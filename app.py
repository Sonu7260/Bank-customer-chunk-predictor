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

# Single monolithic template housing responsive HTML, CSS, and UI logic
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
        
        /* Results Section Styling */
        .result-panel {
            grid-column: 1 / -1;
            margin-top: 30px;
            border-radius: 16px;
            padding: 24px;
            animation: fadeIn 0.5s ease-out;
        }
        .churn-high { background: rgba(239, 68, 68, 0.12); border: 1px solid rgba(239, 68, 68, 0.25); }
        .churn-low { background: rgba(34, 197, 94, 0.12); border: 1px solid rgba(34, 197, 94, 0.25); }
        
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
        }
        .result-title { font-size: 1.3rem; font-weight: 700; }
        .churn-high .result-title { color: #fca5a5; }
        .churn-low .result-title { color: #86efac; }
        
        .prob-badge {
            font-size: 1.4rem;
            font-weight: 800;
            padding: 4px 12px;
            border-radius: 8px;
            background: rgba(15, 23, 42, 0.4);
        }
        .churn-high .prob-badge { color: #f87171; border: 1px solid rgba(239, 68, 68, 0.2); }
        .churn-low .prob-badge { color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.2); }

        /* Dynamic Visual Probability Track Meter */
        .meter-container {
            width: 100%;
            height: 10px;
            background: rgba(15, 23, 42, 0.6);
            border-radius: 999px;
            overflow: hidden;
            margin-bottom: 12px;
            border: 1px solid var(--border-ui);
        }
        .meter-fill {
            height: 100%;
            border-radius: 999px;
            transition: width 1s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .churn-high .meter-fill { background: linear-gradient(90deg, #f87171, #ef4444); }
        .churn-low .meter-fill { background: linear-gradient(90deg, #4ade80, #22c55e); }
        
        .result-desc { font-size: 0.95rem; color: var(--text-muted); line-height: 1.5; }
        
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>Customer Churn Analytics</h1>
        <p>Input account parameters below to evaluate real-time model retention risk profiles.</p>
    </header>

    <form method="POST" action="/" class="grid-form">
        <div class="form-group">
            <label for="credit_score">Credit Score</label>
            <input type="number" id="credit_score" name="credit_score" min="300" max="850" value="{{ form_values.credit_score if form_values else 650 }}" required>
        </div>
        
        <div class="form-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="age" min="18" max="100" value="{{ form_values.age if form_values else 38 }}" required>
        </div>

        <div class="form-group">
            <label for="tenure">Tenure (Years)</label>
            <input type="number" id="tenure" name="tenure" min="0" max="10" value="{{ form_values.tenure if form_values else 5 }}" required>
        </div>

        <div class="form-group">
            <label for="balance">Account Balance ($)</label>
            <input type="number" step="0.01" id="balance" name="balance" value="{{ form_values.balance if form_values else 50000.00 }}" required>
        </div>

        <div class="form-group">
            <label for="num_products">Number of Products</label>
            <input type="number" id="num_products" name="num_products" min="1" max="4" value="{{ form_values.num_products if form_values else 1 }}" required>
        </div>

        <div class="form-group">
            <label for="estimated_salary">Estimated Salary ($)</label>
            <input type="number" step="0.01" id="estimated_salary" name="estimated_salary" value="{{ form_values.estimated_salary if form_values else 85000.00 }}" required>
        </div>

        <div class="form-group">
            <label for="geography">Geography Region</label>
            <select id="geography" name="geography">
                <option value="France" {% if form_values and form_values.geography == 'France' %}selected{% endif %}>France</option>
                <option value="Germany" {% if form_values and form_values.geography == 'Germany' %}selected{% endif %}>Germany</option>
                <option value="Spain" {% if form_values and form_values.geography == 'Spain' %}selected{% endif %}>Spain</option>
            </select>
        </div>

        <div class="form-group">
            <label for="gender">Gender</label>
            <select id="gender" name="gender">
                <option value="Male" {% if form_values and form_values.gender == 'Male' %}selected{% endif %}>Male</option>
                <option value="Female" {% if form_values and form_values.gender == 'Female' %}selected{% endif %}>Female</option>
            </select>
        </div>

        <div class="form-group">
            <label for="has_crcard">Has Credit Card?</label>
            <select id="has_crcard" name="has_crcard">
                <option value="1" {% if form_values and form_values.has_crcard == '1' %}selected{% endif %}>Yes</option>
                <option value="0" {% if form_values and form_values.has_crcard == '0' %}selected{% endif %}>No</option>
            </select>
        </div>

        <div class="form-group">
            <label for="is_active">Is Active Member?</label>
            <select id="is_active" name="is_active">
                <option value="1" {% if form_values and form_values.is_active == '1' %}selected{% endif %}>Yes</option>
                <option value="0" {% if form_values and form_values.is_active == '0' %}selected{% endif %}>No</option>
            </select>
        </div>

        <div class="btn-container">
            <button type="submit" class="submit-btn">Run Probability Analysis</button>
        </div>
        
        {% if churn_probability is not none %}
            {% if churn_probability >= 50.0 %}
                <div class="result-panel churn-high">
                    <div class="result-header">
                        <div class="result-title">High Churn Risk Trend Detected</div>
                        <div class="prob-badge">{{ churn_probability }}%</div>
                    </div>
                    <div class="meter-container">
                        <div class="meter-fill" style="width: {{ churn_probability }}%;"></div>
                    </div>
                    <div class="result-desc">
                        The random forest system indicates a strong likelihood of customer exit. Targeted proactive retention protocols and direct account engagement are highly recommended.
                    </div>
                </div>
            {% else %}
                <div class="result-panel churn-low">
                    <div class="result-header">
                        <div class="result-title">Customer Stable (Low Churn Risk)</div>
                        <div class="prob-badge">{{ churn_probability }}%</div>
                    </div>
                    <div class="meter-container">
                        <div class="meter-fill" style="width: {{ churn_probability }}%;"></div>
                    </div>
                    <div class="result-desc">
                        Account signatures fall securely within standard target benchmarks. The predictive data patterns suggest strong baseline account loyalty.
                    </div>
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
    churn_probability = None
    form_values = None
    
    if request.method == "POST":
        if model is None:
            return "Error: Serialized model file ('random1_pxl.pxl') not found on server.", 500

        # Preserve selected form data to maintain UI state after refresh
        form_values = request.form

        # Extract features from the POST payload
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
        
        # Categorical Mapping Logic (One-Hot Encoding to match the model's 10 expected features)
        geo_germany = 1.0 if geography == "Germany" else 0.0
        geo_spain = 1.0 if geography == "Spain" else 0.0

        # Map vector structure explicitly to the 10 core inputs
        input_vector = np.array([[
            credit_score, age, tenure, balance, num_products, 
            has_crcard, is_active, estimated_salary, 
            geo_germany, geo_spain
        ]])
        
        # Calculate class probabilities
        # model.predict_proba returns an array: [[prob_class_0, prob_class_1]]
        probabilities = model.predict_proba(input_vector)
        
        # Target probability of class 1 (Churned) and convert to a clean percentage
        churn_probability = round(float(probabilities[0][1]) * 100, 2)

    return render_template_string(HTML_TEMPLATE, churn_probability=churn_probability, form_values=form_values)

if __name__ == "__main__":
    # Run application on port 5000, open to public traffic for EC2 routing
    app.run(host="0.0.0.0", port=5000)
