import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# ----------------------------------------------------
# 1. Configuration & Model Loading
# ----------------------------------------------------
MODEL_PATH = "random1_pxl.pxl"

def load_model():
    """Loads the serialized random forest model safely."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please place it in the app directory.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    # Safely retrieve expected feature dimension (Model contains 10 features based on metadata)
    n_features = getattr(model, "n_features_in_", 10) 
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    n_features = 10

# ----------------------------------------------------
# 2. Categorical UI Mapping Configuration
# ----------------------------------------------------
# Replace these placeholders with your actual dataset features and categorical text options.
FEATURE_MAPPING = {
    "Feature_1": {"Low": 0, "Medium": 1, "High": 2},
    "Feature_2": {"No": 0, "Yes": 1},
    "Feature_3": {"Type A": 0, "Type B": 1, "Type C": 2, "Type D": 3},
    "Feature_4": {"Urban": 0, "Suburban": 1, "Rural": 2},
    "Feature_5": {"Single": 0, "Married": 1, "Divorced": 2},
    "Feature_6": {"Private": 0, "Public": 1, "Government": 2},
    "Feature_7": {"None": 0, "Basic": 1, "Premium": 2},
    "Feature_8": {"Tier 1": 0, "Tier 2": 1},
    "Feature_9": {"Spring": 0, "Summer": 1, "Autumn": 2, "Winter": 3},
    "Feature_10": {"Low Risk": 0, "High Risk": 1}
}

# Mapping output target class indexes back to human-readable names
TARGET_CLASSES = {
    0: "Class Alpha (Negative / Healthy)",
    1: "Class Beta (Positive / Alert)"
}

# ----------------------------------------------------
# 3. Unified Elegant Glassmorphism Frontend (HTML/CSS)
# ----------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Prediction Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <style>
        body {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            min-height: 100vh;
        }
        .glass-card {
            background: rgba(255, 255, 255, 0.03);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
    </style>
</head>
<body class="text-slate-200 font-sans antialiased flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">

    <div class="max-w-4xl w-full space-y-8">
        <div class="text-center">
            <h1 class="text-4xl font-extrabold tracking-tight text-white sm:text-5xl bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent">
                Random Forest Classifier
            </h1>
            <p class="mt-2 text-sm text-slate-400">
                AWS Deployment Interface • High Performance Machine Learning Inference
            </p>
        </div>

        {% if error %}
        <div class="rounded-xl bg-red-950/50 border border-red-500/30 p-4 text-sm text-red-300 glass-card">
            <div class="flex items-center space-x-2">
                <span class="font-bold">System Alert:</span> <span>{{ error }}</span>
            </div>
        </div>
        {% endif %}

        {% if prediction is not none %}
        <div class="rounded-2xl p-6 border {% if prediction == 1 or 'Beta' in prediction_label %}border-amber-500/30 bg-amber-950/20{% else %}border-emerald-500/30 bg-emerald-950/20{% endif %} glass-card shadow-2xl transition duration-500 transform scale-102">
            <h3 class="text-xs uppercase tracking-wider text-slate-400 font-semibold">Inference Engine Result</h3>
            <div class="mt-2 flex flex-col sm:flex-row justify-between items-start sm:items-center">
                <div>
                    <span class="text-2xl font-bold text-white block">{{ prediction_label }}</span>
                    <span class="text-xs text-slate-400">Model Output Node Index: {{ prediction }}</span>
                </div>
                {% if probability %}
                <div class="mt-3 sm:mt-0 bg-slate-900/60 px-4 py-2 rounded-xl border border-slate-700/50 text-right">
                    <span class="text-xs text-slate-400 block">Confidence Score</span>
                    <span class="text-xl font-mono font-bold text-indigo-400">{{ probability }}%</span>
                </div>
                {% endif %}
            </div>
        </div>
        {% endif %}

        <div class="glass-card rounded-3xl p-8 shadow-xl">
            <form action="/" method="POST" class="space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {% for feature, choices in mapping.items() %}
                    <div class="flex flex-col space-y-2">
                        <label for="{{ feature }}" class="text-sm font-medium text-slate-300">
                            {{ feature | replace('_', ' ') }}
                        </label>
                        <select name="{{ feature }}" id="{{ feature }}" required
                                class="block w-full rounded-xl bg-slate-900/60 border border-slate-700/60 px-4 py-3 text-slate-200 shadow-sm focus:border-indigo-500 focus:ring-2 focus:ring-indigo-500/20 focus:outline-none transition-all duration-200 cursor-pointer">
                            {% for choice in choices.keys() %}
                            <option value="{{ choice }}" {% if previous_inputs and previous_inputs.get(feature) == choice %}selected{% endif %}>
                                {{ choice }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>
                    {% endfor %}
                </div>

                <div class="pt-4">
                    <button type="submit" 
                            class="w-full flex justify-center py-4 px-6 rounded-xl border border-transparent text-sm font-semibold text-white bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 focus:ring-offset-slate-900 shadow-lg shadow-indigo-600/20 transform active:scale-98 transition-all duration-150 cursor-pointer">
                        Execute Real-Time Prediction
                    </button>
                </div>
            </form>
        </div>
    </div>

</body>
</html>
"""

# ----------------------------------------------------
# 4. Routing & Inference Controllers
# ----------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    prediction_label = None
    probability = None
    error = None
    previous_inputs = {}

    if request.method == "POST":
        previous_inputs = request.form.to_dict()
        if model is None:
            error = "Inference aborted: Model artifact is missing or corrupted on the server."
            return render_template_string(HTML_TEMPLATE, mapping=FEATURE_MAPPING, error=error)
        
        try:
            # Process values sequentially through the mapped categorical dropdown keys
            features_ordered = []
            for feat_name in sorted(FEATURE_MAPPING.keys(), key=lambda x: int(x.split('_')[1])):
                selected_str_val = request.form.get(feat_name)
                # Map option string back to numeric label index
                numeric_val = FEATURE_MAPPING[feat_name][selected_str_val]
                features_ordered.append(numeric_val)

            # Build matrix array for Scikit-Learn
            input_array = np.array([features_ordered], dtype=np.float64)

            # Execute predictions
            pred_idx = int(model.predict(input_array)[0])
            prediction = pred_idx
            prediction_label = TARGET_CLASSES.get(pred_idx, f"Class Key {pred_idx}")

            # Extract class probability if supported
            if hasattr(model, "predict_proba"):
                prob_scores = model.predict_proba(input_array)[0]
                probability = round(float(prob_scores[pred_idx]) * 100, 2)

        except Exception as e:
            error = f"Inference execution failure: {str(e)}"

    return render_template_string(
        HTML_TEMPLATE,
        mapping=FEATURE_MAPPING,
        prediction=prediction,
        prediction_label=prediction_label,
        probability=probability,
        error=error,
        previous_inputs=previous_inputs
    )

if __name__ == "__main__":
    # WSGI production configurations for local fallback testing
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
