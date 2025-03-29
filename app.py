import os
from flask import Flask, render_template, request, jsonify
import pandas as pd
from joblib import load
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the model
model_path = "model_joblib.pkl"

if os.path.exists(model_path):
    LGBM = load(model_path)
    print("✅ Model loaded successfully!")
else:
    raise FileNotFoundError(f"⚠️ Model file not found: {model_path}. Please upload it.")

# Ensure the 'uploads' folder exists
os.makedirs("uploads", exist_ok=True)

@app.route("/")
def index():
    return render_template("Index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template("Index.html", prediction_text="⚠️ No file uploaded!")

    file = request.files['file']
    if file.filename == '':
        return render_template("Index.html", prediction_text="⚠️ No file selected!")

    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    file.save(file_path)

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return render_template("Index.html", prediction_text=f"⚠️ Error reading file: {str(e)}")

    # Ensure input columns match model training columns
    required_features = LGBM.feature_name_  # Get required feature names from the model
    missing_features = [col for col in required_features if col not in df.columns]

    if missing_features:
        return render_template("Index.html", prediction_text=f"⚠️ Missing columns: {missing_features}")

    # Make predictions
    try:
        y_test_pred_lgbm = LGBM.predict_proba(df[required_features])[:, 1]
    except Exception as e:
        return render_template("Index.html", prediction_text=f"⚠️ Error in prediction: {str(e)}")

    # Generate prediction result
    prediction_results = ["Transaction is fraud" if p > 0.5 else "Transaction is non-fraud" for p in y_test_pred_lgbm]

    return render_template("Index.html", prediction_text=f"✅ {prediction_results[0]}")  # Show first result

if __name__ == "__main__":
    app.run(debug=True)  # No Ngrok, run locally
