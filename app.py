# from flask import Flask, request, jsonify, render_template
# import joblib
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # Load trained model, label encoder, and scaler
# model = joblib.load("mental_health_knn_model.pkl")
# label_encoder = joblib.load("label_encoder.pkl")
# scaler = joblib.load("scaler.pkl")

# # Feature columns used during training
# feature_columns = [
#     "Sexual Activity", "Concentration", "Optimisim", "Mood Swing", "Suicidal Thought",
#     "Anorxia", "Authority Respect", "Try-Explanation", "Aggressive Response",
#     "Ignore & Move-On", "Nervous Break-down", "Admit Mistakes", "Overthinking",
#     "Sadness", "Euphoric", "Exhausted", "Sleep Dissorder"
# ]

# # Mapping for categorical variables
# binary_mapping = {"Yes": 1, "No": 0}
# ordinal_mapping = {"Seldom": 1, "Sometimes": 2, "Usually": 3, "Most Often": 4}

# @app.route('/')
# def home():
#     return render_template("index.html")

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.json
#         input_data = []
        
#         for col in feature_columns:
#             value = data[col]
#             if col in binary_mapping:
#                 input_data.append(binary_mapping[value])
#             elif col in ordinal_mapping:
#                 input_data.append(ordinal_mapping[value])
#             else:
#                 input_data.append(float(value))
        
#         # Convert to numpy array and scale
#         input_array = np.array(input_data).reshape(1, -1)
#         input_scaled = scaler.transform(input_array)
        
#         # Make prediction
#         prediction = model.predict(input_scaled)
#         predicted_label = label_encoder.inverse_transform(prediction)[0]
        
#         return jsonify({"Predicted Mental Health Condition": predicted_label})
    
#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == "__main__":
#     app.run(debug=True, host="0.0.0.0", port=5000)

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__, template_folder="templates")

# Load the trained model, scaler, and label encoder
model = joblib.load("mental_health_knn_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = [float(request.form.get(key)) for key in request.form.keys()]
        
        # Transform input using scaler
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_data)

        # Decode the prediction label
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        return render_template("index.html", prediction=predicted_label)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
