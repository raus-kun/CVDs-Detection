from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load models
models = {
    "RandomForest": joblib.load("models/random_forest_model.pkl"),
    "LogisticRegression": joblib.load("models/logreg_model.pkl"),
    "SVC": joblib.load("models/svc_model.pkl"),
    "KNN": joblib.load("models/knn_model.pkl"),
    "DecisionTree": joblib.load("models/decision_tree_model.pkl"),
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Receive input data
        data = request.form
        features = [
            int(data["age"]),
            int(data["gender"]),
            int(data["height"]),
            float(data["weight"]),
            int(data["ap_hi"]),
            int(data["ap_lo"]),
            int(data["cholesterol"]),
            int(data["gluc"]),
            int(data["smoke"]),
            int(data["alco"]),
            int(data["active"]),
            float(data["bmi"]),
        ]

        # Generate predictions for each model
        predictions = {
            name: int(model.predict([features])[0])
            for name, model in models.items()
        }

        # Average predictions (binary classification: 0 or 1)
        final_prediction = round(sum(predictions.values()) / len(predictions))

        # Render result page
        return render_template("result.html", 
                               final_prediction=final_prediction,
                               individual_predictions=predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
