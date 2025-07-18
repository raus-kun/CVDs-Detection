from flask import Flask, request, render_template
import joblib
import os
import numpy as np

app = Flask(__name__)

# This dictionary maps the value from the HTML form to the model's filename.
MODEL_FILENAMES = {
    "DecisionTree": "decision_tree_model.pkl",
    "KNN": "knn_model.pkl",
    "LogisticRegression": "logreg_model.pkl",
    "RandomForest": "random_forest_model.pkl",
    "SVC": "svc_model.pkl",
    "XGBoost": "xgboost_model.pkl"
}

@app.route('/')
def home():
    # Pass the model names to the homepage template for the dropdown menu.
    return render_template('index.html', model_names=MODEL_FILENAMES.keys())

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get the user's model choice from the form (e.g., "RandomForest").
        model_key = request.form['model']
        
        # 2. Look up the correct filename using the dictionary.
        model_filename = MODEL_FILENAMES.get(model_key)
        
        # If the key isn't found, return an error.
        if not model_filename:
            return "Error: Invalid model selected.", 400
            
        # 3. Construct the full path and load only the selected model.
        model_path = os.path.join('models', model_filename)
        model = joblib.load(model_path)
        
        # 4. Get the rest of the form data for prediction.
        features = [
                    int(request.form['age']),
                    int(request.form['gender']),
                    int(request.form['height']),
                    float(request.form['weight']),
                    int(request.form['ap_hi']),
                    int(request.form['ap_lo']),
                    int(request.form['cholesterol']),
                    int(request.form['gluc']),
                    int(request.form['smoke']),
                    int(request.form['alco']),
                    int(request.form['active']),
                    float(request.form['bmi'])
]
        final_features = [np.array(features)]
        
        # 5. Make the prediction.
        prediction = model.predict(final_features)
        
        # 6. Return the result to the results page.
        return render_template('result.html', model_used=model_key, prediction_result=prediction[0])
        
    except Exception as e:
        # This will catch any other errors and display them.
        return str(e)

if __name__ == '__main__':
    app.run(debug=True)