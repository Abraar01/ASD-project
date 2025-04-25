'''
from flask import Flask, render_template, request
import joblib
import numpy as np
import sqlite3

app = Flask(__name__)

# Load artifacts
try:
    imputer = joblib.load('model/imputer.pkl')
    scaler = joblib.load('model/scaler.pkl')
    model = joblib.load('model/model.pkl')
    print("Model artifacts loaded successfully.")
    print(f"Scaler expects: {scaler.n_features_in_} features")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    
    if request.method == 'POST':
        try:
            # Get all 12 features in EXACT order used during training
            input_data = [
                float(request.form['social_responsiveness']),
                float(request.form['age']),
                int(request.form['speech_delay']),
                int(request.form['learning_disorder']),
                int(request.form['genetic_disorders']),
                int(request.form['depression']),
                int(request.form['developmental_delay']),
                int(request.form['behavioral_issues']),
                int(request.form['anxiety']),
                int(request.form['sex']),
                int(request.form['jaundice']),
                int(request.form['family_asd'])
            ]
            
            # Verify feature count
            if len(input_data) != 12:
                raise ValueError(f"Expected 12 features, got {len(input_data)}")
            
            # Preprocess
            X = np.array(input_data).reshape(1, -1)
            X_imputed = imputer.transform(X)  # Handle any missing values
            X_scaled = scaler.transform(X_imputed)
            
            # Predict
            prediction = model.predict(X_scaled)[0]
            result = "ASD Detected" if prediction == 1 else "No ASD Detected"
            
        except Exception as e:
            result = f"Error: {str(e)}"
            print(result)
    
    return render_template('ASD_project.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
'''

'''
from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Load model artifacts
try:
    imputer = joblib.load('model/imputer.pkl')
    scaler = joblib.load('model/scaler.pkl')
    model = joblib.load('model/model.pkl')
    print("Model artifacts loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/screening', methods=['GET', 'POST'])
def screening():
    result = None
    if request.method == 'POST':
        try:
            input_data = [
                float(request.form.get('social_responsiveness', 0)),
                float(request.form.get('age', 0)),
                int(request.form.get('speech_delay', 0)),
                int(request.form.get('learning_disorder', 0)),
                int(request.form.get('genetic_disorders', 0)),
                int(request.form.get('depression', 0)),
                int(request.form.get('developmental_delay', 0)),
                int(request.form.get('behavioral_issues', 0)),
                int(request.form.get('anxiety', 0)),
                int(request.form.get('sex', 0)),
                int(request.form.get('jaundice', 0)),
                int(request.form.get('family_asd', 0))
            ]
            
            X = np.array(input_data).reshape(1, -1)
            X_imputed = imputer.transform(X)
            X_scaled = scaler.transform(X_imputed)
            prediction = model.predict(X_scaled)[0]
            
            result = {
                'prediction': "ASD Detected" if prediction == 1 else "No ASD Detected",
                'probability': float(model.predict_proba(X_scaled)[0][1]),
                'input_values': input_data
            }
            
        except Exception as e:
            result = {'error': str(e)}
    
    return render_template('screening.html', result=result)

@app.route('/research')
def research():
    return render_template('research.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
'''

# from flask import Flask, render_template, request
# import joblib
# import numpy as np
# import sys

# app = Flask(__name__)

# def load_models():
#     try:
#         model = joblib.load('model/model.pkl')
#         scaler = joblib.load('model/scaler.pkl')
#         print("✅ Models loaded successfully", file=sys.stderr)
#         return model, scaler
#     except Exception as e:
#         print(f"❌ Model loading failed: {e}", file=sys.stderr)
#         raise e

# try:
#     model, scaler = load_models()
# except:
#     print("⚠️ Continuing without models - prediction will fail", file=sys.stderr)

# @app.route('/')
# def home():
#     print("🏠 Homepage accessed", file=sys.stderr)
#     return render_template('home.html')

# @app.route('/screening')
# def screening():
#     return render_template('screening.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         features = [
#             float(request.form['social_responsiveness']),
#             float(request.form['age']),
#             int(request.form['speech_delay']),
#             int(request.form['learning_disorder']),
#             int(request.form['genetic_disorders']),
#             int(request.form['depression']),
#             int(request.form['developmental_delay']),
#             int(request.form['behavioral_issues']),
#             int(request.form['anxiety']),
#             int(request.form['sex']),
#             int(request.form['jaundice']),
#             int(request.form['family_asd'])
#         ]
        
#         print(f"🔮 Prediction request with features: {features}", file=sys.stderr)
        
#         X = scaler.transform(np.array(features).reshape(1, -1))
#         pred = model.predict(X)[0]
#         proba = model.predict_proba(X)[0][1]
        
#         return render_template('screening.html', 
#                             result={
#                                 'diagnosis': 'ASD Detected' if pred == 1 else 'No ASD Detected',
#                                 'probability': round(proba*100, 1)
#                             })
    
#     except Exception as e:
#         print(f"🔥 Prediction error: {e}", file=sys.stderr)
#         return render_template('screening.html', error=str(e))

# if __name__ == '__main__':
#     print("🚀 Starting Flask server...", file=sys.stderr)
#     app.run(host='0.0.0.0', port=5000, debug=True)
#     print("🛑 Server stopped", file=sys.stderr)

from flask import Flask, render_template, request, url_for
import joblib
import numpy as np
import logging
from datetime import datetime
import sys

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='asd_detection.log'
)

logger = logging.getLogger(__name__)

# Define expected features in exact order used during training
EXPECTED_FEATURES = [
    'Social_Responsiveness_Scale',
    'Age_Years',
    'Speech Delay/Language Disorder',
    'Learning disorder',
    'Genetic_Disorders',
    'Depression',
    'Global developoental delay/intellectual disability',
    'Social/Behavioural Issues',
    'Anxiety_disorder',
    'Sex',
    'Jaundice',
    'Family_member_with_ASD'
]

def load_artifacts():
    """Load all required ML artifacts with error handling"""
    try:
        imputer = joblib.load('model/imputer.pkl')
        scaler = joblib.load('model/scaler.pkl')
        model = joblib.load('model/model.pkl')
        
        # Verify feature counts match
        assert imputer.n_features_in_ == len(EXPECTED_FEATURES)
        assert scaler.n_features_in_ == len(EXPECTED_FEATURES)
        assert model.n_features_in_ == len(EXPECTED_FEATURES)
        
        logger.info("All artifacts loaded successfully")
        return imputer, scaler, model
        
    except Exception as e:
        logger.error(f"Error loading artifacts: {str(e)}")
        raise e

try:
    imputer, scaler, model = load_artifacts()
except Exception as e:
    logger.critical(f"Failed to load ML artifacts: {str(e)}")
    sys.exit(1)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/asd-detection')
def asd_detection():
    return render_template('screening.html', 
                         features=EXPECTED_FEATURES,
                         page_title="ASD Screening Tool")
    
@app.route('/our-team')
def our_team():
    return render_template('research.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Extract and validate form data
        form_data = {
    'Social_Responsiveness_Scale': request.form.get('Social_Responsiveness_Scale', '0'),
    'Age_Years': request.form.get('Age_Years', '0'),
    'Speech Delay/Language Disorder': int(request.form.get('Speech_Delay_Language_Disorder', 0)),
    'Learning disorder': int(request.form.get('Learning_disorder', 0)),
    'Genetic_Disorders': int(request.form.get('Genetic_Disorders', 0)),
    'Depression': int(request.form.get('Depression', 0)),
    'Global developoental delay/intellectual disability': int(request.form.get('Global_developmental_delay_intellectual_disability', 0)),
    'Social/Behavioural Issues': int(request.form.get('Social_Behavioural_Issues', 0)),
    'Anxiety_disorder': int(request.form.get('Anxiety_disorder', 0)),
    'Sex': int(request.form.get('Sex', 0)),  # 0=Female, 1=Male
    'Jaundice': int(request.form.get('Jaundice', 0)),
    'Family_member_with_ASD': int(request.form.get('Family_member_with_ASD', 0))
}
        
        logger.info(f"Raw form data: {form_data}")

        # 2. Convert to array in correct feature order
        features = np.array([form_data[feat] for feat in EXPECTED_FEATURES]).reshape(1, -1)
        
        # 3. Process exactly like training pipeline
        features_imputed = imputer.transform(features)
        features_scaled = scaler.transform(features_imputed)
        
        # 4. Make prediction
        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0][1]
        
        # 5. Prepare result
        result = {
            'diagnosis': 'ASD Detected' if pred == 1 else 'No ASD Detected',
            'probability': round(proba*100, 1),
            'confidence': 'high' if proba > 0.85 else 'medium' if proba > 0.6 else 'low',
            'features': {
                feat: float(form_data[feat]) if feat in ['Social_Responsiveness_Scale', 'Age_Years'] 
                        else int(form_data[feat]) 
                for feat in EXPECTED_FEATURES
            },
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return render_template('screening.html',
                            result=result,
                            features=EXPECTED_FEATURES,
                            form_data=form_data)

    except ValueError as ve:
        error_msg = f"Invalid input: {str(ve)}"
        logger.error(error_msg)
        return render_template('screening.html',
                            error=error_msg,
                            features=EXPECTED_FEATURES,
                            form_data=request.form)

    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        logger.error(error_msg)
        return render_template('screening.html',
                            error="An error occurred during processing. Please try again.",
                            features=EXPECTED_FEATURES,
                            form_data=request.form)

if __name__ == '__main__':
    print("\n\n=== Server starting ===")
    print(f"Access your app at: http://127.0.0.1:5003")
    app.run(host='0.0.0.0', port=5003, debug=True)