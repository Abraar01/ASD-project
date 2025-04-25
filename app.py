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
    app.run(host='0.0.0.0', port=10000, debug=True)
