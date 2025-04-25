import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import joblib
import os
import sys

os.makedirs('model', exist_ok=True)

def prepare_data():
    try:
        print("\n=== Loading Dataset ===")
        df = pd.read_csv('asd_data_csv.csv')
        print("Data loaded successfully. Shape:", df.shape)
        
        # Print column names for verification
        print("\nColumns in dataset:", df.columns.tolist())
        
        # Check for missing values
        print("\nMissing values before cleaning:")
        print(df.isnull().sum())

        # Convert binary traits
        binary_features = [
            'Speech Delay/Language Disorder',
            'Learning disorder',
            'Genetic_Disorders',
            'Depression',
            'Global developoental delay/intellectual disability',
            'Social/Behavioural Issues',
            'Anxiety_disorder',
            'Jaundice',
            'Family_member_with_ASD'
        ]
        
        print("\n=== Processing Binary Features ===")
        for feature in binary_features:
            if feature in df.columns:
                print(f"Processing {feature}")
                if df[feature].dtype == 'object':
                    df[feature] = df[feature].map({'Present': 1, 'Not present': 0})
                # Fill missing with 0 (assuming "Not present")
                df[feature] = df[feature].fillna(0)
            else:
                print(f"Warning: Column '{feature}' not found in dataset")

        # Handle Sex/Gender
        print("\n=== Processing Gender ===")
        if 'Sex' in df.columns:
            print("Found Sex column")
            # Convert to numeric first
            df['Sex'] = df['Sex'].replace({'Male': 1, 'Female': 0})
            # Fill missing with mode
            sex_mode = df['Sex'].mode()[0]
            print(f"Gender mode: {sex_mode}")
            df['Sex'] = df['Sex'].fillna(sex_mode)
            print("Gender missing values after filling:", df['Sex'].isnull().sum())
        else:
            print("Warning: 'Sex' column not found")

        # Define all expected features
        features = [
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

        print("\n=== Final Feature Check ===")
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            print(f"Error: Missing required features: {missing_features}")
            return None, None, None

        X = df[features]
        y = df['Outcome']

        print("\n=== Imputing Missing Values ===")
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        
        print("Data preparation completed successfully")
        return X_imputed, y, imputer

    except Exception as e:
        print(f"\n!!! Error in prepare_data: {str(e)}")
        return None, None, None

def train_and_save():
    print("\n=== Starting Training ===")
    X, y, imputer = prepare_data()
    
    if X is None or y is None:
        print("\n!!! Failed to prepare data. Cannot continue training.")
        sys.exit(1)

    print("\n=== Splitting Data ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    print("\n=== Scaling Features ===")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    print("\n=== Training Model ===")
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_scaled, y_train)
    
    # Verify dimensions
    print("\n=== Feature Verification ===")
    print(f"Expected Features: 12")
    print(f"Actual Features - X_train: {X_train.shape[1]}, Scaler: {scaler.n_features_in_}, Model: {model.n_features_in_}")
    
    if scaler.n_features_in_ != 12 or model.n_features_in_ != 12:
        print("\n!!! Error: Feature count mismatch!")
        sys.exit(1)
    
    # Evaluate
    print("\n=== Evaluating Model ===")
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.1%}")
    
    # Save artifacts
    print("\n=== Saving Artifacts ===")
    joblib.dump(imputer, 'model/imputer.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(model, 'model/model.pkl')
    print("Model artifacts saved to 'model/' directory")

if __name__ == '__main__':
    train_and_save()