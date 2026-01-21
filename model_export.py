"""
Model Export Script
Run this after training your model in the Jupyter notebook to save it for Streamlit
"""

import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def save_model(model, scaler, model_name='random_forest'):
    """
    Save trained model and scaler to disk
    
    Args:
        model: Trained sklearn or xgboost model
        scaler: Fitted StandardScaler
        model_name: Name for the model file
    """
    # Save model
    model_path = f'models/{model_name}_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to: {model_path}")
    
    # Save scaler
    scaler_path = f'models/{model_name}_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved to: {scaler_path}")

def save_model_metadata(model_name, accuracy, precision, recall, f1_score, training_samples):
    """Save model performance metrics"""
    import json
    
    metadata = {
        'model_name': model_name,
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'training_samples': int(training_samples),
    }
    
    metadata_path = f'models/{model_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"✅ Metadata saved to: {metadata_path}")

# Example usage (run this in your Jupyter notebook after training):
"""
# After training your Random Forest model
from model_export import save_model, save_model_metadata
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assume you have: clf (trained model), scaler (fitted scaler), 
# X_val_scaled, y_val_res (validation data)

# Make predictions to get metrics
y_pred = clf.predict(X_val_scaled)

# Calculate metrics
acc = accuracy_score(y_val_res, y_pred)
prec = precision_score(y_val_res, y_pred)
rec = recall_score(y_val_res, y_pred)
f1 = f1_score(y_val_res, y_pred)

# Save model
save_model(clf, scaler, model_name='random_forest')

# Save metadata
save_model_metadata(
    model_name='random_forest',
    accuracy=acc,
    precision=prec,
    recall=rec,
    f1_score=f1,
    training_samples=len(X_train_scaled)
)

# For XGBoost model (best_model from GridSearchCV)
save_model(best_model, scaler, model_name='xgboost')
"""

if __name__ == "__main__":
    import os
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    print("✅ Models directory created/verified")
    print("\n📝 To export your trained model:")
    print("1. Train your model in the Jupyter notebook")
    print("2. Import this script: from model_export import save_model, save_model_metadata")
    print("3. Call save_model(clf, scaler, 'random_forest')")
    print("4. Run the Streamlit app: streamlit run app_streamlit.py")
