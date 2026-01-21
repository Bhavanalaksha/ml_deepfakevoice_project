"""
Model Loader for Streamlit App
Handles loading trained models and making predictions
"""

import pickle
import numpy as np
import os
import json
from pathlib import Path

class ModelLoader:
    """Handles loading and using trained models"""
    
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.models = {}
        self.scalers = {}
        self.metadata = {}
        self._load_available_models()
    
    def _load_available_models(self):
        """Load all available models from models directory"""
        if not os.path.exists(self.models_dir):
            print(f"⚠️ Models directory '{self.models_dir}' not found. Using demo mode.")
            return
        
        # Find all model files
        model_files = Path(self.models_dir).glob('*_model.pkl')
        
        for model_file in model_files:
            model_name = model_file.stem.replace('_model', '')
            
            try:
                # Load model
                with open(model_file, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                
                # Load scaler
                scaler_file = model_file.parent / f"{model_name}_scaler.pkl"
                if scaler_file.exists():
                    with open(scaler_file, 'rb') as f:
                        self.scalers[model_name] = pickle.load(f)
                
                # Load metadata
                metadata_file = model_file.parent / f"{model_name}_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        self.metadata[model_name] = json.load(f)
                
                print(f"✅ Loaded model: {model_name}")
            
            except Exception as e:
                print(f"❌ Error loading {model_name}: {str(e)}")
    
    def get_available_models(self):
        """Return list of available model names"""
        return list(self.models.keys())
    
    def has_models(self):
        """Check if any models are loaded"""
        return len(self.models) > 0
    
    def predict(self, features, model_name='random_forest'):
        """
        Make prediction using specified model
        
        Args:
            features: Feature vector (numpy array)
            model_name: Name of the model to use
            
        Returns:
            prediction: 0 (fake) or 1 (real)
            confidence: Confidence score (0-1)
        """
        # Check if model exists
        if model_name not in self.models:
            # Fallback to demo prediction
            return self._demo_predict(features)
        
        try:
            # Reshape features if needed
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features if scaler is available
            if model_name in self.scalers:
                features = self.scalers[model_name].transform(features)
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(features)[0]
            
            # Get confidence (probability)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)[0]
                confidence = probabilities[prediction]
            else:
                # For models without predict_proba
                confidence = 0.8  # Default confidence
            
            return int(prediction), float(confidence)
        
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return self._demo_predict(features)
    
    def _demo_predict(self, features):
        """
        Demo prediction when no model is available
        Uses simple heuristic for demonstration
        """
        # Simple rule-based prediction for demo
        mean_feature = np.mean(features)
        
        if mean_feature > 0:
            prediction = 1  # Real
            confidence = min(0.5 + abs(mean_feature) * 0.1, 0.95)
        else:
            prediction = 0  # Fake
            confidence = min(0.5 + abs(mean_feature) * 0.1, 0.95)
        
        return prediction, confidence
    
    def get_metadata(self, model_name):
        """Get metadata for a specific model"""
        return self.metadata.get(model_name, {})
    
    def ensemble_predict(self, features, model_names=None):
        """
        Make ensemble prediction using multiple models
        
        Args:
            features: Feature vector
            model_names: List of model names to use (None = use all)
            
        Returns:
            prediction: Majority vote prediction
            confidence: Average confidence
        """
        if model_names is None:
            model_names = self.get_available_models()
        
        if not model_names:
            return self._demo_predict(features)
        
        predictions = []
        confidences = []
        
        for model_name in model_names:
            pred, conf = self.predict(features, model_name)
            predictions.append(pred)
            confidences.append(conf)
        
        # Majority vote
        final_prediction = int(np.round(np.mean(predictions)))
        
        # Average confidence
        final_confidence = np.mean(confidences)
        
        return final_prediction, final_confidence

# Global model loader instance
_model_loader = None

def get_model_loader():
    """Get or create global model loader instance"""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader
