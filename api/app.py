# api/app.py (Production with TensorFlow)
import os
import numpy as np
import joblib
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
import traceback
import pickle

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)
CORS(app)

# Model storage
models = {}
scalers = {}
prediction_count = 0

# Feature columns - matching your model training order
FEATURE_COLUMNS = [
    'voltage_mean', 'voltage_std', 'voltage_start', 'voltage_end', 'voltage_drop',
    'voltage_drop_rate', 'voltage_25pct', 'voltage_50pct', 'voltage_75pct',
    'current_mean', 'current_std', 'temp_mean', 'temp_std', 'temp_range',
    'ambient_temp', 'duration_hours', 'energy_wh'
]

def load_models():
    """Load trained models and preprocessing objects"""
    global models, scalers
    model_dir = 'model'  # Note: using 'model' not 'models' to match your directory
    
    if not os.path.exists(model_dir):
        logger.error(f"Models directory '{model_dir}' not found")
        return
    
    try:
        # Load preprocessing objects
        preprocessing_path = os.path.join(model_dir, 'preprocessing_objects.joblib')
        if os.path.exists(preprocessing_path):
            scalers = joblib.load(preprocessing_path)
            logger.info("Loaded preprocessing objects")
        
        # Load ML features scaler if separate
        ml_scaler_path = os.path.join(model_dir, 'ml_features_scaler.joblib')
        if os.path.exists(ml_scaler_path):
            scalers['ml_features'] = joblib.load(ml_scaler_path)
            logger.info("Loaded ML features scaler")
        
        # Load prepared DL data for reference
        dl_data_path = os.path.join(model_dir, 'prepared_dl_data.pkl')
        if os.path.exists(dl_data_path):
            with open(dl_data_path, 'rb') as f:
                dl_data = pickle.load(f)
            logger.info("Loaded DL data reference")
    
    except Exception as e:
        logger.warning(f"Could not load preprocessing objects: {e}")
    
    # Model files mapping
    model_files = {
        'Random Forest': {
            'file': 'rf_model.joblib',
            'type': 'traditional_ml'
        },
        'XGBoost': {
            'file': 'xgb_model.joblib', 
            'type': 'traditional_ml'
        },
        'Simple LSTM': {
            'file': 'lstm_baseline_model.keras',
            'type': 'deep_learning'
        },
        'TCN': {
            'file': 'tcn_model_final.keras',
            'type': 'deep_learning'
        },
        'BiLSTM': {
            'file': 'bilstm_model_final.keras',
            'type': 'deep_learning'
        }
    }
    
    for model_name, model_info in model_files.items():
        model_path = os.path.join(model_dir, model_info['file'])
        
        try:
            if os.path.exists(model_path):
                if model_info['type'] == 'traditional_ml':
                    # Load scikit-learn/XGBoost models
                    models[model_name] = {
                        'model': joblib.load(model_path),
                        'type': 'traditional_ml'
                    }
                    logger.info(f"Loaded {model_name} from {model_info['file']}")
                    
                elif model_info['type'] == 'deep_learning':
                    # Load TensorFlow/Keras models
                    model = tf.keras.models.load_model(model_path, compile=False)
                    models[model_name] = {
                        'model': model,
                        'type': 'deep_learning'
                    }
                    logger.info(f"Loaded {model_name} from {model_info['file']}")
                    
            else:
                logger.warning(f"Model file not found: {model_info['file']}")
                
        except Exception as e:
            logger.error(f"Failed to load {model_name}: {str(e)}")

def prepare_features(data, model_type='traditional_ml'):
    """Prepare features for model prediction"""
    # Default values based on NASA dataset statistics
    defaults = {
        'voltage_mean': 3.7, 'voltage_std': 0.15, 'voltage_start': 4.1, 'voltage_end': 3.2,
        'voltage_drop': 0.9, 'voltage_drop_rate': 0.025, 'voltage_25pct': 3.45,
        'voltage_50pct': 3.7, 'voltage_75pct': 3.95, 'current_mean': 1.8,
        'current_std': 0.3, 'temp_mean': 25.5, 'temp_std': 2.1, 'temp_range': 8.5,
        'ambient_temp': 24.0, 'duration_hours': 2.5, 'energy_wh': 18.5
    }
    
    features = []
    for feature in FEATURE_COLUMNS:
        if feature in data:
            features.append(float(data[feature]))
        else:
            features.append(defaults.get(feature, 0.0))
    
    feature_array = np.array(features).reshape(1, -1)
    
    # Apply scaling if available
    if model_type == 'traditional_ml' and 'ml_features' in scalers:
        try:
            feature_array = scalers['ml_features'].transform(feature_array)
        except Exception as e:
            logger.warning(f"Could not apply ML scaling: {e}")
    
    elif model_type == 'deep_learning':
        # For deep learning models, we might need sequence data
        # This depends on your specific model architecture
        if feature_array.shape[1] == len(FEATURE_COLUMNS):
            # If models expect sequence input, reshape accordingly
            # Assuming your DL models expect (batch, timesteps, features)
            # Adjust this based on your actual model input shape
            feature_array = feature_array.reshape(1, 1, -1)
    
    return feature_array

def predict_soh(model_info, features, model_name):
    """Make SOH prediction with the given model"""
    try:
        model = model_info['model']
        model_type = model_info['type']
        
        if model_type == 'traditional_ml':
            # Traditional ML models
            prediction = model.predict(features)[0]
            
        elif model_type == 'deep_learning':
            # Deep learning models
            prediction = model.predict(features, verbose=0)[0]
            
            # Handle different output shapes
            if isinstance(prediction, np.ndarray):
                if prediction.shape == (1,):
                    prediction = prediction[0]
                elif len(prediction.shape) > 0:
                    prediction = prediction[0] if len(prediction) > 0 else prediction
        
        # Convert to SOH percentage
        if isinstance(prediction, (list, np.ndarray)):
            prediction = float(prediction[0]) if len(prediction) > 0 else float(prediction)
        else:
            prediction = float(prediction)
        
        # Convert to percentage if needed
        if prediction < 1:
            soh = prediction * 100
        else:
            soh = prediction
        
        # Bound to realistic range
        soh = max(40, min(99, soh))
        return soh
        
    except Exception as e:
        logger.error(f"Prediction error with {model_name}: {str(e)}")
        raise

def calculate_capacity(soh, nominal_capacity=2.0):
    """Calculate capacity based on SOH"""
    return (soh / 100) * nominal_capacity

def calculate_confidence(soh, cycle_number, voltage_std, model_type):
    """Calculate prediction confidence"""
    confidence = 0.85
    
    # Base confidence adjustments
    if cycle_number < 50:
        confidence -= 0.1
    elif cycle_number > 1500:
        confidence -= 0.15
    
    if voltage_std > 0.3:
        confidence -= 0.1
        
    if soh < 50 or soh > 95:
        confidence -= 0.05
    
    # Model-specific confidence adjustments
    if model_type == 'deep_learning':
        confidence += 0.05  # DL models might be more confident
    
    return max(0.5, min(0.98, confidence))

def calculate_degradation_rate(soh, cycle_number):
    """Calculate degradation rate per cycle"""
    if cycle_number > 0:
        return (100 - soh) / cycle_number
    return 0.05

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    models_status = {}
    for model_name in ['Random Forest', 'XGBoost', 'Simple LSTM', 'TCN', 'BiLSTM']:
        models_status[f"{model_name.lower().replace(' ', '_')}_model"] = model_name in models
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': models_status,
        'total_models': len(models),
        'predictions_count': prediction_count,
        'api_version': '1.0.0',
        'tensorflow_version': tf.__version__,
        'scalers_loaded': len(scalers) > 0
    })

@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    model_info = []
    for model_name, model_data in models.items():
        model_info.append({
            'name': model_name,
            'type': model_data['type'],
            'loaded': True,
            'description': f"{model_name} model for SOH prediction"
        })
    
    return jsonify({
        'available_models': model_info,
        'total_loaded': len(models),
        'feature_columns': FEATURE_COLUMNS,
        'scalers_available': list(scalers.keys()) if scalers else []
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    global prediction_count
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'model_type' not in data:
            return jsonify({'error': 'Missing required field: model_type'}), 400
        
        model_type = data['model_type']
        
        if model_type not in models:
            return jsonify({
                'error': f'Model "{model_type}" not available',
                'available_models': list(models.keys())
            }), 400
        
        # Get model info
        model_info = models[model_type]
        
        # Prepare features based on model type
        features = prepare_features(data, model_info['type'])
        
        # Make prediction
        soh = predict_soh(model_info, features, model_type)
        
        # Calculate metrics
        capacity_ah = calculate_capacity(soh)
        confidence_score = calculate_confidence(
            soh, 
            data.get('cycle_number', 100), 
            data.get('voltage_std', 0.15),
            model_info['type']
        )
        degradation_rate = calculate_degradation_rate(soh, data.get('cycle_number', 100))
        
        prediction_count += 1
        
        result = {
            'predicted_soh': round(soh, 1),
            'capacity_ah': round(capacity_ah, 2),
            'model_used': model_type,
            'model_type': model_info['type'],
            'confidence_score': round(confidence_score, 2),
            'degradation_rate': round(degradation_rate, 3),
            'timestamp': datetime.now().isoformat(),
            'prediction_id': prediction_count
        }
        
        logger.info(f"Prediction {prediction_count}: SOH={soh:.1f}% using {model_type} ({model_info['type']})")
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'Internal server error during prediction',
            'details': str(e)
        }), 500

@app.route('/', methods=['GET'])
def root():
    """API documentation"""
    return jsonify({
        'service': 'Battery SOH Prediction API',
        'version': '1.0.0',
        'description': 'ESCL Chungnam University - Machine Learning-Based Battery State of Health Estimation',
        'endpoints': {
            '/health': 'GET - Health check',
            '/models': 'GET - List available models', 
            '/predict': 'POST - Predict battery SOH'
        },
        'models_loaded': len(models),
        'tensorflow_enabled': True,
        'tensorflow_version': tf.__version__,
        'status': 'ready'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Battery SOH Prediction API with TensorFlow")
    logger.info("ESCL Chungnam University Research Project")
    logger.info(f"TensorFlow version: {tf.__version__}")
    
    # Load models and scalers
    load_models()
    
    if not models:
        logger.error("No models loaded! Please check the model directory.")
        exit(1)
    else:
        logger.info(f"Loaded {len(models)} models: {list(models.keys())}")
    
    if scalers:
        logger.info(f"Loaded scalers: {list(scalers.keys())}")
    
    # Production settings
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)