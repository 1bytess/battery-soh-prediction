# api/create_mock_models_no_tf.py
"""
Script to create mock models for testing the Battery SOH API (MacBook compatible)
This version only creates scikit-learn models to avoid TensorFlow issues on macOS
"""

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.datasets import make_regression

print("🔧 Creating mock models for Battery SOH API testing (MacBook compatible)...")
print("📱 Skipping TensorFlow models to avoid macOS compatibility issues")

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Feature count (matching your actual feature engineering)
n_features = 17

print(f"📊 Using {n_features} features for mock models")

# Generate some mock training data
X_mock, y_mock = make_regression(
    n_samples=1000, 
    n_features=n_features, 
    noise=0.1, 
    random_state=42
)

# Scale y to SOH range (40-100%)
y_mock = 40 + (y_mock - y_mock.min()) / (y_mock.max() - y_mock.min()) * 60

print("🌳 Creating Random Forest model...")
# Create and save Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_mock, y_mock)
joblib.dump(rf_model, 'models/rf_model.joblib')
print("   ✅ Saved: models/rf_model.joblib")

print("🚀 Creating XGBoost model...")
# Create and save XGBoost model (if available)
try:
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    xgb_model.fit(X_mock, y_mock)
    joblib.dump(xgb_model, 'models/xgb_model.joblib')
    print("   ✅ Saved: models/xgb_model.joblib")
except ImportError:
    print("   ⚠️  XGBoost not installed, creating Linear Regression as substitute...")
    # Use Linear Regression as substitute for XGBoost
    xgb_substitute = LinearRegression()
    xgb_substitute.fit(X_mock, y_mock)
    joblib.dump(xgb_substitute, 'models/xgb_model.joblib')
    print("   ✅ Saved: models/xgb_model.joblib (Linear Regression substitute)")

print("🧠 Creating substitute models for deep learning...")

# Create SVR model as LSTM substitute
print("   Creating LSTM substitute (SVR)...")
lstm_substitute = SVR(kernel='rbf', C=1.0, gamma='scale')
lstm_substitute.fit(X_mock, y_mock)
joblib.dump(lstm_substitute, 'models/lstm_baseline_model.joblib')  # Note: .joblib instead of .keras
print("   ✅ Saved: models/lstm_baseline_model.joblib")

# Create another Random Forest as TCN substitute
print("   Creating TCN substitute (Random Forest variant)...")
tcn_substitute = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=123)
tcn_substitute.fit(X_mock, y_mock)
joblib.dump(tcn_substitute, 'models/tcn_model_final.joblib')  # Note: .joblib instead of .keras
print("   ✅ Saved: models/tcn_model_final.joblib")

# Create Linear Regression as BiLSTM substitute
print("   Creating BiLSTM substitute (Linear Regression)...")
bilstm_substitute = LinearRegression()
bilstm_substitute.fit(X_mock, y_mock)
joblib.dump(bilstm_substitute, 'models/bilstm_model_final.joblib')  # Note: .joblib instead of .keras
print("   ✅ Saved: models/bilstm_model_final.joblib")

print("\n🎉 Mock models created successfully!")
print("\nCreated files:")
print("📁 models/")

# List created files
for filename in sorted(os.listdir('models')):
    if filename.endswith('.joblib'):
        filepath = os.path.join('models', filename)
        size = os.path.getsize(filepath)
        size_kb = size / 1024
        print(f"   ├── {filename} ({size_kb:.1f} KB)")

print(f"\n📋 Model Summary:")
print(f"   Features: {n_features}")
print(f"   Training samples: {len(X_mock)}")
print(f"   SOH range: {y_mock.min():.1f}% - {y_mock.max():.1f}%")
print(f"   All models: Scikit-learn compatible (.joblib format)")

print(f"\n🔧 Technical Notes:")
print(f"   • Used .joblib format for all models (no TensorFlow)")
print(f"   • LSTM → SVR (Support Vector Regression)")
print(f"   • TCN → Random Forest variant")
print(f"   • BiLSTM → Linear Regression")
print(f"   • All models will work identically in the API")

print(f"\n🚀 Next steps:")
print(f"   1. Run: python3 app.py")
print(f"   2. Test: python3 test_api.py")
print(f"   3. Replace with your actual trained models when ready")

print(f"\n⚠️  Note: These are MOCK models for testing only!")
print(f"   Replace them with your actual trained models for real predictions.")
print(f"   When you get TensorFlow working, you can replace the .joblib files")
print(f"   with your actual .keras files and update the app.py accordingly.")