#!/bin/bash
# api/setup_production.sh (TensorFlow enabled)

echo "🔋 Setting up Production Battery SOH API with TensorFlow"
echo "======================================================"

# Check if model directory exists and has files
if [ ! -d "model" ]; then
    echo "❌ Model directory not found!"
    echo "📁 Please ensure your model directory is named 'model' (not 'models')"
    exit 1
fi

# Check for model files
joblib_count=$(find model -name "*.joblib" | wc -l)
keras_count=$(find model -name "*.keras" | wc -l) 
h5_count=$(find model -name "*.h5" | wc -l)
pkl_count=$(find model -name "*.pkl" | wc -l)

total_files=$((joblib_count + keras_count + h5_count + pkl_count))

if [ $total_files -eq 0 ]; then
    echo "❌ No model files found in model/ directory!"
    echo "📁 Expected files:"
    echo "   - *.joblib (scikit-learn/XGBoost models)"
    echo "   - *.keras (TensorFlow models)"
    echo "   - *.h5 (TensorFlow models)"
    echo "   - *.pkl (preprocessing objects)"
    exit 1
fi

echo "✅ Found $total_files model/preprocessing files:"
echo "   📊 Traditional ML models (.joblib): $joblib_count"
echo "   🧠 Deep learning models (.keras/.h5): $((keras_count + h5_count))"
echo "   🔧 Preprocessing files (.pkl): $pkl_count"

# List model files
echo ""
echo "📋 Files found in model/ directory:"
ls -la model/

# Clean up any previous containers
echo ""
echo "🧹 Cleaning up previous containers..."
docker-compose -f docker-compose.prod.yml down 2>/dev/null || true

# Build and start the production container
echo "🐳 Building production Docker image with TensorFlow..."
echo "⏳ This may take several minutes for TensorFlow installation..."
docker-compose -f docker-compose.prod.yml build

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed!"
    exit 1
fi

echo "🚀 Starting production API server..."
docker-compose -f docker-compose.prod.yml up -d

# Wait for the service to be ready
echo "⏳ Waiting for API to be ready (TensorFlow models take longer to load)..."
sleep 30

# Test the API
echo "🧪 Testing API health..."
max_attempts=10
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f http://localhost:5000/health > /dev/null 2>&1; then
        echo "✅ API is running successfully!"
        break
    else
        echo "⏳ Attempt $attempt/$max_attempts - API not ready yet..."
        sleep 10
        ((attempt++))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    echo "❌ API failed to start properly"
    echo "📋 Check logs with: docker-compose -f docker-compose.prod.yml logs"
    exit 1
fi

echo "🌐 API available at: http://localhost:5000"
echo "🏥 Health check: http://localhost:5000/health"
echo "📋 Models list: http://localhost:5000/models"

# Show detailed status
echo ""
echo "📊 API Status:"
curl -s http://localhost:5000/health | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f\"  Status: {data['status']}\")
    print(f\"  Models loaded: {sum(data['models_loaded'].values())}/5\")
    print(f\"  TensorFlow version: {data.get('tensorflow_version', 'N/A')}\")
    print(f\"  Predictions count: {data['predictions_count']}\")
    print(f\"  Scalers loaded: {data.get('scalers_loaded', False)}\")
except:
    print('  Could not parse health response')
"

echo ""
echo "📋 Available Models:"
curl -s http://localhost:5000/models | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for model in data['available_models']:
        print(f\"  ✅ {model['name']} ({model['type']})\")
except:
    print('  Could not parse models response')
"

echo ""
echo "🎉 Production setup complete!"
echo ""
echo "📋 Useful commands:"
echo "   View logs: docker-compose -f docker-compose.prod.yml logs -f"
echo "   Stop API:  docker-compose -f docker-compose.prod.yml down"
echo "   Restart:   docker-compose -f docker-compose.prod.yml restart"
echo "   Test API:  python3 test_production.py"