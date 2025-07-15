# api/test_production.py

import requests
import json
import time

BASE_URL = "http://localhost:5000"

def test_production_api():
    """Test the production API thoroughly"""
    print("🧪 Production API Test Suite")
    print("=" * 50)
    
    # Test 1: Health Check
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Models loaded: {sum(data['models_loaded'].values())}")
            print(f"   API version: {data['api_version']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False
    
    # Test 2: Models endpoint
    print("\n🔍 Testing models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models endpoint working")
            print(f"   Available models: {data['total_loaded']}")
            for model in data['available_models']:
                print(f"   - {model['name']} ({model['type']})")
        else:
            print(f"❌ Models endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Models endpoint error: {e}")
        return False
    
    # Test 3: Prediction
    print("\n🔍 Testing prediction with realistic battery data...")
    test_data = {
        "voltage_mean": 3.72,
        "voltage_std": 0.14,
        "voltage_start": 4.08,
        "voltage_end": 3.25,
        "voltage_drop": 0.83,
        "voltage_drop_rate": 0.022,
        "voltage_25pct": 3.48,
        "voltage_50pct": 3.72,
        "voltage_75pct": 3.92,
        "current_mean": 1.85,
        "current_std": 0.28,
        "temp_mean": 26.2,
        "temp_std": 2.3,
        "temp_range": 9.1,
        "ambient_temp": 25.0,
        "duration_hours": 2.7,
        "energy_wh": 19.2,
        "battery_id": "B0005",
        "cycle_number": 250,
        "model_type": "Random Forest"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(test_data),
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful")
            print(f"   SOH: {data['predicted_soh']}%")
            print(f"   Capacity: {data['capacity_ah']} Ah")
            print(f"   Model: {data['model_used']}")
            print(f"   Confidence: {data['confidence_score']}")
            print(f"   Prediction ID: {data['prediction_id']}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return False
    
    # Test 4: Load testing (multiple quick requests)
    print("\n🔍 Testing API load handling...")
    start_time = time.time()
    successful_requests = 0
    total_requests = 10
    
    for i in range(total_requests):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                successful_requests += 1
        except:
            pass
    
    end_time = time.time()
    print(f"✅ Load test completed")
    print(f"   Successful requests: {successful_requests}/{total_requests}")
    print(f"   Total time: {end_time - start_time:.2f} seconds")
    print(f"   Average response time: {(end_time - start_time)/total_requests:.3f} seconds")
    
    # Test 5: Error handling
    print("\n🔍 Testing error handling...")
    try:
        # Test invalid model
        invalid_data = {"model_type": "NonExistentModel"}
        response = requests.post(
            f"{BASE_URL}/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps(invalid_data),
            timeout=10
        )
        
        if response.status_code == 400:
            print("✅ Error handling working correctly")
        else:
            print(f"❌ Error handling failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Production test suite completed!")
    return True

if __name__ == "__main__":
    test_production_api()