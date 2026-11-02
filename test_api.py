import requests
import json
import sys
import time

BASE_URL = "http://localhost:8000"

# Test data
EXAMPLE_INPUT = {
    "age": 39,
    "workclass": "State-gov",
    "fnlwgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}

EXAMPLE_INPUT_HIGH_INCOME = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlwgt": 209642,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15000,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States"
}


def test_health():
    """Test health endpoint"""
    print("Testing /health...")
    response = requests.get(f"{BASE_URL}/health", timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert "preprocessor_loaded" in data
    print(f"  [PASS] Health check passed: {data}")
    return True


def test_model_info():
    """Test model info endpoint"""
    print("Testing /model-info...")
    response = requests.get(f"{BASE_URL}/model-info", timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "features_count" in data
    print(f"  [PASS] Model info: {data['model_name']} ({data['features_count']} features)")
    return True


def test_predict():
    """Test single prediction"""
    print("Testing /predict (low income example)...")
    response = requests.post(f"{BASE_URL}/predict", json=EXAMPLE_INPUT, timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "confidence" in data
    assert data["prediction"] in [">50K", "<=50K"]
    assert 0 <= data["probability"] <= 1
    assert data["confidence"] in ["high", "medium", "low"]
    print(f"  [PASS] Prediction: {data['prediction']} (prob={data['probability']:.4f}, confidence={data['confidence']})")
    return True


def test_predict_high_income():
    """Test prediction for likely high income"""
    print("Testing /predict (high income example)...")
    response = requests.post(f"{BASE_URL}/predict", json=EXAMPLE_INPUT_HIGH_INCOME, timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [">50K", "<=50K"]
    print(f"  [PASS] Prediction: {data['prediction']} (prob={data['probability']:.4f}, confidence={data['confidence']})")
    return True


def test_batch_predict():
    """Test batch prediction"""
    print("Testing /predict/batch...")
    batch_input = {"instances": [EXAMPLE_INPUT, EXAMPLE_INPUT_HIGH_INCOME]}
    response = requests.post(f"{BASE_URL}/predict/batch", json=batch_input, timeout=15)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert "prediction" in pred
        assert "probability" in pred
        assert "confidence" in pred
    print(f"  [PASS] Batch prediction: {len(data['predictions'])} results")
    return True


def test_invalid_input():
    """Test invalid input handling"""
    print("Testing invalid input handling...")
    # Missing required field
    invalid_input = EXAMPLE_INPUT.copy()
    del invalid_input["age"]
    response = requests.post(f"{BASE_URL}/predict", json=invalid_input, timeout=10)
    assert response.status_code == 422  # Validation error
    print(f"  [PASS] Invalid input correctly rejected (422)")
    return True


def test_example_endpoint():
    """Test example endpoint"""
    print("Testing /example...")
    response = requests.get(f"{BASE_URL}/example", timeout=10)
    assert response.status_code == 200
    data = response.json()
    assert "example_input" in data
    assert "expected_output" in data
    print(f"  [PASS] Example endpoint works")
    return True


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Running API Tests")
    print("="*60)

    tests = [
        test_health,
        test_model_info,
        test_predict,
        test_predict_high_income,
        test_batch_predict,
        test_invalid_input,
        test_example_endpoint,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {e}")
            failed += 1

    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    # Wait for server to be ready
    print("Waiting for API server to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                print("API server is ready!")
                break
        except:
            pass
        time.sleep(1)
    else:
        print("Timeout waiting for API server")
        sys.exit(1)

    run_all_tests()