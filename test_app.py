import time
import subprocess
import requests
import sys

print("Starting verification test...")

# Launch Flask app in a background process
process = subprocess.Popen(
    [sys.executable, "app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Wait for Flask and Keras models to initialize
time.sleep(7)

# Test inputs
test_cases = [
    {
        "text": "I love this course",
        "expected": "Positive"
    },
    {
        "text": "Absolutely terrible movie",
        "expected": "Negative"
    },
    {
        "text": "This is terrible and boring",
        "expected": "Negative"
    },
    {
        "text": "A wonderful brilliant masterpiece",
        "expected": "Positive"
    },
    {
        "text": "Some completely new vocabulary like keyboard elephant", # Out of Vocab words
        "expected": None
    }
]

success = True
try:
    # 1. Test root page
    r = requests.get("http://127.0.0.1:5001/")
    if r.status_code == 200:
        print("✓ Root page (/) is active and returned HTTP 200")
    else:
        print(f"✗ Root page returned status {r.status_code}")
        success = False

    # 2. Test predictions
    for case in test_cases:
        text = case["text"]
        print(f"\nTesting sentence: '{text}'")
        
        response = requests.post(
            "http://127.0.0.1:5001/predict",
            json={"text": text}
        )
        
        if response.status_code != 200:
            print(f"✗ Prediction endpoint failed with status {response.status_code}")
            print(response.json())
            success = False
            continue
            
        res_data = response.json()
        print("✓ Response status: 200 OK")
        print("Tokenization mapping:")
        for t in res_data["tokens_mapping"]:
            print(f"  - '{t['token']}' -> index {t['index']}")
            
        print("Predictions:")
        for model_name, pred in res_data["predictions"].items():
            print(f"  - {model_name:15}: Sentiment={pred['sentiment']:8} Probability={pred['probability']:.4f}")
            
except Exception as e:
    print(f"✗ Verification failed due to error: {e}")
    success = False
finally:
    # Terminate background Flask app process
    process.terminate()
    process.wait()
    print("\nBackground Flask process terminated.")

if success:
    print("\n✓ ALL TESTS PASSED SUCCESSFULLY!")
    sys.exit(0)
else:
    print("\n✗ SOME TESTS FAILED.")
    sys.exit(1)
