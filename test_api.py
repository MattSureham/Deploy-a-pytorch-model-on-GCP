# Test script for MNIST API
"""
Test the deployed MNIST classifier API
"""

import requests
import base64
import sys
from PIL import Image
import io
import numpy as np


def test_health(endpoint_url):
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{endpoint_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("✅ Health check passed\n")


def test_predict_file(endpoint_url, image_path):
    """Test prediction with file upload"""
    print(f"Testing prediction with file: {image_path}")
    
    with open(image_path, 'rb') as f:
        files = {'file': ('image.png', f, 'image/png')}
        response = requests.post(f"{endpoint_url}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")
    
    assert response.status_code == 200
    assert result['success'] == True
    assert 'predicted_digit' in result
    print(f"✅ Prediction successful: Digit {result['predicted_digit']} with confidence {result['confidence']}\n")
    
    return result


def test_predict_base64(endpoint_url, image_path):
    """Test prediction with base64 encoded image"""
    print(f"Testing prediction with base64: {image_path}")
    
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_string = base64.b64encode(image_data).decode('utf-8')
    
    response = requests.post(
        f"{endpoint_url}/predict_base64",
        json={"image_base64": base64_string}
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {result}")
    
    assert response.status_code == 200
    print("✅ Base64 prediction successful\n")


def create_test_image(digit_size=20):
    """Create a simple test image (a white digit on black background)"""
    # Create a simple test image
    img = Image.new('L', (28, 28), color=0)  # Black background
    
    # Draw a simple pattern (simulates a digit)
    pixels = img.load()
    for i in range(8, 20):
        for j in range(8, 20):
            pixels[i, j] = 255  # White square in middle
    
    return img


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <endpoint_url>")
        print("Example: python test_api.py https://mnist-classifier-xxx-uc.a.run.app")
        sys.exit(1)
    
    endpoint_url = sys.argv[1].rstrip('/')
    
    print("🧪 Testing MNIST Classifier API")
    print("=" * 50)
    print(f"Endpoint: {endpoint_url}\n")
    
    try:
        # Test health
        test_health(endpoint_url)
        
        # Create and save test image
        test_img = create_test_image()
        test_img.save('test_image.png')
        print("Created test image: test_image.png\n")
        
        # Test file upload
        test_predict_file(endpoint_url, 'test_image.png')
        
        # Test base64
        test_predict_base64(endpoint_url, 'test_image.png')
        
        print("=" * 50)
        print("✅ All tests passed!")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        sys.exit(1)
    except AssertionError as e:
        print(f"❌ Test assertion failed: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        import os
        if os.path.exists('test_image.png'):
            os.remove('test_image.png')


if __name__ == "__main__":
    main()
