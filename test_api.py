"""
Test script for YOLOv5 Object Detection API
"""

import requests
import base64
import sys
from PIL import Image
import io
import json


def test_health(endpoint_url):
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    response = requests.get(f"{endpoint_url}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("✅ Health check passed\n")


def test_root(endpoint_url):
    """Test root endpoint"""
    print("ℹ️  Testing root endpoint...")
    response = requests.get(f"{endpoint_url}/")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Service: {data.get('service')}")
    print(f"Model: {data.get('model')}")
    print("✅ Root endpoint working\n")


def test_detect_file(endpoint_url, image_url):
    """Test detection with image from URL"""
    print(f"🔍 Testing detection with image: {image_url}")
    
    # Download image
    img_response = requests.get(image_url)
    img_response.raise_for_status()
    
    # Send to API
    files = {'file': ('test.jpg', img_response.content, 'image/jpeg')}
    response = requests.post(f"{endpoint_url}/detect", files=files)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    
    print(f"Objects detected: {result['num_detections']}")
    for det in result['detections'][:5]:  # Show first 5
        print(f"  - {det['class']}: {det['confidence']:.2%}")
    
    assert response.status_code == 200
    assert result['success'] == True
    print("✅ Detection successful\n")
    return result


def test_detect_visualize(endpoint_url, image_url):
    """Test visualization endpoint"""
    print("🎨 Testing visualization endpoint...")
    
    # Download image
    img_response = requests.get(image_url)
    
    # Send to API
    files = {'file': ('test.jpg', img_response.content, 'image/jpeg')}
    response = requests.post(f"{endpoint_url}/detect/visualize", files=files)
    
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('content-type')}")
    print(f"Detections: {response.headers.get('X-Detections')}")
    
    # Save output
    with open('output_detected.png', 'wb') as f:
        f.write(response.content)
    print("Saved to: output_detected.png")
    
    assert response.status_code == 200
    print("✅ Visualization successful\n")


def test_classes(endpoint_url):
    """Test classes endpoint"""
    print("📋 Testing classes endpoint...")
    response = requests.get(f"{endpoint_url}/classes")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Number of classes: {data['num_classes']}")
    print(f"Sample classes: {', '.join(data['classes'][:10])}...")
    print("✅ Classes endpoint working\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_api.py <endpoint_url>")
        print("Example: python test_api.py http://localhost:8080")
        print("Example: python test_api.py https://myapp-xxx-uc.a.run.app")
        sys.exit(1)
    
    endpoint_url = sys.argv[1].rstrip('/')
    
    # Test image (bus with people)
    test_image_url = "https://ultralytics.com/images/bus.jpg"
    
    print("🧪 Testing YOLOv5 Object Detection API")
    print("=" * 50)
    print(f"Endpoint: {endpoint_url}")
    print(f"Test image: {test_image_url}\n")
    
    try:
        # Run tests
        test_health(endpoint_url)
        test_root(endpoint_url)
        test_classes(endpoint_url)
        test_detect_file(endpoint_url, test_image_url)
        test_detect_visualize(endpoint_url, test_image_url)
        
        print("=" * 50)
        print("✅ All tests passed!")
        print("\nExample usage:")
        print(f"  curl -X POST '{endpoint_url}/detect' -F 'file=@your_image.jpg'")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        sys.exit(1)
    except AssertionError as e:
        print(f"❌ Test assertion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
