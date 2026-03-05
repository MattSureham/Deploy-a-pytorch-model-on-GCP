"""
YOLOv5 Object Detection Model Wrapper
Uses ultralytics YOLOv5 via PyTorch Hub
"""

import torch
import numpy as np
from PIL import Image
import io
from typing import List, Dict, Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOv5Detector:
    """YOLOv5 Object Detection wrapper"""
    
    # COCO dataset class names (80 classes)
    COCO_NAMES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    def __init__(self, model_name: str = 'yolov5s', device: str = None):
        """
        Initialize YOLOv5 model
        
        Args:
            model_name: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x (n=fastest, x=most accurate)
            device: 'cuda', 'cpu', or None (auto)
        """
        self.model_name = model_name
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Loading YOLOv5 model: {model_name} on {self.device}")
        
        # Load model from PyTorch Hub
        self.model = torch.hub.load(
            'ultralytics/yolov5',
            model_name,
            pretrained=True,
            force_reload=False
        )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Set inference size
        self.img_size = 640
        
        logger.info(f"Model loaded successfully!")
        logger.info(f"Classes: {len(self.COCO_NAMES)} COCO classes")
    
    def preprocess(self, image_data: bytes) -> Image.Image:
        """Convert bytes to PIL Image"""
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        return image
    
    def detect(self, image_data: bytes, confidence_threshold: float = 0.25) -> Dict:
        """
        Run object detection on image
        
        Args:
            image_data: Raw image bytes
            confidence_threshold: Minimum confidence score (0-1)
        
        Returns:
            Dictionary with detections
        """
        # Preprocess
        image = self.preprocess(image_data)
        original_size = image.size  # (width, height)
        
        # Run inference
        results = self.model(image, size=self.img_size)
        
        # Parse results
        detections = []
        
        # results.xyxy[0] contains [x1, y1, x2, y2, confidence, class]
        for *box, conf, cls in results.xyxy[0]:
            confidence = float(conf)
            
            # Filter by confidence
            if confidence < confidence_threshold:
                continue
            
            class_id = int(cls)
            class_name = self.COCO_NAMES[class_id]
            
            detection = {
                'class': class_name,
                'class_id': class_id,
                'confidence': round(confidence, 4),
                'bbox': {
                    'x1': round(float(box[0]), 2),
                    'y1': round(float(box[1]), 2),
                    'x2': round(float(box[2]), 2),
                    'y2': round(float(box[3]), 2),
                    'width': round(float(box[2] - box[0]), 2),
                    'height': round(float(box[3] - box[1]), 2),
                    'center_x': round(float((box[0] + box[2]) / 2), 2),
                    'center_y': round(float((box[1] + box[3]) / 2), 2),
                }
            }
            detections.append(detection)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'success': True,
            'model': self.model_name,
            'image_size': {'width': original_size[0], 'height': original_size[1]},
            'num_detections': len(detections),
            'detections': detections,
            'classes_detected': list(set(d['class'] for d in detections))
        }
    
    def detect_batch(self, images_data: List[bytes], confidence_threshold: float = 0.25) -> List[Dict]:
        """Batch detection"""
        results = []
        for img_data in images_data:
            try:
                result = self.detect(img_data, confidence_threshold)
                results.append(result)
            except Exception as e:
                results.append({'success': False, 'error': str(e)})
        return results


if __name__ == "__main__":
    # Test the detector
    import requests
    
    print("Testing YOLOv5 Detector")
    print("=" * 50)
    
    detector = YOLOv5Detector(model_name='yolov5s')
    
    # Download a test image (a photo with objects)
    test_url = "https://ultralytics.com/images/bus.jpg"
    response = requests.get(test_url)
    
    print(f"Testing with image: {test_url}")
    result = detector.detect(response.content)
    
    print(f"\nDetected {result['num_detections']} objects:")
    for det in result['detections']:
        print(f"  - {det['class']}: {det['confidence']:.2%} confidence")
    
    print("\n✅ Test complete!")
