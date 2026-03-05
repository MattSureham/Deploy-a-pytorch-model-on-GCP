"""
FastAPI inference server for YOLOv5 Object Detection
"""

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Optional
import logging
import io
import base64
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from model import YOLOv5Detector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="YOLOv5 Object Detection API",
    description="API for detecting objects in images using YOLOv5",
    version="2.0.0"
)

# Global detector variable
detector = None

# Color palette for different classes (for visualization)
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0),
    (128, 0, 128), (0, 128, 128), (255, 128, 0), (255, 0, 128), (128, 255, 0),
    (0, 255, 128), (128, 0, 255), (0, 128, 255), (255, 128, 128), (128, 255, 255)
]


@app.on_event("startup")
async def load_model():
    """Load YOLOv5 model on startup"""
    global detector
    
    logger.info("Loading YOLOv5 model...")
    
    # Use small model for faster inference (good for Cloud Run)
    # Options: yolov5n (fastest), yolov5s, yolov5m, yolov5l, yolov5x (most accurate)
    model_name = 'yolov5s'
    
    detector = YOLOv5Detector(model_name=model_name)
    logger.info(f"YOLOv5 {model_name} model ready for inference!")


def draw_boxes(image: Image.Image, detections: List[dict]) -> Image.Image:
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Create class-to-color mapping
    class_colors = {}
    for i, det in enumerate(detections):
        class_name = det['class']
        if class_name not in class_colors:
            class_colors[class_name] = COLORS[len(class_colors) % len(COLORS)]
        
        color = class_colors[class_name]
        bbox = det['bbox']
        conf = det['confidence']
        
        # Draw rectangle
        draw.rectangle(
            [(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2'])],
            outline=color,
            width=3
        )
        
        # Draw label background
        label = f"{class_name} {conf:.2%}"
        bbox_text = draw.textbbox((0, 0), label, font=font)
        text_width = bbox_text[2] - bbox_text[0]
        text_height = bbox_text[3] - bbox_text[1]
        
        draw.rectangle(
            [(bbox['x1'], bbox['y1'] - text_height - 4), 
             (bbox['x1'] + text_width, bbox['y1'])],
            fill=color
        )
        
        # Draw label text
        draw.text(
            (bbox['x1'], bbox['y1'] - text_height - 4),
            label,
            fill=(255, 255, 255),
            font=font
        )
    
    return image


@app.get("/")
async def root():
    """Service info"""
    return {
        "status": "healthy",
        "service": "YOLOv5 Object Detection API",
        "version": "2.0.0",
        "model": "yolov5s",
        "classes": 80,
        "endpoints": {
            "health": "/health",
            "detect": "/detect (POST)",
            "detect_visualize": "/detect/visualize (POST)",
            "detect_base64": "/detect/base64 (POST)"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return JSONResponse(
        content={
            "status": "healthy",
            "model_loaded": detector is not None,
            "device": str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else "cpu"
        },
        status_code=200
    )


@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    """
    Detect objects in uploaded image
    
    - **file**: Image file (PNG, JPG, JPEG)
    - **confidence**: Minimum confidence threshold (0-1, default 0.25)
    
    Returns: List of detected objects with bounding boxes
    """
    # Validate file type
    allowed_types = ['image/png', 'image/jpeg', 'image/jpg']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    try:
        contents = await file.read()
        result = detector.detect(contents, confidence_threshold=confidence)
        result['filename'] = file.filename
        return result
    
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/visualize")
async def detect_visualize(
    file: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    """
    Detect objects and return image with bounding boxes drawn
    
    - **file**: Image file
    - **confidence**: Minimum confidence threshold
    
    Returns: Image with bounding boxes (PNG)
    """
    allowed_types = ['image/png', 'image/jpeg', 'image/jpg']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    try:
        contents = await file.read()
        
        # Run detection
        result = detector.detect(contents, confidence_threshold=confidence)
        
        # Load image and draw boxes
        image = detector.preprocess(contents)
        image_with_boxes = draw_boxes(image, result['detections'])
        
        # Convert to bytes
        img_byte_arr = io.BytesIO()
        image_with_boxes.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={"X-Detections": str(result['num_detections'])}
        )
    
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/base64")
async def detect_base64(
    image_base64: str,
    confidence: float = 0.25
):
    """
    Detect objects from base64 encoded image
    
    - **image_base64**: Base64 encoded image string
    - **confidence**: Minimum confidence threshold
    """
    try:
        image_data = base64.b64decode(image_base64)
        result = detector.detect(image_data, confidence_threshold=confidence)
        return result
    
    except Exception as e:
        logger.error(f"Base64 detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect/batch")
async def detect_batch(
    files: List[UploadFile] = File(...),
    confidence: float = Form(0.25)
):
    """
    Batch detection on multiple images
    
    - **files**: List of image files
    - **confidence**: Minimum confidence threshold
    """
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            result = detector.detect(contents, confidence_threshold=confidence)
            result['filename'] = file.filename
            results.append(result)
        except Exception as e:
            results.append({
                'filename': file.filename,
                'success': False,
                'error': str(e)
            })
    
    return {
        'batch_size': len(files),
        'results': results
    }


@app.get("/classes")
async def get_classes():
    """Get list of detectable classes"""
    return {
        "num_classes": len(detector.COCO_NAMES),
        "classes": detector.COCO_NAMES
    }


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
