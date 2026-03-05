"""
FastAPI inference server for MNIST model
"""

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
from typing import List, Optional
import logging

from model import MNISTClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MNIST Digit Classification API",
    description="API for classifying handwritten digits using PyTorch",
    version="1.0.0"
)

# Global model variable
model = None
device = None


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, device
    
    logger.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MNISTClassifier().to(device)
    
    # Load trained weights
    import os
    model_path = os.environ.get('MODEL_PATH', 'model/mnist_model.pth')
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Model accuracy: {checkpoint.get('accuracy', 'N/A')}")
    else:
        logger.warning(f"No model found at {model_path}, using untrained model")
    
    model.eval()
    logger.info("Model ready for inference")


def preprocess_image(image_data: bytes) -> torch.Tensor:
    """Preprocess image for model input"""
    # Load image
    image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to numpy array and normalize
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # Apply MNIST normalization
    img_array = (img_array - 0.1307) / 0.3081
    
    # Convert to tensor (1, 1, 28, 28)
    tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0)
    
    return tensor


def predict_digit(tensor: torch.Tensor) -> dict:
    """Run inference on preprocessed tensor"""
    tensor = tensor.to(device)
    
    with torch.no_grad():
        output = model(tensor)
        probabilities = torch.exp(output).squeeze()
        predicted_class = probabilities.argmax().item()
        confidence = probabilities[predicted_class].item()
    
    # Get all class probabilities
    all_probs = {str(i): round(probabilities[i].item(), 4) for i in range(10)}
    
    return {
        "predicted_digit": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": all_probs
    }


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": "MNIST Classifier",
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Health check for GCP load balancers"""
    return JSONResponse(
        content={"status": "healthy"},
        status_code=200
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict digit from uploaded image
    
    - **file**: Image file (PNG, JPG, JPEG) containing a handwritten digit
    """
    # Validate file type
    allowed_types = ['image/png', 'image/jpeg', 'image/jpg']
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Allowed: {allowed_types}"
        )
    
    try:
        # Read image
        contents = await file.read()
        
        # Preprocess
        tensor = preprocess_image(contents)
        
        # Predict
        result = predict_digit(tensor)
        
        return {
            "success": True,
            "filename": file.filename,
            **result
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_base64")
async def predict_base64(image_base64: str):
    """
    Predict digit from base64 encoded image
    
    - **image_base64**: Base64 encoded image string
    """
    try:
        # Decode base64
        image_data = base64.b64decode(image_base64)
        
        # Preprocess
        tensor = preprocess_image(image_data)
        
        # Predict
        result = predict_digit(tensor)
        
        return {
            "success": True,
            **result
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Batch prediction for multiple images
    
    - **files**: List of image files
    """
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            tensor = preprocess_image(contents)
            result = predict_digit(tensor)
            results.append({
                "filename": file.filename,
                "success": True,
                **result
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return {"results": results}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
