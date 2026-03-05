# Deploy YOLOv5 Object Detection on GCP

A complete production-ready solution for deploying a **YOLOv5 object detection model** to **Google Cloud Platform (GCP)** using Cloud Run.

## 🎯 What This Project Does

This repository demonstrates how to:
1. **Run** YOLOv5 (state-of-the-art object detection) via PyTorch
2. **Build** a REST API with FastAPI to serve predictions
3. **Containerize** the application with Docker
4. **Deploy** to GCP Cloud Run for scalable, serverless inference
5. **Detect 80+ object classes** (people, cars, animals, everyday objects)

---

## 📚 Concepts Explained

### What is **GCP** (Google Cloud Platform)?

GCP is Google's cloud computing platform — like renting computers, storage, and services over the internet instead of buying your own hardware.

| Service | What it does | Real-world analogy |
|---------|--------------|-------------------|
| **Cloud Run** | Runs your app in containers, scales automatically | Like a restaurant that opens more kitchens when busy, closes them when quiet |
| **Cloud Build** | Builds your Docker images automatically | Like a factory that packages your app |
| **Container Registry** | Stores your Docker images | Like a warehouse for your packaged apps |

**Why Cloud Run?**
- 🚀 **Serverless**: No servers to manage
- 💰 **Pay-per-use**: Only pay when handling requests (free tier: 2M requests/month)
- 📈 **Auto-scales**: From 0 to thousands of instances automatically
- 🔒 **HTTPS by default**: Secure endpoints out of the box
- 🌍 **Global**: Deploy close to your users

---

### What is **FastAPI**?

FastAPI is a modern Python framework for building **APIs** (Application Programming Interfaces).

**Think of an API like a restaurant:**

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   You       │────▶│   Waiter    │────▶│   Kitchen   │
│  (Client)   │◄────│   (FastAPI) │◄────│   (Model)   │
└─────────────┘     └─────────────┘     └─────────────┘
     "I want         Takes order       Cooks food
      food"          to kitchen        (runs YOLOv5)
```

**Key features:**
- ⚡ Very fast (built on Starlette)
- 📝 Automatic API docs at `/docs`
- ✅ Type hints prevent bugs
- 🔄 Easy async support

**Example endpoint:**
```python
@app.post("/detect")  # URL endpoint
def detect(file: UploadFile):  # Accepts file upload
    result = model(file)       # Runs YOLOv5
    return {"boxes": result}   # Returns JSON
```

---

### What is **Docker**?

Docker packages your application with **everything it needs** into a "container" — like a shipping container for software.

**Without Docker:**
- "Works on my machine" problems
- Need to install Python, PyTorch, dependencies on every server
- Version conflicts

**With Docker:**
- Package app + Python + all libraries into one file
- Runs identically everywhere
- Just run the container

**Dockerfile** (recipe for building):
```dockerfile
FROM python:3.9              # Start with Python 3.9 image
COPY . /app                  # Copy your code
RUN pip install -r requirements.txt  # Install dependencies
CMD ["python", "main.py"]    # Run app
```

**Key commands:**
```bash
docker build -t myapp .      # Build container
docker run -p 8080:8080 myapp  # Run locally
docker push gcr.io/.../myapp   # Upload to cloud
```

---

### What is **YOLOv5**?

YOLO = "You Only Look Once" — a fast, accurate object detection model.

**What it does:**
- Takes an image → Finds objects → Draws boxes around them
- Detects **80 classes**: person, car, dog, cat, phone, laptop, etc.
- Runs in real-time (~30+ FPS on GPU)

**Model sizes:**
| Model | Speed | Accuracy | Use case |
|-------|-------|----------|----------|
| yolov5n | ⚡ Fastest | Good | Edge devices, mobile |
| yolov5s | Fast | Better | **This project** (good balance) |
| yolov5m | Medium | Great | Higher accuracy needs |
| yolov5l | Slow | Excellent | Maximum accuracy |
| yolov5x | Slowest | Best | Research, high-end |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                           USER REQUEST                               │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GCP CLOUD RUN (Serverless)                     │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  Docker Container                                              │  │
│  │  ┌─────────────────────────────────────────────────────────┐  │  │
│  │  │  FastAPI App                                             │  │  │
│  │  │  ┌───────────────────────────────────────────────────┐  │  │  │
│  │  │  │  YOLOv5 Model                                      │  │  │  │
│  │  │  │  • Loads pre-trained weights                       │  │  │  │
│  │  │  │  • Processes image                                 │  │  │  │
│  │  │  │  • Returns detections                              │  │  │  │
│  │  │  └───────────────────────────────────────────────────┘  │  │  │
│  │  └─────────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         JSON RESPONSE                                │
│  {                                                                    │
│    "detections": [                                                   │
│      {"class": "person", "confidence": 0.95, "bbox": {...}},         │
│      {"class": "car", "confidence": 0.89, "bbox": {...}}             │
│    ]                                                                 │
│  }                                                                    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
.
├── model.py              # YOLOv5 wrapper class
├── main.py               # FastAPI application (API endpoints)
├── requirements.txt      # Python dependencies
├── Dockerfile           # Container configuration
├── cloudbuild.yaml      # GCP Cloud Build CI/CD
├── deploy.sh            # One-command deployment script
├── test_api.py          # API testing script
├── .github/workflows/   # GitHub Actions
│   └── deploy.yml
├── .gitignore
├── .gcloudignore
└── README.md            # This file
```

---

## 🚀 Quick Start

### Prerequisites

1. **Python 3.9+**
2. **Google Cloud SDK** (`gcloud`) — [Install here](https://cloud.google.com/sdk/docs/install)
3. **Docker** (optional, for local testing) — [Install here](https://docs.docker.com/get-docker/)
4. **GCP account** with billing enabled

### Step 1: Clone & Setup

```bash
# Clone repository
git clone https://github.com/MattSureham/Deploy-a-pytorch-model-on-GCP.git
cd Deploy-a-pytorch-model-on-GCP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Run Locally

```bash
# Start the server
python main.py
```

Server starts at `http://localhost:8080`

### Step 3: Test the API

**Test 1: Health check**
```bash
curl http://localhost:8080/health
```

**Test 2: Detect objects in an image**
```bash
# Download a test image
curl -o test_image.jpg https://ultralytics.com/images/bus.jpg

# Send to API
curl -X POST "http://localhost:8080/detect" \
  -F "file=@test_image.jpg"
```

**Test 3: Get image with boxes drawn**
```bash
curl -X POST "http://localhost:8080/detect/visualize" \
  -F "file=@test_image.jpg" \
  --output output.png
```

**Or use the test script:**
```bash
python test_api.py http://localhost:8080
```

---

## ☁️ Deploy to GCP

### Option 1: One-Command Deploy (Recommended)

```bash
# Set your GCP project ID
export GCP_PROJECT_ID=your-project-id

# Run deployment script
./deploy.sh
```

This script will:
1. Build Docker image using Cloud Build
2. Push to Container Registry
3. Deploy to Cloud Run
4. Give you the HTTPS URL

### Option 2: Step-by-Step Manual

```bash
# 1. Set GCP project
gcloud config set project YOUR_PROJECT_ID

# 2. Build container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/yolov5-api .

# 3. Deploy to Cloud Run
gcloud run deploy yolov5-api \
  --image gcr.io/YOUR_PROJECT_ID/yolov5-api \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300

# 4. Get the URL
gcloud run services describe yolov5-api --region us-central1 --format 'value(status.url)'
```

### Option 3: GitHub Actions (CI/CD)

1. Fork this repository
2. Add GitHub Secrets:
   - `GCP_PROJECT_ID`: Your project ID
   - `WORKLOAD_IDENTITY_PROVIDER`: Workload identity provider
   - `SERVICE_ACCOUNT_EMAIL`: Service account email
3. Push to `main` branch → Automatic deployment!

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/classes` | GET | List of 80 detectable classes |
| `/detect` | POST | Detect objects in image |
| `/detect/visualize` | POST | Return image with bounding boxes |
| `/detect/base64` | POST | Detect from base64 string |
| `/detect/batch` | POST | Batch detection on multiple images |

### Example Requests

**Detect objects:**
```bash
curl -X POST "https://your-api-url/detect" \
  -F "file=@photo.jpg" \
  -F "confidence=0.5"
```

**Response:**
```json
{
  "success": true,
  "model": "yolov5s",
  "num_detections": 3,
  "detections": [
    {
      "class": "person",
      "confidence": 0.9534,
      "bbox": {
        "x1": 100.5,
        "y1": 200.0,
        "x2": 350.2,
        "y2": 600.5
      }
    },
    {
      "class": "car",
      "confidence": 0.8912,
      "bbox": {...}
    }
  ],
  "classes_detected": ["person", "car"]
}
```

**Visualize (get image with boxes):**
```bash
curl -X POST "https://your-api-url/detect/visualize" \
  -F "file=@photo.jpg" \
  --output result.png
```

---

## 🐳 Using Docker

### Build & Run Locally

```bash
# Build Docker image
docker build -t yolov5-api .

# Run container
docker run -p 8080:8080 yolov5-api

# Test
curl http://localhost:8080/health
```

### Push to Docker Hub (optional)

```bash
# Tag image
docker tag yolov5-api yourusername/yolov5-api:latest

# Login
docker login

# Push
docker push yourusername/yolov5-api:latest
```

---

## 💰 GCP Cost Estimation

| Component | Free Tier | Paid (Est.) |
|-----------|-----------|-------------|
| Cloud Run | 2M requests/month | $0.40/million requests |
| Container Registry | 0.5GB storage | $0.026/GB/month |
| Cloud Build | 120 min/day | $0.003/minute |
| Egress (outbound data) | 1GB/month | $0.12/GB |

**Typical cost for low traffic: $0-5/month**

---

## 🛠️ Troubleshooting

### "Out of memory" error
```bash
# Increase memory allocation
gcloud run services update yolov5-api --memory 4Gi
```

### Cold start latency (first request slow)
```bash
# Keep minimum 1 instance warm
gcloud run services update yolov5-api --min-instances 1
```

### Model download fails
The Dockerfile pre-downloads the model. If you see download errors:
```bash
# Manually download first
python -c "import torch; torch.hub.load('ultralytics/yolov5', 'yolov5s')"
```

### Build fails
```bash
# Check Cloud Build logs
gcloud builds list
gcloud builds log [BUILD_ID]
```

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Model | YOLOv5s |
| Classes | 80 COCO classes |
| Input | Any size (resized to 640x640) |
| Inference time | ~100-300ms (CPU) |
| Cold start | ~5-10 seconds |
| Throughput | ~3-10 req/sec per instance |

---

## 🎯 Use Cases

- **Security cameras**: Detect people, vehicles
- **Retail**: Count customers, track products
- **Wildlife**: Animal detection in camera traps
- **Manufacturing**: Quality control, defect detection
- **Smart cities**: Traffic monitoring
- **Drones**: Aerial object detection

---

## 📚 Resources

- [YOLOv5 GitHub](https://github.com/ultralytics/yolov5)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [GCP Cloud Run](https://cloud.google.com/run)
- [Docker Docs](https://docs.docker.com/)

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

---

## 📝 License

MIT License — Free to use for personal and commercial projects!

---

**Questions?** Open an issue or reach out!

**Happy Deploying! 🚀**
