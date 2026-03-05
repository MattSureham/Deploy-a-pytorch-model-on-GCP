# Deploy a PyTorch Model on GCP

A complete production-ready solution for deploying a PyTorch MNIST digit classifier to Google Cloud Platform using Cloud Run.

## 🎯 What This Project Does

This repository demonstrates how to:
1. **Train** a Convolutional Neural Network (CNN) on MNIST dataset
2. **Containerize** the PyTorch model with Docker
3. **Deploy** to GCP Cloud Run for serverless inference
4. **Serve** predictions via a REST API using FastAPI

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Client App    │────▶│  GCP Cloud Run  │────▶│  PyTorch Model  │
│  (Web/Mobile)   │◄────│  (FastAPI API)  │◄────│  (MNIST CNN)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

**Tech Stack:**
- **PyTorch** - Deep learning framework
- **FastAPI** - Modern, fast web framework for APIs
- **Docker** - Containerization
- **GCP Cloud Run** - Serverless container platform
- **Cloud Build** - CI/CD pipeline
- **GitHub Actions** - Automated deployment

## 📁 Project Structure

```
.
├── model.py                 # PyTorch CNN model definition
├── train.py                 # Training script
├── main.py                  # FastAPI inference server
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
├── cloudbuild.yaml         # GCP Cloud Build config
├── deploy.sh               # Local deployment script
├── test_api.py             # API testing script
├── .github/workflows/      # GitHub Actions CI/CD
│   └── deploy.yml
├── .gitignore
├── .gcloudignore
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Google Cloud SDK (`gcloud`)
- Docker (optional, for local testing)
- GCP account with billing enabled

### 1. Local Development

```bash
# Clone the repository
git clone https://github.com/MattSureham/Deploy-a-pytorch-model-on-GCP.git
cd Deploy-a-pytorch-model-on-GCP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train the MNIST classifier
python train.py --epochs 10 --batch-size 64

# Model will be saved to model/mnist_model.pth
```

**Expected output:**
```
Using device: cpu (or cuda)
Model parameters: 103,050
Train Epoch: 1 [0/60000 (0%)]  Loss: 2.312
...
Test set: Average loss: 0.0234, Accuracy: 9923/10000 (99.23%)
Saved best model to model/mnist_model.pth (Accuracy: 99.23%)
```

### 3. Run Local Server

```bash
# Start the FastAPI server
python main.py

# Or with uvicorn directly
uvicorn main:app --reload --port 8080
```

Server will start at `http://localhost:8080`

### 4. Test the API

```bash
# Health check
curl http://localhost:8080/health

# Test prediction (create a test image first)
python test_api.py http://localhost:8080
```

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/predict` | POST | Predict digit from image file |
| `/predict_base64` | POST | Predict from base64 image |
| `/predict_batch` | POST | Batch prediction |

### API Usage Examples

**Single prediction:**
```bash
curl -X POST "https://your-service-url/predict" \
  -F "file=@digit.png"
```

**Response:**
```json
{
  "success": true,
  "filename": "digit.png",
  "predicted_digit": 7,
  "confidence": 0.9823,
  "probabilities": {
    "0": 0.0012,
    "1": 0.0003,
    ...
  }
}
```

## ☁️ Deploy to GCP

### Option 1: Using Cloud Build (Recommended)

```bash
# Set your GCP project
gcloud config set project YOUR_PROJECT_ID

# Submit build
gcloud builds submit --config cloudbuild.yaml .
```

This will:
1. Build the Docker image
2. Push to Container Registry
3. Deploy to Cloud Run

### Option 2: Using Deployment Script

```bash
# Set environment variable
export GCP_PROJECT_ID=your-project-id

# Run deployment script
./deploy.sh
```

### Option 3: Manual Steps

```bash
# Build container
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/mnist-classifier .

# Deploy to Cloud Run
gcloud run deploy mnist-classifier \
  --image gcr.io/YOUR_PROJECT_ID/mnist-classifier \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1
```

## 🔧 CI/CD with GitHub Actions

The repository includes GitHub Actions workflow for automatic deployment:

1. **Push to main** → Triggers deployment
2. **Run tests** → Validates model loading
3. **Build & push** → Creates container image
4. **Deploy** → Updates Cloud Run service

### Setup GitHub Secrets

Add these secrets to your GitHub repository:

- `GCP_PROJECT_ID` - Your GCP project ID
- `WORKLOAD_IDENTITY_PROVIDER` - Workload identity provider
- `SERVICE_ACCOUNT_EMAIL` - Service account for deployment

## 🧪 Testing

```bash
# Run all tests
python test_api.py https://your-service-url

# Or test locally
python test_api.py http://localhost:8080
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Parameters | ~103K |
| Input Size | 28×28 grayscale |
| Output | 10 classes (digits 0-9) |
| Expected Accuracy | ~99%+ |
| Inference Time | <100ms (CPU) |

## 💰 Cost Estimation (GCP)

| Component | Free Tier | Paid (Estimated) |
|-----------|-----------|------------------|
| Cloud Run | 2M requests/month | $0.40/million requests |
| Container Registry | 0.5GB storage | $0.026/GB/month |
| Cloud Build | 120 build-minutes/day | $0.003/minute |

**Typical cost for low traffic:** <$5/month

## 🔒 Security Considerations

- API is publicly accessible (`--allow-unauthenticated`)
- For production, consider:
  - API key authentication
  - Rate limiting
  - VPC connector for private services
  - Cloud Armor for DDoS protection

## 🐛 Troubleshooting

### Model not found
```
WARNING: No model found at model/mnist_model.pth
```
**Solution:** Run `python train.py` first to generate the model file.

### Out of memory
```
Memory limit of 512M exceeded
```
**Solution:** Increase memory in Cloud Run:
```bash
gcloud run services update mnist-classifier --memory 1Gi
```

### Cold start latency
**Solution:** Set minimum instances:
```bash
gcloud run services update mnist-classifier --min-instances 1
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [GCP Cloud Run](https://cloud.google.com/run)
- [PyTorch Deployment](https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html)

## 📝 License

MIT License - feel free to use this for your own projects!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Happy Deploying! 🚀**
