#!/bin/bash
# Deploy YOLOv5 Object Detection API to GCP Cloud Run

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-"your-project-id"}
SERVICE_NAME="yolov5-api"
REGION="us-central1"
IMAGE_TAG="latest"

echo "🚀 Deploying YOLOv5 Object Detection API to GCP Cloud Run"
echo "=========================================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install Google Cloud SDK:"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if project ID is set
if [ "$PROJECT_ID" = "your-project-id" ]; then
    echo "❌ Please set your GCP project ID:"
    echo "   export GCP_PROJECT_ID=your-actual-project-id"
    exit 1
fi

# Set project
echo "📋 Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Enable required APIs
echo "🔧 Enabling required APIs..."
gcloud services enable cloudbuild.googleapis.com run.googleapis.com containerregistry.googleapis.com

# Build container
echo "🔨 Building Docker container (this may take a few minutes)..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/$SERVICE_NAME:$IMAGE_TAG .

# Deploy to Cloud Run
echo "☁️  Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image gcr.io/$PROJECT_ID/$SERVICE_NAME:$IMAGE_TAG \
    --region $REGION \
    --platform managed \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 1 \
    --timeout 300 \
    --concurrency 40 \
    --max-instances 10

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format 'value(status.url)')

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "Quick test:"
echo "  curl $SERVICE_URL/health"
echo ""
echo "Test with image:"
echo "  curl -X POST '$SERVICE_URL/detect' -F 'file=@your_image.jpg'"
echo ""
echo "View logs:"
echo "  gcloud logging tail --service=$SERVICE_NAME --region=$REGION"
echo ""
