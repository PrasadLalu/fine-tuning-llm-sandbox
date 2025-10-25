#!/bin/bash
# H100 Docker Build and Deployment Commands
# Run these on your H100 GPU server after training

echo "🐳 Building H100 Docker Image..."

# Build the Docker image with H100 optimization
docker build -t llm-api-h100:latest .

echo "🚀 Starting H100 Docker Container..."

# Run with GPU support and H100 optimizations
docker run -d \
    --name llm-api-h100 \
    --gpus all \
    -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/datasets:/app/datasets \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    -e TOKENIZERS_PARALLELISM=false \
    --restart unless-stopped \
    llm-api-h100:latest

echo "📊 Checking container status..."
docker ps | grep llm-api-h100

echo "📝 Viewing logs..."
docker logs llm-api-h100

echo "🧪 Testing API..."
sleep 10
curl http://localhost:8000/health

echo "✅ H100 Docker deployment complete!"
echo "🌐 API available at: http://localhost:8000"
echo "📚 API docs at: http://localhost:8000/docs"
