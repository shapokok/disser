#!/bin/bash

# Crop Disease Detection System - Docker Start Script
# This script builds and starts the Docker containers

set -e

echo "=========================================="
echo "Crop Disease Detection System - Docker"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/uploads data/test_images models results/heatmaps

# Check if we should use production profile
if [ "$1" == "production" ] || [ "$1" == "prod" ]; then
    echo -e "${YELLOW}🚀 Starting in PRODUCTION mode (with Nginx)${NC}"
    PROFILE="--profile production"
else
    echo -e "${GREEN}🔧 Starting in DEVELOPMENT mode${NC}"
    PROFILE=""
fi

# Build and start containers
echo ""
echo "🐳 Building Docker images..."
docker-compose build

echo ""
echo "🚀 Starting containers..."
docker-compose up -d $PROFILE

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check if backend is healthy
echo ""
echo "🔍 Checking service health..."

MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:5000/ > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is healthy!${NC}"
        break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting for backend... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${YELLOW}⚠ Backend took longer than expected to start${NC}"
    echo "Check logs with: docker-compose logs backend"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}✓ System is running!${NC}"
echo "=========================================="
echo ""
echo "📍 Access points:"
echo "   Backend API: http://localhost:5000"
if [ ! -z "$PROFILE" ]; then
    echo "   Frontend:    http://localhost (via Nginx)"
else
    echo "   Frontend:    Open crop_monitoring_app/frontend/index.html in browser"
fi
echo ""
echo "📊 Useful commands:"
echo "   View logs:        docker-compose logs -f"
echo "   Stop:             docker-compose stop"
echo "   Stop & remove:    docker-compose down"
echo "   Restart:          docker-compose restart"
echo ""
