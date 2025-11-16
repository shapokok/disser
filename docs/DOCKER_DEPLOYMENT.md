# Docker Deployment Guide

## 🐳 Quick Start

### Prerequisites
- Docker (v20.10+)
- Docker Compose (v2.0+)
- 4GB RAM minimum
- 10GB disk space

### Installation

```bash
# Install Docker (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt-get install docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

---

## 🚀 Deployment Options

### Option 1: Simple Start (Development)

```bash
# Start the system
./docker-start.sh

# Access the application
# Backend API: http://localhost:5000
# Frontend: Open crop_monitoring_app/frontend/index.html in browser
```

### Option 2: Production Mode (with Nginx)

```bash
# Start with Nginx reverse proxy
./docker-start.sh production

# Access the application
# Frontend: http://localhost
# Backend API: http://localhost/api/
```

### Option 3: Manual Docker Compose

```bash
# Development mode
docker-compose up -d

# Production mode (with Nginx)
docker-compose --profile production up -d

# View logs
docker-compose logs -f backend

# Stop
docker-compose stop

# Stop and remove containers
docker-compose down
```

---

## 📁 Project Structure

```
disser/
├── Dockerfile                      # Multi-stage Docker build
├── docker-compose.yml             # Container orchestration
├── nginx.conf                     # Nginx configuration (production)
├── .dockerignore                  # Files to exclude from build
├── docker-start.sh               # Helper script to start
├── docker-stop.sh                # Helper script to stop
│
├── crop_monitoring_app/
│   ├── backend/
│   │   ├── app.py                # Flask application
│   │   ├── requirements.txt      # Python dependencies
│   │   └── ...
│   └── frontend/
│       └── ...
│
├── data/
│   ├── uploads/                  # Uploaded images (persisted)
│   └── test_images/             # Test images
│
├── models/                       # Model files (persisted)
└── results/                      # Analysis results (persisted)
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=0

# Upload Configuration
MAX_FILE_SIZE=16777216  # 16MB in bytes
ALLOWED_EXTENSIONS=png,jpg,jpeg,jfif

# Model Configuration
MODELS_DIR=/app/models
UPLOAD_DIR=/app/data/uploads

# Server Configuration
HOST=0.0.0.0
PORT=5000
WORKERS=4
```

### Docker Compose Override

Create `docker-compose.override.yml` for local customizations:

```yaml
version: '3.8'

services:
  backend:
    environment:
      - FLASK_DEBUG=1
    volumes:
      # Mount source code for development
      - ./crop_monitoring_app:/app/crop_monitoring_app:ro
```

---

## 📊 Management Commands

### View Logs

```bash
# All services
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail=100 backend

# Nginx (production)
docker-compose logs -f nginx
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart backend only
docker-compose restart backend

# Rebuild and restart
docker-compose up -d --build
```

### Shell Access

```bash
# Access backend container
docker-compose exec backend bash

# Run Python commands
docker-compose exec backend python -c "import torch; print(torch.__version__)"

# Check model files
docker-compose exec backend ls -lh /app/models/
```

### Resource Monitoring

```bash
# Container stats
docker stats

# Disk usage
docker system df

# Container resource limits
docker-compose exec backend cat /sys/fs/cgroup/memory/memory.limit_in_bytes
```

---

## 🗂️ Data Persistence

Data is persisted in Docker volumes mapped to host directories:

### Uploaded Images
```bash
# Host: ./data/uploads/
# Container: /app/data/uploads/
# Persisted: Yes
```

### Model Files
```bash
# Host: ./models/
# Container: /app/models/
# Persisted: Yes

# Add your trained models here:
cp my_trained_model.pth models/efficientnet.pth
```

### Results
```bash
# Host: ./results/
# Container: /app/results/
# Persisted: Yes
```

---

## 🔒 Security Best Practices

### Production Deployment

1. **Use HTTPS**
   ```bash
   # Get SSL certificate (Let's Encrypt)
   sudo apt-get install certbot
   sudo certbot certonly --standalone -d yourdomain.com

   # Update nginx.conf with SSL configuration
   # Uncomment the HTTPS server block
   ```

2. **Set Strong Secrets**
   ```bash
   # Generate secret key
   python -c "import secrets; print(secrets.token_hex(32))"

   # Add to .env
   echo "SECRET_KEY=your_generated_key" >> .env
   ```

3. **Limit Resources**
   ```yaml
   # In docker-compose.yml
   services:
     backend:
       deploy:
         resources:
           limits:
             cpus: '2'
             memory: 4G
           reservations:
             cpus: '1'
             memory: 2G
   ```

4. **Run as Non-Root**
   ```dockerfile
   # Add to Dockerfile
   RUN useradd -m -u 1000 appuser
   USER appuser
   ```

5. **Update Dependencies**
   ```bash
   # Regularly update base image
   docker pull python:3.9-slim
   docker-compose build --no-cache
   ```

---

## 🚀 Production Deployment

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml crop-disease

# Scale services
docker service scale crop-disease_backend=3

# View services
docker service ls

# Remove stack
docker stack rm crop-disease
```

### Using Kubernetes

```bash
# Convert to Kubernetes manifests
kompose convert -f docker-compose.yml

# Apply manifests
kubectl apply -f backend-deployment.yaml
kubectl apply -f backend-service.yaml

# Check status
kubectl get pods
kubectl get services
```

---

## 🐛 Troubleshooting

### Backend Won't Start

```bash
# Check logs
docker-compose logs backend

# Common issues:
# 1. Port 5000 already in use
lsof -i :5000
kill -9 <PID>

# 2. Out of memory
docker stats

# 3. Missing dependencies
docker-compose build --no-cache
```

### Models Not Loading

```bash
# Check models directory
docker-compose exec backend ls -lh /app/models/

# Check permissions
docker-compose exec backend ls -lah /app/models/

# Copy models to container
docker cp my_model.pth crop-disease-backend:/app/models/
```

### Connection Refused

```bash
# Check if container is running
docker-compose ps

# Check port mapping
docker-compose port backend 5000

# Test from inside container
docker-compose exec backend curl http://localhost:5000/

# Check firewall
sudo ufw allow 5000
```

### High Memory Usage

```bash
# Monitor resources
docker stats crop-disease-backend

# Limit memory
# Edit docker-compose.yml:
services:
  backend:
    mem_limit: 4g
    mem_reservation: 2g
```

---

## 📈 Performance Optimization

### Multi-Stage Build

The Dockerfile uses multi-stage build to reduce image size:

```dockerfile
# Stage 1: Dependencies
FROM python:3.9-slim as base
# Install dependencies

# Stage 2: Application
FROM base as app
# Copy only necessary files
```

**Result:** ~1.5GB instead of ~3GB

### Layer Caching

Order in Dockerfile optimized for caching:

1. System dependencies (rarely change)
2. Python requirements (change occasionally)
3. Application code (changes frequently)

### Resource Limits

```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

### Model Loading Optimization

```python
# Add to app.py startup
# Warm up models to avoid cold start
for model in models:
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model(dummy_input)
```

---

## 🧪 Testing

### Health Checks

```bash
# Manual health check
curl http://localhost:5000/

# Docker health status
docker inspect crop-disease-backend | grep -A 10 Health

# Automated testing
docker-compose exec backend python -m pytest tests/
```

### Load Testing

```bash
# Install apache-bench
sudo apt-get install apache2-utils

# Test upload endpoint
ab -n 100 -c 10 http://localhost:5000/api/stats

# Test with authentication
ab -n 100 -c 10 -H "Authorization: Bearer token" http://localhost:5000/api/predict
```

---

## 🔄 Updates and Maintenance

### Updating the Application

```bash
# Pull latest code
git pull origin main

# Rebuild containers
docker-compose build --no-cache

# Restart with new image
docker-compose up -d

# Verify
docker-compose ps
```

### Database Migrations (if added later)

```bash
# Run migrations
docker-compose exec backend flask db upgrade

# Create migration
docker-compose exec backend flask db migrate -m "description"
```

### Backup and Restore

```bash
# Backup uploaded images
tar -czf backup-uploads-$(date +%Y%m%d).tar.gz data/uploads/

# Backup models
tar -czf backup-models-$(date +%Y%m%d).tar.gz models/

# Restore
tar -xzf backup-uploads-20231115.tar.gz
tar -xzf backup-models-20231115.tar.gz
```

---

## 🌐 Cloud Deployment

### AWS (EC2 + ECS)

```bash
# Install AWS CLI
pip install awscli

# Login to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t crop-disease .
docker tag crop-disease:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/crop-disease:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/crop-disease:latest

# Deploy to ECS
aws ecs update-service --cluster crop-cluster --service crop-service --force-new-deployment
```

### Google Cloud (Cloud Run)

```bash
# Build and submit
gcloud builds submit --tag gcr.io/<project-id>/crop-disease

# Deploy
gcloud run deploy crop-disease \
  --image gcr.io/<project-id>/crop-disease \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

### Azure (Container Instances)

```bash
# Login to ACR
az acr login --name myregistry

# Build and push
docker build -t myregistry.azurecr.io/crop-disease .
docker push myregistry.azurecr.io/crop-disease

# Deploy
az container create \
  --resource-group myResourceGroup \
  --name crop-disease \
  --image myregistry.azurecr.io/crop-disease \
  --cpu 2 \
  --memory 4 \
  --ports 5000
```

---

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
- [Nginx Documentation](https://nginx.org/en/docs/)
- [Flask Deployment](https://flask.palletsprojects.com/en/2.3.x/deploying/)
- [PyTorch Docker](https://pytorch.org/get-started/locally/)

---

## 💡 Tips and Tricks

### Development Workflow

```bash
# 1. Make code changes
# 2. Restart without rebuilding (if only Python code changed)
docker-compose restart backend

# 3. Rebuild if requirements changed
docker-compose build backend
docker-compose up -d backend
```

### Debugging

```bash
# Run backend in interactive mode
docker-compose run --rm --service-ports backend bash
python app.py  # Manual start for debugging

# Check environment variables
docker-compose exec backend env

# View Python packages
docker-compose exec backend pip list
```

### Clean Up

```bash
# Remove stopped containers
docker-compose rm

# Remove all unused images
docker image prune -a

# Remove all unused volumes
docker volume prune

# Complete cleanup (CAUTION: removes everything)
docker system prune -a --volumes
```

---

## ✅ Checklist for Production

- [ ] SSL/TLS certificates configured
- [ ] Environment variables set
- [ ] Resource limits configured
- [ ] Logging to external service (e.g., CloudWatch, Datadog)
- [ ] Monitoring dashboard (Grafana, Prometheus)
- [ ] Backup strategy in place
- [ ] Health checks configured
- [ ] Firewall rules set
- [ ] Security scan completed
- [ ] Load testing performed
- [ ] Documentation updated
- [ ] Team trained on deployment

---

**Need help?** Check the main README or open an issue on GitHub.
