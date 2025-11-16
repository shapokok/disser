# 🐳 Docker Deployment Guide

Complete guide for deploying the Crop Disease Detection System using Docker.

---

## 🚀 Quick Start (Recommended)

The easiest way to run the entire application with one command:

```bash
docker-compose up --build
```

**That's it!** The application will be available at:
- **Frontend:** http://localhost:8080
- **Backend API:** http://localhost:5000

---

## 📋 Prerequisites

1. **Docker** installed (version 20.10+)
   ```bash
   docker --version
   ```

2. **Docker Compose** installed (version 1.29+)
   ```bash
   docker-compose --version
   ```

3. **Minimum System Requirements:**
   - 4GB RAM (8GB recommended for model inference)
   - 10GB disk space
   - Modern CPU (multi-core recommended)

---

## 🎯 Deployment Options

### Option 1: Docker Compose (Full Stack)

**Best for:** Complete deployment with frontend + backend

```bash
cd /home/user/disser/crop_monitoring_app

# Build and start all services
docker-compose up --build

# Run in background (detached mode)
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

**Services Started:**
- ✅ Backend API (Flask) on port 5000
- ✅ Frontend (Nginx) on port 8080
- ✅ Automatic networking between services
- ✅ Volume mounts for persistent data

### Option 2: Docker Only (Backend)

**Best for:** API-only deployment or custom frontend

```bash
# Build the image
docker build -t crop-disease-app .

# Run the container
docker run -p 5000:5000 \
  -v $(pwd)/data/uploads:/app/data/uploads \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  --name crop-disease-backend \
  crop-disease-app

# Run in background
docker run -d -p 5000:5000 \
  -v $(pwd)/data/uploads:/app/data/uploads \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  --name crop-disease-backend \
  crop-disease-app
```

### Option 3: Development Mode

**Best for:** Development with live code changes

```bash
docker-compose -f docker-compose.dev.yml up
```

*(Note: Create docker-compose.dev.yml with volume mounts for live reload)*

---

## 📁 Docker Architecture

### Multi-Stage Build

The Dockerfile uses a multi-stage build for optimal image size:

```
Stage 1 (Builder):  Install dependencies → 2GB
Stage 2 (Production): Copy only runtime → 800MB
```

**Benefits:**
- Smaller final image (60% size reduction)
- Faster deployment
- Better security (no build tools in production)

### Volume Mounts

| Volume | Purpose | Persistence |
|--------|---------|-------------|
| `./data/uploads` | User uploaded images | ✅ Persistent |
| `./models` | ML model weights | ✅ Persistent |
| `./results` | Generated heatmaps | ✅ Persistent |
| `./data/sample_images` | Test images | ✅ Persistent |

---

## 🔧 Configuration

### Environment Variables

Create `.env` file for custom configuration:

```env
# Backend
FLASK_ENV=production
PORT=5000
HOST=0.0.0.0

# Frontend
NGINX_PORT=8080

# Models
USE_GPU=false
MODEL_PATH=/app/models
```

### Custom Ports

Change ports in `docker-compose.yml`:

```yaml
services:
  backend:
    ports:
      - "8000:5000"  # Change 8000 to your desired port

  frontend:
    ports:
      - "3000:80"    # Change 3000 to your desired port
```

---

## 🧪 Testing Docker Deployment

### 1. Health Checks

```bash
# Check backend health
curl http://localhost:5000/

# Check frontend
curl http://localhost:8080/
```

### 2. API Testing

```bash
# Upload test image
curl -X POST http://localhost:5000/api/upload \
  -F 'files=@/path/to/image.jpg'

# Test prediction
curl -X POST http://localhost:5000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"image.jpg","model":"efficientnet"}'
```

### 3. Container Status

```bash
# List running containers
docker ps

# View container logs
docker logs crop-disease-backend
docker logs crop-disease-frontend

# Inspect container
docker inspect crop-disease-backend
```

---

## 🐛 Troubleshooting

### Issue 1: Port Already in Use

```bash
# Find process using port 5000
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or use different port in docker-compose.yml
```

### Issue 2: Permission Denied

```bash
# Fix volume permissions
sudo chown -R $USER:$USER ./data ./models ./results

# Or run with sudo
sudo docker-compose up
```

### Issue 3: Container Crashes

```bash
# View logs
docker-compose logs backend

# Common fixes:
# 1. Increase Docker memory limit (Docker Desktop settings)
# 2. Check model files exist in ./models/
# 3. Verify dependencies in requirements.txt
```

### Issue 4: Cannot Connect to Backend

```bash
# Check network
docker network ls
docker network inspect crop-disease_crop-disease-network

# Restart services
docker-compose restart
```

---

## 🔄 Updates and Maintenance

### Update Application Code

```bash
# Pull latest changes
git pull

# Rebuild containers
docker-compose up --build

# Or rebuild specific service
docker-compose up --build backend
```

### Clean Up Old Images

```bash
# Remove old images
docker image prune -a

# Remove all stopped containers
docker container prune

# Remove unused volumes
docker volume prune
```

### Backup Data

```bash
# Backup models and data
tar -czf backup_$(date +%Y%m%d).tar.gz ./models ./data

# Restore from backup
tar -xzf backup_20250116.tar.gz
```

---

## 📊 Resource Monitoring

### View Resource Usage

```bash
# Real-time stats
docker stats

# Specific container
docker stats crop-disease-backend
```

### Optimize Performance

1. **Increase Memory:**
   ```yaml
   # In docker-compose.yml
   services:
     backend:
       deploy:
         resources:
           limits:
             memory: 4G
           reservations:
             memory: 2G
   ```

2. **Use GPU (if available):**
   ```yaml
   services:
     backend:
       runtime: nvidia
       environment:
         - NVIDIA_VISIBLE_DEVICES=all
   ```

---

## 🌐 Production Deployment

### SSL/HTTPS Setup

1. **Install Let's Encrypt:**
   ```bash
   sudo apt-get install certbot python3-certbot-nginx
   ```

2. **Get Certificate:**
   ```bash
   sudo certbot --nginx -d yourdomain.com
   ```

3. **Update nginx.conf:**
   ```nginx
   server {
       listen 443 ssl;
       ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
       ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
       ...
   }
   ```

### Reverse Proxy

For production, use nginx as reverse proxy:

```nginx
upstream backend {
    server localhost:5000;
}

server {
    listen 80;
    server_name yourdomain.com;

    location / {
        root /var/www/frontend;
        try_files $uri /index.html;
    }

    location /api {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

---

## 📝 Docker Commands Cheat Sheet

```bash
# Build
docker build -t crop-disease-app .
docker-compose build

# Run
docker run -p 5000:5000 crop-disease-app
docker-compose up
docker-compose up -d

# Stop
docker stop crop-disease-backend
docker-compose down

# Logs
docker logs crop-disease-backend
docker logs -f crop-disease-backend
docker-compose logs -f

# Clean up
docker rm crop-disease-backend
docker rmi crop-disease-app
docker system prune -a

# Inspect
docker ps
docker ps -a
docker images
docker inspect crop-disease-backend
```

---

## ✅ Verification Checklist

After deployment, verify:

- [ ] Backend responds at http://localhost:5000
- [ ] Frontend loads at http://localhost:8080
- [ ] Can upload images
- [ ] Predictions work with all 4 models
- [ ] Ensemble prediction works
- [ ] Export to CSV/JSON/Excel works
- [ ] Visualization (Grad-CAM/LIME) displays
- [ ] No errors in `docker logs`

---

## 🆘 Getting Help

If you encounter issues:

1. Check logs: `docker-compose logs -f`
2. Verify all files exist (Dockerfile, docker-compose.yml, nginx.conf)
3. Ensure ports 5000 and 8080 are available
4. Check Docker has enough resources (memory, disk)
5. Review this guide's troubleshooting section

---

**Happy Dockerizing! 🐳**
