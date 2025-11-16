# 🐳 Docker Quick Start - 5 Minutes

## Prerequisites
- Docker installed ([Get Docker](https://docs.docker.com/get-docker/))
- 4GB RAM available
- 10GB disk space

## 🚀 Start in 3 Commands

```bash
# 1. Make scripts executable (first time only)
chmod +x docker-start.sh docker-stop.sh

# 2. Start the system
./docker-start.sh

# 3. Open your browser
# Backend API: http://localhost:5000
# Frontend: Open crop_monitoring_app/frontend/index.html
```

**That's it!** 🎉 The system is now running.

---

## 📝 What Just Happened?

The script:
1. ✅ Created necessary directories
2. ✅ Built Docker images (~2-3 minutes)
3. ✅ Started the backend container
4. ✅ Loaded all 4 ML models
5. ✅ Made API available on port 5000

---

## 🎯 Quick Tests

### Test 1: Check API Health
```bash
curl http://localhost:5000/
```

Expected: `{"status": "online", ...}`

### Test 2: List Available Models
```bash
curl http://localhost:5000/api/models
```

Expected: `{"models": ["baseline", "efficientnet", "mobilenet", "hybrid"]}`

### Test 3: Upload an Image
```bash
curl -X POST http://localhost:5000/api/upload \
  -F "files=@path/to/your/image.jpg"
```

---

## 🛑 Stop the System

```bash
# Stop containers (keeps data)
./docker-stop.sh

# Stop and remove everything (including uploaded images)
./docker-stop.sh --clean
```

---

## 📊 Useful Commands

```bash
# View logs
docker-compose logs -f backend

# Check status
docker-compose ps

# Restart
docker-compose restart

# Shell access
docker-compose exec backend bash
```

---

## 🐛 Problems?

### Port 5000 Already in Use
```bash
# Find what's using it
lsof -i :5000

# Kill the process
kill -9 <PID>

# Or change the port in docker-compose.yml
```

### Models Not Loading
```bash
# Check logs
docker-compose logs backend | grep -i "error"

# Models will download automatically on first run
# This is normal and takes 2-3 minutes
```

### Out of Memory
```bash
# Check available memory
free -h

# Restart Docker
sudo systemctl restart docker

# Or increase Docker memory limit in Docker Desktop settings
```

---

## 🚀 Production Deployment

```bash
# Start with Nginx reverse proxy
./docker-start.sh production

# Now access:
# Frontend: http://localhost
# Backend: http://localhost/api/
```

---

## 📚 Need More Details?

See full documentation:
- **[DOCKER_DEPLOYMENT.md](DOCKER_DEPLOYMENT.md)** - Complete deployment guide
- **[TESTING_GUIDE.md](crop_monitoring_app/TESTING_GUIDE.md)** - Testing procedures
- **[SESSION_SUMMARY.md](crop_monitoring_app/SESSION_SUMMARY.md)** - All recent changes

---

## ✨ What's Next?

1. Upload some plant images via the frontend
2. Try different models (efficientnet, hybrid, mobilenet)
3. Compare Grad-CAM vs LIME visualizations
4. Export results and analysis

**Enjoy your containerized ML system!** 🌱🔬
