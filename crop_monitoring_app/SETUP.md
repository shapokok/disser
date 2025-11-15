# 🚀 Quick Setup Guide

Complete setup instructions for the Crop Disease Detection System.

## ⏱️ Estimated Setup Time: 15-20 minutes

---

## 📋 Prerequisites Checklist

Before starting, ensure you have:

- [ ] **Python 3.8 or higher** installed
  ```bash
  python --version  # Should show Python 3.8+
  ```

- [ ] **pip** (Python package manager) installed
  ```bash
  pip --version
  ```

- [ ] **At least 2GB free disk space** for dependencies

- [ ] **Modern web browser** (Chrome, Firefox, Safari, Edge)

- [ ] **(Optional) Git** for version control

---

## 🔧 Installation Steps

### Step 1: Navigate to Project Directory

```bash
cd crop_monitoring_app
```

### Step 2: Create Virtual Environment

**Why?** Keeps dependencies isolated from your system Python.

#### On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### Step 3: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

**Note:** This will download ~2GB of packages. It may take 5-10 minutes.

If you encounter errors:

**Option A - CPU Only (Faster install):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-cpu
pip install -r requirements.txt
```

**Option B - GPU Support (if you have NVIDIA GPU with CUDA):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow[and-cuda]
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import torch; import tensorflow; import flask; print('✓ All packages installed successfully!')"
```

---

## 🎯 Running the Application

### Terminal 1: Start Backend Server

```bash
# Make sure you're in the backend directory
cd backend

# Activate virtual environment if not already activated
# Windows: venv\Scripts\activate
# macOS/Linux: source venv/bin/activate

# Run the server
python app.py
```

**Expected Output:**
```
Loading models...
✓ Baseline model loaded
✓ EfficientNet model loaded
✓ MobileNet model loaded
✓ Hybrid CNN-Transformer model loaded
All models loaded successfully!

==================================================
Crop Disease Detection System - Backend Server
==================================================
Device: cpu
Loaded models: ['baseline', 'efficientnet', 'mobilenet', 'hybrid']
==================================================

 * Running on http://0.0.0.0:5000
 * Press CTRL+C to quit
```

### Terminal 2: Serve Frontend (Optional but Recommended)

#### Option A: Python HTTP Server
```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Python 3
python -m http.server 8000

# Or Python 2
python -m SimpleHTTPServer 8000
```

Then open: **http://localhost:8000**

#### Option B: Direct File Access
Simply open `frontend/index.html` in your web browser.

**Note:** Option A is recommended to avoid CORS issues.

---

## 🧪 Testing the Installation

### 1. Test Backend API

Open a new terminal and run:

```bash
curl http://localhost:5000
```

**Expected Response:**
```json
{
  "status": "online",
  "message": "Crop Disease Detection API",
  "models_loaded": ["baseline", "efficientnet", "mobilenet", "hybrid"]
}
```

Or open in browser: http://localhost:5000

### 2. Test Frontend

1. Open http://localhost:8000 (or your frontend URL)
2. You should see the home page with a green header
3. Click "Start Analysis"
4. Try uploading any image (even a non-plant image for testing)

### 3. Test Complete Workflow

1. Find or download a plant leaf image
2. Go to "Analyze" page
3. Select model: **EfficientNet-B0**
4. Select explanation: **Grad-CAM**
5. Upload your image
6. Click "Analyze Images"
7. Wait 5-10 seconds
8. You should see results with a heatmap visualization

---

## 🐛 Common Issues & Solutions

### Issue 1: Port Already in Use

**Error:** `Address already in use` or `Port 5000 is already in use`

**Solution:**

**Windows:**
```bash
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

**macOS/Linux:**
```bash
lsof -ti:5000 | xargs kill -9
```

Or change the port in `backend/app.py`:
```python
app.run(host='0.0.0.0', port=5001, debug=True)  # Change 5000 to 5001
```

### Issue 2: CORS Errors

**Error:** Browser console shows `CORS policy` errors

**Solution:**
1. Ensure Flask-CORS is installed: `pip install Flask-CORS`
2. Use the Python HTTP server for frontend (Option A above)
3. Check that backend URL in `frontend/script.js` is correct

### Issue 3: Module Not Found

**Error:** `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
# Make sure virtual environment is activated
# Then reinstall
pip install -r requirements.txt
```

### Issue 4: Out of Memory

**Error:** `CUDA out of memory` or system freezes

**Solution:**
The app automatically uses CPU if GPU is unavailable. If still having issues:
1. Close other applications
2. Process one image at a time
3. Use MobileNet model (smallest)

### Issue 5: Slow Loading

**Symptom:** Models take long to load or predict

**Explanation:**
- First load is slow (loading model weights)
- Subsequent predictions are faster
- CPU inference is slower than GPU
- Expected: 5-10 seconds for first prediction, 1-2 seconds after

**This is normal!** For thesis demo, load models before presentation.

---

## 📦 Adding Your Trained Models

If you have trained PyTorch models:

1. Save your model state dict:
   ```python
   torch.save(model.state_dict(), 'my_model.pth')
   ```

2. Copy to models directory:
   ```bash
   cp my_model.pth crop_monitoring_app/models/efficientnet_model.pth
   ```

3. Restart the backend server

**Expected model files:**
- `baseline_model.pth`
- `efficientnet_model.pth`
- `mobilenet_model.pth`
- `hybrid_model.pth`

**Note:** If model files don't exist, the app uses pre-trained ImageNet weights (for EfficientNet/MobileNet) or random initialization (for custom models). This is fine for demonstration!

---

## 🎓 For Thesis Defense

### Preparation Checklist

1. **Before Defense:**
   - [ ] Test all functionality thoroughly
   - [ ] Prepare 5-10 sample images (various diseases)
   - [ ] Take screenshots of key results
   - [ ] Test on the presentation computer
   - [ ] Have backup images ready
   - [ ] Print/save key visualizations

2. **During Setup:**
   - [ ] Start backend server (5 min before)
   - [ ] Open frontend in browser
   - [ ] Test with one image to warm up models
   - [ ] Keep terminal visible (shows it's running)

3. **Demo Flow:**
   - Show home page (overview)
   - Navigate to Analyze page
   - Upload 2-3 different disease images
   - Show Grad-CAM visualizations
   - Navigate to Statistics page
   - Explain model comparisons
   - (Optional) Live demo with committee's test image

4. **Backup Plan:**
   - Keep pre-generated screenshots
   - Have results saved as images
   - Prepare PDF with visualizations

### Impressive Demo Points

✨ **Highlight These:**
- Multiple model architectures implemented
- Explainable AI (Grad-CAM showing "why")
- Real-time web application
- Professional UI/UX
- Comparative analysis
- 38 disease classes supported
- Both controlled and field image support

---

## 🌐 Deployment Options (Optional)

### Local Network Access

Allow others on your network to access:

```python
# In app.py, it's already set:
app.run(host='0.0.0.0', port=5000)
```

Find your IP:
- **Windows:** `ipconfig`
- **macOS/Linux:** `ifconfig` or `ip addr`

Share: `http://YOUR_IP:5000`

### Cloud Deployment

For permanent deployment:

**Option 1: Heroku**
- Free tier available
- Easy deployment

**Option 2: AWS EC2**
- More control
- Requires server management

**Option 3: Google Cloud Run**
- Container-based
- Auto-scaling

(Deployment guides available online)

---

## 📝 Development Tips

### Modifying the Code

1. **Backend Changes:**
   - Edit `backend/app.py` for API changes
   - Edit `backend/model.py` for model changes
   - Restart server to see changes

2. **Frontend Changes:**
   - Edit HTML/CSS/JS files
   - Just refresh browser (no restart needed)

3. **Adding New Models:**
   ```python
   # In backend/model.py
   model_manager.load_model('my_new_model', 'model_type')
   ```

### Debugging

**Enable debug mode:**
Backend already has `debug=True`

**Check logs:**
- Backend: Terminal output
- Frontend: Browser Developer Console (F12)

---

## ✅ Post-Setup Checklist

After setup, verify:

- [ ] Backend server starts without errors
- [ ] Frontend loads correctly
- [ ] Can upload images
- [ ] Can run predictions
- [ ] Visualizations appear
- [ ] Statistics page loads
- [ ] All 4 models are available
- [ ] Both Grad-CAM and LIME work

---

## 🎉 You're Ready!

Everything should now be working.

**Next Steps:**
1. Read the main README.md for detailed documentation
2. Test with various plant images
3. Explore different models and settings
4. Prepare your presentation materials
5. Practice your demo

**Good luck with your thesis defense! 🌱🎓**

---

## 📞 Need Help?

1. Check error messages in terminal
2. Review browser console (F12)
3. Ensure all steps above are completed
4. Verify Python version and dependencies
5. Check README.md troubleshooting section

**Remember:** The system works with or without trained model files - perfect for demonstration!
