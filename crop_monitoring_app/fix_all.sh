#!/bin/bash

# Comprehensive Fix Script for Crop Disease Detection System
# This script fixes all common issues and sets up the environment

echo "=========================================="
echo "  🔧 Crop Disease Detection - Fix All"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to script directory
cd "$(dirname "$0")"

echo "📁 Step 1: Creating missing directories..."
mkdir -p backend
mkdir -p frontend
mkdir -p models
mkdir -p data/uploads
mkdir -p data/sample_images
mkdir -p data/field_images
mkdir -p results/heatmaps
echo -e "${GREEN}✓${NC} Directories created"
echo ""

echo "📦 Step 2: Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗${NC} Python 3 not found! Please install Python 3.8+"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo -e "${GREEN}✓${NC} $PYTHON_VERSION installed"
echo ""

echo "📚 Step 3: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${YELLOW}→${NC} Virtual environment already exists"
fi
echo ""

echo "🔌 Step 4: Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓${NC} Virtual environment activated"
echo ""

echo "📥 Step 5: Installing core dependencies..."
pip install --no-cache-dir Flask Flask-CORS Werkzeug --quiet
echo -e "${GREEN}✓${NC} Core web framework installed"
echo ""

echo "🧮 Step 6: Installing scientific computing packages..."
pip install --no-cache-dir numpy scipy scikit-learn --quiet
echo -e "${GREEN}✓${NC} Scientific packages installed"
echo ""

echo "🖼️  Step 7: Installing image processing libraries..."
pip install --no-cache-dir Pillow opencv-python matplotlib seaborn --quiet
echo -e "${GREEN}✓${NC} Image processing libraries installed"
echo ""

echo "🤖 Step 8: Installing machine learning frameworks..."
echo "   (This may take several minutes...)"
pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo -e "${GREEN}✓${NC} PyTorch installed"
echo ""

echo "📊 Step 9: Installing additional dependencies..."
pip install --no-cache-dir reportlab python-multipart --quiet
echo -e "${GREEN}✓${NC} Additional packages installed"
echo ""

echo "🗂️  Step 10: Initializing models directory..."
cd models
if [ ! -f "class_names.json" ]; then
    python3 -c "
import json
class_names = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]
with open('class_names.json', 'w') as f:
    json.dump(class_names, f, indent=2)
print('Created class_names.json')
" && echo -e "${GREEN}✓${NC} class_names.json created"
else
    echo -e "${YELLOW}→${NC} class_names.json already exists"
fi
cd ..
echo ""

echo "✅ Step 11: Making scripts executable..."
chmod +x main.py 2>/dev/null
chmod +x run.sh 2>/dev/null
chmod +x download_samples.py 2>/dev/null
chmod +x fix_all.sh 2>/dev/null
echo -e "${GREEN}✓${NC} Scripts are executable"
echo ""

echo "=========================================="
echo "  ✨ Setup Complete!"
echo "=========================================="
echo ""
echo "📋 Summary:"
echo "  • All directories created"
echo "  • Virtual environment ready"
echo "  • All dependencies installed"
echo "  • Models directory initialized"
echo ""
echo "🚀 Next Steps:"
echo "  1. Start the backend:"
echo "     ./run.sh"
echo "     OR"
echo "     python main.py --server"
echo ""
echo "  2. Open the frontend:"
echo "     Open frontend/index.html in your browser"
echo ""
echo "  3. (Optional) Download sample images:"
echo "     python download_samples.py"
echo ""
echo "=========================================="
echo ""
