#!/usr/bin/env python3
"""
Crop Disease Detection System - Main Entry Point
Centralized launcher that starts the backend server and provides setup guidance
"""

import os
import sys

def print_banner():
    """Print application banner"""
    print("=" * 70)
    print("  🌱 CROP DISEASE DETECTION SYSTEM")
    print("  Intelligent Agricultural Monitoring with Deep Learning")
    print("=" * 70)
    print()

def check_dependencies():
    """Check if required packages are installed"""
    print("📦 Checking dependencies...")

    required_packages = {
        'flask': 'Flask',
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'matplotlib': 'Matplotlib'
    }

    missing = []
    for module, name in required_packages.items():
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            missing.append(name)

    print()

    if missing:
        print("❌ Missing dependencies detected!")
        print()
        print("To install missing packages, run:")
        print("  cd backend")
        print("  pip install -r requirements.txt")
        print()
        print("Or install individually:")
        print(f"  pip install {' '.join(missing.lower() for missing in missing)}")
        print()
        return False

    print("✅ All dependencies installed!")
    print()
    return True

def check_directories():
    """Check if required directories exist"""
    print("📁 Checking directory structure...")

    required_dirs = [
        'backend',
        'frontend',
        'models',
        'data/uploads',
        'data/sample_images',
        'data/field_images',
        'results/heatmaps'
    ]

    missing_dirs = []
    for dir_path in required_dirs:
        full_path = os.path.join(os.path.dirname(__file__), dir_path)
        if os.path.exists(full_path):
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ - MISSING")
            missing_dirs.append(full_path)

    print()

    if missing_dirs:
        print("Creating missing directories...")
        for dir_path in missing_dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"  ✓ Created {os.path.relpath(dir_path)}/")
        print()

    return True

def check_backend_files():
    """Check if backend files exist"""
    print("🔍 Checking backend files...")

    backend_files = [
        'backend/app.py',
        'backend/model.py',
        'backend/utils.py',
        'backend/config.py',
        'backend/validation.py',
        'backend/report_generator.py',
        'backend/requirements.txt'
    ]

    all_exist = True
    for file_path in backend_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_exist = False

    print()
    return all_exist

def check_frontend_files():
    """Check if frontend files exist"""
    print("🌐 Checking frontend files...")

    frontend_files = [
        'frontend/index.html',
        'frontend/analyze.html',
        'frontend/stats.html',
        'frontend/style.css',
        'frontend/script.js'
    ]

    all_exist = True
    for file_path in frontend_files:
        full_path = os.path.join(os.path.dirname(__file__), file_path)
        if os.path.exists(full_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_exist = False

    print()
    return all_exist

def print_usage_instructions():
    """Print usage instructions"""
    print("🚀 QUICK START GUIDE")
    print("-" * 70)
    print()
    print("1. Start the backend server:")
    print("   python main.py --server")
    print("   OR")
    print("   cd backend && python app.py")
    print()
    print("2. Open the frontend in your browser:")
    print("   • Navigate to: frontend/index.html")
    print("   • Or use a local server:")
    print("     python -m http.server 8080")
    print("     Then visit: http://localhost:8080/frontend/")
    print()
    print("3. Upload images and start analyzing!")
    print()
    print("-" * 70)
    print()

def start_server():
    """Start the Flask backend server"""
    print("🚀 Starting backend server...")
    print()

    # Change to backend directory
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend')
    os.chdir(backend_dir)

    # Import and run app
    try:
        from backend import app as application
        print("Backend server starting at http://localhost:5000")
        print("Press Ctrl+C to stop")
        print()
        application.app.run(host='0.0.0.0', port=5000, debug=True)
    except ImportError:
        # Fallback: execute app.py directly
        import subprocess
        subprocess.run([sys.executable, 'app.py'])

def main():
    """Main entry point"""
    print_banner()

    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--server' or sys.argv[1] == '-s':
            start_server()
            return
        elif sys.argv[1] == '--help' or sys.argv[1] == '-h':
            print("Usage:")
            print("  python main.py          # Run diagnostics and show instructions")
            print("  python main.py --server # Start backend server")
            print("  python main.py --help   # Show this help message")
            print()
            return

    # Run diagnostics
    deps_ok = check_dependencies()
    check_directories()
    backend_ok = check_backend_files()
    frontend_ok = check_frontend_files()

    print("=" * 70)
    print()

    if deps_ok and backend_ok and frontend_ok:
        print("✅ System check passed! Your application is ready to run.")
        print()
        print_usage_instructions()
    else:
        print("⚠️ Some issues were detected. Please fix them before running.")
        print()
        if not deps_ok:
            print("1. Install Python dependencies (see above)")
        if not backend_ok:
            print("2. Restore missing backend files from backup")
        if not frontend_ok:
            print("3. Restore missing frontend files from backup")
        print()
        print("For detailed setup instructions, see:")
        print("  • README.md")
        print("  • SETUP.md")
        print()

if __name__ == "__main__":
    main()
