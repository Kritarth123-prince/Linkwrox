import uvicorn
import sys
import subprocess
import importlib
from pathlib import Path

def check_package_installed(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def install_requirements():
    packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn[standard]", 
        "multipart": "python-multipart"
    }
    
    missing_packages = []
    
    for import_name, install_name in packages.items():
        if not check_package_installed(import_name):
            missing_packages.append(install_name)
        else:
            print(f"Package {import_name} already installed")
    
    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("All required packages installed")
    else:
        print("All required packages already available")

def main():
    print("=" * 60)
    print("LINKWROX SERVER LAUNCHER")
    print("Copyright 2025 Kritarth Ranjan - All Rights Reserved")
    print("Version: 1.0.0-Optimized")
    print("=" * 60)
    
    install_requirements()
    
    if not Path("linkwrox_api.py").exists():
        print("Error: linkwrox_api.py not found!")
        return
    
    print("\nStarting Linkwrox server...")
    print("Access at: http://127.0.0.1:8080")
    print("API docs at: http://127.0.0.1:8080/docs")
    print("Press Ctrl+C to stop")
    print("-" * 40)
    
    try:
        uvicorn.run(
            "linkwrox_api:app",
            host="127.0.0.1",
            port=8080,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nLinkwrox server stopped")

if __name__ == "__main__":
    main()