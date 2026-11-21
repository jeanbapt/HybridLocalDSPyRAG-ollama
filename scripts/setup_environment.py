"""
Setup script for Analytics-STeeR environment

This script sets up the Python environment with all required dependencies.
"""

import subprocess
import sys
from pathlib import Path

def setup_environment():
    """Set up the Python environment with required dependencies"""
    print("Setting up Analytics-STeeR environment...")
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Create virtual environment
    print("Creating virtual environment...")
    subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
    
    # Activate virtual environment and install dependencies
    if sys.platform == "win32":
        activate_script = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_script = "venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install dependencies
    print("Installing dependencies...")
    subprocess.run([
        pip_cmd, "install", "-r", str(project_root / "requirements.txt")
    ], check=True)
    
    # Install development dependencies
    print("Installing development dependencies...")
    subprocess.run([
        pip_cmd, "install", "pre-commit"
    ], check=True)
    
    # Set up pre-commit hooks
    print("Setting up pre-commit hooks...")
    subprocess.run(["pre-commit", "install"], check=True)
    
    print("\nEnvironment setup complete!")
    print("\nTo activate the environment:")
    if sys.platform == "win32":
        print("    .\\venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")
    print("\nTo deactivate the environment:")
    print("    deactivate")

if __name__ == "__main__":
    setup_environment() 