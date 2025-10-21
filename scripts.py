#!/usr/bin/env python3
"""
Development scripts for the stock screener project
"""
import subprocess
import sys
from pathlib import Path

def run_app():
    """Run the Streamlit app"""
    app_path = Path(__file__).parent / "src" / "stock_screener" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])

def install():
    """Install dependencies"""
    subprocess.run(["uv", "sync"])

def test():
    """Run tests (placeholder)"""
    print("No tests configured yet. Add pytest and create tests/")

def lint():
    """Run linting (placeholder)"""
    print("No linting configured yet. Add black, flake8, or ruff")

def main():
    """Main script runner"""
    if len(sys.argv) < 2:
        print("Available commands: app, install, test, lint")
        return
    
    command = sys.argv[1]
    
    if command == "app":
        run_app()
    elif command == "install":
        install()
    elif command == "test":
        test()
    elif command == "lint":
        lint()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: app, install, test, lint")

if __name__ == "__main__":
    main()