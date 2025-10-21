#!/usr/bin/env python3
"""
Script to run the Streamlit app
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Run the Streamlit app"""
    app_path = Path(__file__).parent / "src" / "stock_screener" / "app.py"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_path)
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nApp stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()