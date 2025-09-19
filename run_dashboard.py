#!/usr/bin/env python3
"""
Simple script to run the Healthcare DSS Streamlit Dashboard
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit dashboard"""
    try:
        print("Starting Healthcare DSS Dashboard...")
        print("The dashboard will open in your default web browser.")
        print("If it doesn't open automatically, go to: http://localhost:8501")
        print("\nPress Ctrl+C to stop the dashboard.")
        
        # Run Streamlit app
        subprocess.run([sys.executable, "-m", "streamlit", "run", "healthcare_dss/streamlit_app.py"], check=True)
        
    except KeyboardInterrupt:
        print("\nDashboard stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")
        print("Make sure Streamlit is installed: pip install streamlit")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
