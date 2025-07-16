#!/usr/bin/env python3
"""
Complete Banking ML Application Runner
Starts both API and Dashboard services
"""

import subprocess
import sys
import os
import time
import signal
from multiprocessing import Process

def start_api():
    """Start the FastAPI backend"""
    print("🚀 Starting API server...")
    os.chdir('/mnt/c/Users/Semah Kadri/Desktop/new/banking-ml-project')
    sys.path.append('/mnt/c/Users/Semah Kadri/Desktop/new/banking-ml-project/src')
    
    from src.api.app import app
    import uvicorn
    
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")

def start_dashboard():
    """Start the Dash dashboard"""
    print("🎨 Starting Dashboard...")
    os.chdir('/mnt/c/Users/Semah Kadri/Desktop/new/banking-ml-project')
    sys.path.append('/mnt/c/Users/Semah Kadri/Desktop/new/banking-ml-project/src')
    
    from src.dashboard.app import app
    
    app.run(debug=False, host="0.0.0.0", port=8050)

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n🛑 Shutting down services...")
    sys.exit(0)

if __name__ == "__main__":
    # Handle Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    print("🏦 Banking ML Platform - Complete Application")
    print("=" * 50)
    
    # Start both services
    api_process = Process(target=start_api)
    dashboard_process = Process(target=start_dashboard)
    
    try:
        api_process.start()
        print("✅ API server started on http://localhost:8001")
        
        # Wait a bit for API to start
        time.sleep(3)
        
        dashboard_process.start()
        print("✅ Dashboard started on http://localhost:8050")
        
        print("\n🌟 Application is ready!")
        print("- API Documentation: http://localhost:8001/docs")
        print("- Dashboard: http://localhost:8050")
        print("\nPress Ctrl+C to stop all services")
        
        # Wait for processes
        api_process.join()
        dashboard_process.join()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        api_process.terminate()
        dashboard_process.terminate()
        api_process.join()
        dashboard_process.join()
        print("✅ All services stopped")