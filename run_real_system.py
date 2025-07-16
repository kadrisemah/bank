#!/usr/bin/env python3
"""
Run the real banking ML system with proper dependencies
"""

import subprocess
import sys
import os
import time
import threading

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    dependencies = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn', 
        'xgboost', 'lightgbm', 'dash', 'dash-bootstrap-components',
        'plotly', 'joblib', 'openpyxl', 'requests'
    ]
    
    for dep in dependencies:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"✅ {dep} installed")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {dep}")
            return False
    
    return True

def run_api_server():
    """Run the API server"""
    print("🚀 Starting API server...")
    
    try:
        # Change to project directory
        os.chdir('/mnt/c/Users/Semah Kadri/Desktop/new/banking-ml-project')
        
        # Run the API server
        subprocess.run([
            sys.executable, '-m', 'uvicorn', 
            'src.api.app:app', 
            '--host', '0.0.0.0', 
            '--port', '8001',
            '--reload'
        ])
        
    except Exception as e:
        print(f"❌ Error running API server: {e}")

def run_dashboard():
    """Run the dashboard"""
    print("🎨 Starting dashboard...")
    
    try:
        # Change to project directory
        os.chdir('/mnt/c/Users/Semah Kadri/Desktop/new/banking-ml-project')
        
        # Run the dashboard
        subprocess.run([sys.executable, 'src/dashboard/app.py'])
        
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

def test_models():
    """Test if models are working"""
    print("🧪 Testing models...")
    
    try:
        os.chdir('/mnt/c/Users/Semah Kadri/Desktop/new/banking-ml-project')
        
        # Test model loading
        result = subprocess.run([
            sys.executable, '-c', '''
import sys
sys.path.append(".")
from src.models.ml_models import PerformancePredictor, ProductRecommender, ChurnPredictor
import pandas as pd

print("Testing model loading...")

# Test manager performance model
manager_predictor = PerformancePredictor()
try:
    manager_predictor.load_model("data/models/manager_performance")
    print("✅ Manager performance model loaded")
except Exception as e:
    print(f"❌ Manager performance model failed: {e}")

# Test agency performance model
agency_predictor = PerformancePredictor()
try:
    agency_predictor.load_model("data/models/agency_performance")
    print("✅ Agency performance model loaded")
except Exception as e:
    print(f"❌ Agency performance model failed: {e}")

# Test product recommender
recommender = ProductRecommender()
try:
    recommender.load_model("data/models/product_recommender")
    print("✅ Product recommender model loaded")
    print(f"   Matrix shape: {recommender.user_item_matrix.shape}")
except Exception as e:
    print(f"❌ Product recommender failed: {e}")

# Test churn predictor
churn_predictor = ChurnPredictor()
try:
    churn_predictor.load_model("data/models/churn_predictor")
    print("✅ Churn predictor model loaded")
except Exception as e:
    print(f"❌ Churn predictor failed: {e}")

print("🎉 Model testing completed!")
'''], capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
    except Exception as e:
        print(f"❌ Error testing models: {e}")

def main():
    """Main function"""
    print("🏦 Banking ML System - Real Implementation")
    print("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return
    
    # Test models
    print("\n" + "=" * 50)
    test_models()
    
    # Ask user what to run
    print("\n" + "=" * 50)
    print("What would you like to run?")
    print("1. API Server only")
    print("2. Dashboard only") 
    print("3. Both API and Dashboard")
    print("4. Test models only")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        run_api_server()
    elif choice == "2":
        run_dashboard()
    elif choice == "3":
        # Run both in separate threads
        api_thread = threading.Thread(target=run_api_server)
        dashboard_thread = threading.Thread(target=run_dashboard)
        
        api_thread.start()
        time.sleep(3)  # Give API time to start
        dashboard_thread.start()
        
        api_thread.join()
        dashboard_thread.join()
    elif choice == "4":
        test_models()
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()