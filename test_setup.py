"""
Test script to verify the Banking ML project setup
Place this in the root directory and run: python test_setup.py
"""

import os
import sys
import importlib
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_directories():
    """Test if all required directories exist"""
    print("📁 Testing directory structure...")
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'data/models',
        'src/data_processing',
        'src/models',
        'src/api',
        'src/dashboard',
        'notebooks'
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - Missing!")
            all_exist = False
    
    return all_exist

def test_data_files():
    """Test if Excel files are present"""
    print("\n📊 Testing data files...")
    
    required_files = [
        'agences.xlsx',
        'Clients_DOU_replaced_DDMMYYYY.xlsx',
        'Comptes_DFE_replaced_DDMMYYYY.xlsx',
        'Produits_DFSOU_replaced_DDMMYYYY.xlsx',
        'eerp_formatted_eer_sortie.xlsx',
        'gestionnaires.xlsx'
    ]
    
    data_path = 'data/raw'
    all_exist = True
    
    for file_name in required_files:
        file_path = os.path.join(data_path, file_name)
        if os.path.exists(file_path):
            print(f"  ✅ {file_name}")
        else:
            print(f"  ❌ {file_name} - Missing!")
            all_exist = False
    
    return all_exist

def test_imports():
    """Test if all modules can be imported"""
    print("\n🐍 Testing Python imports...")
    
    modules_to_test = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('sklearn', 'scikit-learn'),
        ('xgboost', 'xgboost'),
        ('lightgbm', 'lightgbm'),
        ('fastapi', 'fastapi'),
        ('dash', 'dash'),
        ('plotly', 'plotly')
    ]
    
    all_imported = True
    for module_name, package_name in modules_to_test:
        try:
            importlib.import_module(module_name)
            print(f"  ✅ {package_name}")
        except ImportError:
            print(f"  ❌ {package_name} - Not installed!")
            all_imported = False
    
    return all_imported

def test_source_files():
    """Test if all source files exist"""
    print("\n📄 Testing source files...")
    
    source_files = [
        'src/data_processing/data_processor.py',
        'src/models/ml_models.py',
        'src/api/app.py',
        'src/dashboard/app.py',
        'main.py',
        'requirements.txt'
    ]
    
    all_exist = True
    for file_path in source_files:
        if os.path.exists(file_path):
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} - Missing!")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🔍 Banking ML Project Setup Test")
    print("=" * 50)
    
    results = []
    
    # Run tests
    results.append(("Directories", test_directories()))
    results.append(("Data Files", test_data_files()))
    results.append(("Python Packages", test_imports()))
    results.append(("Source Files", test_source_files()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Your project is ready to run.")
        print("\nNext steps:")
        print("1. Run: python main.py --all")
        print("2. Start API: cd src/api && uvicorn app:app --reload")
        print("3. Start Dashboard: python src/dashboard/app.py")
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("- Missing directories: Run 'python main.py --setup'")
        print("- Missing packages: Run 'pip install -r requirements.txt'")
        print("- Missing data files: Copy Excel files to data/raw/")

if __name__ == "__main__":
    main()