#!/usr/bin/env python3
"""
Test API functionality directly
"""

import sys
import os
sys.path.append('.')

# Test imports
try:
    from src.models.ml_models import PerformancePredictor, ProductRecommender, ChurnPredictor
    print("✅ ML models imported successfully")
except Exception as e:
    print(f"❌ ML models import failed: {e}")
    sys.exit(1)

try:
    import pandas as pd
    print("✅ Pandas imported successfully")
except Exception as e:
    print(f"❌ Pandas import failed: {e}")
    sys.exit(1)

def test_manager_performance():
    """Test manager performance model"""
    print("\n🧪 Testing Manager Performance Model...")
    
    try:
        # Load model
        predictor = PerformancePredictor()
        predictor.load_model("data/models/manager_performance")
        print("✅ Manager performance model loaded")
        
        # Test prediction
        test_data = pd.DataFrame({
            'total_clients': [100],
            'agencies_covered': [3],
            'total_products_managed': [250],
            'active_products_managed': [200],
            'products_per_client': [2.5],
            'active_products_ratio': [0.8]
        })
        
        prediction = predictor.predict(test_data)[0]
        print(f"✅ Prediction: {prediction:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Manager performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_product_recommender():
    """Test product recommender"""
    print("\n🧪 Testing Product Recommender...")
    
    try:
        # Load model
        recommender = ProductRecommender()
        recommender.load_model("data/models/product_recommender")
        print("✅ Product recommender loaded")
        print(f"   Matrix shape: {recommender.user_item_matrix.shape}")
        
        # Test with a real client
        client_ids = recommender.user_item_matrix.index[:5].tolist()
        test_client = client_ids[0]
        print(f"   Testing with client: {test_client}")
        
        recommendations = recommender.get_recommendations(test_client, 5)
        print(f"✅ Generated {len(recommendations)} recommendations")
        
        for i, rec in enumerate(recommendations[:3]):
            print(f"   {i+1}. Product {rec['product_id']}: {rec['score']}")
            
        return True
        
    except Exception as e:
        print(f"❌ Product recommender test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_churn_predictor():
    """Test churn predictor"""
    print("\n🧪 Testing Churn Predictor...")
    
    try:
        # Load model
        predictor = ChurnPredictor()
        predictor.load_model("data/models/churn_predictor")
        print("✅ Churn predictor loaded")
        
        # Test prediction
        test_data = pd.DataFrame({
            'age': [35],
            'client_seniority_days': [730],
            'total_products': [3],
            'active_products': [2],
            'active_products_ratio': [0.67],
            'avg_product_duration': [365],
            'total_accounts': [2],
            'closed_accounts_ratio': [0.0],
            'unique_account_types': [2],
            'SEXT_encoded': [1],
            'age_group_encoded': [2],
            'Segment Client_encoded': [1],
            'District_encoded': [1]
        })
        
        # Only use columns that exist in the model
        available_cols = [col for col in test_data.columns if col in predictor.feature_columns]
        if available_cols:
            test_data = test_data[available_cols]
            
            churn_prob = predictor.predict_proba(test_data)[0]
            print(f"✅ Churn probability: {churn_prob:.3f}")
            
            return True
        else:
            print("❌ No matching feature columns found")
            return False
        
    except Exception as e:
        print(f"❌ Churn predictor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🏦 Testing Banking ML Models")
    print("=" * 50)
    
    # Test all models
    tests = [
        test_manager_performance,
        test_product_recommender,
        test_churn_predictor
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n🎉 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("✅ All models are working correctly!")
        print("   Your API should work with real ML predictions")
    else:
        print("❌ Some models failed - check the errors above")

if __name__ == "__main__":
    main()