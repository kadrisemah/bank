#!/usr/bin/env python3
"""
Test the API recommendation system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.ml_models import ProductRecommender
import pandas as pd

def test_recommender():
    """Test the product recommender directly"""
    print("Testing Product Recommender...")
    
    # Load the trained model
    recommender = ProductRecommender()
    try:
        recommender.load_model('data/models/product_recommender')
        print("✅ Model loaded successfully")
        print(f"   User-item matrix shape: {recommender.user_item_matrix.shape}")
        print(f"   SVD model components: {recommender.svd_model.n_components}")
        print(f"   Available clients: {len(recommender.user_item_matrix.index)}")
        
        # Test with a few client IDs
        test_clients = recommender.user_item_matrix.index[:5].tolist()
        print(f"   Testing with clients: {test_clients}")
        
        for client_id in test_clients:
            recommendations = recommender.get_recommendations(client_id, 5)
            print(f"   Client {client_id}: {len(recommendations)} recommendations")
            for i, rec in enumerate(recommendations[:3]):
                print(f"     {i+1}. Product {rec['product_id']}: {rec['score']}")
        
        # Test with a non-existent client (cold start)
        fake_client = 99999999
        recommendations = recommender.get_recommendations(fake_client, 5)
        print(f"   Cold start client {fake_client}: {len(recommendations)} recommendations")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recommender()