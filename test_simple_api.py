#!/usr/bin/env python3
"""
Simple test to check if models exist and can be loaded
"""

import os
import joblib

def test_models():
    """Test if models are properly saved and can be loaded"""
    print("Testing Model Files...")
    
    model_path = "data/models"
    
    # Check product recommender files
    recommender_files = [
        f"{model_path}/product_recommender_svd.pkl",
        f"{model_path}/product_recommender_knn.pkl", 
        f"{model_path}/product_recommender_matrix.pkl"
    ]
    
    print("\nProduct Recommender Files:")
    for file in recommender_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
            try:
                data = joblib.load(file)
                if 'matrix' in file:
                    print(f"   Matrix shape: {data.shape}")
                    print(f"   Number of clients: {len(data.index)}")
                    print(f"   Number of products: {len(data.columns)}")
                    print(f"   Sample clients: {data.index[:5].tolist()}")
                elif 'svd' in file:
                    print(f"   SVD components: {data.n_components}")
                elif 'knn' in file:
                    print(f"   KNN neighbors: {data.n_neighbors}")
            except Exception as e:
                print(f"   ❌ Error loading: {e}")
        else:
            print(f"❌ {file} missing")
    
    # Check client data
    client_files = [
        "data/raw/Clients_DOU_replaced_DDMMYYYY.xlsx",
        "data/raw/Produits_DFSOU_replaced_DDMMYYYY.xlsx"
    ]
    
    print("\nData Files:")
    for file in client_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")

if __name__ == "__main__":
    test_models()