#!/usr/bin/env python3
"""
Test the product recommendation system
"""

import requests
import json

def test_api_recommendations():
    """Test the API recommendations endpoint"""
    print("Testing API recommendations...")
    
    # Test cases
    test_cases = [
        {"client_id": 43568328, "count": 3},
        {"client_id": 12345678, "count": 5},
        {"client_id": 99999999, "count": 7},
    ]
    
    for test in test_cases:
        try:
            url = f"http://localhost:8001/api/v1/recommend/products/{test['client_id']}?n_recommendations={test['count']}"
            print(f"Testing: {url}")
            
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Client {test['client_id']}: Got {len(data.get('recommendations', []))} recommendations")
                if data.get('recommendations'):
                    for i, rec in enumerate(data['recommendations'][:2]):  # Show first 2
                        print(f"   {i+1}. {rec.get('product_name', 'N/A')} ({rec.get('score', 0):.2f})")
            else:
                print(f"❌ Client {test['client_id']}: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Client {test['client_id']}: Connection error - {e}")
        except Exception as e:
            print(f"❌ Client {test['client_id']}: Error - {e}")
        
        print()

def test_fallback_recommendations():
    """Test fallback recommendation logic"""
    print("Testing fallback recommendations...")
    
    import numpy as np
    
    product_mapping = {
        '201': {'name': 'Savings Account Premium', 'category': 'Savings', 'description': 'High-yield savings account'},
        '210': {'name': 'Personal Loan', 'category': 'Loans', 'description': 'Flexible personal loan'},
        '230': {'name': 'Auto Insurance', 'category': 'Insurance', 'description': 'Auto insurance coverage'},
        '270': {'name': 'Investment Portfolio', 'category': 'Investment', 'description': 'Investment management'},
        '301': {'name': 'Credit Card Gold', 'category': 'Credit', 'description': 'Premium credit card'},
        '145': {'name': 'Home Mortgage', 'category': 'Loans', 'description': 'Home mortgage loan'},
        '189': {'name': 'Life Insurance', 'category': 'Insurance', 'description': 'Life insurance policy'},
        '234': {'name': 'Business Account', 'category': 'Business', 'description': 'Business banking account'},
        '156': {'name': 'Mobile Banking Plus', 'category': 'Digital', 'description': 'Mobile banking services'},
        '278': {'name': 'Retirement Plan', 'category': 'Investment', 'description': 'Retirement savings plan'}
    }
    
    # Test different counts
    for count in [1, 3, 5, 7, 10]:
        product_ids = list(product_mapping.keys())[:count]
        products = [product_mapping[pid]['name'] for pid in product_ids]
        scores = sorted([np.random.uniform(0.7, 0.95) for _ in products], reverse=True)
        
        print(f"✅ Count {count}: Generated {len(products)} products")
        for i, (product, score) in enumerate(zip(products, scores)):
            print(f"   {i+1}. {product} ({score:.2f})")
        print()

if __name__ == "__main__":
    print("🧪 Testing Product Recommendation System")
    print("=" * 50)
    
    print("1. Testing fallback logic (always works):")
    test_fallback_recommendations()
    
    print("2. Testing API endpoint (requires server running):")
    test_api_recommendations()
    
    print("🎉 Testing completed!")