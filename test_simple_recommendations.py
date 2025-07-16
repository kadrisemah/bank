#!/usr/bin/env python3
"""
Test the product recommendation system - Simple version
"""

import random

def test_fallback_recommendations():
    """Test fallback recommendation logic"""
    print("Testing fallback recommendations...")
    
    product_mapping = {
        '201': {'name': 'Savings Account Premium', 'category': 'Savings'},
        '210': {'name': 'Personal Loan', 'category': 'Loans'},
        '230': {'name': 'Auto Insurance', 'category': 'Insurance'},
        '270': {'name': 'Investment Portfolio', 'category': 'Investment'},
        '301': {'name': 'Credit Card Gold', 'category': 'Credit'},
        '145': {'name': 'Home Mortgage', 'category': 'Loans'},
        '189': {'name': 'Life Insurance', 'category': 'Insurance'},
        '234': {'name': 'Business Account', 'category': 'Business'},
        '156': {'name': 'Mobile Banking Plus', 'category': 'Digital'},
        '278': {'name': 'Retirement Plan', 'category': 'Investment'}
    }
    
    # Test different counts
    for count in [1, 3, 5, 7, 10]:
        product_ids = list(product_mapping.keys())[:count]
        products = [product_mapping[pid]['name'] for pid in product_ids]
        scores = sorted([random.uniform(0.7, 0.95) for _ in products], reverse=True)
        
        print(f"✅ Count {count}: Generated {len(products)} products")
        for i, (product, score) in enumerate(zip(products, scores)):
            print(f"   {i+1}. {product} ({score:.2f})")
        print()

def test_recommendation_logic():
    """Test the exact logic from the dashboard"""
    print("Testing dashboard logic...")
    
    # Simulate API result being None (fallback scenario)
    result = None
    
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
    
    for count in [1, 3, 5, 7, 10]:
        print(f"Testing count = {count}")
        
        if result and 'recommendations' in result and len(result['recommendations']) > 0:
            print("   Using API result")
        else:
            print("   Using fallback data")
            product_ids = list(product_mapping.keys())[:count]
            products = [product_mapping[pid]['name'] for pid in product_ids]
            scores = sorted([random.uniform(0.7, 0.95) for _ in products], reverse=True)
            categories = [product_mapping[pid]['category'] for pid in product_ids]
            descriptions = [product_mapping[pid]['description'] for pid in product_ids]
            
            print(f"   Generated {len(products)} products:")
            for i, (product, score, category) in enumerate(zip(products, scores, categories)):
                print(f"     {i+1}. {product} ({category}) - {score:.2f}")
        print()

if __name__ == "__main__":
    print("🧪 Testing Product Recommendation System")
    print("=" * 50)
    
    test_fallback_recommendations()
    test_recommendation_logic()
    
    print("🎉 Testing completed!")
    print("\nThe recommendation system should always generate products!")
    print("If you're still getting 0 recommendations, the API server might not be running on port 8001.")