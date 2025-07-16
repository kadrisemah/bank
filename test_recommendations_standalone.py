#!/usr/bin/env python3
"""
Test recommendations without API dependencies
"""

import pandas as pd

def get_client_recommendations(client_id, n_recommendations=5):
    """Get recommendations for a client"""
    
    # Product mapping
    product_mapping = {
        '201': {'name': 'Savings Account Premium', 'category': 'Savings', 'description': 'High-yield savings account with premium benefits'},
        '210': {'name': 'Personal Loan', 'category': 'Loans', 'description': 'Flexible personal loan with competitive rates'},
        '230': {'name': 'Auto Insurance', 'category': 'Insurance', 'description': 'Comprehensive auto insurance coverage'},
        '270': {'name': 'Investment Portfolio', 'category': 'Investment', 'description': 'Diversified investment portfolio management'},
        '301': {'name': 'Credit Card Gold', 'category': 'Credit', 'description': 'Premium credit card with rewards program'},
        '145': {'name': 'Home Mortgage', 'category': 'Loans', 'description': 'Fixed-rate home mortgage loan'},
        '189': {'name': 'Life Insurance', 'category': 'Insurance', 'description': 'Term life insurance policy'},
        '234': {'name': 'Business Account', 'category': 'Business', 'description': 'Professional business banking account'},
        '156': {'name': 'Mobile Banking Plus', 'category': 'Digital', 'description': 'Enhanced mobile banking services'},
        '278': {'name': 'Retirement Plan', 'category': 'Investment', 'description': 'Long-term retirement savings plan'},
        '221': {'name': 'Current Account', 'category': 'Banking', 'description': 'Standard current account'},
        '222': {'name': 'Business Current Account', 'category': 'Business', 'description': 'Business current account'},
        '665': {'name': 'Investment Fund', 'category': 'Investment', 'description': 'Investment fund portfolio'}
    }
    
    try:
        # Load product data
        products_df = pd.read_csv('data/processed/products_cleaned.csv')
        
        # Get products this client already has (active products)
        client_products = products_df[
            (products_df['CLI'] == client_id) & 
            (products_df['ETA'] == 'VA')
        ]['CPRO'].unique()
        
        # Get all available products
        all_products = products_df['CPRO'].unique()
        
        # Find products this client doesn't have (for recommendations)
        available_products = [p for p in all_products if p not in client_products and str(p).strip()]
        
        # Calculate popularity scores based on usage
        product_popularity = products_df[products_df['ETA'] == 'VA'].groupby('CPRO').size().sort_values(ascending=False)
        
        # Generate recommendations
        recommendations = []
        recommended_products = available_products[:n_recommendations]
        
        for i, product_id in enumerate(recommended_products):
            product_str = str(product_id).strip()
            
            # Get product info
            product_info = product_mapping.get(product_str, {
                'name': f'Product {product_str}',
                'category': 'Banking',
                'description': 'Banking product'
            })
            
            # Calculate score based on popularity and position
            popularity_score = product_popularity.get(product_id, 0)
            base_score = 0.9 - (i * 0.1)  # Decreasing score
            popularity_factor = min(popularity_score / 1000, 0.2)  # Bonus for popular products
            final_score = min(base_score + popularity_factor, 0.95)
            
            recommendations.append({
                'product_id': product_str,
                'score': final_score,
                'product_name': product_info['name'],
                'category': product_info['category'],
                'description': product_info['description']
            })
        
        return recommendations
        
    except Exception as e:
        print(f"Error: {e}")
        return []

def test_recommendations():
    """Test recommendations with real client IDs"""
    print("Testing Recommendations...")
    
    try:
        # Load client data
        clients_df = pd.read_excel('data/raw/Clients_DOU_replaced_DDMMYYYY.xlsx')
        
        # Test with first 5 clients
        test_clients = clients_df['CLI'].head(5).tolist()
        print(f"Testing with clients: {test_clients}")
        
        for client_id in test_clients:
            recommendations = get_client_recommendations(client_id, 5)
            print(f"\n📊 Client {client_id}: {len(recommendations)} recommendations")
            
            for i, rec in enumerate(recommendations):
                print(f"  {i+1}. {rec['product_name']} ({rec['category']}) - Score: {rec['score']:.2f}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recommendations()