#!/usr/bin/env python3
"""
Test the recommendation logic
"""

import pandas as pd

def test_recommendation_logic():
    """Test the recommendation logic with actual data"""
    print("Testing recommendation logic...")
    
    try:
        # Load actual data
        products_df = pd.read_csv('data/processed/products_cleaned.csv')
        clients_df = pd.read_excel('data/raw/Clients_DOU_replaced_DDMMYYYY.xlsx')
        
        print(f"✅ Loaded {len(products_df)} products and {len(clients_df)} clients")
        
        # Get a real client ID from the data
        test_client_id = clients_df['CLI'].iloc[0]
        print(f"Testing with client ID: {test_client_id}")
        
        # Check if client exists
        client_exists = test_client_id in clients_df['CLI'].values
        print(f"Client exists: {client_exists}")
        
        # Get products this client has
        client_products = products_df[
            (products_df['CLI'] == test_client_id) & 
            (products_df['ETA'] == 'VA')
        ]['CPRO'].unique()
        
        print(f"Client has {len(client_products)} products: {client_products}")
        
        # Get all available products
        all_products = products_df['CPRO'].unique()
        print(f"Total products available: {len(all_products)}")
        
        # Find products this client doesn't have
        available_products = [p for p in all_products if p not in client_products and str(p).strip()]
        print(f"Products available for recommendation: {len(available_products)}")
        print(f"Sample available products: {available_products[:10]}")
        
        # Test with multiple clients
        print("\nTesting with multiple clients:")
        for i in range(5):
            client_id = clients_df['CLI'].iloc[i]
            client_products = products_df[
                (products_df['CLI'] == client_id) & 
                (products_df['ETA'] == 'VA')
            ]['CPRO'].unique()
            
            available_products = [p for p in all_products if p not in client_products and str(p).strip()]
            print(f"  Client {client_id}: has {len(client_products)} products, {len(available_products)} available for recommendation")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_recommendation_logic()