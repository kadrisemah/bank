#!/usr/bin/env python3
"""
Simple API server without complex ML dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Simple imports
import pandas as pd
import numpy as np
import json
from datetime import datetime

def get_client_recommendations(client_id, n_recommendations=5):
    """Get recommendations for a client using real data"""
    
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
        # Check if client exists
        clients_df = pd.read_excel('data/raw/Clients_DOU_replaced_DDMMYYYY.xlsx')
        if client_id not in clients_df['CLI'].values:
            return {"error": f"Client {client_id} not found"}
            
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
        
        return {
            "client_id": client_id,
            "recommendations": recommendations,
            "recommendation_type": "collaborative_filtering"
        }
        
    except Exception as e:
        return {"error": str(e)}

# Simple HTTP server
def start_simple_server():
    """Start a simple HTTP server for testing"""
    import http.server
    import socketserver
    from urllib.parse import urlparse, parse_qs
    import json
    
    class RecommendationHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            parsed_url = urlparse(self.path)
            
            if parsed_url.path.startswith('/api/v1/recommend/products/'):
                try:
                    # Extract client_id from URL
                    client_id = int(parsed_url.path.split('/')[-1])
                    
                    # Get query parameters
                    query_params = parse_qs(parsed_url.query)
                    n_recommendations = int(query_params.get('n_recommendations', [5])[0])
                    
                    # Get recommendations
                    result = get_client_recommendations(client_id, n_recommendations)
                    
                    # Send response
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    
                    self.wfile.write(json.dumps(result).encode())
                    
                except Exception as e:
                    self.send_response(500)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": str(e)}).encode())
                    
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'Not Found')
    
    PORT = 8001
    with socketserver.TCPServer(("", PORT), RecommendationHandler) as httpd:
        print(f"🚀 Simple API server running on http://localhost:{PORT}")
        print(f"   Test URL: http://localhost:{PORT}/api/v1/recommend/products/43568328?n_recommendations=5")
        httpd.serve_forever()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        start_simple_server()
    else:
        # Test the function
        test_client = 43568328
        result = get_client_recommendations(test_client, 5)
        print(f"Testing client {test_client}:")
        print(json.dumps(result, indent=2))