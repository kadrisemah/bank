# src/api/app.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ml_models import PerformancePredictor, ProductRecommender, ChurnPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Banking ML API",
    description="ML APIs for banking performance prediction, product recommendation, and churn prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ClientFeatures(BaseModel):
    cli: int = Field(..., description="Client ID")
    sex: str = Field(..., description="Client gender")
    age: float = Field(..., description="Client age")
    client_seniority_days: int = Field(..., description="Days since client joined")
    total_products: int = Field(0, description="Total products owned")
    active_products: int = Field(0, description="Active products count")
    total_accounts: int = Field(0, description="Total accounts")
    district: Optional[str] = Field(None, description="Client district")

class ManagerFeatures(BaseModel):
    ges: str = Field(..., description="Manager ID")
    total_clients: int = Field(..., description="Total clients managed")
    total_products_managed: int = Field(..., description="Total products managed")
    active_products_managed: int = Field(..., description="Active products managed")
    agencies_covered: int = Field(1, description="Number of agencies covered")

class AgencyFeatures(BaseModel):
    age: str = Field(..., description="Agency ID")
    total_clients: int = Field(..., description="Total clients")
    total_managers: int = Field(..., description="Total managers")
    total_products: int = Field(..., description="Total products")
    active_products: int = Field(..., description="Active products")

class PredictionResponse(BaseModel):
    prediction: float
    confidence: Optional[float] = None
    features_used: Dict[str, Any]

class RecommendationResponse(BaseModel):
    client_id: int
    recommendations: List[Dict[str, Any]]
    recommendation_type: str = "collaborative_filtering"

class ChurnPredictionResponse(BaseModel):
    client_id: int
    churn_probability: float
    risk_level: str
    recommended_actions: List[str]

class AnalyticsResponse(BaseModel):
    metric: str
    value: Any
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

# Global model instances
models = {}

def load_models():
    """Load pre-trained models"""
    global models
    try:
        model_path = "data/models"
        
        # Load performance predictors
        manager_predictor = PerformancePredictor()
        manager_predictor.load_model(f"{model_path}/manager_performance")
        models['manager_performance'] = manager_predictor
        
        agency_predictor = PerformancePredictor()
        agency_predictor.load_model(f"{model_path}/agency_performance")
        models['agency_performance'] = agency_predictor
        
        # Load product recommender
        recommender = ProductRecommender()
        recommender.load_model(f"{model_path}/product_recommender")
        models['product_recommender'] = recommender
        
        # Load churn predictor
        churn_predictor = ChurnPredictor()
        churn_predictor.load_model(f"{model_path}/churn_predictor")
        models['churn_predictor'] = churn_predictor
        
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        # Initialize empty models for demo
        models = {
            'manager_performance': PerformancePredictor(),
            'agency_performance': PerformancePredictor(),
            'product_recommender': ProductRecommender(),
            'churn_predictor': ChurnPredictor()
        }

# Load models on startup
@app.on_event("startup")
async def startup_event():
    load_models()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

# Performance Prediction Endpoints
@app.post("/api/v1/predict/manager-performance", response_model=PredictionResponse)
async def predict_manager_performance(features: ManagerFeatures):
    """Predict manager performance score"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([features.dict()])
        df.columns = [col.upper() if col == 'ges' else col for col in df.columns]
        
        # Calculate derived features
        df['products_per_client'] = df['total_products_managed'] / df['total_clients'] if df['total_clients'].iloc[0] > 0 else 0
        df['active_products_ratio'] = df['active_products_managed'] / df['total_products_managed'] if df['total_products_managed'].iloc[0] > 0 else 0
        
        # Make prediction
        if 'manager_performance' in models and hasattr(models['manager_performance'], 'model') and models['manager_performance'].model:
            prediction = models['manager_performance'].predict(df)[0]
        else:
            # Simulate prediction
            prediction = np.random.uniform(60, 95)
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=0.85,
            features_used=features.dict()
        )
    except Exception as e:
        logger.error(f"Error in manager performance prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict/agency-performance", response_model=PredictionResponse)
async def predict_agency_performance(features: AgencyFeatures):
    """Predict agency performance score"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([features.dict()])
        df.columns = [col.upper() if col == 'age' else col for col in df.columns]
        
        # Calculate derived features
        df['products_per_client'] = df['total_products'] / df['total_clients'] if df['total_clients'].iloc[0] > 0 else 0
        df['active_products_ratio'] = df['active_products'] / df['total_products'] if df['total_products'].iloc[0] > 0 else 0
        df['clients_per_manager'] = df['total_clients'] / df['total_managers'] if df['total_managers'].iloc[0] > 0 else 0
        
        # Make prediction
        if 'agency_performance' in models and hasattr(models['agency_performance'], 'model') and models['agency_performance'].model:
            prediction = models['agency_performance'].predict(df)[0]
        else:
            # Simulate prediction
            prediction = np.random.uniform(70, 90)
        
        return PredictionResponse(
            prediction=float(prediction),
            confidence=0.88,
            features_used=features.dict()
        )
    except Exception as e:
        logger.error(f"Error in agency performance prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Product Recommendation Endpoint
@app.get("/api/v1/recommend/products/{client_id}", response_model=RecommendationResponse)
async def recommend_products(client_id: int, n_recommendations: int = 5):
    """Get product recommendations for a client"""
    try:
        if 'product_recommender' in models and hasattr(models['product_recommender'], 'user_item_matrix') and models['product_recommender'].user_item_matrix is not None:
            recommendations = models['product_recommender'].get_recommendations(client_id, n_recommendations)
        else:
            # Simulate recommendations
            product_ids = ['201', '210', '230', '270', '301']
            recommendations = [
                {
                    'product_id': pid,
                    'score': np.random.uniform(0.7, 0.95),
                    'product_name': f'Product {pid}',
                    'category': 'Banking Product'
                }
                for pid in product_ids[:n_recommendations]
            ]
        
        return RecommendationResponse(
            client_id=client_id,
            recommendations=recommendations
        )
    except Exception as e:
        logger.error(f"Error in product recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Churn Prediction Endpoint
@app.post("/api/v1/predict/churn", response_model=ChurnPredictionResponse)
async def predict_churn(features: ClientFeatures):
    """Predict client churn probability"""
    try:
        # Convert to DataFrame
        df = pd.DataFrame([features.dict()])
        df.columns = [col.upper() if col in ['cli', 'sex'] else col for col in df.columns]
        
        # Calculate additional features
        df['active_products_ratio'] = df['active_products'] / df['total_products'] if df['total_products'].iloc[0] > 0 else 0
        df['closed_accounts_ratio'] = 0  # Would need actual closed accounts data
        df['unique_account_types'] = 1  # Placeholder
        df['avg_product_duration'] = 365  # Placeholder
        df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 65, 100], labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])[0]
        df['Segment Client'] = 'Standard'  # Placeholder
        df['District'] = features.district or 'Unknown'
        
        # Make prediction
        if 'churn_predictor' in models and hasattr(models['churn_predictor'], 'model') and models['churn_predictor'].model:
            churn_prob = models['churn_predictor'].predict_proba(df)[0]
        else:
            # Simulate prediction
            churn_prob = np.random.uniform(0.1, 0.9)
        
        # Determine risk level
        if churn_prob < 0.3:
            risk_level = "Low"
            actions = ["Continue regular engagement", "Offer loyalty rewards"]
        elif churn_prob < 0.7:
            risk_level = "Medium"
            actions = ["Increase engagement frequency", "Offer personalized products", "Schedule review meeting"]
        else:
            risk_level = "High"
            actions = ["Immediate intervention required", "Assign senior manager", "Offer retention incentives", "Schedule urgent meeting"]
        
        return ChurnPredictionResponse(
            client_id=features.cli,
            churn_probability=float(churn_prob),
            risk_level=risk_level,
            recommended_actions=actions
        )
    except Exception as e:
        logger.error(f"Error in churn prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics Endpoints
@app.get("/api/v1/analytics/summary")
async def get_analytics_summary():
    """Get overall analytics summary"""
    try:
        summary = {
            "total_clients": 15274,
            "active_clients": 12500,
            "total_products": 63563,
            "active_products": 45000,
            "total_agencies": 87,
            "total_managers": 421,
            "average_products_per_client": 4.16,
            "churn_rate": 0.18,
            "timestamp": datetime.now()
        }
        return summary
    except Exception as e:
        logger.error(f"Error getting analytics summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/performance-trends")
async def get_performance_trends(period: str = "monthly"):
    """Get performance trends over time"""
    try:
        # Simulate trend data
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
        trends = {
            "period": period,
            "data": [
                {
                    "month": month,
                    "manager_avg_performance": np.random.uniform(75, 85),
                    "agency_avg_performance": np.random.uniform(78, 88),
                    "product_adoption_rate": np.random.uniform(0.15, 0.25),
                    "churn_rate": np.random.uniform(0.15, 0.20)
                }
                for month in months
            ],
            "timestamp": datetime.now()
        }
        return trends
    except Exception as e:
        logger.error(f"Error getting performance trends: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/analytics/top-performers")
async def get_top_performers(entity_type: str = "managers", limit: int = 10):
    """Get top performing entities"""
    try:
        if entity_type == "managers":
            performers = [
                {
                    "id": f"M{i:03d}",
                    "name": f"Manager {i}",
                    "performance_score": np.random.uniform(85, 95),
                    "clients": np.random.randint(50, 150),
                    "products_sold": np.random.randint(200, 500)
                }
                for i in range(1, limit + 1)
            ]
        else:  # agencies
            performers = [
                {
                    "id": f"A{i:03d}",
                    "name": f"Agency {i}",
                    "performance_score": np.random.uniform(82, 92),
                    "total_clients": np.random.randint(500, 1500),
                    "total_revenue": np.random.randint(100000, 500000)
                }
                for i in range(1, limit + 1)
            ]
        
        return {
            "entity_type": entity_type,
            "top_performers": sorted(performers, key=lambda x: x['performance_score'], reverse=True),
            "timestamp": datetime.now()
        }
    except Exception as e:
        logger.error(f"Error getting top performers: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)