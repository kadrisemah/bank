# src/models/ml_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import xgboost as xgb
import lightgbm as lgb
import joblib
import logging
from typing import Dict, List, Tuple, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformancePredictor:
    """Predict manager and agency performance"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for training"""
        # Select numerical features
        feature_cols = [
            'total_clients', 'agencies_covered', 'total_products_managed',
            'active_products_managed', 'products_per_client', 'active_products_ratio'
        ]
        
        # For agencies, use different features
        if 'total_managers' in df.columns:
            feature_cols = [
                'total_clients', 'total_managers', 'total_products',
                'active_products', 'products_per_client', 'active_products_ratio',
                'clients_per_manager'
            ]
        
        self.feature_columns = feature_cols
        return df[feature_cols].fillna(0)
    
    def train(self, df: pd.DataFrame, target_col: str, model_type: str = 'xgboost'):
        """Train performance prediction model"""
        logger.info(f"Training performance predictor with {model_type}...")
        
        # Prepare features
        X = self.prepare_features(df)
        y = df[target_col] if target_col in df.columns else np.random.rand(len(df)) * 100  # Simulated target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        logger.info(f"Model trained. RMSE: {rmse:.2f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info(f"Top features:\n{feature_importance.head()}")
        
        return {'rmse': rmse, 'model': self.model}
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def save_model(self, path: str):
        """Save model and scaler"""
        joblib.dump(self.model, f"{path}_model.pkl")
        joblib.dump(self.scaler, f"{path}_scaler.pkl")
        joblib.dump(self.feature_columns, f"{path}_features.pkl")
    
    def load_model(self, path: str):
        """Load model and scaler"""
        self.model = joblib.load(f"{path}_model.pkl")
        self.scaler = joblib.load(f"{path}_scaler.pkl")
        self.feature_columns = joblib.load(f"{path}_features.pkl")


class ProductRecommender:
    """Recommend products to clients using collaborative filtering"""
    
    def __init__(self):
        self.user_item_matrix = None
        self.product_features = None
        self.product_names = None  # Store product names mapping
        self.svd_model = None
        self.knn_model = None
        
    def create_user_item_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item interaction matrix"""
        # Store product names mapping if available
        if 'product_name' in df.columns:
            self.product_names = df.groupby('CPRO')['product_name'].first().to_dict()
            logger.info(f"Stored product names for {len(self.product_names)} products")
        
        # Assuming df has CLI (client), CPRO (product), and is_active columns
        interactions = df.groupby(['CLI', 'CPRO'])['is_active'].max().reset_index()
        
        # Pivot to create matrix
        user_item_matrix = interactions.pivot(
            index='CLI', 
            columns='CPRO', 
            values='is_active'
        ).fillna(0)
        
        self.user_item_matrix = user_item_matrix
        return user_item_matrix
    
    def train_svd(self, n_components: int = 50):
        """Train SVD for collaborative filtering"""
        logger.info("Training SVD recommender...")
        
        # Apply SVD
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        user_features = self.svd_model.fit_transform(self.user_item_matrix)
        
        # Train KNN for finding similar users
        self.knn_model = NearestNeighbors(n_neighbors=10, metric='cosine')
        self.knn_model.fit(user_features)
        
        logger.info(f"SVD trained with {n_components} components")
        
        return user_features
    
    def get_recommendations(self, client_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Get product recommendations for a client"""
        logger.info(f"Getting recommendations for client {client_id}")
        logger.info(f"Matrix has {len(self.user_item_matrix.index)} clients")
        
        if client_id not in self.user_item_matrix.index:
            logger.info(f"Client {client_id} not in matrix - using cold start")
            # Cold start - recommend popular products
            popular_products = self.user_item_matrix.sum().sort_values(ascending=False).head(n_recommendations)
            logger.info(f"Popular products: {popular_products}")
            result = [{'product_id': prod, 'score': float(score)} for prod, score in popular_products.items()]
            logger.info(f"Cold start recommendations: {result}")
            return result
        
        logger.info(f"Client {client_id} found in matrix - using collaborative filtering")
        
        # Get client features
        client_idx = self.user_item_matrix.index.get_loc(client_id)
        client_vector = self.svd_model.transform(self.user_item_matrix.iloc[client_idx:client_idx+1])
        
        # Find similar users
        distances, indices = self.knn_model.kneighbors(client_vector)
        similar_users = self.user_item_matrix.index[indices[0][1:]]  # Exclude self
        logger.info(f"Found {len(similar_users)} similar users")
        
        # Get products used by similar users but not by this client
        client_products = set(self.user_item_matrix.columns[self.user_item_matrix.loc[client_id] > 0])
        logger.info(f"Client has {len(client_products)} existing products")
        
        recommendations = {}
        for similar_user in similar_users:
            similar_user_products = set(
                self.user_item_matrix.columns[self.user_item_matrix.loc[similar_user] > 0]
            )
            new_products = similar_user_products - client_products
            
            for product in new_products:
                if product not in recommendations:
                    recommendations[product] = 0
                recommendations[product] += 1
        
        logger.info(f"Found {len(recommendations)} candidate products from similar users")
        
        # If no recommendations from similar users, fall back to popular products
        if not recommendations:
            logger.info("No products from similar users, using popular products fallback")
            # Get all products this client doesn't have
            all_products = set(self.user_item_matrix.columns)
            available_products = all_products - client_products
            
            if available_products:
                # Get popularity scores for available products
                product_popularity = self.user_item_matrix.sum()
                available_popularity = {prod: product_popularity[prod] for prod in available_products if prod in product_popularity.index}
                
                # Sort by popularity
                sorted_by_popularity = sorted(
                    available_popularity.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:n_recommendations]
                
                result = [{'product_id': prod, 'score': float(score)} for prod, score in sorted_by_popularity]
                logger.info(f"Popular product recommendations: {result}")
                return result
            else:
                logger.warning(f"Client {client_id} already has all available products!")
                return []
        
        # Sort by frequency from similar users
        sorted_recommendations = sorted(
            recommendations.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:n_recommendations]
        
        result = [{'product_id': prod, 'score': float(score)} for prod, score in sorted_recommendations]
        logger.info(f"Collaborative filtering recommendations: {result}")
        return result
    
    def save_model(self, path: str):
        """Save recommender models"""
        joblib.dump(self.svd_model, f"{path}_svd.pkl")
        joblib.dump(self.knn_model, f"{path}_knn.pkl")
        joblib.dump(self.user_item_matrix, f"{path}_matrix.pkl")
        joblib.dump(self.product_names, f"{path}_names.pkl")  # Save product names
    
    def load_model(self, path: str):
        """Load recommender models"""
        self.svd_model = joblib.load(f"{path}_svd.pkl")
        self.knn_model = joblib.load(f"{path}_knn.pkl")
        self.user_item_matrix = joblib.load(f"{path}_matrix.pkl")
        try:
            self.product_names = joblib.load(f"{path}_names.pkl")  # Load product names
            logger.info(f"Loaded product names for {len(self.product_names)} products")
        except:
            logger.warning("Product names not found in saved model")
            self.product_names = None


class ChurnPredictor:
    """Predict client churn probability"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = None
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Prepare features for churn prediction"""
        features = df.copy()
        
        # Numerical features
        numerical_features = [
            'age', 'client_seniority_days', 'total_products', 'active_products',
            'active_products_ratio', 'avg_product_duration', 'total_accounts',
            'closed_accounts_ratio', 'unique_account_types'
        ]
        
        # Categorical features
        categorical_features = ['SEXT', 'age_group', 'Segment Client', 'District']
        
        # Encode categorical variables
        for col in categorical_features:
            if col in features.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        features[col].fillna('Unknown')
                    )
                else:
                    features[f'{col}_encoded'] = self.label_encoders[col].transform(
                        features[col].fillna('Unknown')
                    )
        
        # Select final features
        encoded_categorical = [f'{col}_encoded' for col in categorical_features if col in features.columns]
        self.feature_columns = numerical_features + encoded_categorical
        
        # Filter existing columns
        self.feature_columns = [col for col in self.feature_columns if col in features.columns]
        
        return features[self.feature_columns].fillna(0), self.label_encoders
    
    def create_churn_label(self, df: pd.DataFrame) -> pd.Series:
        """Create churn label based on activity"""
        # Define churn as: no active products and client seniority > 180 days
        churn_conditions = (
            (df['active_products'] == 0) & 
            (df['client_seniority_days'] > 180)
        )
        return churn_conditions.astype(int)
    
    def train(self, df: pd.DataFrame, model_type: str = 'xgboost'):
        """Train churn prediction model"""
        logger.info(f"Training churn predictor with {model_type}...")
        
        # Prepare features
        X, _ = self.prepare_features(df)
        y = self.create_churn_label(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
            )
        elif model_type == 'lightgbm':
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                class_weight='balanced'
            )
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                class_weight='balanced'
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        auc_score = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"Model trained. AUC Score: {auc_score:.3f}")
        logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            logger.info(f"Top features:\n{feature_importance.head()}")
        
        return {'auc': auc_score, 'model': self.model}
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict churn probability"""
        X, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def save_model(self, path: str):
        """Save model, scaler, and encoders"""
        joblib.dump(self.model, f"{path}_model.pkl")
        joblib.dump(self.scaler, f"{path}_scaler.pkl")
        joblib.dump(self.label_encoders, f"{path}_encoders.pkl")
        joblib.dump(self.feature_columns, f"{path}_features.pkl")
    
    def load_model(self, path: str):
        """Load model, scaler, and encoders"""
        self.model = joblib.load(f"{path}_model.pkl")
        self.scaler = joblib.load(f"{path}_scaler.pkl")
        self.label_encoders = joblib.load(f"{path}_encoders.pkl")
        self.feature_columns = joblib.load(f"{path}_features.pkl")


class ModelTrainer:
    """Main class to train all models"""
    
    def __init__(self, data_path: str, model_path: str):
        self.data_path = data_path
        self.model_path = model_path
        self.performance_predictor = PerformancePredictor()
        self.product_recommender = ProductRecommender()
        self.churn_predictor = ChurnPredictor()
        
    def train_all_models(self):
        """Train all ML models"""
        logger.info("Starting model training...")
        
        # Load processed data
        client_features = pd.read_csv(f"{self.data_path}/client_features.csv")
        manager_features = pd.read_csv(f"{self.data_path}/manager_features.csv")
        agency_features = pd.read_csv(f"{self.data_path}/agency_features.csv")
        
        # Train performance predictors
        logger.info("Training manager performance predictor...")
        self.performance_predictor.train(manager_features, 'performance_score', 'xgboost')
        self.performance_predictor.save_model(f"{self.model_path}/manager_performance")
        
        logger.info("Training agency performance predictor...")
        agency_predictor = PerformancePredictor()
        agency_predictor.train(agency_features, 'performance_score', 'lightgbm')
        agency_predictor.save_model(f"{self.model_path}/agency_performance")
        
        # Train product recommender
        logger.info("Training product recommender...")
        # Use enhanced product data with names
        try:
            # Try to load enhanced product data first
            products_df = pd.read_csv(f"{self.data_path}/products_cleaned.csv")
            logger.info(f"Using enhanced product data: {products_df.shape}")
            logger.info(f"Enhanced data includes: {products_df.columns.tolist()}")
        except:
            # Fallback to raw data if enhanced data not available
            logger.warning("Enhanced product data not found, using raw data")
            products_df = pd.read_excel(f"{self.data_path}/../raw/Produits_DFSOU_replaced_DDMMYYYY.xlsx")
            products_df['is_active'] = (products_df['ETA'] == 'VA').astype(int)
        
        self.product_recommender.create_user_item_matrix(products_df)
        self.product_recommender.train_svd(n_components=30)
        self.product_recommender.save_model(f"{self.model_path}/product_recommender")
        
        # Train churn predictor
        logger.info("Training churn predictor...")
        self.churn_predictor.train(client_features, 'xgboost')
        self.churn_predictor.save_model(f"{self.model_path}/churn_predictor")
        
        logger.info("All models trained successfully!")
        
        return {
            'manager_performance': self.performance_predictor,
            'agency_performance': agency_predictor,
            'product_recommender': self.product_recommender,
            'churn_predictor': self.churn_predictor
        }


# Usage example
if __name__ == "__main__":
    trainer = ModelTrainer('data/processed', 'data/models')
    models = trainer.train_all_models()