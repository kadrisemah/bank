# src/data_processing/data_processor.py

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BankingDataProcessor:
    """Main data processor for banking ML project"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.data_files = {
            'agences': 'agences.xlsx',
            'chapitres': 'chapitres.xlsx',
            'clients': 'Clients_DOU_replaced_DDMMYYYY.xlsx',
            'comptes': 'Comptes_DFE_replaced_DDMMYYYY.xlsx',
            'eerp': 'eerp_formatted_eer_sortie.xlsx',
            'gestionnaires': 'gestionnaires.xlsx',
            'gestionnaires_util': 'gestionnaires_utilisateurs.xlsx',
            'produits': 'Produits_DFSOU_replaced_DDMMYYYY.xlsx',
            'district': 'referenciel_district.xlsx',
            'packs': 'referenciel_packs.xlsx',
            'ref_produits': 'referenciel_produits.xlsx',
            'utilisateurs': 'utilisateurs.xlsx'
        }
        self.data = {}
        
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all Excel files into dataframes"""
        logger.info("Loading all data files...")
        
        for key, filename in self.data_files.items():
            file_path = os.path.join(self.data_path, filename)
            try:
                self.data[key] = pd.read_excel(file_path)
                logger.info(f"Loaded {key}: {self.data[key].shape}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {str(e)}")
                
        return self.data
    
    def clean_clients_data(self) -> pd.DataFrame:
        """Clean and process clients data"""
        df = self.data['clients'].copy()
        
        # Convert dates
        df['DNA'] = pd.to_datetime(df['DNA'], format='%d/%m/%Y', errors='coerce')
        df['DOU'] = pd.to_datetime(df['DOU'], format='%d/%m/%Y', errors='coerce')
        
        # Calculate age
        df['age'] = (datetime.now() - df['DNA']).dt.days / 365.25
        
        # Calculate client seniority
        df['client_seniority_days'] = (datetime.now() - df['DOU']).dt.days
        
        # Clean sex column
        df['SEXT'] = df['SEXT'].fillna('U')
        
        # Create age groups
        df['age_group'] = pd.cut(df['age'], 
                                 bins=[0, 25, 35, 45, 55, 65, 100],
                                 labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'])
        
        # Convert GES to string to ensure consistency
        df['GES'] = df['GES'].astype(str)
        
        return df
    
    def load_product_references(self) -> Dict[str, pd.DataFrame]:
        """Load product and pack reference data"""
        logger.info("Loading product reference data...")
        
        references = {}
        
        try:
            # Load product reference
            product_ref_path = os.path.join(self.data_path, 'referenciel_produits.xlsx')
            references['products'] = pd.read_excel(product_ref_path)
            logger.info(f"Loaded product reference: {references['products'].shape}")
            
            # Load pack reference
            pack_ref_path = os.path.join(self.data_path, 'referenciel_packs.xlsx')
            references['packs'] = pd.read_excel(pack_ref_path)
            logger.info(f"Loaded pack reference: {references['packs'].shape}")
            
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            # Create empty reference dataframes if files don't exist
            references['products'] = pd.DataFrame(columns=['CPRO', 'LIB', 'ATT', 'CGAM'])
            references['packs'] = pd.DataFrame(columns=['CPACK', 'LIB'])
            
        return references
    
    def clean_products_data(self) -> pd.DataFrame:
        """Clean and process products data with reference name mapping"""
        df = self.data['produits'].copy()
        
        # Convert dates
        df['DDSOU'] = pd.to_datetime(df['DDSOU'], format='%d/%m/%Y', errors='coerce')
        if 'DFSOU' in df.columns:
            df['DFSOU'] = pd.to_datetime(df['DFSOU'], format='%d/%m/%Y', errors='coerce')
            # Calculate product duration
            df['product_duration_days'] = (df['DFSOU'] - df['DDSOU']).dt.days
        else:
            df['product_duration_days'] = 365  # Default duration
        
        # Create product status
        df['is_active'] = df['ETA'] == 'VA'
        
        # Load reference data
        references = self.load_product_references()
        
        # Merge product names
        if not references['products'].empty:
            # Ensure CPRO is consistent type
            df['CPRO'] = pd.to_numeric(df['CPRO'], errors='coerce')
            references['products']['CPRO'] = pd.to_numeric(references['products']['CPRO'], errors='coerce')
            
            # Merge product information
            df = df.merge(
                references['products'][['CPRO', 'LIB', 'ATT', 'CGAM']],
                on='CPRO',
                how='left',
                suffixes=('', '_product')
            )
            
            # Rename columns for clarity
            df.rename(columns={
                'LIB': 'product_name',
                'ATT': 'product_attribute',
                'CGAM': 'product_category'
            }, inplace=True)
            
            logger.info(f"Merged product names: {df['product_name'].notna().sum()}/{len(df)} products")
        
        # Merge pack names
        if not references['packs'].empty:
            # Clean and convert CPACK
            df['CPACK'] = df['CPACK'].astype(str).str.strip()
            df['CPACK_numeric'] = pd.to_numeric(df['CPACK'], errors='coerce')
            
            references['packs']['CPACK'] = pd.to_numeric(references['packs']['CPACK'], errors='coerce')
            
            # Merge pack information
            df = df.merge(
                references['packs'][['CPACK', 'LIB']],
                left_on='CPACK_numeric',
                right_on='CPACK',
                how='left',
                suffixes=('', '_pack')
            )
            
            # Rename and clean up
            df.rename(columns={'LIB': 'pack_name'}, inplace=True)
            df.drop(columns=['CPACK_pack', 'CPACK_numeric'], inplace=True)
            
            logger.info(f"Merged pack names: {df['pack_name'].notna().sum()}/{len(df)} packs")
        
        # Fill missing product/pack names with codes
        if 'product_name' in df.columns:
            df['product_name'] = df['product_name'].fillna(f"Product_{df['CPRO']}")
        else:
            df['product_name'] = f"Product_{df['CPRO']}"
            
        if 'pack_name' in df.columns:
            df['pack_name'] = df['pack_name'].fillna(f"Pack_{df['CPACK']}")
        else:
            df['pack_name'] = f"Pack_{df['CPACK']}"
        
        logger.info("Products data cleaned with reference mapping")
        return df
    
    def clean_accounts_data(self) -> pd.DataFrame:
        """Clean and process accounts data"""
        df = self.data['comptes'].copy()
        
        # Convert dates
        if 'DOU' in df.columns:
            df['DOU'] = pd.to_datetime(df['DOU'], format='%d/%m/%Y', errors='coerce')
        
        # Create account status
        df['is_closed'] = df['CFE'] == 'O'
        
        return df
    
    def create_client_features(self) -> pd.DataFrame:
        """Create comprehensive client features for ML"""
        logger.info("Creating client features...")
        
        # Start with cleaned client data
        clients = self.clean_clients_data()
        products = self.clean_products_data()
        accounts = self.clean_accounts_data()
        eerp = self.data['eerp'].copy()
        
        # Aggregate product features per client
        product_features = products.groupby('CLI').agg({
            'CPRO': 'count',  # Total products
            'is_active': ['sum', 'mean'],  # Active products count and ratio
            'product_duration_days': ['mean', 'std', 'max'],  # Product duration stats
            'CPACK': 'nunique'  # Number of unique packs
        }).reset_index()
        
        product_features.columns = ['CLI', 'total_products', 'active_products', 
                                   'active_products_ratio', 'avg_product_duration',
                                   'std_product_duration', 'max_product_duration',
                                   'unique_packs']
        
        # Fill NaN values in std with 0
        product_features['std_product_duration'] = product_features['std_product_duration'].fillna(0)
        
        # Aggregate account features per client
        account_features = accounts.groupby('CLI').agg({
            'CHA': 'count',  # Total accounts
            'is_closed': ['sum', 'mean'],  # Closed accounts count and ratio
            'TYP': 'nunique'  # Number of unique account types
        }).reset_index()
        
        account_features.columns = ['CLI', 'total_accounts', 'closed_accounts',
                                   'closed_accounts_ratio', 'unique_account_types']
        
        # Merge all features
        client_features = clients.merge(product_features, on='CLI', how='left')
        client_features = client_features.merge(account_features, on='CLI', how='left')
        
        # Merge EERP data if columns exist
        eerp_cols = ['CLI']
        if 'Segment Client' in eerp.columns:
            eerp_cols.append('Segment Client')
        if 'Actif/Inactif' in eerp.columns:
            eerp_cols.append('Actif/Inactif')
        if 'District' in eerp.columns:
            eerp_cols.append('District')
            
        if len(eerp_cols) > 1:
            client_features = client_features.merge(eerp[eerp_cols], on='CLI', how='left')
        
        # Fill missing values
        numeric_cols = client_features.select_dtypes(include=[np.number]).columns
        client_features[numeric_cols] = client_features[numeric_cols].fillna(0)
        
        return client_features
    
    def create_manager_features(self) -> pd.DataFrame:
        """Create manager performance features"""
        logger.info("Creating manager features...")
        
        clients = self.clean_clients_data()  # Use cleaned data with consistent GES type
        products = self.clean_products_data()
        gestionnaires = self.data['gestionnaires'].copy()
        
        # Ensure GES is string in gestionnaires
        gestionnaires['GES'] = gestionnaires['GES'].astype(str)
        
        # Count clients per manager
        clients_per_manager = clients.groupby('GES').agg({
            'CLI': 'count',
            'AGE': 'nunique'
        }).reset_index()
        clients_per_manager.columns = ['GES', 'total_clients', 'agencies_covered']
        
        # Count products sold per manager
        manager_products = clients.merge(products[['CLI', 'is_active']], on='CLI')
        products_per_manager = manager_products.groupby('GES').agg({
            'is_active': ['count', 'sum']
        }).reset_index()
        products_per_manager.columns = ['GES', 'total_products_managed', 'active_products_managed']
        
        # Merge features
        manager_features = gestionnaires.merge(clients_per_manager, on='GES', how='left')
        manager_features = manager_features.merge(products_per_manager, on='GES', how='left')
        
        # Fill NaN values
        manager_features = manager_features.fillna(0)
        
        # Calculate ratios
        manager_features['products_per_client'] = np.where(
            manager_features['total_clients'] > 0,
            manager_features['total_products_managed'] / manager_features['total_clients'],
            0
        )
        manager_features['active_products_ratio'] = np.where(
            manager_features['total_products_managed'] > 0,
            manager_features['active_products_managed'] / manager_features['total_products_managed'],
            0
        )
        
        return manager_features
    
    def create_agency_features(self) -> pd.DataFrame:
        """Create agency performance features"""
        logger.info("Creating agency features...")
        
        agences = self.data['agences'].copy()
        clients = self.clean_clients_data()
        products = self.clean_products_data()
        district = self.data['district'].copy()
        
        # Ensure AGE is consistent type
        agences['AGE'] = agences['AGE'].astype(str)
        clients['AGE'] = clients['AGE'].astype(str)
        district['AGE'] = district['AGE'].astype(str)
        
        # Count clients per agency
        clients_per_agency = clients.groupby('AGE').agg({
            'CLI': 'count',
            'GES': 'nunique'
        }).reset_index()
        clients_per_agency.columns = ['AGE', 'total_clients', 'total_managers']
        
        # Count products per agency
        agency_products = clients.merge(products[['CLI', 'is_active']], on='CLI')
        products_per_agency = agency_products.groupby('AGE').agg({
            'is_active': ['count', 'sum']
        }).reset_index()
        products_per_agency.columns = ['AGE', 'total_products', 'active_products']
        
        # Merge features
        agency_features = agences.merge(clients_per_agency, on='AGE', how='left')
        agency_features = agency_features.merge(products_per_agency, on='AGE', how='left')
        agency_features = agency_features.merge(district, on='AGE', how='left')
        
        # Fill NaN values
        agency_features = agency_features.fillna(0)
        
        # Calculate ratios
        agency_features['products_per_client'] = np.where(
            agency_features['total_clients'] > 0,
            agency_features['total_products'] / agency_features['total_clients'],
            0
        )
        agency_features['active_products_ratio'] = np.where(
            agency_features['total_products'] > 0,
            agency_features['active_products'] / agency_features['total_products'],
            0
        )
        agency_features['clients_per_manager'] = np.where(
            agency_features['total_managers'] > 0,
            agency_features['total_clients'] / agency_features['total_managers'],
            0
        )
        
        # Filter out administrative units (agencies with 0 clients)
        agency_features = agency_features[agency_features['total_clients'] > 0]
        
        return agency_features
    
    def process_and_save_all(self, output_path: str):
        """Process all data and save cleaned datasets"""
        logger.info("Processing all data...")
        
        # Load data
        self.load_all_data()
        
        # Create features
        client_features = self.create_client_features()
        manager_features = self.create_manager_features()
        agency_features = self.create_agency_features()
        
        # Process and save enhanced products data
        products_cleaned = self.clean_products_data()
        
        # Save processed data
        os.makedirs(output_path, exist_ok=True)
        
        client_features.to_csv(os.path.join(output_path, 'client_features.csv'), index=False)
        manager_features.to_csv(os.path.join(output_path, 'manager_features.csv'), index=False)
        agency_features.to_csv(os.path.join(output_path, 'agency_features.csv'), index=False)
        products_cleaned.to_csv(os.path.join(output_path, 'products_cleaned.csv'), index=False)
        
        # Save reference mappings for API use
        references = self.load_product_references()
        if not references['products'].empty:
            references['products'].to_csv(os.path.join(output_path, 'product_reference.csv'), index=False)
        if not references['packs'].empty:
            references['packs'].to_csv(os.path.join(output_path, 'pack_reference.csv'), index=False)
        
        logger.info(f"Saved processed data to {output_path}")
        logger.info(f"Products file now includes: product_name, pack_name, product_category")
        
        return {
            'client_features': client_features,
            'manager_features': manager_features,
            'agency_features': agency_features,
            'products_cleaned': products_cleaned
        }

# Usage example
if __name__ == "__main__":
    processor = BankingDataProcessor('data/raw')
    processed_data = processor.process_and_save_all('data/processed')