#!/usr/bin/env python3

import os
import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.data_processor import BankingDataProcessor
from models.ml_models import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'data/models',
        'logs'
    ]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    logger.info("Directories created successfully")

def process_data():
    """Process raw data files"""
    logger.info("Starting data processing...")
    processor = BankingDataProcessor('data/raw')
    processed_data = processor.process_and_save_all('data/processed')
    logger.info("Data processing completed!")
    return processed_data

def train_models():
    """Train all ML models"""
    logger.info("Starting model training...")
    trainer = ModelTrainer('data/processed', 'data/models')
    models = trainer.train_all_models()
    logger.info("Model training completed!")
    return models

def main():
    parser = argparse.ArgumentParser(description='Banking ML Project')
    parser.add_argument('--setup', action='store_true', help='Setup project directories')
    parser.add_argument('--process', action='store_true', help='Process raw data')
    parser.add_argument('--train', action='store_true', help='Train ML models')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    
    args = parser.parse_args()
    
    if args.setup or args.all:
        setup_directories()
    
    if args.process or args.all:
        process_data()
    
    if args.train or args.all:
        train_models()
    
    if not any([args.setup, args.process, args.train, args.all]):
        parser.print_help()

if __name__ == "__main__":
    main()