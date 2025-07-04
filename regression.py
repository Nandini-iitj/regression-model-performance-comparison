# regression.py
import pandas as pd
import numpy as np
from utils import (
    load_data, 
    preprocess_data, 
    train_regression_models, 
    hyperparameter_tuning,
    plot_results,
    save_results_to_csv,
    explore_data
)
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description='Housing Price Prediction')
    parser.add_argument('--mode', choices=['basic', 'tuned'], default='basic',
                       help='Mode: basic regression or hyperparameter tuning')
    parser.add_argument('--explore', action='store_true',
                       help='Perform data exploration')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HOUSING PRICE PREDICTION - MLOPS ASSIGNMENT")
    print("=" * 60)
    
    # Load data
    print("Loading Boston Housing dataset...")
    df = load_data()
    
    if df is None:
        print("Error: Could not load dataset")
        sys.exit(1)
    
    print(f"Dataset loaded successfully: {df.shape}")
    
    # Data exploration
    if args.explore:
        print("\nPerforming data exploration...")
        explore_data(df)
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    if args.mode == 'basic':
        print("\n" + "=" * 60)
        print("BASIC REGRESSION MODELS")
        print("=" * 60)
        
        # Train basic models
        models, results = train_regression_models(X_train, X_test, y_train, y_test)
        
        # Plot results
        plot_results(results, "Basic Regression Models Performance")
        
        # Save results
        save_results_to_csv(results, "basic_regression_results.csv")
        
    elif args.mode == 'tuned':
        print("\n" + "=" * 60)
        print("HYPERPARAMETER TUNING")
        print("=" * 60)
        
        # Hyperparameter tuning
        best_models, results = hyperparameter_tuning(X_train, X_test, y_train, y_test)
        
        # Plot results
        plot_results(results, "Hyperparameter Tuned Models Performance")
        
        # Save results
        save_results_to_csv(results, "hyperparameter_tuning_results.csv")
    
    print("\nAnalysis completed successfully!")
    print("Check the generated files for detailed results.")

if __name__ == "__main__":
    main()
