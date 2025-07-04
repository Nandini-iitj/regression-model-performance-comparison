# utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """
    Load Boston Housing dataset from CMU repository
    Returns: pandas DataFrame with features and target
    """
    try:
        data_url = "http://lib.stat.cmu.edu/datasets/boston"
        raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
        
        # Split data and target
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        
        # Feature names from original dataset
        feature_names = [
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
        ]
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=feature_names)
        df['MEDV'] = target  # Target variable
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the dataset
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    # Separate features and target
    X = df.drop('MEDV', axis=1)
    y = df['MEDV']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance
    Returns: dictionary with metrics
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        'Model': model_name,
        'MSE': mse,
        'RMSE': np.sqrt(mse),
        'R²': r2
    }
    
    return results

def train_regression_models(X_train, X_test, y_train, y_test):
    """
    Train multiple regression models
    Returns: dictionary with trained models and results
    """
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42)
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate model
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)
        trained_models[name] = model
        
        print(f"{name}:")
        print(f"  MSE: {result['MSE']:.4f}")
        print(f"  RMSE: {result['RMSE']:.4f}")
        print(f"  R²: {result['R²']:.4f}")
        print("-" * 30)
    
    return trained_models, results

def hyperparameter_tuning(X_train, X_test, y_train, y_test):
    """
    Perform hyperparameter tuning for regression models
    Returns: dictionary with best models and results
    """
    # Define hyperparameter grids
    param_grids = {
        'Ridge Regression': {
            'alpha': [0.1, 1.0, 10.0],
            'solver': ['auto', 'svd', 'cholesky'],
            'max_iter': [1000, 2000, 3000]
        },
        'Random Forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 5, 10]
        },
        'SVR': {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto'],
            'kernel': ['rbf', 'linear', 'poly']
        }
    }
    
    models = {
        'Ridge Regression': Ridge(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'SVR': SVR()
    }
    
    best_models = {}
    results = []
    
    for name, model in models.items():
        print(f"Tuning {name}...")
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, 
            param_grids[name], 
            cv=5, 
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model
        best_model = grid_search.best_estimator_
        best_models[name] = best_model
        
        # Evaluate best model
        result = evaluate_model(best_model, X_test, y_test, name)
        result['Best_Params'] = grid_search.best_params_
        results.append(result)
        
        print(f"  Best parameters: {grid_search.best_params_}")
        print(f"  MSE: {result['MSE']:.4f}")
        print(f"  R²: {result['R²']:.4f}")
        print("-" * 50)
    
    return best_models, results

def plot_results(results, title="Model Performance Comparison"):
    """
    Plot model performance comparison
    """
    df_results = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MSE comparison
    ax1.bar(df_results['Model'], df_results['MSE'])
    ax1.set_title('Mean Squared Error Comparison')
    ax1.set_ylabel('MSE')
    ax1.tick_params(axis='x', rotation=45)
    
    # R² comparison
    ax2.bar(df_results['Model'], df_results['R²'])
    ax2.set_title('R² Score Comparison')
    ax2.set_ylabel('R²')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results_to_csv(results, filename):
    """
    Save results to CSV file
    """
    df_results = pd.DataFrame(results)
    df_results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def explore_data(df):
    """
    Perform basic data exploration
    """
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    print("\nDataset Description:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Correlation matrix
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
