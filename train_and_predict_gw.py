import os
import json
import psutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge, ElasticNet
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import sys

# Function to prepare the dataset by dropping missing values and resetting the index
def prepare_data(df, lag_columns, step_columns):
    df = df.dropna().reset_index(drop=True)
    return df

# Function to calculate the Symmetric Mean Absolute Percentage Error (SMAPE)
def smape(a, f):
    a, f = np.asarray(a), np.asarray(f)
    mask = np.isfinite(a) & np.isfinite(f) & ((np.abs(a) + np.abs(f)) != 0)
    if np.any(mask):
        return 100 * np.mean(2 * np.abs(f[mask] - a[mask]) / (np.abs(a[mask]) + np.abs(f[mask])))
    return np.nan

# Create a custom scorer based on SMAPE
smape_scorer = make_scorer(smape, greater_is_better=False)

# Function to load the best model parameters from a JSON file
def load_best_params(json_file, id_loc):
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            results = json.load(f)
        if id_loc in results:
            return results[id_loc]['model'], results[id_loc]['hyperparameters'], results[id_loc]['vars_set']
    return None, None, None

# Function to retrieve a model object by its name
def get_model_by_name(model_name):
    if model_name == 'ridge':
        return Ridge()
    if model_name == 'elasticnet':
        return ElasticNet()
    if model_name == 'lightgbm':
        return LGBMRegressor(verbosity=-1, n_jobs=-1)
    if model_name == 'randomforest':
        return RandomForestRegressor(n_jobs=-1)
    if model_name == 'catboost':
        return CatBoostRegressor(verbose=False, task_type='CPU', thread_count=-1)
    if model_name == 'xgboost':
        return XGBRegressor(tree_method="hist", verbosity=0, n_jobs=-1)
    else:
        return None

# Function to check memory usage and ensure it doesn't exceed the limit (adjust depending on hardware)
def check_memory_usage(limit_gb=11):
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_used_gb = memory_info.rss / (1024 ** 3)  # Convert memory usage to GB
    return memory_used_gb > limit_gb

# Function to train and evaluate a model using pre-loaded parameters with cross-validation
def train_and_evaluate_model_with_params(model_name, model, best_params, X_train, y_train, n_folds=7, memory_limit_gb=11):
    print(f"\nStarting cross-validation training for {model_name}...")
    cv = KFold(n_splits=n_folds, random_state=42, shuffle=True)
    
    if model_name in ['ridge', 'elasticnet', 'lightgbm', 'randomforest']:
        multi_model = MultiOutputRegressor(model.set_params(**best_params))
        
        if check_memory_usage(memory_limit_gb):
            print(f"\nMemory limit exceeded for {model_name}. Skipping iteration.")
            return None, np.inf
            
        multi_model.fit(X_train, y_train)
        print(f"\nTraining completed for {model_name}")
        return multi_model, -1
        
    else:
        model.set_params(**best_params)
        
        try:
            if check_memory_usage(memory_limit_gb):
                print(f"\nMemory limit exceeded for {model_name}. Skipping iteration.")
                return None, np.inf
            
            model.fit(X_train, y_train)
        except MemoryError:
            print(f"\nMemory limit exceeded for {model_name}. Skipping iteration.")
            return None, np.inf
        
        print(f"\nTraining completed for {model_name}")
        return model, -1  # Placeholder for SMAPE CV

# Function to train and predict for a specific id_loc
def train_and_predict_for_id_loc(training_data_df, lag_columns, step_columns, json_file, id_loc, n_folds=7, memory_limit_gb=11):
    print(f"\nPreparing data for id_loc {id_loc}...")
    training_data_df = prepare_data(training_data_df, lag_columns, step_columns)

    model_name, best_params, vars_set = load_best_params(json_file, id_loc)

    if model_name is None or best_params is None:
        print(f"\nNo hyperparameters found for {id_loc}.")
        return None, None

    feature_columns = vars_dict[vars_set]

    loc_data = training_data_df[training_data_df['id_loc'] == id_loc][feature_columns]

    X = loc_data.drop(step_columns + ['id_loc', 'Zone'], axis=1)
    y = loc_data[step_columns]

    print(f"\nTraining model {model_name} for id_loc {id_loc}...")
    model = get_model_by_name(model_name)
    best_model, smape_cv = train_and_evaluate_model_with_params(model_name, model, best_params, X, y, n_folds, memory_limit_gb)

    if best_model is None:
        print(f"\nMemory issues during training for {id_loc}.")
        return None, None

    print(f"\nGenerating predictions for id_loc {id_loc}...")
    X_test = test_data[test_data['id_loc'] == id_loc][X.columns]
    
    y_pred_test = best_model.predict(X_test)

    if y_pred_test.size == 0:
        print(f"\nNo predictions generated for id_loc {id_loc}. Something went wrong.")
    else:
        lag_values = X_test[lag_columns].values.flatten()
        step_values = y_pred_test.flatten()

        print(f'\nThe predictions for the next 26 steps (Jan 2022 - Jun 2024) are:')
        print(step_values)
        
    return best_model, y_pred_test

# Main function to execute the script
if __name__ == "__main__":
    # Receive id_loc from the command line
    if len(sys.argv) != 2:
        print("\nUsage: python train_predict.py <id_loc>")
        sys.exit(1)
    
    id_loc = sys.argv[1]

    print("\nLoading datasets...")
    training_data_df = pd.read_csv('auxiliary_data/complete_and_fully_processed_data.csv')
    training_data_df['id_loc'] = training_data_df['id_loc'].astype(str)
    test_data = pd.read_csv('auxiliary_data/taikai_test_data_with_exog.csv')  
    test_data['id_loc'] = test_data['id_loc'].astype(str)

    # Configure lag and step columns
    lag_columns = [f'lag_{i}' for i in range(50, 0, -1)]
    step_columns = [f'step_{i}' for i in range(1, 27)]
    
    # Define the JSON file containing the best models and hyperparameters
    json_file = 'auxiliary_data/best_models_combined.json'

    necessary_vars = training_data_df.columns[list(range(0, 77)) + [136]]
    hist_temp_vars = training_data_df.columns[list(range(77, 136))]
    geogr_climate_vars = training_data_df.columns[list(range(137, 282))]
    soil_demogr_vars = training_data_df.columns[list(range(282, 384))]
    
    # Dictionary to store variable sets (you need to define `vars_dict`)
    vars_dict = {
        'necessary_vars': necessary_vars,
        'necessary_vars_hist_temp_vars': pd.Index(necessary_vars.tolist() + hist_temp_vars.tolist()),
        'necessary_vars_hist_temp_vars_geogr_climate_vars': pd.Index(necessary_vars.tolist() + hist_temp_vars.tolist() + geogr_climate_vars.tolist()),
        'necessary_vars_hist_temp_vars_geogr_climate_vars_soil_demogr_vars': pd.Index(necessary_vars.tolist() + hist_temp_vars.tolist() + geogr_climate_vars.tolist() + soil_demogr_vars.tolist())
    }

    print(f"\nStarting the process for id_loc {id_loc}...")
    
    # Call the function to train and predict for the specific id_loc
    trained_model, predictions = train_and_predict_for_id_loc(training_data_df, lag_columns, step_columns, json_file, id_loc)

    print(f"\nProcess completed for id_loc {id_loc}.")