import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# --- Logging Configuration ---

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise



# --- PLACEHOLDER FUNCTION ---
# Note: The load_data function is required by main() but was defined in an earlier script.
# Defining a placeholder here for completeness.
def load_data(file_path: str) -> pd.DataFrame:
    """Placeholder for the function to load data from a CSV file."""
    # This should include logic for reading CSV and handling NaNs/errors.
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise
        
# --- Model Training Function ---

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train the RandomForest model.

    :param X_train: Training features
    :param y_train: Training labels
    :param params: Dictionary of hyperparameters
    :return: Trained RandomForestClassifier
    """

    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samples in X_train and y_train must be the same.")

        logger.debug("Initializing RandomForest model with parameters: %s", params)
        
        # Initialize the classifier using parameters from the dictionary
        clf = RandomForestClassifier(
            n_estimators=params['n_estimators'],
            random_state=params['random_state']
        )

        logger.debug("Model training started with %d samples", X_train.shape[0])
        
        # Train the model
        clf.fit(X_train, y_train)
        
        logger.debug("Model training completed")

        return clf

    except ValueError as e:
        logger.error("ValueError during model training: %s", e)
        raise
    except Exception as e:
        logger.error("Error during model training: %s", e)
        raise
    

# --- Model Saving Function (Indentation Corrected) ---

def save_model(model, file_path: str) -> None:
    """Save the model to a file using pickle."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the model object using pickle
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
            
        logger.debug('Model saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('File path not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error occurred while saving the model: %s', e)
        raise

# --- Main Execution Function ---

def main():
    try:
        # Define model hyperparameters
        params = load_params('params.yaml')['model_training']
        # Load the TF-IDF processed training data
        train_data = load_data('./data/processed/train_tfidf.csv')
        
        # Separate features (X) and target (y)
        # X_train is all columns except the last one (features)
        X_train = train_data.iloc[:, :-1].values
        # y_train is the last column (target/label)
        y_train = train_data.iloc[:, -1].values.astype(int) # Ensure target is integer for classifier
        
        # Train the model
        clf = train_model(X_train, y_train, params)
        
        # Define the path to save the model
        model_save_path = 'models/model.pkl'
        
        # Save the trained model
        save_model(clf, model_save_path)
        
    except Exception as e:
        logger.error("Failed to complete the model building process: %s", e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()