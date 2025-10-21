import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split

# --- Logging Configuration (from image_eb747d.jpg) ---

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File handler
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger (avoid duplicate handlers if this module is reloaded)
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# --- Data Handling Functions (from image_eb12a2.jpg) ---

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        # Drop any trailing unnamed columns if they exist
        unnamed_cols = [c for c in df.columns if c.startswith('Unnamed')]
        if unnamed_cols:
            df.drop(columns=unnamed_cols, inplace=True)

        # Rename expected columns to standardized names
        rename_map = {}
        if 'v1' in df.columns:
            rename_map['v1'] = 'target'
        if 'v2' in df.columns:
            rename_map['v2'] = 'text'
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        # Ensure required columns exist
        if 'target' not in df.columns or 'text' not in df.columns:
            raise KeyError('Required columns "v1/v2" (target/text) not found after preprocessing')
        logger.debug('Data preprocessing completed')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        # Define the raw data path (a subdirectory for saving)
        raw_data_path = os.path.join(data_path, 'raw')
        
        # Ensure the directory exists
        os.makedirs(raw_data_path, exist_ok=True)
        
        # Save train and test data to CSV files
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        
        # Log successful save
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        # Configuration variables
        test_size = 0.2
        # Use local data file in the experiments folder by default
        data_url_path = os.path.join('experiments', 'spam.csv')
        data_path = os.path.join('.', 'data')

        # 1. Load data
        df = load_data(data_url_path)
        
        # 2. Preprocess data
        final_df = preprocess_data(df)
        
        # 3. Split data (using the imported train_test_split)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        
        # 4. Save data (ensure base data directory exists)
        os.makedirs(data_path, exist_ok=True)
        save_data(train_data, test_data, data_path)

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()