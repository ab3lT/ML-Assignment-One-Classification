import os
import zipfile
import pandas as pd
import io

def extract_zip(zip_file_path: str, extract_to: str) -> None:
     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
         zip_ref.extractall(extract_to)

def load_data(outer_zip_path: str, filename: str) -> pd.DataFrame:
    try:
        extract_to = "../data/"
        os.makedirs(extract_to, exist_ok=True)
        
        extract_zip(outer_zip_path, extract_to)
        
        file_path = os.path.join(extract_to, filename)
    
        # Load the .txt file as pipe-separated (|)
        df = pd.read_csv(file_path, delimiter='|', low_memory=False)
        return df
    
    except Exception as e:
        raise RuntimeError(f'Error loading data: {str(e)}')