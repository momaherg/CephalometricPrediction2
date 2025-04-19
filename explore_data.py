import pickle
import pandas as pd
import numpy as np

# Load the dataset
with open('data/train_data_pure_depth.pkl', 'rb') as f:
    data = pickle.load(f)

# Check if it's a pandas DataFrame
if isinstance(data, pd.DataFrame):
    print("Dataset is a pandas DataFrame")
    print(f"Shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    
    # Check the set distribution
    if 'set' in data.columns:
        print(f"Set distribution: {data['set'].value_counts().to_dict()}")
    
    # Sample a row and check its structure
    sample_row = data.iloc[0]
    print("\nSample row data types:")
    for col, val in sample_row.items():
        if col in ['Image', 'depth_feature']:
            print(f"{col}: {type(val)}, shape: {val.shape if hasattr(val, 'shape') else 'N/A'}")
        else:
            print(f"{col}: {type(val)}")
else:
    print(f"Dataset is not a pandas DataFrame. Type: {type(data)}")
    print(f"Data structure: {data}") 