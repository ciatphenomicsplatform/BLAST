import pandas as pd
from imblearn.over_sampling import RandomOverSampler

def delete_columns(dataset):
    """
    Deletes specific unwanted columns from the dataset.
    Columns to delete based on the original notebook:
    'Parcela', 'Muestra', 'Repeticion', 'Variedad', 'P1', 'P2', 'P3', 'P4'
    """
    cols_to_delete = ['Parcela', 'Muestra', 'Repeticion', 'Variedad', 'P1', 'P2', 'P3', 'P4']
    for col in cols_to_delete:
        if col in dataset.columns:
            dataset = dataset.drop(col, axis=1)
    return dataset

def reclassificar_columnas(df, columnas):
    """
    Reclassifies categorical columns (like 'Severity') into integers.
    """
    class_mapping = {
        'H': 1, 'M': 2, 'L': 3, 'S': 4,
        1: 1, 2: 2, 3: 3, 4: 4,
        '1': 1, '2': 2, '3': 3, '4': 4
    }
    
    for col in columnas:
        if col in df.columns:
            df[col] = df[col].map(class_mapping)
    return df

def filter_features_no_canopy(df, feature_columns):
    """
    Keep only exactly the columns intended to be used, e.g. eliminating canopy.
    """
    available_cols = [c for c in feature_columns if c in df.columns]
    return df[available_cols].copy()

def balance_dataset_oversampling(df, target_column='BL', oversampling_factor=0.5):
    """
    Balances the dataset using RandomOverSampler from imblearn.
    """
    from imblearn.over_sampling import RandomOverSampler
    
    # Assuming 'BL' is the label and rest are features
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # oversampling_factor could be a float like 0.5 (meaning minority class gets 50% of majority)
    # or 'auto' (meaning equal parts)
    
    sampler = RandomOverSampler(sampling_strategy=oversampling_factor, random_state=42)
    X_resampled, y_resampled = sampler.fit_resample(X, y)
    
    df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
    return df_resampled
