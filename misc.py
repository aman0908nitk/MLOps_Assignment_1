import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def load_data():
    
    """Fetches the raw Boston Housing dataset from the given url"""
    
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    
    # Splitting this into data and target
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    
    #Feature names based on original datasets
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

    
    # Creating dataframe
    df = pd.DataFrame(data, columns=feature_names)
    df = target  # MEDV is our target variable
    return df

def preprocess_data(df, target_column='MEDV', test_size=0.25, random_state=42, scale=False):
   
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Perform reproducible train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    if scale:
        # Standardize features: z = (x - mean) / std_dev
        scaler = StandardScaler()
        # Fit parameters strictly on training partition to prevent data leakage
        X_train_processed = scaler.fit_transform(X_train)
        # Apply the learned parameters to scale the test partition
        X_test_processed = scaler.transform(X_test)
        return X_train_processed, X_test_processed, y_train, y_test
        
    return X_train.values, X_test.values, y_train, y_test

def train_model(model_instance, X_train, y_train):
    
    model_instance.fit(X_train, y_train)
    return model_instance

def evaluate_model(model_instance, X_test, y_test):
    """
    Generates predictions and evaluates using Mean Squared Error (MSE).
    """
    predictions = model_instance.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse
