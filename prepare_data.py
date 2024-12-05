import numpy as np
import pandas as pd

def prepare_data(df, target_column='Close', window_size=7):
    features = df.drop(columns=['Date', 'Stock Name', target_column])  # Dropping non-numeric columns
    target = df[target_column].values  # Stock price (Close)
    
    # Create data structure with window_size time steps
    x, y = [], []
    for i in range(window_size, len(df)):
        x.append(features.iloc[i-window_size:i, :].values)  # Using last 'window_size' rows as input
        y.append(target[i])  # Predicting the next day's stock price 
    
    return np.array(x), np.array(y)