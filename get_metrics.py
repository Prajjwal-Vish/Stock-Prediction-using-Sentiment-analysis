from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

def get_metrics(true, pred):
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, pred)

    metrics = {"Mean Absolute Error" : mae, "Mean Square Error":mse, "Root Mean Square Error":rmse,"R2-Score":r2}
    return metrics