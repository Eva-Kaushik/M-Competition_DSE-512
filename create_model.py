import numpy as np
import pandas as pd
import matplotlib as mtp
import sys
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    series = np.load(f"series/W{sys.argv[1]}.npy")
    train_X, train_y = get_X_and_Y(series, D)
    model = LinearRegression()
    model.fit(train_X, train_y)
    mse = mean_squared_error(y, preds)
    print(f"W{idx}, MSE: {mse}")
    
def get_X_and_Y(data, D=10):
    """gets window for training data

    Returns:
        (df,array): training data
    """
    X = np.column_stack([data[i:len(data)-D+i] for i in range(D)])
    Y = values[D:]
    return X, Y