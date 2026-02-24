import numpy as np
import pandas as pd
import matplotlib as mtp
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def get_X_and_Y(data, D=10):
    """gets window for training data

    Returns:
        (df,array): training data
    """
    X = np.column_stack([data[i:len(data)-D+i] for i in range(D)])
    Y = data[D:]
    return X, Y

if __name__ == "__main__":
    series = np.load(f"series/W{sys.argv[1]}.npy")
    if not sys.argv[2].isdigit():
        print("numRows must be an int")
        print("Usage: <train> <numRows>")
        sys.exit(1)
    D = int(sys.argv[2])
    train_X, train_y = get_X_and_Y(series, D)
    model = LinearRegression()
    model.fit(train_X, train_y)
    preds = model.predict(train_X)
    mse = mean_squared_error(y, preds)
    print(f"W{sys.argv[1]}, MSE: {mse}")
    
