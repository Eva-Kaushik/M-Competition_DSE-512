import numpy as np
import pandas as pd
import matplotlib as mtp
import sys
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_model.py <train.csv> <test.csv>")
        sys.exit(1)
    train = pd.read_csv(sys.argv[1])
    
    test = pd.read_csv(sys.argv[2])
    split = test.shape[1] - 1
    
def get_X_and_Y(data, D=10):
    values = np.array([row[1] for row in data])
    X = np.column_stack([np.ones(len(values) - D)] + 
                        [values[i:len(values)-D+i] for i in range(D)])
    Y = values[D:]
    return X, Y