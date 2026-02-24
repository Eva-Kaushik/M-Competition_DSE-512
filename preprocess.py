import numpy as np
import pandas as pd
import os
import sys

def get_X_and_Y(data, D=10):
    """gets window for training data

    Returns:
        (df,array): training data
    """
    X = np.column_stack([values[i:len(data)-D+i] for i in range(D)])
    Y = values[D:]
    return X, Y

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python create_model.py <train.csv> <test.csv> <numRows>")
        sys.exit(1)
    train = pd.read_csv(sys.argv[1])
    
    test = pd.read_csv(sys.argv[2])
    if not sys.argv[3].isDigit():
        print("numRows must be numeric")
        sys.exit(1)
    numRows = sys.argv[3]
    h = test.shape[1] - 1
    os.makedirs("series", exist_ok=True)
    for i in range(numRows):
        series = train.iloc[i, 1:].dropna().values
        np.save(f"series/W{i}.npy", series)
        
    