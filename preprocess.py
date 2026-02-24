import numpy as np
import pandas as pd
import os
import sys


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python create_model.py <train.csv> <test.csv> <numRows>")
        sys.exit(1)
    train = pd.read_csv(sys.argv[1])
    
    test = pd.read_csv(sys.argv[2])
    if not sys.argv[3].isdigit():
        print("numRows must be numeric")
        sys.exit(1)
    numRows = int(sys.argv[3])
    h = test.shape[1] - 1
    os.makedirs("series", exist_ok=True)
    for i in range(numRows):
        series = train.iloc[i, 1:].dropna().values
        np.save(f"series/W{i}_Train.npy", series.astype(float))
        
    