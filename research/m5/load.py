import sys; sys.path.insert(0, '/home/claude/lab')
import numpy as np, pandas as pd
from core import data as D
def m1():
    d = D.load_m1(); d = d[d.time >= "2012-12-01"].reset_index(drop=True)
    return d
if __name__ == "__main__":
    d = m1(); print(len(d), d.time.min(), d.time.max(), d.columns.tolist())
    d[["time","open","high","low","close"]].to_parquet("/home/claude/m5r/m1.parquet")
