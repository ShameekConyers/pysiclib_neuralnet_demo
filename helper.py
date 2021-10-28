import requests
import io
import csv
import pandas as pd

v = pd.read_csv("mnist_test_base.csv")
t = pd.read_csv("mnist_train_base.csv")
print(t)
print(v)

ts = t.sample(n=5000)
vs = v.sample(n=2000)
print(ts)

ts.to_csv("mnist_train_5000.csv", index=False)
vs.to_csv("mnist_test_2000.csv", index=False)
