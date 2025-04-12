"""
Usage:
    python test/test_pickle.py

"""

import pickle


path = "DATA/reasonings.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

print(data[0])