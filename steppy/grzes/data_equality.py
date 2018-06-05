import numpy as np
import pandas as pd

def data_equals(s, t):
    if s.__class__ != t.__class__:
        return False
    if isinstance(s, dict):
        return dict_equals(s, t)
    if isinstance(s, list):
        return list_equals(s, t)
    if isinstance(s, tuple):
        return list_equals(list(s), list(t))
    if isinstance(s, set):
        return list_equals(sorted(list(s)), sorted(list(t)))
    if isinstance(s, np.ndarray):
        return np.array_equal(s, t)
    if isinstance(s, pd.DataFrame):
        return s.equals(t)

    return s == t


def dict_equals(s, t):
    if s.keys() != t.keys():
        return False
    for k, v in s.items():
        if not data_equals(v, t[k]):
            return False
    return True


def list_equals(s, t):
    for a, b in zip(s, t):
        if not data_equals(a, b):
            return False
    return True
