import numpy as np


def run(fs, op, data, other=None, axis=None):
    a = np.asarray(data)
    operations = {'array': lambda: a, 'sum': lambda: a.sum(axis=axis),
                  'mean': lambda: a.mean(axis=axis), 'min': lambda: a.min(axis=axis),
                  'max': lambda: a.max(axis=axis), 'std': lambda: a.std(axis=axis),
                  'median': lambda: np.median(a, axis=axis),
                  'transpose': lambda: a.T, 'reshape': lambda: a.reshape(other),
                  'dot': lambda: a.dot(other), 'unique': lambda: np.unique(a)}
    value = operations[op]()
    return value.tolist()
