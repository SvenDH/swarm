import json

import pandas as pd

from .files import source, save


def run(fs, op, data=None, path=None, columns=None):
    if op in ('read_csv', 'read_excel'):
        frame = (pd.read_csv(source(fs, path)) if op == 'read_csv'
                 else pd.read_excel(source(fs, path)))
    else:
        frame = pd.DataFrame(data)
        if op in ('to_csv', 'to_excel'):
            writer = frame.to_csv if op == 'to_csv' else frame.to_excel
            return save(fs, path, lambda output: writer(output, index=False))
        operations = {'head': lambda: frame.head(5 if columns is None else columns),
                      'describe': frame.describe, 'select': lambda: frame[columns],
                      'sort': lambda: frame.sort_values(columns),
                      'drop_duplicates': frame.drop_duplicates}
        frame = operations[op]()
    return json.loads(frame.to_json(orient='records'))
