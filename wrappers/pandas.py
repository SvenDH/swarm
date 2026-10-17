import json
import io
from datetime import date, timedelta

import pandas as pd
import numpy as np
from pandas.core.indexing import _LocationIndexer

from .files import source, save
from .bridge import descriptor
from . import numpy as arrays


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


class Group:
    def __init__(self, frame, args, kwargs, selection=None):
        self.frame, self.args, self.kwargs, self.selection = frame, args, kwargs, selection

    def __getitem__(self, selection):
        return Group(self.frame, self.args, self.kwargs, selection)

    def native(self):
        group = self.frame.groupby(*self.args, **self.kwargs)
        return group if self.selection is None else group[self.selection]


def aggregate(value, function, *args, **kwargs):
    def validate(item):
        if isinstance(item, dict):
            return all(validate(part) for part in item.values())
        if isinstance(item, (list, tuple)):
            return all(validate(part) for part in item)
        return isinstance(item, str) and item in {'sum', 'mean', 'median', 'min', 'max', 'std',
            'var', 'count', 'size', 'first', 'last', 'nunique', 'prod', 'any', 'all'}
    if not validate(function):
        raise ValueError('agg accepts only supported reduction names, lists or dictionaries')
    return value.agg(function, *args, **kwargs)


def checked(kwargs, allowed):
    unknown = set(kwargs) - set(allowed.split())
    if unknown:
        raise ValueError('Unsupported IO options: ' + ', '.join(sorted(unknown)))
    return kwargs


def exports(fs):
    def read_csv(filepath_or_buffer, **kwargs):
        checked(kwargs, 'sep delimiter header names index_col usecols dtype nrows skiprows '
                       'encoding na_values keep_default_na parse_dates skip_blank_lines decimal thousands')
        return pd.read_csv(source(fs, filepath_or_buffer), **kwargs)
    def read_excel(io, **kwargs):
        checked(kwargs, 'sheet_name header names index_col usecols dtype nrows skiprows '
                       'na_values keep_default_na parse_dates')
        return pd.read_excel(source(fs, io), **kwargs)
    return {'DataFrame': pd.DataFrame, 'Series': pd.Series, 'Index': pd.Index,
            'read_csv': read_csv, 'read_excel': read_excel, 'concat': pd.concat,
            'merge': pd.merge, 'isna': pd.isna, 'notna': pd.notna,
            'to_numeric': pd.to_numeric, 'NA': None}


def members(value, fs, name=None):
    if isinstance(value, (np.ndarray, np.dtype)):
        return arrays.members(value, fs)
    if isinstance(value, Group):
        group = value.native()
        return {'sum': group.sum, 'mean': group.mean, 'median': group.median, 'min': group.min,
                'max': group.max, 'count': group.count, 'size': group.size, 'first': group.first,
                'last': group.last, 'nunique': group.nunique, 'std': group.std,
                'agg': lambda function, *a, **kw: aggregate(group, function, *a, **kw)}
    if isinstance(value, (pd.DataFrame, pd.Series)):
        def to_csv(path_or_buf=None, **kwargs):
            checked(kwargs, 'sep na_rep float_format columns header index index_label encoding '
                           'quoting quotechar lineterminator date_format doublequote escapechar decimal')
            if path_or_buf is None:
                return value.to_csv(**kwargs)
            return save(fs, path_or_buf, lambda stream: value.to_csv(stream, **kwargs))
        def to_excel(excel_writer, **kwargs):
            checked(kwargs, 'sheet_name na_rep float_format columns header index index_label '
                           'startrow startcol merge_cells inf_rep freeze_panes')
            return save(fs, excel_writer, lambda stream: value.to_excel(stream, **kwargs))
        result = {'shape': value.shape, 'size': value.size, 'ndim': value.ndim,
                  'index': value.index, 'values': value.values, 'empty': value.empty,
                  'loc': value.loc, 'iloc': value.iloc, 'head': value.head, 'tail': value.tail,
                  'copy': value.copy, 'describe': value.describe, 'to_dict': value.to_dict,
                  'to_numpy': value.to_numpy, 'to_csv': to_csv, 'to_excel': to_excel,
                  'sum': value.sum, 'mean': value.mean, 'min': value.min, 'max': value.max,
                  'median': value.median, 'std': value.std, 'count': value.count,
                  'sort_values': value.sort_values, 'sort_index': value.sort_index,
                  'drop': value.drop, 'dropna': value.dropna, 'fillna': value.fillna,
                  'drop_duplicates': value.drop_duplicates, 'astype': value.astype,
                  'reset_index': value.reset_index, 'rename': value.rename, 'isna': value.isna,
                  'notna': value.notna, 'round': value.round, 'nunique': value.nunique,
                  'agg': lambda function, *a, **kw: aggregate(value, function, *a, **kw)}
        if isinstance(value, pd.DataFrame):
            def iterrows():
                if len(value) > 10_000:
                    raise ValueError('iterrows is limited to 10,000 rows; prefer vectorized operations')
                return list(value.iterrows())
            result.update(columns=value.columns, dtypes=value.dtypes, T=value.T,
                          assign=value.assign, set_index=value.set_index, merge=value.merge,
                          iterrows=iterrows, groupby=lambda *a, **kw: Group(value, a, kw))
        else:
            result.update(name=value.name, dtype=str(value.dtype), tolist=value.tolist,
                          unique=value.unique, value_counts=value.value_counts,
                          isin=value.isin, between=value.between)
        if isinstance(value, pd.DataFrame) and name not in result and name in value.columns:
            result[name] = value[name]
        return result
    if isinstance(value, pd.Index):
        return {'tolist': value.tolist, 'to_list': value.tolist, 'name': value.name,
                'size': value.size, 'dtype': str(value.dtype)}
    return {}


def pack(value):
    if isinstance(value, Group):
        return descriptor('pandas', 'group', [pack(value.frame)['data'], value.args, value.kwargs, value.selection])
    if isinstance(value, (pd.DataFrame, pd.Series)):
        frame = value.to_frame(name='__series_value__') if isinstance(value, pd.Series) else value
        if frame.size > 1_000_000:
            raise ValueError('dataframes are limited to 1,000,000 cells')
        def plain(item):
            if item is None or item is pd.NA or item is pd.NaT or isinstance(item, (str, bool, int, float, date, timedelta, np.generic)):
                return True
            if isinstance(item, (list, tuple)):
                return all(plain(part) for part in item)
            if isinstance(item, dict):
                return all(isinstance(key, (str, int, float, bool)) and plain(part) for key, part in item.items())
            return False
        if not all(plain(item) for item in frame.select_dtypes(include=['object']).to_numpy().flat):
            raise ValueError('DataFrame object cells must contain plain data, not native library objects')
        if not all(plain(item) for item in [*frame.columns, *frame.index, *frame.columns.names, *frame.index.names]):
            raise ValueError('DataFrame labels must contain plain data')
        if isinstance(value, pd.Series) and not plain(value.name):
            raise ValueError('Series name must contain plain data')
        encoded = frame.copy()
        encoded.columns = [f'__column_{i}__' for i in range(len(frame.columns))]
        encoded.index.names = [f'__index_{i}__' for i in range(frame.index.nlevels)]
        table = encoded.to_json(orient='table', date_format='iso')
        data = {'table': table, 'columns': frame.columns.tolist(), 'column_names': list(frame.columns.names),
                'multi': isinstance(frame.columns, pd.MultiIndex), 'index_names': list(frame.index.names)}
        if len(table.encode()) > 32 * 1024**2:
            raise ValueError('dataframe exceeds 32 MiB')
        return descriptor('pandas', 'series', [data, value.name]) if isinstance(value, pd.Series) else descriptor('pandas', 'frame', data)
    if isinstance(value, pd.Index):
        return descriptor('pandas', 'index', value.tolist())
    if isinstance(value, np.ndarray):
        return arrays.pack(value)
    if isinstance(value, np.generic):
        return descriptor('pandas', 'scalar', value.item())


def restore(kind, data):
    if kind == 'index':
        return pd.Index(data)
    if kind == 'group':
        frame, args, kwargs, selection = data
        return Group(restore('frame', frame), args, kwargs, selection)
    if kind not in ('frame', 'series'):
        raise ValueError('Unsupported pandas object')
    encoded = data[0] if kind == 'series' else data
    frame = pd.read_json(io.StringIO(encoded['table']), orient='table')
    frame.columns = (pd.MultiIndex.from_tuples(encoded['columns'], names=encoded['column_names'])
                     if encoded['multi'] else pd.Index(encoded['columns'], name=encoded['column_names'][0]))
    frame.index.names = encoded['index_names']
    if kind == 'series':
        series = frame.iloc[:, 0]
        series.name = data[1]
        return series
    return frame


def is_view(value):
    return isinstance(value, (pd.Series, pd.DataFrame, Group, _LocationIndexer))


def iterate(value):
    return iter(value.native() if isinstance(value, Group) else value)


def view(root, value):
    return None


def assign(value, name, item):
    if isinstance(value, pd.DataFrame) and name == 'columns':
        value.columns = item
    elif isinstance(value, (pd.Series, pd.DataFrame)) and name == 'index':
        value.index = item
    elif isinstance(value, pd.Series) and name == 'name':
        value.name = item
    else:
        raise AttributeError('Unsupported pandas attribute assignment')
