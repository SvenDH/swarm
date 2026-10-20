import pandas as pd
import numpy as np
from pandas.core.indexing import _LocationIndexer
from pandas.core.strings.accessor import StringMethods
from pandas.core.indexes.accessors import DatetimeProperties, TimedeltaProperties
from inspect import signature
from .api import select

from .files import source, save
from . import numpy as arrays


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
    return {**select(vars(pd), '''pivot melt get_dummies cut qcut factorize
                unique value_counts isnull notnull to_datetime to_timedelta date_range bdate_range
                timedelta_range infer_freq merge_ordered merge_asof wide_to_long'''),
            'DataFrame': pd.DataFrame, 'Series': pd.Series, 'Index': pd.Index,
            'Categorical': pd.Categorical,
            'pivot_table': lambda *a, **kw: summarized(pd.pivot_table, *a, **kw),
            'crosstab': lambda *a, **kw: summarized(pd.crosstab, *a, **kw),
            'read_csv': read_csv, 'read_excel': read_excel, 'concat': pd.concat,
            'merge': pd.merge, 'isna': pd.isna, 'notna': pd.notna,
            'to_numeric': pd.to_numeric, 'NA': None}


def members(value, fs, name=None):
    if isinstance(value, StringMethods):
        return {'lower': value.lower, 'upper': value.upper, 'casefold': value.casefold,
                'title': value.title, 'capitalize': value.capitalize, 'strip': value.strip,
                'lstrip': value.lstrip, 'rstrip': value.rstrip, 'len': value.len,
                'contains': value.contains, 'startswith': value.startswith, 'endswith': value.endswith,
                'replace': value.replace, 'split': value.split, 'rsplit': value.rsplit,
                'extract': value.extract, 'extractall': value.extractall, 'find': value.find,
                'count': value.count, 'slice': value.slice, 'get': value.get,
                'pad': value.pad, 'zfill': value.zfill, 'cat': value.cat}
    if isinstance(value, TimedeltaProperties):
        return {'days': value.days, 'seconds': value.seconds, 'microseconds': value.microseconds,
                'nanoseconds': value.nanoseconds, 'total_seconds': value.total_seconds,
                'components': value.components, 'floor': value.floor, 'ceil': value.ceil, 'round': value.round}
    if isinstance(value, DatetimeProperties):
        return {'year': value.year, 'month': value.month, 'day': value.day, 'hour': value.hour,
                'minute': value.minute, 'second': value.second, 'dayofweek': value.dayofweek,
                'dayofyear': value.dayofyear, 'quarter': value.quarter,
                'date': value.date, 'strftime': value.strftime, 'floor': value.floor,
                'ceil': value.ceil, 'round': value.round, 'normalize': value.normalize}
    if isinstance(value, pd.Categorical):
        return {'categories': value.categories, 'codes': value.codes, 'ordered': value.ordered,
                'tolist': value.tolist, 'isna': value.isna, 'notna': value.notna,
                'unique': value.unique, 'value_counts': value.value_counts,
                'remove_unused_categories': value.remove_unused_categories}
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
                  'var': value.var, 'prod': value.prod, 'any': value.any, 'all': value.all,
                  'cumsum': value.cumsum, 'cumprod': value.cumprod,
                  'cummin': value.cummin, 'cummax': value.cummax,
                  'quantile': value.quantile, 'skew': value.skew, 'kurt': value.kurt,
                  'rank': value.rank, 'diff': value.diff, 'pct_change': value.pct_change,
                  'shift': value.shift, 'ffill': value.ffill, 'bfill': value.bfill,
                  'duplicated': value.duplicated, 'clip': value.clip,
                  'sample': lambda *a, **kw: sample(value, *a, **kw),
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
                          iterrows=iterrows, groupby=lambda *a, **kw: Group(value, a, kw),
                          pivot=value.pivot, pivot_table=lambda *a, **kw: pivot_table(value, *a, **kw),
                          melt=value.melt, stack=value.stack, unstack=value.unstack,
                          transpose=value.transpose, select_dtypes=value.select_dtypes,
                          corr=value.corr, cov=value.cov, isin=value.isin)
        else:
            result.update(name=value.name, dtype=str(value.dtype), tolist=value.tolist,
                          unique=value.unique, value_counts=value.value_counts,
                          isin=value.isin, between=value.between, to_frame=value.to_frame,
                          nlargest=value.nlargest, nsmallest=value.nsmallest)
            if name == 'str':
                result['str'] = value.str
            if name == 'dt':
                result['dt'] = value.dt
        if isinstance(value, pd.DataFrame) and name not in result and name in value.columns:
            result[name] = value[name]
        return result
    if isinstance(value, pd.Index):
        return {'tolist': value.tolist, 'to_list': value.tolist, 'name': value.name,
                'size': value.size, 'dtype': str(value.dtype)}
    return {}


def sample(value, *args, **kwargs):
    kwargs.setdefault('random_state', 0)
    return value.sample(*args, **kwargs)


def pivot_table(value, *args, **kwargs):
    return summarized(pd.pivot_table, value, *args, **kwargs)


def summarized(function, *args, **kwargs):
    bound = signature(function).bind(*args, **kwargs)
    reducer = bound.arguments.get('aggfunc')
    def valid(item):
        if isinstance(item, dict):
            return all(valid(x) for x in item.values())
        if isinstance(item, (tuple, list)):
            return all(valid(x) for x in item)
        return isinstance(item, str) and item in {'sum', 'mean', 'median', 'min', 'max', 'count',
            'size', 'std', 'var', 'prod', 'first', 'last', 'nunique', 'any', 'all'}
    if reducer is not None and not valid(reducer):
        raise ValueError('aggfunc must use supported reduction names')
    return function(*args, **kwargs)


def is_view(value):
    return isinstance(value, (pd.Series, pd.DataFrame, Group, _LocationIndexer,
                              StringMethods, DatetimeProperties, TimedeltaProperties, pd.Categorical))


def iterate(value):
    return iter(value.native() if isinstance(value, Group) else value)


def assign(value, name, item):
    if isinstance(value, pd.DataFrame) and name == 'columns':
        value.columns = item
    elif isinstance(value, (pd.Series, pd.DataFrame)) and name == 'index':
        value.index = item
    elif isinstance(value, pd.Series) and name == 'name':
        value.name = item
    else:
        raise AttributeError('Unsupported pandas attribute assignment')
