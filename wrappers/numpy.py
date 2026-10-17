import numpy as np
import io
import math
import json
import operator

from .bridge import descriptor
from .files import source, save


def checked_size(shape, dtype='float64'):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    if len(shape) > 64 or any(operator.index(size) < 0 or size > 1_000_000 for size in shape):
        raise ValueError('array dimensions must be 0..1,000,000 with at most 64 axes')
    count = math.prod(shape)
    dtype = np.dtype(dtype)
    if count > 1_000_000 or count * dtype.itemsize > 32 * 1024**2 or dtype.hasobject:
        raise ValueError('array exceeds size limits or uses object dtype')


def array(data, dtype=None, **kwargs):
    initial = np.asarray(data)
    checked_size(initial.shape, initial.dtype if dtype is None else dtype)
    return np.array(initial, dtype=dtype, **kwargs)


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


def bounded(factory):
    def create(shape, *args, **kwargs):
        dtype = kwargs.get('dtype', args[1] if factory is np.full and len(args) > 1 else
                           args[0] if factory is not np.full and args else 'float64')
        checked_size(shape, dtype)
        return factory(shape, *args, **kwargs)
    return create


def arange(start, stop=None, step=1, **kwargs):
    first, last = (0, start) if stop is None else (start, stop)
    if step == 0 or max(0, math.ceil((last - first) / step)) > 1_000_000:
        raise ValueError('arange exceeds 1,000,000 elements or has zero step')
    return np.arange(first, last, step, **kwargs)


def linspace(start, stop, num=50, **kwargs):
    if not 0 <= num <= 1_000_000:
        raise ValueError('linspace num must be 0..1,000,000')
    return np.linspace(start, stop, num=num, **kwargs)


def tile(A, reps):
    value = np.asarray(A)
    checked_size(value.shape, value.dtype)
    try:
        repetitions = (operator.index(reps),)
    except TypeError:
        repetitions = tuple(operator.index(item) for item in reps)
    rank = max(value.ndim, len(repetitions))
    if rank > 64 or any(item < 0 for item in repetitions):
        raise ValueError('tile requires nonnegative repetitions and at most 64 dimensions')
    shape = (1,) * (rank - value.ndim) + value.shape
    repetitions = (1,) * (rank - len(repetitions)) + repetitions
    # NumPy repeats axes sequentially: a later zero must not hide an oversized
    # intermediate allocation. Check the conservative peak before calling tile.
    peak = tuple(size * max(count, 1) for size, count in zip(shape, repetitions))
    if any(size > np.iinfo(np.intp).max for size in peak):
        raise ValueError('tile dimension exceeds platform limits')
    checked_size(peak, value.dtype)
    return np.tile(value, repetitions)


def repeat(a, repeats, axis=None):
    value = np.asarray(a)
    counts = np.asarray(repeats)
    if counts.ndim > 1 or counts.dtype.kind not in 'iu' or np.any(counts < 0):
        raise ValueError('repeat counts must be nonnegative integers')
    length = value.size if axis is None else value.shape[operator.index(axis)]
    if counts.ndim == 0:
        total = length * int(counts)
    else:
        if counts.size not in (1, length):
            raise ValueError('repeat counts must match the selected axis')
        total = (length * int(counts[0]) if counts.size == 1 else
                 sum(int(item) for item in counts))
    shape = [total] if axis is None else list(value.shape)
    if axis is not None:
        shape[axis] = total
    checked_size(shape, value.dtype)
    return np.repeat(value, counts, axis=axis)


def eye(N, M=None, k=0, dtype='float64', order='C'):
    checked_size((operator.index(N), operator.index(N if M is None else M)), dtype)
    return np.eye(N, M, k=k, dtype=dtype, order=order)


def diag(v, k=0):
    value = np.asarray(v)
    if value.ndim == 1:
        checked_size((value.size + abs(operator.index(k)),) * 2, value.dtype)
    return np.diag(value, k=k)


def meshgrid(*xi, copy=True, sparse=False, indexing='xy'):
    arrays = [np.asarray(item) for item in xi]
    shape = tuple(item.size for item in arrays)
    if not sparse:
        # All coordinate arrays coexist, not just one grid.
        checked_size((len(arrays), *shape), np.result_type(*arrays) if arrays else 'float64')
    else:
        for item in arrays:
            checked_size(item.shape, item.dtype)
    return np.meshgrid(*arrays, copy=copy, sparse=sparse, indexing=indexing)


def histogram(a, bins=10, range=None, density=None, weights=None):
    if isinstance(bins, str):
        raise ValueError('use an explicit bin count or bin edges')
    checked_size((operator.index(bins) if np.isscalar(bins) else len(bins),))
    return np.histogram(a, bins=bins, range=range, density=density, weights=weights)


def exports(fs):
    def load(path, *, allow_pickle=False):
        if allow_pickle:
            raise ValueError('pickle-backed arrays are not supported')
        return restore('array', source(fs, path).getvalue())
    def write(path, array, *, allow_pickle=False):
        if allow_pickle:
            raise ValueError('pickle-backed arrays are not supported')
        path = str(path)
        return save(fs, path if path.endswith('.npy') else path + '.npy',
                    lambda stream: np.save(stream, array, allow_pickle=False))
    return {'array': array, 'asarray': array, 'arange': arange, 'linspace': linspace,
            'zeros': bounded(np.zeros), 'ones': bounded(np.ones), 'full': bounded(np.full),
            'zeros_like': np.zeros_like, 'ones_like': np.ones_like, 'sum': np.sum,
            'mean': np.mean, 'median': np.median, 'min': np.min, 'max': np.max,
            'std': np.std, 'var': np.var, 'dot': np.dot, 'matmul': np.matmul,
            'reshape': np.reshape, 'transpose': np.transpose, 'tile': tile, 'repeat': repeat,
            'eye': eye, 'identity': lambda n, dtype='float64': eye(n, dtype=dtype),
            'diag': diag, 'diagonal': np.diagonal, 'trace': np.trace, 'meshgrid': meshgrid,
            'squeeze': np.squeeze, 'expand_dims': np.expand_dims, 'moveaxis': np.moveaxis,
            'swapaxes': np.swapaxes, 'flip': np.flip, 'fliplr': np.fliplr, 'flipud': np.flipud,
            'rot90': np.rot90, 'roll': np.roll, 'ravel': np.ravel, 'concatenate': np.concatenate,
            'stack': np.stack, 'vstack': np.vstack, 'hstack': np.hstack, 'unique': np.unique,
            'where': np.where, 'clip': np.clip, 'sqrt': np.sqrt, 'abs': np.abs,
            'exp': np.exp, 'log': np.log, 'sin': np.sin, 'cos': np.cos, 'isnan': np.isnan,
            'isfinite': np.isfinite, 'all': np.all, 'any': np.any, 'sort': np.sort,
            'argsort': np.argsort, 'argmin': np.argmin, 'argmax': np.argmax,
            'cumsum': np.cumsum, 'cumprod': np.cumprod, 'prod': np.prod,
            'diff': np.diff, 'gradient': np.gradient, 'percentile': np.percentile,
            'quantile': np.quantile, 'nanmean': np.nanmean, 'nansum': np.nansum,
            'nanmedian': np.nanmedian, 'nanstd': np.nanstd, 'nanmin': np.nanmin, 'nanmax': np.nanmax,
            'nonzero': np.nonzero, 'flatnonzero': np.flatnonzero, 'count_nonzero': np.count_nonzero,
            'isclose': np.isclose, 'allclose': np.allclose, 'array_equal': np.array_equal,
            'isinf': np.isinf, 'nan_to_num': lambda x, **kw: np.nan_to_num(x, copy=True, **kw),
            'log10': np.log10, 'log2': np.log2, 'expm1': np.expm1, 'log1p': np.log1p,
            'tan': np.tan, 'arctan2': np.arctan2, 'floor': np.floor, 'ceil': np.ceil,
            'maximum': np.maximum, 'minimum': np.minimum, 'sign': np.sign,
            'logical_and': np.logical_and, 'logical_or': np.logical_or, 'logical_not': np.logical_not,
            'histogram': histogram,
            'corrcoef': np.corrcoef, 'cov': np.cov, 'interp': np.interp,
            'split': np.split, 'array_split': np.array_split,
            'round': np.round, 'dtype': np.dtype, 'load': load, 'save': write,
            'pi': np.pi, 'e': np.e, 'nan': np.nan, 'inf': np.inf,
            'int32': 'int32', 'int64': 'int64', 'float32': 'float32', 'float64': 'float64',
            'bool_': 'bool', 'linalg': {'norm': np.linalg.norm, 'solve': np.linalg.solve,
                                      'inv': np.linalg.inv, 'det': np.linalg.det,
                                      'svd': np.linalg.svd, 'eig': np.linalg.eig, 'eigh': np.linalg.eigh,
                                      'eigvals': np.linalg.eigvals, 'eigvalsh': np.linalg.eigvalsh,
                                      'qr': np.linalg.qr, 'cholesky': np.linalg.cholesky,
                                      'pinv': np.linalg.pinv, 'lstsq': np.linalg.lstsq,
                                      'matrix_rank': np.linalg.matrix_rank, 'cond': np.linalg.cond},
            'random': {'default_rng': np.random.default_rng}}


def members(value, fs, name=None):
    if isinstance(value, np.random.Generator):
        def distribution(function, size_position):
            def generate(*args, **kwargs):
                size = kwargs.get('size', args[size_position] if len(args) > size_position else None)
                checked_size(() if size is None else size, kwargs.get('dtype', 'float64'))
                return function(*args, **kwargs)
            return generate
        return {'random': distribution(value.random, 0), 'normal': distribution(value.normal, 2),
                'uniform': distribution(value.uniform, 2), 'integers': distribution(value.integers, 2),
                'standard_normal': distribution(value.standard_normal, 0)}
    if isinstance(value, np.ndarray):
        return {'shape': value.shape, 'size': value.size, 'ndim': value.ndim,
                'real': value.real, 'imag': value.imag,
                'dtype': value.dtype, 'T': value.T, 'tolist': value.tolist, 'item': value.item,
                'reshape': value.reshape, 'flatten': value.flatten, 'ravel': value.ravel,
                'astype': lambda dtype, **kwargs: array(value, dtype=dtype, **kwargs),
                'copy': value.copy, 'sum': value.sum, 'mean': value.mean,
                'repeat': lambda repeats, axis=None: repeat(value, repeats, axis),
                'squeeze': value.squeeze, 'swapaxes': value.swapaxes,
                'diagonal': value.diagonal, 'trace': value.trace,
                'prod': value.prod, 'cumsum': value.cumsum, 'cumprod': value.cumprod,
                'nonzero': value.nonzero,
                'min': value.min, 'max': value.max, 'std': value.std, 'var': value.var,
                'dot': value.dot, 'transpose': value.transpose, 'all': value.all, 'any': value.any,
                'round': value.round, 'clip': value.clip, 'sort': value.sort,
                'argsort': value.argsort, 'argmin': value.argmin, 'argmax': value.argmax}
    if isinstance(value, np.dtype):
        return {'name': value.name, 'kind': value.kind, 'itemsize': value.itemsize}
    return {}


def pack(value):
    if isinstance(value, np.random.Generator):
        return descriptor('numpy', 'rng', json.dumps(value.bit_generator.state))
    if isinstance(value, np.ndarray):
        checked_size(value.shape, value.dtype)
        output = io.BytesIO()
        np.save(output, value, allow_pickle=False)
        return descriptor('numpy', 'array', output.getvalue())
    if isinstance(value, np.dtype):
        return descriptor('numpy', 'dtype', value.str)
    if isinstance(value, np.generic):
        return descriptor('numpy', 'scalar', value.item())


def restore(kind, data):
    if kind == 'rng':
        value = np.random.default_rng(0)
        state = json.loads(data)
        if state['bit_generator'] != 'PCG64':
            raise ValueError('Only PCG64 generators are supported')
        value.bit_generator.state = state
        return value
    if kind == 'scalar':
        return data
    if kind == 'dtype':
        return np.dtype(data)
    if kind != 'array':
        raise ValueError('unsupported numpy object')
    stream = io.BytesIO(data)
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError('Only NPY versions 1 and 2 are supported')
    checked_size(shape, dtype)
    result = np.load(io.BytesIO(data), allow_pickle=False)
    if not isinstance(result, np.ndarray) or result.size > 1_000_000:
        raise ValueError('invalid array')
    return result


def is_view(value):
    return isinstance(value, np.ndarray)


def iterate(value):
    return iter(value)


def view(root, value):
    return None


def shares(root, value):
    return isinstance(root, np.ndarray) and isinstance(value, np.ndarray) and np.shares_memory(root, value)


def assign(value, name, item):
    raise AttributeError('Assign array elements with indexing, not attributes')
