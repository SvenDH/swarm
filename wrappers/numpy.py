import numpy as np
import io
import math
import json
import operator
from numpy._core._exceptions import _ArrayMemoryError

from .bridge import descriptor
from .files import source, save
from .limits import check_memory
from .api import select
from .common import Record, freeze, thaw

# Initialize native FFT code on the Python import worker, not a Monty callback.
np.fft.fft([0.0])
SCALARS = {name: np.dtype(name).type for name in (
    'bool', 'uint8', 'uint16', 'uint32', 'uint64', 'int8', 'int16', 'int32', 'int64',
    'float16', 'float32', 'float64', 'complex64', 'complex128')}

UFUNCS = {'add': np.add, 'subtract': np.subtract, 'multiply': np.multiply,
          'divide': np.divide, 'true_divide': np.true_divide, 'floor_divide': np.floor_divide,
          'power': np.power, 'remainder': np.remainder, 'mod': np.mod,
          'negative': np.negative, 'positive': np.positive, 'square': np.square,
          'absolute': np.absolute, 'equal': np.equal, 'not_equal': np.not_equal,
          'less': np.less, 'less_equal': np.less_equal, 'greater': np.greater,
          'greater_equal': np.greater_equal, 'bitwise_and': np.bitwise_and,
          'bitwise_or': np.bitwise_or, 'bitwise_xor': np.bitwise_xor,
          'bitwise_not': np.bitwise_not, 'hypot': np.hypot,
          'maximum': np.maximum, 'minimum': np.minimum,
          'logical_and': np.logical_and, 'logical_or': np.logical_or, 'logical_not': np.logical_not,
          'sqrt': np.sqrt, 'exp': np.exp, 'log': np.log, 'sin': np.sin, 'cos': np.cos}
UFUNCS.update(select(vars(np), '''arccos arcsin arctan arccosh arcsinh arctanh cosh sinh tanh
    tan arctan2 degrees radians deg2rad rad2deg exp2 expm1 log2 log10 log1p logaddexp
    logaddexp2 cbrt reciprocal float_power fmod divmod conjugate conj fabs fmax fmin
    rint floor ceil trunc spacing nextafter copysign sign signbit heaviside ldexp
    gcd lcm invert left_shift right_shift logical_xor isfinite isinf isnan'''))

PURE = '''allclose array_equal array_equiv atleast_1d atleast_2d atleast_3d broadcast_arrays
    broadcast_shapes hsplit vsplit dsplit compress extract
    trim_zeros flatnonzero unravel_index lexsort partition argpartition
    union1d intersect1d setdiff1d setxor1d isin unique_counts unique_inverse unique_values
    nanprod nanvar nanpercentile nanquantile nancumsum nancumprod nanargmin nanargmax ptp
    average trapezoid ediff1d real imag angle real_if_close unwrap sinc i0 convolve correlate
    vdot einsum_path'''


def numeric_dtype(*values, dtype=None):
    result = np.dtype(dtype) if dtype is not None else np.result_type(*[np.asarray(value).dtype for value in values])
    if result.kind not in 'buifcmM':
        raise ValueError('ufuncs support numeric and datetime dtypes only')
    return result


class Ufunc:
    def __init__(self, name):
        self.name = name

    def __call__(self, *args, **kwargs):
        return ufunc_call(UFUNCS[self.name], *args, **kwargs)


def ufunc_call(function, *args, **kwargs):
    if len(args) != function.nin:
        raise ValueError('supply input arguments using positional arguments')
    shapes = [np.shape(arg) for arg in args]
    if 'where' in kwargs:
        if kwargs.get('out') is None and not np.all(kwargs['where']):
            raise ValueError('masked ufunc calls without initialized out arrays are not supported')
        shapes.append(np.shape(kwargs['where']))
    dtype = numeric_dtype(*args, dtype=kwargs.get('dtype'))
    checked_size(np.broadcast_shapes(*shapes), dtype)
    return function(*args, **kwargs)


class Grid:
    def __init__(self, sparse=False):
        self.sparse = sparse

    def __getitem__(self, key):
        axes = key if isinstance(key, tuple) else (key,)
        if not 1 <= len(axes) <= 64:
            raise ValueError('grids support 1..64 axes')
        counts = []
        for axis in axes:
            if not isinstance(axis, slice) or axis.stop is None:
                raise TypeError('grids require slices with a stop value')
            start, stop, step = 0 if axis.start is None else axis.start, axis.stop, 1 if axis.step is None else axis.step
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in (start, stop)):
                raise ValueError('grid endpoints must be finite real numbers')
            if isinstance(step, complex):
                if not math.isfinite(abs(step)):
                    raise ValueError('grid step must be finite')
                count = int(abs(step))
            else:
                if not isinstance(step, (int, float)) or not math.isfinite(step) or step == 0:
                    raise ValueError('grid step must be finite and nonzero')
                count = max(0, math.ceil((stop - start) / step))
            checked_size((count,))
            counts.append(count)
        checked_size((sum(counts),) if self.sparse else (len(axes), *counts))
        return (np.ogrid if self.sparse else np.mgrid)[key]


def bincount(x, weights=None, minlength=0):
    values = np.asarray(x)
    if values.ndim != 1 or (values.size and (values.dtype.kind not in 'iu' or np.any(values < 0))):
        raise ValueError('bincount requires a one-dimensional array of nonnegative integers')
    minlength = operator.index(minlength)
    if minlength < 0:
        raise ValueError('minlength must be nonnegative')
    checked_size((max(minlength, int(values.max()) + 1 if values.size else 0),))
    return np.bincount(values.astype(np.intp, copy=False), weights=weights, minlength=minlength)


def indices(dimensions, dtype='int64', sparse=False):
    shape = tuple(operator.index(size) for size in dimensions)
    if len(shape) > 64 or any(size < 0 for size in shape):
        raise ValueError('indices requires nonnegative dimensions and at most 64 axes')
    checked_size((sum(shape),) if sparse else (len(shape), *shape), dtype)
    return np.indices(shape, dtype=dtype, sparse=sparse)


def broadcast_to(value, shape):
    value = np.asarray(value)
    checked_size(shape, value.dtype)
    return np.broadcast_to(value, shape)


def pad(value, pad_width, mode='constant', **kwargs):
    value = np.asarray(value)
    widths = np.broadcast_to(np.asarray(pad_width), (value.ndim, 2))
    if widths.dtype.kind not in 'iu' or np.any(widths < 0):
        raise ValueError('pad widths must be nonnegative integers')
    if mode not in ('constant', 'edge', 'linear_ramp', 'maximum', 'mean', 'median', 'minimum',
                    'reflect', 'symmetric', 'wrap'):
        raise ValueError('Unsupported pad mode; callbacks and uninitialized padding are not exposed')
    checked_size(tuple(size + int(width[0]) + int(width[1]) for size, width in zip(value.shape, widths)), value.dtype)
    return np.pad(value, pad_width, mode=mode, **kwargs)


def take(value, indices, axis=None, out=None, mode='raise'):
    value = np.asarray(value)
    if axis is None:
        shape = np.shape(indices)
    else:
        axis = operator.index(axis)
        if not -value.ndim <= axis < value.ndim:
            raise ValueError('invalid take axis')
        axis %= value.ndim
        shape = (*value.shape[:axis], *np.shape(indices), *value.shape[axis+1:])
    checked_size(shape, value.dtype)
    return np.take(value, indices, axis=axis, out=out, mode=mode)


def machine_info(dtype, integer=False):
    info = np.iinfo(dtype) if integer else np.finfo(dtype)
    return Record({'bits': info.bits, 'min': info.min, 'max': info.max,
                   **({} if integer else {'eps': info.eps, 'epsneg': info.epsneg, 'tiny': info.tiny,
                                         'smallest_normal': info.smallest_normal,
                                         'smallest_subnormal': info.smallest_subnormal,
                                         'precision': info.precision, 'resolution': info.resolution})})


def checked_size(shape, dtype='float64'):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    if len(shape) > 64 or any(operator.index(size) < 0 for size in shape):
        raise ValueError('array dimensions must be nonnegative with at most 64 axes')
    count = math.prod(shape)
    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValueError('object dtype is not supported')
    try:
        check_memory(count * max(1, dtype.itemsize))
    except MemoryError:
        raise _ArrayMemoryError(shape, dtype) from None


def preflight_binary(name, left, right):
    if not isinstance(left, np.ndarray) and not isinstance(right, np.ndarray):
        return
    a, b = np.shape(left), np.shape(right)
    if name == 'MatMult':
        if not a or not b:
            raise ValueError('matmul inputs need at least one dimension')
        shape = (*np.broadcast_shapes(a[:-2] if len(a)>1 else (), b[:-2] if len(b)>1 else ()),
                 *((a[-2],) if len(a)>1 else ()), *((b[-1],) if len(b)>1 else ()))
    else:
        shape = np.broadcast_shapes(a, b)
    checked_size(shape, numeric_dtype(left, right))


def matmul(a, b):
    a, b = np.asarray(a), np.asarray(b)
    preflight_binary('MatMult', a, b)
    return np.matmul(a, b)


def dot(a, b):
    a, b = np.asarray(a), np.asarray(b)
    shape = (np.broadcast_shapes(a.shape, b.shape) if not a.ndim or not b.ndim else
             a.shape[:-1] if b.ndim == 1 else (*a.shape[:-1], *b.shape[:-2], b.shape[-1]))
    checked_size(shape, numeric_dtype(a, b))
    return np.dot(a, b)


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


def bounded_like(factory):
    def create(a, dtype=None, order='K', subok=True, shape=None, **kwargs):
        value = np.asarray(a)
        checked_size(value.shape if shape is None else shape, value.dtype if dtype is None else dtype)
        return factory(value, dtype=dtype, order=order, subok=subok, shape=shape, **kwargs)
    return create


def arange(start, stop=None, step=1, **kwargs):
    first, last = (0, start) if stop is None else (start, stop)
    if step == 0:
        raise ValueError('arange step must not be zero')
    checked_size((max(0, math.ceil((last - first) / step)),), kwargs.get('dtype', 'float64'))
    return np.arange(first, last, step, **kwargs)


def linspace(start, stop, num=50, **kwargs):
    checked_size((num, *np.broadcast_shapes(np.shape(start), np.shape(stop))), kwargs.get('dtype', 'float64'))
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


def fft_transform(function, inverse_real=False):
    def transform(a, n=None, axis=-1, norm=None):
        value = np.asarray(a)
        axis = operator.index(axis)
        shape = list(value.shape)
        shape[axis] = operator.index(n) if n is not None else (
            2 * (shape[axis] - 1) if inverse_real else shape[axis])
        checked_size(shape, 'complex128')
        return function(value, n=n, axis=axis, norm=norm)
    return transform


def fft_frequencies(function):
    def frequencies(n, d=1.0):
        checked_size((operator.index(n),))
        return function(n, d=d)
    return frequencies


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
    return {**select(vars(np), PURE),
            'pad': pad, 'take': take, 'finfo': machine_info, 'iinfo': lambda dtype: machine_info(dtype, True),
            'result_type': np.result_type, 'promote_types': np.promote_types, 'can_cast': np.can_cast,
            'iscomplex': np.iscomplex, 'isreal': np.isreal, 'iscomplexobj': np.iscomplexobj, 'isrealobj': np.isrealobj,
            'array': array, 'asarray': array, 'arange': arange, 'linspace': linspace,
            'mgrid': Grid(), 'ogrid': Grid(True), 'bincount': bincount,
            'indices': indices, 'broadcast_to': broadcast_to,
            'digitize': np.digitize, 'searchsorted': np.searchsorted, 'argwhere': np.argwhere,
            'zeros': bounded(np.zeros), 'ones': bounded(np.ones), 'full': bounded(np.full),
            'empty': bounded(np.zeros), 'empty_like': bounded_like(np.zeros_like),
            'zeros_like': bounded_like(np.zeros_like), 'ones_like': bounded_like(np.ones_like), 'sum': np.sum,
            'mean': np.mean, 'median': np.median, 'min': np.min, 'max': np.max,
            'std': np.std, 'var': np.var, 'dot': dot, 'matmul': matmul,
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
            **SCALARS, 'bool_': SCALARS['bool'], 'linalg': {'norm': np.linalg.norm, 'solve': np.linalg.solve,
                                      'inv': np.linalg.inv, 'det': np.linalg.det,
                                      'svd': np.linalg.svd, 'eig': np.linalg.eig, 'eigh': np.linalg.eigh,
                                      'eigvals': np.linalg.eigvals, 'eigvalsh': np.linalg.eigvalsh,
                                      'qr': np.linalg.qr, 'cholesky': np.linalg.cholesky,
                                      'pinv': np.linalg.pinv, 'lstsq': np.linalg.lstsq,
                                      'matrix_rank': np.linalg.matrix_rank, 'cond': np.linalg.cond},
            'random': {'default_rng': np.random.default_rng},
            'fft': {'fft': fft_transform(np.fft.fft), 'ifft': fft_transform(np.fft.ifft),
                    'rfft': fft_transform(np.fft.rfft), 'irfft': fft_transform(np.fft.irfft, True),
                    'fftfreq': fft_frequencies(np.fft.fftfreq),
                    'rfftfreq': fft_frequencies(np.fft.rfftfreq),
                    'fftshift': np.fft.fftshift, 'ifftshift': np.fft.ifftshift},
            **{name: Ufunc(name) for name in UFUNCS}}


def members(value, fs, name=None):
    if isinstance(value, Record):
        return value.fields
    if isinstance(value, Ufunc):
        function = UFUNCS[value.name]
        def aggregate(method):
            def invoke(a, axis=0, dtype=None, out=None, *args, **kwargs):
                checked_size(np.shape(a), numeric_dtype(a, dtype=dtype))
                return method(a, axis, dtype, out, *args, **kwargs)
            return invoke
        def outer(a, b, **kwargs):
            dtype = numeric_dtype(a, b, dtype=kwargs.get('dtype'))
            shape = (*np.shape(a), *np.shape(b))
            if 'where' in kwargs:
                shape = np.broadcast_shapes(shape, np.shape(kwargs['where']))
            checked_size(shape, dtype)
            if kwargs.get('out') is not None:
                raise ValueError('out arrays are not supported')
            if 'where' in kwargs and not np.all(kwargs['where']):
                raise ValueError('masked ufunc calls without initialized out arrays are not supported')
            return function.outer(a, b, **kwargs)
        def at(a, indices, b=None, /):
            if not isinstance(a, np.ndarray):
                raise TypeError('at requires a NumPy array')
            # Validate the indexed selection without allocating a[indices].
            advanced, basic, axis = [], [], 0
            for key in indices if isinstance(indices, tuple) else (indices,):
                if isinstance(key, slice):
                    basic.append(len(range(*key.indices(a.shape[axis]))))
                    axis += 1
                else:
                    index = np.asarray(key)
                    if index.size and index.dtype.kind not in 'iub':
                        raise ValueError('at indices must be integers, booleans or slices')
                    advanced.append((int(np.count_nonzero(index)),) if index.dtype.kind == 'b' else index.shape)
                    axis += index.ndim if index.dtype.kind == 'b' else 1
            checked_size((*np.broadcast_shapes(*advanced), *basic, *a.shape[axis:]), a.dtype)
            return function.at(a, indices) if b is None else function.at(a, indices, b)
        def reduceat(a, indices, axis=0, dtype=None):
            shape = list(np.shape(a))
            shape[operator.index(axis)] = len(indices)
            checked_size(shape, numeric_dtype(a, dtype=dtype))
            return function.reduceat(a, indices, axis=axis, dtype=dtype)
        return {'reduce': aggregate(function.reduce), 'accumulate': aggregate(function.accumulate),
                'reduceat': reduceat, 'outer': outer, 'at': at, 'nin': function.nin, 'nout': function.nout}
    if isinstance(value, np.random.Generator):
        def distribution(function, size_position):
            def generate(*args, **kwargs):
                size = kwargs.get('size', args[size_position] if len(args) > size_position else None)
                if kwargs.get('out') is not None:
                    raise ValueError('out arrays are not supported')
                if size is None:
                    params = [*args[:size_position],
                              *(v for k, v in kwargs.items() if k not in ('size', 'dtype', 'out', 'method'))]
                    size = np.broadcast_shapes(*(np.shape(v) for v in params))
                checked_size(size, kwargs.get('dtype', 'float64'))
                return function(*args, **kwargs)
            return generate
        return {'random': distribution(value.random, 0), 'normal': distribution(value.normal, 2),
                'uniform': distribution(value.uniform, 2), 'integers': distribution(value.integers, 2),
                'standard_normal': distribution(value.standard_normal, 0),
                'exponential': distribution(value.exponential, 1),
                'standard_exponential': distribution(value.standard_exponential, 0),
                'gamma': distribution(value.gamma, 2), 'beta': distribution(value.beta, 2),
                'poisson': distribution(value.poisson, 1), 'binomial': distribution(value.binomial, 2),
                'lognormal': distribution(value.lognormal, 2), 'logistic': distribution(value.logistic, 2),
                'laplace': distribution(value.laplace, 2), 'chisquare': distribution(value.chisquare, 1),
                'standard_t': distribution(value.standard_t, 1), 'weibull': distribution(value.weibull, 1),
                'geometric': distribution(value.geometric, 1), 'triangular': distribution(value.triangular, 3)}
    if isinstance(value, np.ndarray):
        return {'shape': value.shape, 'size': value.size, 'ndim': value.ndim, 'nbytes': value.nbytes,
                'tobytes': value.tobytes,
                'real': value.real, 'imag': value.imag,
                'dtype': value.dtype, 'T': value.T, 'tolist': value.tolist, 'item': value.item,
                'reshape': value.reshape, 'flatten': value.flatten, 'ravel': value.ravel,
                'astype': lambda dtype, **kwargs: array(value, dtype=dtype, **kwargs),
                'copy': value.copy, 'sum': value.sum, 'mean': value.mean,
                'repeat': lambda repeats, axis=None: repeat(value, repeats, axis),
                'squeeze': value.squeeze, 'swapaxes': value.swapaxes,
                'diagonal': value.diagonal, 'trace': value.trace,
                'prod': value.prod, 'cumsum': value.cumsum, 'cumprod': value.cumprod,
                'itemsize': value.itemsize, 'strides': value.strides, 'real': value.real,
                'take': lambda *a, **kw: take(value, *a, **kw),
                'compress': value.compress, 'searchsorted': value.searchsorted,
                'conj': value.conj, 'conjugate': value.conjugate,
                'nonzero': value.nonzero,
                'min': value.min, 'max': value.max, 'std': value.std, 'var': value.var,
                'dot': lambda other: dot(value, other), 'transpose': value.transpose, 'all': value.all, 'any': value.any,
                'round': value.round, 'clip': value.clip, 'sort': value.sort,
                'argsort': value.argsort, 'argmin': value.argmin, 'argmax': value.argmax}
    if isinstance(value, np.dtype):
        return {'name': value.name, 'kind': value.kind, 'itemsize': value.itemsize}
    return {}


def pack(value):
    if isinstance(value, type) and value in SCALARS.values():
        return descriptor('numpy', 'scalar_type', np.dtype(value).name)
    if isinstance(value, Record):
        return descriptor('numpy', 'record', freeze(value.fields))
    if isinstance(value, Ufunc):
        return descriptor('numpy', 'ufunc', value.name)
    if isinstance(value, Grid):
        return descriptor('numpy', 'grid', value.sparse)
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


def restore(kind, data, *, copy=True):
    if kind == 'scalar_type' and data in SCALARS:
        return SCALARS[data]
    if kind == 'record':
        return Record(thaw(data))
    if kind == 'ufunc' and data in UFUNCS:
        return Ufunc(data)
    if kind == 'grid' and type(data) is bool:
        return Grid(data)
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
    offset = stream.tell()
    size = math.prod(shape) * dtype.itemsize
    if len(data) - offset != size:
        raise ValueError('invalid array payload size')
    # Decode directly from the validated buffer, then make one writable copy.
    # np.load(BytesIO(...)) otherwise adds stream/read-buffer copies per RPC.
    result = np.ndarray(shape, dtype=dtype, buffer=data, offset=offset,
                        order='F' if fortran else 'C')
    return result.copy(order='K') if copy else result


def is_view(value):
    return isinstance(value, (np.ndarray, Ufunc)) or (isinstance(value, type) and value in SCALARS.values())


def is_grid(value):
    return isinstance(value, Grid)


def iterate(value):
    return iter(value)


def view(root, value):
    return None


def shares(root, value):
    return isinstance(root, np.ndarray) and isinstance(value, np.ndarray) and np.shares_memory(root, value)


def assign(value, name, item):
    raise AttributeError('Assign array elements with indexing, not attributes')
