"""Pure numeric SciPy APIs; Python callbacks and native file IO are excluded."""
from scipy import stats, linalg, signal, fft, interpolate, optimize
from scipy.spatial import distance
from .bridge import descriptor
from .common import Record, freeze, thaw, view, iterate, assign

# Initialize PocketFFT on the Python IO worker during the first requested import.
# First-time initialization on Monty's foreign callback thread can segfault.
fft.fft([0.0])


class Distribution:
    def __init__(self, name, args=(), kwargs=None):
        self.name, self.args, self.kwargs = name, args, kwargs or {}

    def __call__(self, *args, **kwargs):
        return Distribution(self.name, args, kwargs)


DISTRIBUTIONS = {'norm': stats.norm, 'uniform': stats.uniform, 'binom': stats.binom,
                 'poisson': stats.poisson, 't': stats.t, 'chi2': stats.chi2,
                 'expon': stats.expon, 'beta': stats.beta}


def test(function):
    def invoke(*args, **kwargs):
        result = function(*args, **kwargs)
        return Record({'statistic': result.statistic, 'pvalue': result.pvalue})
    return invoke


def exports(fs):
    return {'stats': {
        **{name: Distribution(name) for name in DISTRIBUTIONS},
        'ttest_ind': test(stats.ttest_ind), 'ttest_1samp': test(stats.ttest_1samp),
        'ttest_rel': test(stats.ttest_rel), 'pearsonr': test(stats.pearsonr),
        'spearmanr': test(stats.spearmanr), 'mannwhitneyu': test(stats.mannwhitneyu),
        'chi2_contingency': stats.chi2_contingency, 'zscore': stats.zscore,
        'describe': lambda a, **kw: Record(dict(zip(
            ('nobs', 'minmax', 'mean', 'variance', 'skewness', 'kurtosis'), stats.describe(a, **kw)))),
        'linregress': lambda *a, **kw: regression(*a, **kw)},
        'linalg': {'solve': linalg.solve, 'inv': linalg.inv, 'det': linalg.det,
                   'norm': linalg.norm, 'svd': linalg.svd, 'eig': linalg.eig,
                   'eigh': linalg.eigh, 'lstsq': linalg.lstsq},
        'signal': {'find_peaks': signal.find_peaks, 'convolve': signal.convolve,
                   'correlate': signal.correlate, 'savgol_filter': signal.savgol_filter,
                   'butter': signal.butter, 'filtfilt': signal.filtfilt,
                   'resample': signal.resample, 'detrend': signal.detrend},
        'fft': {'fft': fft.fft, 'ifft': fft.ifft, 'rfft': fft.rfft,
                'irfft': fft.irfft, 'fftfreq': fft.fftfreq, 'rfftfreq': fft.rfftfreq},
        'interpolate': {'interp1d': lambda *a, **kw: Interpolator('linear', a, kw),
                        'CubicSpline': lambda *a, **kw: Interpolator('cubic', a, kw)},
        'optimize': {'linear_sum_assignment': optimize.linear_sum_assignment,
                     'linprog': lambda *a, **kw: linear_program(*a, **kw), 'nnls': optimize.nnls},
        'spatial': {'distance': {'cdist': distance.cdist, 'pdist': distance.pdist,
                                  'squareform': distance.squareform, 'euclidean': distance.euclidean,
                                  'cosine': distance.cosine}}}


def linear_program(*args, **kwargs):
    if kwargs.get('callback') is not None:
        raise ValueError('Python callbacks are not supported')
    if kwargs.get('method', 'highs') not in ('highs', 'highs-ds', 'highs-ipm'):
        raise ValueError('Only HiGHS linear programming methods are exposed')
    options = kwargs.get('options') or {}
    if set(options) - {'maxiter', 'presolve', 'time_limit', 'dual_feasibility_tolerance',
                        'primal_feasibility_tolerance', 'ipm_optimality_tolerance',
                        'simplex_dual_edge_weight_strategy', 'mip_rel_gap', 'mip_max_nodes'}:
        raise ValueError('Unsupported solver options (native log/file output is not exposed)')
    result = optimize.linprog(*args, **kwargs)
    return Record({'x': result.x, 'fun': result.fun, 'success': result.success,
                   'message': result.message, 'status': result.status, 'nit': result.nit})


def regression(*args, **kwargs):
    r = stats.linregress(*args, **kwargs)
    return Record({'slope': r.slope, 'intercept': r.intercept, 'rvalue': r.rvalue,
                   'pvalue': r.pvalue, 'stderr': r.stderr, 'intercept_stderr': r.intercept_stderr})


class Interpolator:
    def __init__(self, kind, args, kwargs):
        self.kind, self.args, self.kwargs = kind, args, kwargs

    def __call__(self, x, **kwargs):
        factory = interpolate.interp1d if self.kind == 'linear' else interpolate.CubicSpline
        return factory(*self.args, **self.kwargs)(x, **kwargs)


def members(value, fs, name=None):
    if isinstance(value, Record):
        return value.fields
    if isinstance(value, Distribution):
        native = DISTRIBUTIONS[value.name](*value.args, **value.kwargs) if value.args or value.kwargs else DISTRIBUTIONS[value.name]
        def sample(*args, **kwargs):
            from .numpy import checked_size
            size = kwargs.get('size', ())
            checked_size(() if size is None else size)
            kwargs.setdefault('random_state', 0)
            return native.rvs(*args, **kwargs)
        return {'rvs': sample, 'pdf': native.pdf, 'cdf': native.cdf, 'ppf': native.ppf,
                'mean': native.mean, 'var': native.var, 'std': native.std,
                'sf': native.sf, 'logpdf': native.logpdf} if value.name not in ('binom', 'poisson') else {
                    'rvs': sample, 'pmf': native.pmf, 'cdf': native.cdf, 'ppf': native.ppf,
                    'mean': native.mean, 'var': native.var, 'std': native.std, 'sf': native.sf}
    return {}


def pack(value):
    if isinstance(value, Record):
        return descriptor('scipy', 'record', freeze(value.fields))
    if isinstance(value, Distribution):
        return descriptor('scipy', 'distribution', freeze([value.name, value.args, value.kwargs]))
    if isinstance(value, Interpolator):
        return descriptor('scipy', 'interpolator', freeze([value.kind, value.args, value.kwargs]))


def restore(kind, data):
    if kind == 'record':
        return Record(thaw(data))
    if kind == 'distribution':
        return Distribution(*thaw(data))
    if kind == 'interpolator':
        return Interpolator(*thaw(data))
    raise ValueError('Unsupported SciPy object')


def is_view(value):
    return isinstance(value, (Distribution, Interpolator))
