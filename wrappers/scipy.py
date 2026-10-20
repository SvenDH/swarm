"""Pure numeric SciPy APIs; Python callbacks and native file IO are excluded."""
from scipy import stats, linalg, signal, fft, interpolate, optimize, special, integrate, constants, ndimage
import numpy as np
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
from inspect import signature
import operator
from .common import Record, iterate, assign
from .api import select

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
DISTRIBUTIONS.update(select(vars(stats), '''gamma lognorm loggamma weibull_min weibull_max
    exponnorm exponweib rayleigh logistic laplace cauchy halfnorm halfcauchy pareto powerlaw
    triang truncnorm invgamma invgauss f nct ncx2 ncf genextreme gumbel_r gumbel_l
    bernoulli geom nbinom hypergeom randint skellam betabinom zipf logser'''))
DISCRETE = {'binom', 'poisson', 'bernoulli', 'geom', 'nbinom', 'hypergeom', 'randint',
            'skellam', 'betabinom', 'zipf', 'logser'}


def zoom(input, zoom, **kwargs):
    from .numpy import checked_size
    value = np.asarray(input)
    factors = np.broadcast_to(zoom, (value.ndim,))
    if not np.all(np.isfinite(factors)) or np.any(factors < 0):
        raise ValueError('zoom factors must be finite and nonnegative')
    checked_size(tuple(round(n * float(f)) for n, f in zip(value.shape, factors)), value.dtype)
    return ndimage.zoom(value, zoom, **kwargs)


def numeric_ufunc(function):
    def invoke(*args, **kwargs):
        from .numpy import ufunc_call
        return ufunc_call(function, *args, **kwargs)
    return invoke


def square_factory(function):
    def invoke(n, *args, **kwargs):
        from .numpy import checked_size
        checked_size((operator.index(n),) * 2, 'complex128')
        return function(n, *args, **kwargs)
    return invoke


def window(function, position=0):
    def invoke(*args, **kwargs):
        from .numpy import checked_size
        size = args[position] if len(args) > position else kwargs['Nx' if position else 'M']
        checked_size((operator.index(size),))
        return function(*args, **kwargs)
    return invoke


def convolution(function):
    def invoke(in1, in2, *args, **kwargs):
        from .numpy import checked_size
        a, b = np.asarray(in1), np.asarray(in2)
        if a.ndim != b.ndim:
            raise ValueError('convolution inputs need the same number of axes')
        mode = args[0] if args else kwargs.get('mode', 'full')
        shape = a.shape if mode == 'same' else tuple(
            abs(x-y)+1 if mode == 'valid' else x+y-1 for x, y in zip(a.shape, b.shape))
        checked_size(shape, 'complex128')
        return function(in1, in2, *args, **kwargs)
    return invoke


def image_filter(function):
    def invoke(*args, **kwargs):
        from .numpy import checked_size
        bound = signature(function).bind(*args, **kwargs)
        bound.apply_defaults()
        options = {**bound.arguments, **bound.arguments.get('kwargs', {})}
        if 'sigma' in options:
            sigma = np.asarray(options['sigma'])
            truncate = options.get('truncate', 4.)
            if not np.all(np.isfinite(sigma)) or np.any(sigma < 0) or not np.isfinite(truncate) or truncate < 0:
                raise ValueError('filter sigma/truncate must be finite and nonnegative')
            radius = options.get('radius')
            radius = np.asarray(radius) if radius is not None else np.floor(sigma * truncate + 0.5)
            if not np.all(np.isfinite(radius)) or np.any(radius < 0):
                raise ValueError('filter radius must be finite and nonnegative')
            checked_size((int(np.max(radius)) * 2 + 1,))
        if 'size' in options and options['size'] is not None:
            ndim = np.ndim(args[0] if args else options['input'])
            size = np.broadcast_to(options['size'], (ndim,))
            checked_size(tuple(operator.index(x) for x in size), 'bool')
        return function(*args, **kwargs)
    return invoke


def test(function):
    def invoke(*args, **kwargs):
        result = function(*args, **kwargs)
        return Record({'statistic': result.statistic, 'pvalue': result.pvalue})
    return invoke


def exports(fs):
    return {'constants': {**{name: value for name, value in vars(constants).items()
                             if not name.startswith('_') and type(value) in (int, float)},
                          'physical_constants': dict(constants.physical_constants),
                          **select(vars(constants), 'value unit precision convert_temperature lambda2nu nu2lambda')},
        'ndimage': {**{name: image_filter(function) for name, function in select(vars(ndimage), '''gaussian_filter gaussian_filter1d gaussian_laplace
            gaussian_gradient_magnitude uniform_filter uniform_filter1d median_filter minimum_filter
            maximum_filter percentile_filter rank_filter''').items()},
            **select(vars(ndimage), '''sobel prewitt laplace convolve correlate
            convolve1d correlate1d binary_dilation binary_erosion binary_opening binary_closing
            binary_fill_holes binary_hit_or_miss grey_dilation grey_erosion grey_opening grey_closing
            label find_objects center_of_mass sum_labels mean variance standard_deviation
            distance_transform_edt distance_transform_cdt
            shift spline_filter map_coordinates'''), 'zoom': zoom},
        'stats': {
        **select(vars(stats), '''variation moment median_abs_deviation tmean tvar tstd tsem
            trim_mean trimboth trim1 percentileofscore scoreatpercentile rankdata tiecorrect
            hmean gmean pmean differential_entropy wasserstein_distance energy_distance
            circmean circstd circvar combine_pvalues fisher_exact kendalltau f_oneway kruskal
            friedmanchisquare levene bartlett fligner anderson_ksamp'''),
        **{name: Distribution(name) for name in DISTRIBUTIONS},
        'ttest_ind': test(stats.ttest_ind), 'ttest_1samp': test(stats.ttest_1samp),
        'ttest_rel': test(stats.ttest_rel), 'pearsonr': test(stats.pearsonr),
        'spearmanr': test(stats.spearmanr), 'mannwhitneyu': test(stats.mannwhitneyu),
        'chi2_contingency': stats.chi2_contingency, 'zscore': stats.zscore,
        'skew': stats.skew, 'kurtosis': stats.kurtosis, 'sem': stats.sem,
        'iqr': stats.iqr, 'rankdata': stats.rankdata, 'entropy': stats.entropy,
        'gmean': stats.gmean, 'hmean': stats.hmean,
        'shapiro': test(stats.shapiro), 'normaltest': test(stats.normaltest),
        'ks_2samp': test(stats.ks_2samp), 'wilcoxon': test(stats.wilcoxon),
        'describe': lambda a, **kw: Record(dict(zip(
            ('nobs', 'minmax', 'mean', 'variance', 'skewness', 'kurtosis'), stats.describe(a, **kw)))),
        'linregress': lambda *a, **kw: regression(*a, **kw)},
        'special': {**{name: numeric_ufunc(function) for name, function in vars(special).items()
                       if not name.startswith('_') and isinstance(function, np.ufunc) and function.signature is None},
                    'softmax': special.softmax, 'logsumexp': special.logsumexp,
                    'factorial': special.factorial, 'comb': special.comb},
        'linalg': {**select(vars(linalg), '''pinv pinvh qr rq lu lu_factor lu_solve cholesky cho_factor
            cho_solve solve_triangular solve_banded eigvals eigvalsh svdvals diagsvd orth null_space
            companion leslie det expm logm sqrtm fractional_matrix_power matrix_balance'''),
                   **{name: square_factory(function) for name, function in select(vars(linalg),
                       'hadamard hilbert invhilbert pascal invpascal dft helmert').items()},
                   'solve': linalg.solve, 'inv': linalg.inv, 'det': linalg.det,
                   'norm': linalg.norm, 'svd': linalg.svd, 'eig': linalg.eig,
                   'eigh': linalg.eigh, 'lstsq': linalg.lstsq},
        'signal': {**select(vars(signal), '''
            peak_prominences peak_widths medfilt medfilt2d wiener hilbert hilbert2 envelope
            lfilter sosfilt sosfiltfilt freqz sosfreqz welch periodogram spectrogram csd coherence
            cheby1 cheby2 ellip bessel firwin firwin2 iirnotch iirpeak
            square sawtooth chirp gausspulse'''),
                   **{name: convolution(function) for name, function in select(vars(signal),
                       'fftconvolve oaconvolve correlate2d convolve2d convolve correlate').items()},
                   'windows': {**{name: window(function) for name, function in select(vars(signal.windows), '''boxcar triang blackman hamming hann bartlett
                       flattop parzen bohman blackmanharris nuttall barthann cosine exponential tukey
                       taylor lanczos kaiser gaussian general_gaussian general_hamming''').items()},
                               'get_window': window(signal.windows.get_window, 1)},
                   'find_peaks': signal.find_peaks, 'savgol_filter': signal.savgol_filter,
                   'butter': signal.butter, 'filtfilt': signal.filtfilt,
                   'resample': signal.resample, 'detrend': signal.detrend},
        'fft': {'fft': fft.fft, 'ifft': fft.ifft, 'rfft': fft.rfft,
                'irfft': fft.irfft, 'fftfreq': fft.fftfreq, 'rfftfreq': fft.rfftfreq},
        'interpolate': {'interp1d': lambda *a, **kw: Interpolator('linear', a, kw),
                        'CubicSpline': lambda *a, **kw: Interpolator('cubic', a, kw)},
        'optimize': {'linear_sum_assignment': optimize.linear_sum_assignment,
                     'linprog': lambda *a, **kw: linear_program(*a, **kw), 'nnls': optimize.nnls},
        'integrate': {'trapezoid': integrate.trapezoid, 'simpson': integrate.simpson,
                      'cumulative_trapezoid': integrate.cumulative_trapezoid,
                      'cumulative_simpson': integrate.cumulative_simpson},
        'spatial': {'transform': {'Rotation': {
                        'from_quat': Rotation.from_quat, 'from_euler': Rotation.from_euler,
                        'from_rotvec': Rotation.from_rotvec, 'from_matrix': Rotation.from_matrix}},
                    'distance': {'cdist': distance.cdist, 'pdist': distance.pdist,
                                  'squareform': distance.squareform, 'euclidean': distance.euclidean,
                                  'cosine': distance.cosine}}}


def linear_program(*args, **kwargs):
    bound = signature(optimize.linprog).bind(*args, **kwargs)
    kwargs = bound.arguments
    args = ()
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
    if isinstance(value, Rotation):
        return {'as_quat': value.as_quat, 'as_euler': value.as_euler,
                'as_matrix': value.as_matrix, 'as_rotvec': value.as_rotvec,
                'apply': value.apply, 'inv': value.inv, 'magnitude': value.magnitude}
    if isinstance(value, Record):
        return value.fields
    if isinstance(value, Distribution):
        native = DISTRIBUTIONS[value.name](*value.args, **value.kwargs) if value.args or value.kwargs else DISTRIBUTIONS[value.name]
        def sample(*args, **kwargs):
            from .numpy import checked_size
            frozen = bool(value.args or value.kwargs)
            if frozen:
                if len(args) > 1:
                    raise ValueError('pass random_state by keyword')
                size = kwargs.get('size', args[0] if args else None)
                params = [*value.args, *value.kwargs.values()]
            else:
                maximum = DISTRIBUTIONS[value.name].numargs + (1 if value.name in DISCRETE else 2)
                if len(args) > maximum:
                    raise ValueError('pass size and random_state by keyword')
                size = kwargs.get('size')
                params = [*args, *(v for k, v in kwargs.items() if k not in ('size', 'random_state'))]
            if size is None:
                size = np.broadcast_shapes(*(np.shape(v) for v in params))
            checked_size(size)
            kwargs.setdefault('random_state', 0)
            return native.rvs(*args, **kwargs)
        return {'rvs': sample, 'pdf': native.pdf, 'cdf': native.cdf, 'ppf': native.ppf,
                'mean': native.mean, 'var': native.var, 'std': native.std,
                'sf': native.sf, 'logsf': native.logsf, 'logcdf': native.logcdf, 'isf': native.isf,
                'stats': native.stats, 'moment': native.moment, 'entropy': native.entropy,
                'interval': native.interval, 'logpdf': native.logpdf} if value.name not in DISCRETE else {
                    'rvs': sample, 'pmf': native.pmf, 'cdf': native.cdf, 'ppf': native.ppf,
                    'logpmf': native.logpmf, 'logcdf': native.logcdf, 'logsf': native.logsf,
                    'isf': native.isf, 'stats': native.stats, 'moment': native.moment,
                    'entropy': native.entropy, 'interval': native.interval,
                    'mean': native.mean, 'var': native.var, 'std': native.std, 'sf': native.sf}
    return {}


def is_view(value):
    return isinstance(value, (Distribution, Interpolator, Rotation))
