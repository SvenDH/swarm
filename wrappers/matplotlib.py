"""Session-local plotting recipes. No pyplot globals, GUI, pickle or host paths."""
import threading
from pathlib import PurePosixPath
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .bridge import descriptor
from .common import freeze, thaw, record, iterate, assign
from .files import save

LOCK = threading.RLock()  # Matplotlib font/render internals are not thread safe.
PLOTS = {'plot', 'scatter', 'bar', 'barh', 'hist', 'imshow', 'pie', 'boxplot',
         'fill_between', 'errorbar', 'axhline', 'axvline'}
LABELS = {'set_title', 'set_xlabel', 'set_ylabel', 'set_xlim', 'set_ylim',
          'set_xticks', 'set_yticks', 'set_xticklabels', 'set_yticklabels',
          'legend', 'grid', 'set_xscale', 'set_yscale'}


class Chart:
    def __init__(self, nrows=1, ncols=1, figsize=(6.4, 4.8), dpi=100, commands=None):
        if nrows * ncols > 64 or min(nrows, ncols) < 1 or max(figsize) > 32 or min(figsize) <= 0 or not 1 <= dpi <= 300:
            raise ValueError('Figures need 1..64 axes, dimensions <=32 inches and dpi <=300')
        self.nrows, self.ncols, self.figsize, self.dpi = nrows, ncols, figsize, dpi
        self.commands = commands or []

    def data(self):
        return [self.nrows, self.ncols, self.figsize, self.dpi, self.commands]

    def add(self, index, name, args, kwargs):
        def check(value):
            if isinstance(value, dict):
                if set(value) & {'usetex', 'fontproperties', 'fname', 'url'}:
                    raise ValueError('TeX, host font files and URLs are not exposed')
                for item in value.values():
                    check(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    check(item)
        check(kwargs)
        record(self.commands, [index, name], args, kwargs)


class Axis:
    def __init__(self, chart, index=0):
        self.chart, self.index = chart, index


class Session:
    def __init__(self, figures=None, current=0):
        self.figures, self.current = figures or [], current

    def chart(self):
        if not self.figures:
            self.figures.append(Chart())
        return self.figures[self.current]

    def figure(self, figsize=(6.4, 4.8), dpi=100):
        if len(self.figures) >= 128:
            raise ValueError('Plotting sessions are limited to 128 figures')
        chart = Chart(figsize=figsize, dpi=dpi)
        self.figures.append(chart)
        self.current = len(self.figures) - 1
        return chart

    def subplots(self, nrows=1, ncols=1, *, figsize=(6.4, 4.8), dpi=100, squeeze=True):
        if len(self.figures) >= 128:
            raise ValueError('Plotting sessions are limited to 128 figures')
        chart = Chart(nrows, ncols, figsize, dpi)
        self.figures.append(chart)
        self.current = len(self.figures) - 1
        axes = [Axis(chart, i) for i in range(nrows * ncols)]
        if squeeze and nrows * ncols == 1:
            result = axes[0]
        elif squeeze and (nrows == 1 or ncols == 1):
            result = axes
        else:
            result = [axes[i * ncols:(i + 1) * ncols] for i in range(nrows)]
        return chart, result


def exports(fs):
    return Session()


def render(chart, fs, filename, **kwargs):
    if set(kwargs) - {'format', 'dpi', 'bbox_inches', 'transparent', 'facecolor'}:
        raise ValueError('Unsupported savefig options')
    fmt = kwargs.pop('format', None) or PurePosixPath(str(filename)).suffix.lstrip('.') or 'png'
    if fmt not in ('png', 'svg', 'pdf'):
        raise ValueError('Figures support PNG, SVG and PDF')
    dpi = kwargs.get('dpi', chart.dpi)
    if not isinstance(dpi, (int, float)) or not 1 <= dpi <= 300 or chart.figsize[0] * chart.figsize[1] * dpi**2 > 4_000_000:
        raise ValueError('Figures are limited to 4,000,000 pixels and dpi <=300')
    with LOCK:
        figure = Figure(figsize=chart.figsize, dpi=chart.dpi)
        FigureCanvasAgg(figure)
        axes = [figure.add_subplot(chart.nrows, chart.ncols, i + 1) for i in range(chart.nrows * chart.ncols)]
        for (index, name), args, options in chart.commands:
            args, options = thaw(args), thaw(options)
            target = axes[index] if index is not None else figure
            if name.startswith('sns:'):
                from .seaborn import FUNCTIONS
                FUNCTIONS[name[4:]](*args, ax=target, **options)
            else:
                functions = axis_functions(target) if index is not None else {
                    'suptitle': figure.suptitle, 'tight_layout': figure.tight_layout}
                functions[name](*args, **options)
        try:
            return save(fs, filename, lambda stream: figure.savefig(stream, format=fmt, **kwargs))
        finally:
            figure.clear()


def axis_functions(ax):
    return {'plot': ax.plot, 'scatter': ax.scatter, 'bar': ax.bar, 'barh': ax.barh,
            'hist': ax.hist, 'imshow': ax.imshow, 'pie': ax.pie, 'boxplot': ax.boxplot,
            'fill_between': ax.fill_between, 'errorbar': ax.errorbar,
            'axhline': ax.axhline, 'axvline': ax.axvline, 'set_title': ax.set_title,
            'set_xlabel': ax.set_xlabel, 'set_ylabel': ax.set_ylabel,
            'set_xlim': ax.set_xlim, 'set_ylim': ax.set_ylim,
            'set_xticks': ax.set_xticks, 'set_yticks': ax.set_yticks,
            'set_xticklabels': ax.set_xticklabels, 'set_yticklabels': ax.set_yticklabels,
            'legend': ax.legend, 'grid': ax.grid, 'set_xscale': ax.set_xscale, 'set_yscale': ax.set_yscale}


def members(value, fs, name=None):
    if isinstance(value, Session):
        def draw(method):
            return lambda *a, **kw: value.chart().add(0, method, a, kw)
        return {'pyplot': value, 'figures': value.figures, 'figure': value.figure,
                'subplots': value.subplots, 'gcf': value.chart,
                'gca': lambda: Axis(value.chart()), 'show': lambda: None,
                'close': lambda *args: None,
                'savefig': lambda filename, **kw: render(value.chart(), fs, filename, **kw),
                **{method: draw(method) for method in PLOTS | {'legend', 'grid'}},
                **{method: draw('set_' + method) for method in ('title', 'xlabel', 'ylabel', 'xlim', 'ylim', 'xticks', 'yticks', 'xscale', 'yscale')},
                'tight_layout': lambda **kw: value.chart().add(None, 'tight_layout', (), kw)}
    if isinstance(value, Chart):
        return {'axes': [Axis(value, i) for i in range(value.nrows * value.ncols)],
                'savefig': lambda filename, **kw: render(value, fs, filename, **kw),
                'suptitle': lambda *a, **kw: value.add(None, 'suptitle', a, kw),
                'tight_layout': lambda **kw: value.add(None, 'tight_layout', (), kw)}
    if isinstance(value, Axis):
        return {'figure': value.chart,
                **{method: (lambda *a, _method=method, **kw: value.chart.add(value.index, _method, a, kw))
                   for method in PLOTS | LABELS},
                'set': lambda **kw: set_axis(value, kw)}
    return {}


def set_axis(axis, options):
    for name, value in options.items():
        if 'set_' + name not in LABELS:
            raise ValueError('Unsupported axes property: ' + name)
        axis.chart.add(axis.index, 'set_' + name, (value,), {})


def pack(value):
    if isinstance(value, Session):
        return descriptor('matplotlib', 'module', [[chart.data() for chart in value.figures], value.current])
    if isinstance(value, Chart):
        return descriptor('matplotlib', 'chart', value.data())
    if isinstance(value, Axis):
        return descriptor('matplotlib', 'axis', [value.chart.data(), value.index])


def restore(kind, data):
    if kind == 'module':
        return Session([Chart(*item) for item in data[0]], data[1])
    if kind == 'chart':
        return Chart(*data)
    if kind == 'axis':
        return Axis(Chart(*data[0]), data[1])
    raise ValueError('Unsupported plotting object')


def view(root, value):
    if isinstance(root, Session):
        chart = value.chart if isinstance(value, Axis) else value
        for i, figure in enumerate(root.figures):
            if figure is chart:
                prefix = [['attr', 'figures'], ['item', i]]
                return [*prefix, ['attr', 'axes'], ['item', value.index]] if isinstance(value, Axis) else prefix
    if isinstance(root, Chart) and isinstance(value, Axis) and value.chart is root:
        return [['attr', 'axes'], ['item', value.index]]
    if isinstance(root, Axis) and isinstance(value, Chart) and root.chart is value:
        return [['attr', 'figure']]


def is_view(value):
    return isinstance(value, (Chart, Axis, Session))
