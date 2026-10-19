"""Session-local plotting recipes. No pyplot globals, GUI, pickle or host paths."""
import threading
import math
from pathlib import PurePosixPath
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import colors, colormaps
from .bridge import descriptor
from .common import freeze, thaw, record, iterate, assign
from .files import save
from .limits import check_memory

LOCK = threading.RLock()  # Matplotlib font/render internals are not thread safe.
PLOTS = {'plot', 'scatter', 'bar', 'barh', 'hist', 'imshow', 'pie', 'boxplot',
         'fill_between', 'errorbar', 'axhline', 'axvline', 'step', 'stairs',
         'stackplot', 'fill', 'fill_betweenx', 'hlines', 'vlines', 'axhspan', 'axvspan',
         'semilogx', 'semilogy', 'loglog', 'contour', 'contourf', 'pcolormesh',
         'text', 'annotate'}
LABELS = {'set_title', 'set_xlabel', 'set_ylabel', 'set_xlim', 'set_ylim',
          'set_xticks', 'set_yticks', 'set_xticklabels', 'set_yticklabels',
          'legend', 'grid', 'set_xscale', 'set_yscale', 'axis', 'set_aspect',
          'invert_xaxis', 'invert_yaxis', 'tick_params', 'minorticks_on',
          'minorticks_off', 'margins', 'set_axis_off', 'set_axis_on',
          'set_facecolor', 'set_position', 'set_anchor', 'set_box_aspect',
          'set_xmargin', 'set_ymargin', 'set_autoscale_on', 'set_autoscalex_on', 'set_autoscaley_on'}
FIGURE = {'suptitle', 'supxlabel', 'supylabel', 'tight_layout', 'subplots_adjust', 'text',
          'legend', 'set_facecolor', 'set_edgecolor'}
CM = {name: (lambda *a, _name=name, **kw: colormaps[_name](*a, **kw)) for name in colormaps}


class Chart:
    def __init__(self, nrows=1, ncols=1, figsize=(6.4, 4.8), dpi=100, commands=None):
        if ((nrows, ncols) != (0, 0) and (nrows * ncols > 64 or min(nrows, ncols) < 1)) or any(
                not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0 for x in (*figsize, dpi)):
            raise ValueError('Figures need 1..64 axes and finite positive dimensions/dpi')
        check_memory(math.ceil(figsize[0] * dpi) * math.ceil(figsize[1] * dpi) * 4)
        self.nrows, self.ncols, self.figsize, self.dpi = nrows, ncols, figsize, dpi
        self.commands = commands or []

    def data(self):
        return [self.nrows, self.ncols, self.figsize, self.dpi, self.commands]

    def axes_count(self):
        return self.nrows * self.ncols + sum(name == 'add_axes' for (_, name), _, _ in self.commands)

    def axis(self):
        if not self.axes_count():
            self.nrows = self.ncols = 1
        return Axis(self, self.axes_count() - 1)

    def add_axes(self, rect, **kwargs):
        if (len(rect) != 4 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in rect)
                or not all(0 <= x <= 1 for x in rect) or min(rect[2:]) <= 0):
            raise ValueError('axes rectangle needs finite figure coordinates in 0..1 and positive dimensions')
        if self.axes_count() >= 64:
            raise ValueError('Figures are limited to 64 axes')
        if set(kwargs) - {'facecolor', 'frameon', 'label'}:
            raise ValueError('Unsupported add_axes options')
        axis = Axis(self, self.axes_count())
        self.add(None, 'add_axes', (tuple(rect),), kwargs)
        return axis

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
        check(args)
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
        chart = Chart(0, 0, figsize=figsize, dpi=dpi)
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
    if set(kwargs) - {'format', 'dpi', 'bbox_inches', 'transparent', 'facecolor', 'pil_kwargs'}:
        raise ValueError('Unsupported savefig options')
    pillow = kwargs.get('pil_kwargs', {})
    if type(pillow) is not dict or set(pillow) - {'compress_level', 'optimize', 'quality', 'progressive', 'subsampling'}:
        raise ValueError('Unsupported pil_kwargs; only compression/quality settings are exposed')
    for name, value in pillow.items():
        low, high = {'compress_level': (0, 9), 'optimize': (0, 1), 'quality': (1, 100),
                     'progressive': (0, 1), 'subsampling': (0, 2)}[name]
        if not isinstance(value, int) or not low <= value <= high:
            raise ValueError('Invalid pil_kwargs setting: ' + name)
    fmt = kwargs.pop('format', None) or PurePosixPath(str(filename)).suffix.lstrip('.') or 'png'
    if fmt not in ('png', 'svg', 'pdf'):
        raise ValueError('Figures support PNG, SVG and PDF')
    dpi = kwargs.get('dpi', chart.dpi)
    if not isinstance(dpi, (int, float)) or not math.isfinite(dpi) or dpi <= 0:
        raise ValueError('dpi must be finite and positive')
    check_memory(math.ceil(chart.figsize[0] * dpi) * math.ceil(chart.figsize[1] * dpi) * 4)
    with LOCK:
        figure = Figure(figsize=chart.figsize, dpi=chart.dpi)
        FigureCanvasAgg(figure)
        axes = [figure.add_subplot(chart.nrows, chart.ncols, i + 1) for i in range(chart.nrows * chart.ncols)]
        for (index, name), args, options in chart.commands:
            args, options = thaw(args), thaw(options)
            if name == 'add_axes' and index is None:
                axes.append(figure.add_axes(*args, **options))
                continue
            target = axes[index] if index is not None else figure
            if name.startswith('sns:'):
                from .seaborn import FUNCTIONS
                FUNCTIONS[name[4:]](*args, ax=target, **options)
            else:
                functions = axis_functions(target) if index is not None else {
                    'suptitle': figure.suptitle, 'tight_layout': figure.tight_layout,
                    'supxlabel': figure.supxlabel, 'supylabel': figure.supylabel,
                    'subplots_adjust': figure.subplots_adjust, 'text': figure.text,
                    'legend': figure.legend, 'set_facecolor': figure.set_facecolor,
                    'set_edgecolor': figure.set_edgecolor}
                functions[name](*args, **options)
        try:
            return save(fs, filename, lambda stream: figure.savefig(stream, format=fmt, **kwargs))
        finally:
            figure.clear()


def axis_functions(ax):
    return {'plot': ax.plot, 'scatter': ax.scatter, 'bar': ax.bar, 'barh': ax.barh,
            'hist': ax.hist, 'imshow': ax.imshow, 'pie': ax.pie, 'boxplot': ax.boxplot,
            'fill_between': ax.fill_between, 'errorbar': ax.errorbar,
            'step': ax.step, 'stairs': ax.stairs, 'stackplot': ax.stackplot,
            'fill': ax.fill, 'fill_betweenx': ax.fill_betweenx,
            'hlines': ax.hlines, 'vlines': ax.vlines, 'axhspan': ax.axhspan, 'axvspan': ax.axvspan,
            'semilogx': ax.semilogx, 'semilogy': ax.semilogy, 'loglog': ax.loglog,
            'contour': ax.contour, 'contourf': ax.contourf, 'pcolormesh': ax.pcolormesh,
            'text': ax.text, 'annotate': ax.annotate, 'axis': ax.axis,
            'set_aspect': ax.set_aspect, 'invert_xaxis': ax.invert_xaxis, 'invert_yaxis': ax.invert_yaxis,
            'tick_params': ax.tick_params, 'minorticks_on': ax.minorticks_on,
            'minorticks_off': ax.minorticks_off, 'margins': ax.margins,
            'set_axis_off': ax.set_axis_off, 'set_axis_on': ax.set_axis_on,
            'axhline': ax.axhline, 'axvline': ax.axvline, 'set_title': ax.set_title,
            'set_xlabel': ax.set_xlabel, 'set_ylabel': ax.set_ylabel,
            'set_xlim': ax.set_xlim, 'set_ylim': ax.set_ylim,
            'set_xticks': ax.set_xticks, 'set_yticks': ax.set_yticks,
            'set_xticklabels': ax.set_xticklabels, 'set_yticklabels': ax.set_yticklabels,
            'legend': ax.legend, 'grid': ax.grid, 'set_xscale': ax.set_xscale, 'set_yscale': ax.set_yscale,
            'set_facecolor': ax.set_facecolor, 'set_position': ax.set_position, 'set_anchor': ax.set_anchor,
            'set_box_aspect': ax.set_box_aspect, 'set_xmargin': ax.set_xmargin, 'set_ymargin': ax.set_ymargin,
            'set_autoscale_on': ax.set_autoscale_on, 'set_autoscalex_on': ax.set_autoscalex_on,
            'set_autoscaley_on': ax.set_autoscaley_on}


def use(backend, *, force=True):
    """Compatibility only: figures always use our isolated Agg canvas."""
    if not isinstance(backend, str) or backend.lower() != 'agg':
        raise ValueError('Only the headless Agg backend is exposed')


def members(value, fs, name=None):
    if isinstance(value, Session):
        if name == 'cm':
            return {'cm': dict(CM)}
        def draw(method):
            def invoke(*args, **kwargs):
                axis = value.chart().axis()
                axis.chart.add(axis.index, method, args, kwargs)
            return invoke
        return {'pyplot': value, 'figures': value.figures, 'figure': value.figure,
                'use': use, 'get_backend': lambda: 'Agg', 'is_interactive': lambda: False,
                'colors': {'to_rgb': colors.to_rgb, 'to_rgba': colors.to_rgba,
                           'to_hex': colors.to_hex, 'is_color_like': colors.is_color_like,
                           'to_rgba_array': colors.to_rgba_array, 'same_color': colors.same_color,
                           'CSS4_COLORS': dict(colors.CSS4_COLORS),
                           'TABLEAU_COLORS': dict(colors.TABLEAU_COLORS), 'BASE_COLORS': dict(colors.BASE_COLORS),
                           'rgb_to_hsv': colors.rgb_to_hsv, 'hsv_to_rgb': colors.hsv_to_rgb},
                'subplots': value.subplots, 'gcf': value.chart,
                'gca': lambda: value.chart().axis(), 'show': lambda: None,
                'close': lambda *args: None,
                'savefig': lambda filename, **kw: render(value.chart(), fs, filename, **kw),
                **{method: draw(method) for method in PLOTS | {'legend', 'grid', 'axis', 'tick_params', 'margins', 'minorticks_on', 'minorticks_off'}},
                **{method: draw('set_' + method) for method in ('title', 'xlabel', 'ylabel', 'xlim', 'ylim', 'xticks', 'yticks', 'xscale', 'yscale')},
                **{method: record_method(value.chart, None, method) for method in FIGURE - {'text', 'legend'}}}
    if isinstance(value, Chart):
        return {'axes': [Axis(value, i) for i in range(value.axes_count())], 'add_axes': value.add_axes,
                'savefig': lambda filename, **kw: render(value, fs, filename, **kw),
                **{method: record_method(lambda: value, None, method) for method in FIGURE}}
    if isinstance(value, Axis):
        return {'figure': value.chart,
                **{method: record_method(lambda: value.chart, value.index, method)
                   for method in PLOTS | LABELS},
                'set': lambda **kw: set_axis(value, kw)}
    return {}


def record_method(chart, index, method):
    def invoke(*args, **kwargs):
        return chart().add(index, method, args, kwargs)
    return invoke


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
