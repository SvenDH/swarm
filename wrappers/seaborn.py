"""Axes-level Seaborn plots, recorded into isolated Matplotlib figures."""
import seaborn as sns
from inspect import signature
from .matplotlib import Chart, Axis
from .common import is_view, iterate, assign

FUNCTIONS = {'scatterplot': sns.scatterplot, 'lineplot': sns.lineplot,
             'histplot': sns.histplot, 'boxplot': sns.boxplot, 'barplot': sns.barplot,
             'heatmap': sns.heatmap, 'violinplot': sns.violinplot, 'kdeplot': sns.kdeplot,
             'stripplot': sns.stripplot, 'swarmplot': sns.swarmplot, 'pointplot': sns.pointplot,
             'countplot': sns.countplot, 'boxenplot': sns.boxenplot, 'ecdfplot': sns.ecdfplot,
             'rugplot': sns.rugplot, 'regplot': sns.regplot, 'residplot': sns.residplot,
             'despine': sns.despine}
PALETTES = {'color_palette': sns.color_palette, 'light_palette': sns.light_palette,
            'dark_palette': sns.dark_palette, 'diverging_palette': sns.diverging_palette,
            'cubehelix_palette': sns.cubehelix_palette, 'blend_palette': sns.blend_palette,
            'hls_palette': sns.hls_palette, 'husl_palette': sns.husl_palette}


def plot(name, *args, ax=None, **kwargs):
    if ax is None:
        ax = Axis(Chart())
    if not isinstance(ax, Axis):
        raise TypeError('ax must be a wrapped matplotlib axes')
    if name in ('lineplot', 'barplot'):
        # Avoid nondeterministic bootstrap resampling on each render/restart.
        kwargs.setdefault('errorbar', None)
    ax.chart.add(ax.index, 'sns:' + name, args, kwargs)
    return ax


def exports(fs):
    def palette(function):
        def invoke(*args, **kwargs):
            bound = signature(function).bind(*args, **kwargs)
            if bound.arguments.get('as_cmap'):
                raise ValueError('Use palette colors or a matplotlib colormap name; native colormap objects are not exposed')
            return list(function(*args, **kwargs))
        return invoke
    return {**{name: (lambda *a, _name=name, **kw: plot(_name, *a, **kw)) for name in FUNCTIONS},
            **{name: palette(function) for name, function in PALETTES.items()}}


def members(value, fs, name=None):
    return {}
