"""Axes-level Seaborn plots, recorded into isolated Matplotlib figures."""
import seaborn as sns
from .matplotlib import Chart, Axis
from .common import view, is_view, iterate, assign

FUNCTIONS = {'scatterplot': sns.scatterplot, 'lineplot': sns.lineplot,
             'histplot': sns.histplot, 'boxplot': sns.boxplot, 'barplot': sns.barplot,
             'heatmap': sns.heatmap, 'violinplot': sns.violinplot, 'kdeplot': sns.kdeplot}


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
    return {name: (lambda *a, _name=name, **kw: plot(_name, *a, **kw)) for name in FUNCTIONS}


def members(value, fs, name=None):
    return {}


def pack(value):
    return None  # The bridge encodes returned axes using the matplotlib adapter.


def restore(kind, data):
    raise ValueError('Seaborn returns matplotlib axes, not standalone Seaborn objects')
