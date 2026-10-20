"""Shared adapter defaults and RAM-only document commands."""
from .bridge import PRIMITIVES
from .limits import check_memory


def is_view(value):
    return False


def iterate(value):
    return iter(value)


def assign(value, name, item):
    raise AttributeError('Attribute is read-only')


class Adapter:
    is_view = staticmethod(is_view)
    iterate = staticmethod(iterate)
    assign = staticmethod(assign)


def snapshot(value):
    """Capture command data without encoding it or retaining mutable aliases."""
    if isinstance(value, PRIMITIVES):
        return value
    if type(value) in (list, tuple):
        return type(value)(snapshot(item) for item in value)
    if type(value) is dict:
        return {snapshot(key): snapshot(item) for key, item in value.items()}
    from .bridge import backend
    from .registry import available
    if 'numpy' in available():
        np = backend('numpy').np
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.dtype.hasobject and any(not isinstance(item, PRIMITIVES) for item in value.flat):
                raise TypeError('Object arrays may contain only primitive category labels')
            check_memory(value.nbytes)
            return value.copy()
    if 'pandas' in available():
        pd = backend('pandas').pd
        if isinstance(value, (pd.DataFrame, pd.Series, pd.Index, pd.Categorical)):
            return value.copy()
    raise TypeError(f'Unsupported command data: {type(value).__name__}')


def record(commands, name, args, kwargs):
    if len(commands) >= 1000:
        raise ValueError('Documents/figures are limited to 1,000 recorded operations')
    commands.append([name, snapshot(args), snapshot(kwargs)])


class Record:
    """Named numeric results exposed through reviewed member maps."""
    def __init__(self, fields):
        self.fields = fields

    def __repr__(self):
        return repr(self.fields)
