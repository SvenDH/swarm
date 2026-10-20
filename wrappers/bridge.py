"""Reviewed member maps and arithmetic shared by the in-memory bridge."""
import operator
from importlib import import_module
from functools import lru_cache
from importlib.metadata import version
from datetime import date, time, timedelta
from .registry import MODULES, REQUIRES, STDLIB, require

PRIMITIVES = (str, bytes, bool, int, float, complex, date, time, timedelta, type(None))

OPERATORS = {name: function for name, function in (
    ('Add', operator.add), ('Sub', operator.sub), ('Mult', operator.mul),
    ('Div', operator.truediv), ('FloorDiv', operator.floordiv), ('Mod', operator.mod),
    ('Pow', operator.pow), ('MatMult', operator.matmul), ('BitAnd', operator.and_),
    ('BitOr', operator.or_), ('BitXor', operator.xor), ('LShift', operator.lshift),
    ('RShift', operator.rshift), ('Eq', operator.eq), ('NotEq', operator.ne),
    ('Lt', operator.lt), ('LtE', operator.le), ('Gt', operator.gt), ('GtE', operator.ge),
    ('USub', operator.neg), ('UAdd', operator.pos), ('Invert', operator.invert))}
INPLACE = {'Add': operator.iadd, 'Sub': operator.isub, 'Mult': operator.imul,
           'Div': operator.itruediv, 'FloorDiv': operator.ifloordiv, 'Mod': operator.imod,
           'Pow': operator.ipow, 'MatMult': operator.imatmul, 'BitAnd': operator.iand,
           'BitOr': operator.ior, 'BitXor': operator.ixor, 'LShift': operator.ilshift,
           'RShift': operator.irshift}


class TypeRef:
    """An identification token, not a callable host class."""
    def __init__(self, name, cls):
        self.name, self.cls = name, cls


def backend(module):
    if module not in MODULES:
        raise ImportError(f'Unsupported library: {module}')
    require(module)
    return _backend(module)


@lru_cache(maxsize=None)
def _backend(module):
    # Native initialization (including cross-library result adapters) happens
    # with the requested backend on the Python IO worker, never during replay.
    for dependency in REQUIRES.get(module, ()):
        backend(dependency)
    api = import_module('.' + MODULES[module], 'wrappers')
    return api.Adapter(module) if module in STDLIB else api


def member(value, name, module, fs):
    if isinstance(value, TypeRef) and name in ('__name__', '__module__'):
        return value.name if name == '__name__' else value.cls.__module__
    if name == '__version__' and module not in STDLIB and (type(value) is dict or
            (module == 'numpy' and isinstance(value, backend(module).Session))):
        distribution = {'PIL': 'Pillow', 'docx': 'python-docx', 'pptx': 'python-pptx',
                        'bs4': 'beautifulsoup4', 'sklearn': 'scikit-learn'}.get(module, module)
        return version(distribution)
    if name.startswith('_'):
        raise AttributeError(f"module '{module}' has no attribute '{name}'")
    members = value if type(value) is dict else backend(module).members(value, fs, name)
    if name not in members:
        if type(value) is dict:
            raise AttributeError(f"module '{module}' has no attribute '{name}'")
        raise AttributeError(f"'{type(value).__name__}' object has no attribute '{name}'")
    return members[name]
