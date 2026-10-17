"""Symbolic expressions without sympify/parse_expr's Python evaluation."""
import sympy as sp
from .bridge import descriptor
from .common import is_view, view, iterate, assign

CONSTRUCTORS = {name: value for name, value in (
    ('Symbol', sp.Symbol), ('Integer', sp.Integer), ('Float', sp.Float),
    ('Rational', sp.Rational), ('Add', sp.Add), ('Mul', sp.Mul), ('Pow', sp.Pow),
    ('sin', sp.sin), ('cos', sp.cos), ('tan', sp.tan), ('exp', sp.exp), ('log', sp.log),
    ('Abs', sp.Abs), ('Equality', sp.Eq), ('Unequality', sp.Ne),
    ('StrictLessThan', sp.Lt), ('LessThan', sp.Le), ('StrictGreaterThan', sp.Gt),
    ('GreaterThan', sp.Ge), ('Tuple', sp.Tuple))}


def checked(function):
    def invoke(*args, **kwargs):
        def validate(value):
            if isinstance(value, str):
                raise ValueError('Use Symbol/symbols and expressions, not expression strings')
            if isinstance(value, (list, tuple)):
                for item in value:
                    validate(item)
            if isinstance(value, dict):
                for key, item in value.items():
                    validate(key)
                    validate(item)
        validate(args)
        for value in kwargs.values():
            validate(value)
        return function(*args, **kwargs)
    return invoke


def exports(fs):
    return {'Symbol': sp.Symbol, 'symbols': sp.symbols, 'Integer': sp.Integer,
            'Float': sp.Float, 'Rational': sp.Rational,
            **{name: checked(function) for name, function in (
                ('Eq', sp.Eq), ('solve', sp.solve), ('expand', sp.expand), ('factor', sp.factor),
                ('simplify', sp.simplify), ('diff', sp.diff), ('integrate', sp.integrate),
                ('limit', sp.limit), ('sin', sp.sin), ('cos', sp.cos), ('tan', sp.tan),
                ('sqrt', sp.sqrt), ('exp', sp.exp), ('log', sp.log), ('Abs', sp.Abs),
                ('Matrix', sp.Matrix))}, 'pi': sp.pi, 'E': sp.E, 'I': sp.I, 'oo': sp.oo}


def members(value, fs, name=None):
    if isinstance(value, sp.MatrixBase):
        return {'shape': value.shape, 'T': value.T, 'det': value.det, 'inv': value.inv,
                'tolist': value.tolist, 'eigenvals': value.eigenvals}
    if isinstance(value, sp.Basic):
        return {'free_symbols': list(value.free_symbols), 'args': value.args,
                'subs': checked(value.subs), 'evalf': checked(value.evalf),
                'simplify': checked(value.simplify), 'expand': checked(value.expand),
                'factor': checked(value.factor), 'diff': checked(value.diff),
                'integrate': checked(value.integrate), 'is_number': value.is_number}
    return {}


def tree(value):
    if value in (sp.true, sp.false):
        return ['bool', bool(value)]
    constants = {sp.pi: 'pi', sp.E: 'E', sp.I: 'I', sp.oo: 'oo', -sp.oo: '-oo'}
    if value in constants:
        return ['constant', constants[value]]
    if isinstance(value, sp.Symbol):
        return ['Symbol', [value.name], {key: flag for key, flag in value.assumptions0.items() if flag is not None}]
    if isinstance(value, sp.Integer):
        return ['Integer', [int(value)], {}]
    if isinstance(value, sp.Rational):
        return ['Rational', [int(value.p), int(value.q)], {}]
    if isinstance(value, sp.Float):
        return ['Float', [str(value), value._prec], {'precision': True}]
    for name, factory in CONSTRUCTORS.items():
        if value.func is factory:
            return [name, [tree(item) for item in value.args], None]
    raise TypeError(f'Unsupported symbolic expression: {type(value).__name__}')


def untree(data):
    name = data[0]
    if name == 'bool':
        return sp.true if data[1] else sp.false
    if name == 'constant':
        return {'pi': sp.pi, 'E': sp.E, 'I': sp.I, 'oo': sp.oo, '-oo': -sp.oo}[data[1]]
    if name == 'Float':
        return sp.Float(data[1][0], precision=data[1][1])
    args = [untree(item) for item in data[1]] if data[2] is None else data[1]
    return CONSTRUCTORS[name](*args, **(data[2] or {}))


def pack(value):
    if isinstance(value, dict):
        return descriptor('sympy', 'mapping', [[tree(key), tree(sp.Integer(item) if isinstance(item, int) else item)]
                                              for key, item in value.items()])
    if isinstance(value, sp.MatrixBase):
        return descriptor('sympy', 'matrix', [[tree(item) for item in row] for row in value.tolist()])
    if isinstance(value, sp.Basic):
        return descriptor('sympy', 'expression', tree(value))


def restore(kind, data):
    if kind == 'mapping':
        return {untree(key): untree(item) for key, item in data}
    if kind == 'expression':
        return untree(data)
    if kind == 'matrix':
        return sp.Matrix([[untree(item) for item in row] for row in data])
    raise ValueError('Unsupported symbolic object')
