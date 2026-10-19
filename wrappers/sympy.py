"""Symbolic expressions without sympify/parse_expr's Python evaluation."""
import sympy as sp
from fractions import Fraction
from keyword import iskeyword
from .bridge import descriptor
from .common import is_view, view, iterate, assign
from .api import select
from .limits import check_memory
import operator

CONSTRUCTORS = {name: value for name, value in (
    ('Symbol', sp.Symbol), ('Integer', sp.Integer), ('Float', sp.Float),
    ('Rational', sp.Rational), ('Add', sp.Add), ('Mul', sp.Mul), ('Pow', sp.Pow),
    ('sin', sp.sin), ('cos', sp.cos), ('tan', sp.tan), ('exp', sp.exp), ('log', sp.log),
    ('Abs', sp.Abs), ('Equality', sp.Eq), ('Unequality', sp.Ne),
    ('StrictLessThan', sp.Lt), ('LessThan', sp.Le), ('StrictGreaterThan', sp.Gt),
    ('GreaterThan', sp.Ge), ('Tuple', sp.Tuple))}
ELEMENTARY = select(vars(sp), '''asin acos atan atan2 sinh cosh tanh asinh acosh atanh
    sign floor ceiling conjugate re im arg factorial binomial gamma loggamma erf erfc Min Max''')
CONSTRUCTORS.update(ELEMENTARY)
SETS = select(vars(sp), 'FiniteSet Interval Union Intersection Complement Contains And Or Not')
CONSTRUCTORS.update(SETS)
CONSTANTS = {'pi': sp.pi, 'E': sp.E, 'I': sp.I, 'oo': sp.oo, '-oo': -sp.oo,
             'EulerGamma': sp.EulerGamma, 'Catalan': sp.Catalan, 'GoldenRatio': sp.GoldenRatio,
             'zoo': sp.zoo, 'nan': sp.nan, 'EmptySet': sp.EmptySet, 'UniversalSet': sp.UniversalSet}


def validate(value, depth=0):
    if depth > 64:
        raise ValueError('Symbolic values are nested too deeply')
    if isinstance(value, str):
        raise ValueError('Use Symbol/symbols and expressions, not expression strings')
    if value is None or type(value) in (bool, int, float, complex) or isinstance(value, (sp.Basic, sp.MatrixBase)):
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            validate(item, depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            validate(key, depth + 1)
            validate(item, depth + 1)
        return
    raise TypeError('Unsupported symbolic argument')


def checked(function):
    def invoke(*args, **kwargs):
        validate(args)
        for value in kwargs.values():
            validate(value)
        return function(*args, **kwargs)
    return invoke


def named(factory, names, **assumptions):
    if not isinstance(names, str) or any(type(flag) not in (bool, type(None)) or key.startswith('_')
                                          for key, flag in assumptions.items()):
        raise ValueError('Symbols need a name string and boolean assumptions')
    result = factory(names, **assumptions)
    symbols = result if isinstance(result, (tuple, list)) else (result,)
    if any(not isinstance(symbol, sp.Symbol) or not symbol.name.isidentifier()
           or symbol.name.startswith('__') or iskeyword(symbol.name) for symbol in symbols):
        raise ValueError('Symbol names must be ordinary identifiers')
    return result


def number(factory, *args, **kwargs):
    for item in args:
        if isinstance(item, str):
            if factory is sp.Rational:
                Fraction(item)
            elif factory is sp.Integer:
                int(item)
            else:
                float(item)
        else:
            validate(item)
    if set(kwargs) - {'dps', 'precision'} or any(type(value) is not int or not 1 <= value <= 10000 for value in kwargs.values()):
        raise ValueError('Unsupported numeric precision options')
    return factory(*args, **kwargs)


def matrix_factory(function):
    def invoke(rows, cols=None):
        rows, cols = operator.index(rows), operator.index(rows if cols is None else cols)
        if min(rows, cols) < 0:
            raise ValueError('symbolic matrix dimensions must be nonnegative')
        check_memory(rows * cols * 64)  # Conservative minimum per symbolic cell.
        return function(rows, cols)
    return invoke


def exports(fs):
    return {'Symbol': lambda name, **kw: named(sp.Symbol, name, **kw),
            **CONSTANTS, 'true': sp.true, 'false': sp.false,
            **{name: checked(function) for name, function in SETS.items()},
            **{name: checked(function) for name, function in ELEMENTARY.items()},
            'zeros': matrix_factory(sp.zeros), 'ones': matrix_factory(sp.ones), 'eye': matrix_factory(sp.eye),
            **{name: checked(function) for name, function in select(vars(sp), '''cancel together apart
                collect trigsimp expand_trig powsimp powdenest radsimp rational_interpolate
                resultant discriminant gcd lcm div rem quo invert det trace
                linear_eq_to_matrix linsolve''').items()},
            'symbols': lambda names, **kw: named(sp.symbols, names, **kw),
            'Integer': lambda *a, **kw: number(sp.Integer, *a, **kw),
            'Float': lambda *a, **kw: number(sp.Float, *a, **kw),
            'Rational': lambda *a, **kw: number(sp.Rational, *a, **kw),
            **{name: checked(function) for name, function in (
                ('Eq', sp.Eq), ('solve', sp.solve), ('expand', sp.expand), ('factor', sp.factor),
                ('simplify', sp.simplify), ('diff', sp.diff), ('integrate', sp.integrate),
                ('limit', sp.limit), ('sin', sp.sin), ('cos', sp.cos), ('tan', sp.tan),
                ('sqrt', sp.sqrt), ('exp', sp.exp), ('log', sp.log), ('Abs', sp.Abs),
                ('Matrix', sp.Matrix))}, 'pi': sp.pi, 'E': sp.E, 'I': sp.I, 'oo': sp.oo}


def members(value, fs, name=None):
    if isinstance(value, sp.MatrixBase):
        return {'shape': value.shape, 'T': value.T, 'det': value.det, 'inv': value.inv,
                'rows': value.rows, 'cols': value.cols, 'rank': checked(value.rank), 'trace': value.trace,
                'transpose': value.transpose, 'rref': checked(value.rref), 'nullspace': checked(value.nullspace),
                'columnspace': checked(value.columnspace), 'LUsolve': checked(value.LUsolve),
                'eigenvects': checked(value.eigenvects),
                'tolist': value.tolist, 'eigenvals': checked(value.eigenvals)}
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
    constants = {value: name for name, value in CONSTANTS.items()}
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


def untree(data, depth=0):
    if depth > 64 or type(data) not in (list, tuple) or not data:
        raise ValueError('Invalid symbolic tree')
    name = data[0]
    if name == 'bool':
        if len(data) != 2 or type(data[1]) is not bool:
            raise ValueError('Invalid symbolic boolean')
        return sp.true if data[1] else sp.false
    if name == 'constant':
        if len(data) != 2 or type(data[1]) is not str:
            raise ValueError('Invalid symbolic constant')
        return CONSTANTS[data[1]]
    if len(data) != 3 or type(data[1]) not in (list, tuple):
        raise ValueError('Invalid symbolic tree arguments')
    args, options = data[1:]
    if name == 'Symbol':
        if len(args) != 1 or type(options) is not dict:
            raise ValueError('Invalid symbol')
        return named(sp.Symbol, args[0], **options)
    if name in ('Integer', 'Rational'):
        if len(args) != (1 if name == 'Integer' else 2) or options != {} or any(type(item) is not int for item in args):
            raise ValueError('Invalid symbolic number')
        return CONSTRUCTORS[name](*args)
    if name == 'Float':
        if len(args) != 2 or type(args[0]) is not str or type(args[1]) is not int or not 1 <= args[1] <= 10000:
            raise ValueError('Invalid symbolic float')
        float(args[0])  # Numeric text only, never a symbolic expression string.
        return sp.Float(args[0], precision=args[1])
    if options is not None or name not in CONSTRUCTORS:
        raise ValueError('Composite symbolic nodes require typed child trees')
    return CONSTRUCTORS[name](*(untree(item, depth + 1) for item in args))


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
