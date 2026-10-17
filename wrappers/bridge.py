"""Allowlisted native operations over checkpoint-safe values, never host handles."""
import operator
from importlib import import_module
from itertools import islice
from datetime import date, time, timedelta
from .registry import MODULES, require

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
MUTATING = {'numpy': {'sort', 'random', 'normal', 'uniform', 'integers', 'standard_normal'},
            'pandas': set(), 'PIL': {'putpixel', 'paste', 'thumbnail'},
            'openpyxl': {'create_sheet', 'remove', 'append', 'cell', 'iter_rows', 'merge_cells',
                         'unmerge_cells', 'delete_rows', 'delete_cols'},
            'pypdf': {'add_page', 'add_blank_page', 'append', 'add_metadata', 'rotate', 'scale_to', 'merge_page'},
            'docx': {'add_paragraph', 'add_heading', 'add_table', 'add_page_break', 'add_picture',
                     'add_run', 'add_row', 'add_column'},
            'scipy': set(), 'sklearn': {'fit', 'fit_transform'}, 'seaborn': set(),
            'pdfplumber': set(),
            'bs4': {'clear', 'append', 'decompose', 'extract', 'replace_with'},
            'pymupdf': {'new_page', 'delete_page', 'insert_pdf', 'insert_text', 'set_rotation'},
            'pptx': {'add_slide', 'add_textbox', 'add_shape', 'add_table', 'add_picture',
                     'add_paragraph', 'add_run', 'clear'},
            'reportlab': {'drawString', 'drawCentredString', 'drawRightString', 'setFont',
                          'setFillColorRGB', 'setStrokeColorRGB', 'setFillColor', 'setStrokeColor',
                          'setLineWidth', 'line', 'rect', 'circle', 'ellipse', 'showPage',
                          'translate', 'scale', 'rotate', 'saveState', 'restoreState',
                          'setTitle', 'setAuthor', 'setPageSize', 'drawImage'},
            'xlsxwriter': {'add_worksheet', 'add_format', 'add_chart', 'write', 'write_row',
                           'write_column', 'write_string', 'write_number', 'write_formula',
                           'write_blank', 'set_column', 'set_row', 'merge_range', 'freeze_panes',
                           'autofilter', 'conditional_format', 'insert_chart', 'insert_image',
                           'add_series', 'set_title', 'set_x_axis', 'set_y_axis', 'set_legend',
                           'set_style', 'set_bold', 'set_italic', 'set_num_format', 'set_bg_color',
                           'set_font_color', 'set_align'}}
CROSS = {'scipy': ('numpy',), 'sklearn': ('numpy',), 'seaborn': ('matplotlib',)}


def backend(module):
    if module not in MODULES:
        raise ImportError(f'Unsupported library: {module}')
    require(module)
    return import_module('.' + MODULES[module], 'wrappers')


def descriptor(module, kind, data):
    return {'module': module, 'kind': kind, 'data': data}


def decode(value, cache=None, fs=None):
    if cache is None:
        cache = {}
    if isinstance(value, dict):
        if set(value) == {'__library_value__', 'path'}:
            root = value['__library_value__']
            if root['kind'] == 'module' and root['module'] != 'matplotlib':
                raise TypeError('Library namespaces cannot be passed as data')
            identity = (root['module'], root['identity'])
            if identity not in cache:
                cache[identity] = backend(root['module']).restore(root['kind'], root['data'])
            result = resolve(cache[identity], value['path'], root['module'], fs, cache)
            if callable(result):
                raise TypeError('Native library callbacks cannot be passed as data')
            return result
        if set(value) == {'__library_slice__'}:
            return slice(*value['__library_slice__'])
        return {key: decode(item, cache, fs) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(decode(item, cache, fs) for item in value)
    return value


def member(value, name, module, fs):
    if name.startswith('_'):
        raise AttributeError('Private library attributes are not exposed')
    members = value if type(value) is dict else backend(module).members(value, fs, name)
    if name not in members:
        raise AttributeError(f'{module}.{type(value).__name__}.{name} is not supported by the wrappers')
    return members[name]


def resolve(root, path, module, fs, cache=None):
    value = root
    for step in path:
        if step[0] == 'attr':
            value = member(value, step[1], module, fs)
        elif step[0] == 'item':
            value = value[decode(step[1], cache, fs)]
        elif step[0] == 'call':
            value = member(value, step[1], module, fs)(*decode(step[2], cache, fs), **decode(step[3], cache, fs))
        elif step[0] == 'numpy_function' and module == 'numpy':
            value = member(backend(module).exports(fs), step[1], module, fs)(
                value, *decode(step[2], cache, fs), **decode(step[3], cache, fs))
        else:
            raise ValueError('Invalid library object path')
    return value


def encode(value, module, root=None, path=None):
    if isinstance(value, PRIMITIVES):
        return value
    # Children of document/workbook roots keep paths into the same persisted root.
    if root is not None:
        view = backend(module).view(root, value)
        if view is not None:
            return {'__library_view__': view}
    if isinstance(value, (list, tuple)):
        return (tuple if isinstance(value, tuple) else list)(encode(item, module, root) for item in value)
    if isinstance(value, dict):
        if module == 'sympy' and any(not isinstance(key, PRIMITIVES) for key in value):
            return {'__library_object__': backend(module).pack(value)}
        # User mappings must never be confused with host-created protocol tags.
        return {'__library_dict__': [[key, encode(item, module, root)] for key, item in value.items()]}
    result = backend(module).pack(value)
    if result is None:
        # Scientific/plotting results often belong to another supported library.
        from .registry import available
        for other in CROSS.get(module, ()):
            if other != module and other in available():
                result = backend(other).pack(value)
                if result is not None:
                    break
    if result is not None:
        if result['kind'] == 'scalar':
            return encode(result['data'], module)
        return {'__library_object__': result}
    if path is not None:
        return {'__library_view__': path}
    raise TypeError(f'Unsupported {module} result: {type(value).__name__}')


def dispatch(fs, action, spec, path=(), args=(), kwargs=None):
    """Every attribute/call is resolved through explicit backend member maps."""
    module = spec['module']
    api = backend(module)
    stateful = module == 'matplotlib'
    if spec['kind'] == 'module':
        root = api.restore('module', spec['data']) if stateful and spec['data'] is not None else api.exports(fs)
    else:
        root = api.restore(spec['kind'], spec['data'])
    cache = {} if spec['kind'] == 'module' else {(module, spec['identity']): root}
    value = resolve(root, path, module, fs, cache)
    wire_args, wire_kwargs = args, kwargs or {}
    args, kwargs = decode(args, cache, fs), decode(wire_kwargs, cache, fs)
    result_path = None
    if action == 'get':
        name = args[0]
        result = member(value, name, module, fs)
        if callable(result) and not api.is_view(result):
            return {'call': True}
        result_path = [*path, ['attr', name]]
        # Constants/distributions are independent values, not views of namespaces.
        if spec['kind'] == 'module' and not stateful and api.pack(result) is not None:
            return {'result': encode(result, module)}
        # Learned/statistical arrays belong to NumPy, not to the estimator backend.
        if module in ('scipy', 'sklearn'):
            numeric = backend('numpy').pack(result)
            if numeric is not None:
                return {'result': encode(result, module)}
        # Collections/indexers remain views so nested writes update their root.
        plain_dict = type(result) is dict and spec['kind'] != 'module'
        if not plain_dict and not isinstance(result, (*PRIMITIVES, tuple, list)):
            response = {'result': {'__library_view__': api.view(root, result) or result_path}}
            if stateful and spec['kind'] == 'module':
                response['state'] = api.pack(root)
            return response
    elif action == 'call':
        if not callable(value):
            raise TypeError('Library value is not callable')
        if module == 'numpy' and kwargs.get('out') is not None:
            raise ValueError('out= is not supported; assign the returned array instead')
        result = value(*args, **kwargs)
        if result is root and spec['kind'] != 'module':
            result_path = []
        if module == 'numpy' and api.shares(root, result):
            return {'result': {'__library_view__': [*path[:-1],
                ['call', path[-1][1], wire_args, wire_kwargs]]}}
        # Module functions such as np.flip/reshape also return live array views.
        # Retain the authenticated argument root rather than checkpointing a copy.
        if (module == 'numpy' and spec['kind'] == 'module' and args and
                api.shares(args[0], result) and isinstance(wire_args[0], dict) and
                '__library_value__' in wire_args[0] and len(path) == 1):
            original = wire_args[0]
            return {'argument_result': {
                'module': module, 'identity': original['__library_value__']['identity'],
                'path': [*original['path'], ['numpy_function', path[0][1], wire_args[1:], wire_kwargs]]}}
    elif action == 'item':
        result = value[args[0]]
        result_path = [*path, ['item', wire_args[0]]]
        if api.is_view(result):
            return {'result': {'__library_view__': api.view(root, result) or result_path}}
    elif action == 'setitem':
        if module == 'sympy':
            api.validate(args[1])
        value[args[0]] = args[1]
        result = None
    elif action == 'setattr':
        api.assign(value, args[0], args[1])
        result = None
    elif action == 'len':
        result = len(value)
    elif action == 'iterate':
        items = list(islice(api.iterate(value), 10_001))
        if len(items) > 10_000:
            raise ValueError('library iteration is limited to 10,000 items; prefer vectorized operations')
        result = []
        for i, item in enumerate(items):
            view = api.view(root, item)
            if view is None and module == 'numpy' and api.is_view(item):
                view = [*path, ['item', i]]
            result.append({'__library_view__': view} if view is not None else encode(item, module, root))
        return {'result': result}
    elif action == 'repr':
        result = repr(value)
    elif action == 'convert':
        result = {'float': float, 'int': int, 'bool': bool}[args[0]](value)
    elif action == 'binary':
        name, other, reverse = args
        if module == 'sympy':
            api.validate(other)
        result = OPERATORS[name](other, value) if reverse else OPERATORS[name](value, other)
    elif action == 'unary':
        result = OPERATORS[args[0]](value)
    elif action == 'inplace':
        if module == 'sympy':
            api.validate(args[1])
        result = INPLACE[args[0]](value, args[1])
    else:
        raise ValueError(f'Unknown library action: {action}')
    if action == 'call' and result is root and spec['kind'] != 'module':
        encoded = {'__library_view__': []}
    elif action == 'inplace' and result is value:
        encoded = {'__library_view__': list(path)}
    else:
        encoded = encode(result, module, root, result_path)
    response = {'result': encoded}
    changed = action in ('setitem', 'setattr', 'inplace') or (
        action == 'call' and (kwargs.get('inplace', False) or (path and path[-1][1] in MUTATING.get(module, set())) or module not in MUTATING))
    if (spec['kind'] != 'module' and changed) or stateful:
        response['state'] = api.pack(root)
    if module == 'seaborn':
        response['updates'] = [{'module': owner, 'identity': identity, 'data': backend(owner).pack(obj)['data']}
                               for (owner, identity), obj in cache.items() if owner == 'matplotlib']
        for (owner, identity), obj in cache.items():
            if owner == 'matplotlib':
                result_view = backend(owner).view(obj, result)
                if result_view is not None:
                    response['argument_result'] = {'module': owner, 'identity': identity, 'path': result_view}
                    break
    return response
