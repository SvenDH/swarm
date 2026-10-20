"""Translate library syntax, leaving ordinary code alone until libraries are used."""
import ast
import builtins
import hashlib
import textwrap
from importlib import import_module
from pathlib import Path

from .registry import MODULES, ALIASES, NATIVE, require

IMPORTS = (*MODULES, *ALIASES, 'irc')

BOOTSTRAP = Path(__file__).with_name('sandbox.py').read_text()
CHECK = 'try:\n    _library_modules\nexcept NameError:\n    _library_ready=None\nelse:\n    _library_ready=list(_library_modules)\n_library_ready'
BUILTINS = frozenset(vars(builtins))
DELETE_BOOTSTRAP = '''try:
    _library_deleted
except NameError:
    class _LibraryDeleted:
        pass
    _library_deleted = _LibraryDeleted()
def _library_name(name, value):
    if value is _library_deleted:
        raise NameError("name '" + name + "' is not defined")
    return value
try:
    _LibraryGlobals
except NameError:
    class _LibraryGlobals:
        def __init__(self, values):
            self._values = values
        def __getitem__(self, key):
            return self._values[key]
        def __iter__(self):
            return iter(self._values)
        def __len__(self):
            return len(self._values)
        def __contains__(self, key):
            return key in self._values
        def keys(self):
            return self._values.keys()
        def values(self):
            return self._values.values()
        def items(self):
            return self._values.items()
        def get(self, key, default=None):
            return self._values.get(key, default)
        def copy(self):
            return self._values.copy()
def _library_globals():
    _lib_global_values = {}
    for _lib_global_key in _library_global_names:
        if not _lib_global_key.startswith(('_library_', '_Library', '_lib_global_', '_prepare_exec', '_object_fields')):
            try:
                _lib_global_value = eval(_lib_global_key)
            except NameError:
                continue
            if _lib_global_value is not _library_deleted:
                _lib_global_values[_lib_global_key] = _lib_global_value
    return _LibraryGlobals(_lib_global_values)
def _library_contains(container, item):
    if isinstance(container, _LibraryGlobals):
        return item in container._values
    return item in container
def _library_exec_namespace(value):
    if isinstance(value, _LibraryGlobals):
        value = value.copy()
    if isinstance(value, dict):
        value['_library_exec_scope'] = value
        value['_LibraryValue'] = _LibraryValue
        value['_LibraryRender'] = _LibraryRender
        value['_LibraryGlobals'] = _LibraryGlobals
        value['_LibraryDeleted'] = _LibraryDeleted
        value['_library_deleted'] = _library_deleted
        value['_library_modules'] = _library_modules
        value['_library_rpc'] = _library_rpc
        value['_prepare_exec'] = _prepare_exec
        value['_object_fields'] = _object_fields
        value['display'] = display
        value['_irc_exchange'] = _irc_exchange
    return value
def _library_dunder_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError('Relative imports are not supported')
    module = _library_import_module(name)
    return module if fromlist else _library_import_module(name.split('.')[0])
def _library_exception_only(exc, value=None, **kwargs):
    if value is not None:
        exc = value
    text = str(exc)
    return [type(exc).__name__ + (': ' + text if text else '') + '\\n']
def _library_method(_lib_receiver, _lib_method_name, /, *args, **kwargs):
    if (_lib_receiver.root['module'] == 'importlib' and not _lib_receiver.path and
            _lib_method_name == 'import_module'):
        return _library_import_module(*args, **kwargs)
    if _lib_receiver.root['module'] == 'traceback' and _lib_method_name == 'format_exception_only':
        return _library_exception_only(*args, **kwargs)
    if _lib_receiver.root['module'] == 'PIL' and _lib_method_name == 'point' and args and _library_callable(args[0]):
        # Evaluate callbacks in the interpreter; native code receives only a LUT.
        table = [args[0](i) for i in range(256)]
        bands = _library_request(_lib_receiver, 'method', 'getbands')
        args = (table * len(bands), *args[1:])
    return _library_request(_lib_receiver, 'method', _lib_method_name, *args, **kwargs)
def _library_import_module(name, package=None):
    if not isinstance(name, str):
        raise TypeError('module name must be a string')
    if name.startswith('.'):
        raise ImportError('Relative imports are not supported')
    parts = name.split('.')
    if parts[0] == 'irc':
        return _library_irc_import(name)
    result = _library_native_import(parts[0]) if parts[0] in _library_native_modules else _library_import(parts[0])
    for part in parts[1:]:
        result = _library_get(result, part)
        if not ((isinstance(result, _LibraryValue) and result.root['kind'] == 'module') or
                type(result).__name__ == 'module'):
            raise ModuleNotFoundError("No module named '" + name + "'")
    return result
def _library_getattr(value, name, *default):
    try:
        if isinstance(value, _LibraryValue):
            if name == 'root':
                return value.root
            if name == 'path':
                return value.path
            if name == 'auth':
                return value.auth
            return _library_get(value, name)
        return getattr(value, name)
    except AttributeError:
        if default:
            return default[0]
        raise
def _library_hasattr(value, name):
    try:
        _library_getattr(value, name)
        return True
    except AttributeError:
        return False
def _library_callable(value):
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'callable')
    return (type(value) == type(int) or type(value) == type(_LibraryValue) or type(value) == type(_library_callable) or
            type(value) == type(len) or type(value) == type(_library_rpc) or hasattr(value, '__call__'))
def _library_vars(value):
    if isinstance(value, _LibraryValue):
        return {name: _library_get(value, name) for name in _library_request(value, 'dir')}
    if type(value).__name__ == 'module':
        return {name: getattr(value, name) for name in _library_dir(value)}
    return {name: getattr(value, name) for name in _object_fields(value, True)}
def _library_dir(value):
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'dir')
    if type(value).__name__ == 'module':
        return [name for name in _library_native_modules.get(str(value), ()) if hasattr(value, name)]
    return sorted(_object_fields(value, False))
def _library_isinstance(value, classes):
    if isinstance(classes, tuple):
        return any(_library_isinstance(value, item) for item in classes)
    if isinstance(classes, _LibraryValue):
        if not _library_request(classes, 'is_type'):
            raise TypeError('isinstance() arg 2 must be a type or tuple of types')
        return _library_request(value, 'isinstance', classes) if isinstance(value, _LibraryValue) else False
    return isinstance(value, classes)
def _library_delitem(value, key):
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'delitem', key)
    if isinstance(value, dict):
        value.pop(key)
    elif isinstance(value, list):
        if isinstance(key, slice):
            for index in sorted(list(range(len(value)))[key], reverse=True):
                value.pop(index)
        else:
            value.pop(key)
    else:
        raise TypeError("object does not support item deletion")
'''
# Only names cross this boundary. Attribute lookup stays inside Monty, so native
# modules never become their string wire representation during introspection.
_native_names = {f"<module '{name}'>": sorted(key for key in vars(import_module(name)) if not key.startswith('_'))
                 for name in NATIVE}
DELETE_BOOTSTRAP += '\n_library_native_modules = ' + repr({**_native_names, **{name: () for name in NATIVE}}) + '\n'
DELETE_BOOTSTRAP += 'def _library_native_import(name):\n' + ''.join(
    f'    if name == {name!r}:\n        import {name}\n        return {name}\n' for name in NATIVE)
DELETE_BOOTSTRAP += "    raise ModuleNotFoundError(\"No module named '\" + name + \"'\")\n"
DELETE_BOOTSTRAP += '\n' + Path(__file__).with_name('irc_client.py').read_text() + '\n'
# Defining functions on every call retains old interpreter code in checkpoints.
# Refresh once per helper revision, including sessions restored after an update.
_revision = hashlib.sha256((BOOTSTRAP + DELETE_BOOTSTRAP).encode()).hexdigest()
CHECK += ("\ntry:\n    _library_revision\nexcept NameError:\n    _library_revision=None\n"
          f"(_library_ready, _library_revision == {_revision!r})")
DELETE_BOOTSTRAP = ("try:\n    _library_revision\nexcept NameError:\n    _library_revision=None\n"
                    f"if _library_revision != {_revision!r}:\n" +
                    textwrap.indent(DELETE_BOOTSTRAP, '    ') +
                    f"\n    _library_revision={_revision!r}\n")


def call(name, *args):
    return ast.Call(ast.Name('_library_' + name, ast.Load()), list(args), [])


def key(node):
    if isinstance(node, ast.Constant) and node.value is Ellipsis:
        return ast.Dict([ast.Constant('__library_ellipsis__')], [ast.Constant(True)])
    if isinstance(node, ast.Slice):
        # Monty's slice constructor rejects floats/complex steps used by NumPy
        # grids. Keep ordinary slices native; send extended ones as wire data.
        # The lambda evaluates each bound once and works with old checkpoints.
        factory = ast.parse(
            "lambda a,b,c: slice(a,b,c) if "
            "(a is None or isinstance(a,int)) and (b is None or isinstance(b,int)) and "
            "(c is None or isinstance(c,int)) else {'__library_slice__':[a,b,c]}", mode='eval').body
        return ast.Call(factory, [node.lower or ast.Constant(None), node.upper or ast.Constant(None),
                                  node.step or ast.Constant(None)], [])
    if isinstance(node, ast.Tuple):
        return ast.Tuple([key(item) for item in node.elts], ast.Load())
    return node


def module(name):
    if name.split('.')[0] == 'irc':
        return call('irc_import', ast.Constant(name))
    parts = name.split('.')
    parts[0] = ALIASES.get(parts[0], parts[0])
    require(parts[0])
    # prepare() runs on the Runtime's Python IO pool. Initialize native extensions
    # here, not on Monty's foreign callback threads (SciPy FFT requires this).
    from .bridge import backend
    backend(parts[0])
    result = call('import', ast.Constant(parts[0]))
    for part in parts[1:]:
        result = call('get', result, ast.Constant(part))
    return result


class LibrarySyntax(ast.NodeTransformer):
    def __init__(self, tree):
        self.names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.counter = 0

    def visit_Module(self, node):
        node = self.generic_visit(node)
        names = set()
        def collect(item):
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(item.name)
                return
            if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store):
                names.add(item.id)
            if isinstance(item, (ast.Import, ast.ImportFrom)):
                names.update(alias.asname or alias.name.split('.')[0] for alias in item.names)
            for child in ast.iter_child_nodes(item):
                collect(child)
        collect(node)
        for item in ast.walk(node):
            if isinstance(item, ast.Global):
                names.update(item.names)
        node.body[:0] = ast.parse('_library_global_names = list(set(list(locals()) + ' + repr(sorted(names)) + '))').body
        return node

    def visit_Name(self, node):
        if (isinstance(node.ctx, ast.Load) and node.id not in BUILTINS and
                node.id != 'display' and not node.id.startswith(('_library_', '_prepare_exec'))):
            return call('name', ast.Constant(node.id), node)
        return node

    def visit_Delete(self, node):
        result = []
        for target in node.targets:
            if isinstance(target, (ast.Tuple, ast.List)):
                result.extend(self.visit_Delete(ast.Delete(target.elts)))
            elif isinstance(target, ast.Name):
                result.extend([ast.Expr(call('name', ast.Constant(target.id), ast.Name(target.id, ast.Load()))),
                    ast.Assign([ast.Name(target.id, ast.Store())], ast.Name('_library_deleted', ast.Load()))])
            elif isinstance(target, ast.Subscript):
                result.append(ast.Expr(call('delitem', self.visit(target.value), key(self.visit(target.slice)))))
            else:
                raise SyntaxError('attribute deletion is not supported')
        return result

    def visit_Import(self, node):
        result = []
        for alias in node.names:
            if alias.name.split('.')[0] in IMPORTS:
                name = alias.asname or alias.name.split('.')[0]
                result.append(ast.Assign([ast.Name(name, ast.Store())],
                    module(alias.name if alias.asname else name)))
            else:
                result.append(ast.Import([alias]))
        return result

    def visit_ImportFrom(self, node):
        if node.level or not node.module or node.module.split('.')[0] not in IMPORTS:
            return node
        if any(alias.name == '*' for alias in node.names):
            raise ImportError('Use explicit library imports, not import *')
        return [ast.Assign([ast.Name(alias.asname or alias.name, ast.Store())],
                           call('get', module(node.module), ast.Constant(alias.name))) for alias in node.names]

    def visit_Attribute(self, node):
        self.generic_visit(node)
        return call('get', node.value, ast.Constant(node.attr)) if isinstance(node.ctx, ast.Load) else node

    def visit_Subscript(self, node):
        self.generic_visit(node)
        return call('item', node.value, key(node.slice)) if isinstance(node.ctx, ast.Load) else node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, (ast.Tuple, ast.List)):
                node.value = call('iter', node.value)
            if isinstance(target, ast.Subscript):
                return ast.Expr(call('setitem', target.value, key(target.slice), node.value))
            if isinstance(target, ast.Attribute):
                return ast.Expr(call('setattr', target.value, ast.Constant(target.attr), node.value))
        return node

    def visit_AugAssign(self, node):
        target, value = self.visit(node.target), self.visit(node.value)
        name = ast.Constant(type(node.op).__name__)
        if isinstance(target, ast.Subscript):
            return ast.Expr(call('augitem', target.value, key(target.slice), name, value))
        if isinstance(target, ast.Attribute):
            return ast.Expr(call('augattr', target.value, ast.Constant(target.attr), name, value))
        if isinstance(target, ast.Name):
            return ast.Assign([target], call('inplace', name, ast.Name(target.id, ast.Load()), value))
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        return call('binary', ast.Constant(type(node.op).__name__), node.left, node.right)

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, (ast.USub, ast.UAdd, ast.Invert)):
            return call('unary', ast.Constant(type(node.op).__name__), node.operand)
        return node

    def visit_Compare(self, node):
        self.generic_visit(node)
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            result = call('contains', node.comparators[0], node.left)
            return ast.UnaryOp(ast.Not(), result) if isinstance(node.ops[0], ast.NotIn) else result
        if len(node.ops) == 1 and type(node.ops[0]).__name__ in ('Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE'):
            return call('binary', ast.Constant(type(node.ops[0]).__name__), node.left, node.comparators[0])
        return node

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name) and node.func.id == '__import__':
            return ast.Call(ast.Name('_library_dunder_import', ast.Load()),
                            [self.visit(arg) for arg in node.args],
                            [self.visit(item) for item in node.keywords])
        if isinstance(node.func, ast.Name) and node.func.id in ('iter', 'zip', 'enumerate'):
            name = 'builtin_iter' if node.func.id == 'iter' else node.func.id
            return ast.Call(ast.Name('_library_' + name, ast.Load()),
                            [self.visit(arg) for arg in node.args], [self.visit(item) for item in node.keywords])
        if isinstance(node.func, ast.Name) and node.func.id == 'globals':
            node.func = ast.Name('_library_globals', ast.Load())
            node.args = [self.visit(arg) for arg in node.args]
            return node
        if isinstance(node.func, ast.Attribute):
            # Monty can CALL builtin methods but cannot get their bound value
            # (even getattr([], 'append') fails). Leave native method calls intact.
            self.counter += 1
            name = '_library_receiver_' + str(self.counter)
            while name in self.names:
                name += '_'
            receiver = self.visit(node.func.value)
            args = [self.visit(arg) for arg in node.args]
            keywords = [self.visit(item) for item in node.keywords]
            target = ast.Name(name, ast.Load())
            body = ast.IfExp(call('is_value', target),
                ast.Call(ast.Name('_library_method', ast.Load()),
                         [target, ast.Constant(node.func.attr), *args], keywords),
                ast.Call(ast.Attribute(target, node.func.attr, ast.Load()), args, keywords))
            arguments = ast.arguments(posonlyargs=[], args=[ast.arg(name)],
                vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
            return ast.Call(ast.Lambda(arguments, body), [receiver], [])
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'exec' and node.args:
            isolated = len(node.args) > 1
            node.args[0] = ast.Call(ast.Name('_prepare_exec', ast.Load()), [node.args[0], ast.Constant(isolated)], [])
            if isolated:
                node.args[1] = call('exec_namespace', node.args[1])
            return node
        if isinstance(node.func, ast.Name) and node.func.id == 'display':
            node.args = [call('pack', arg) for arg in node.args]
            return node
        if isinstance(node.func, ast.Name) and node.func.id in ('vars', 'dir') and not node.args:
            listing = node.func.id == 'dir'
            node = ast.Call(ast.Name('locals', ast.Load()), [], [])
            return ast.Call(ast.Name('sorted', ast.Load()), [node], []) if listing else node
        if isinstance(node.func, ast.Name) and node.func.id in ('len', 'list', 'tuple', 'print', 'str', 'float', 'int', 'bool',
                                                              'getattr', 'hasattr', 'callable', 'vars', 'dir', 'isinstance'):
            node.func = ast.Name('_library_' + node.func.id, ast.Load())
        if isinstance(node.func, ast.Name) and node.func.id in BUILTINS:
            return node
        return ast.Call(ast.Name('_library_invoke', ast.Load()), [node.func, *node.args], node.keywords)

    def visit_For(self, node):
        self.generic_visit(node)
        node.iter = call('iter', node.iter)
        return node

    def visit_comprehension(self, node):
        self.generic_visit(node)
        node.iter = call('iter', node.iter)
        return node


def prepare(source, active=False, loaded=()):
    from .bridge import backend
    from .registry import available
    for name in loaded:
        if name in available():
            backend(name)
    tree = ast.parse(source)
    active = active or any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ('exec','getattr','hasattr','callable','vars','dir','globals','__import__','isinstance','iter','zip','enumerate') or
        isinstance(node, ast.Delete) or
        isinstance(node, ast.Import) and any(alias.name.split('.')[0] in IMPORTS for alias in node.names) or
        isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in IMPORTS
        for node in ast.walk(tree))
    if not active:
        return source, False
    tree = LibrarySyntax(tree).visit(tree)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tree.body[-1].value = call('result', tree.body[-1].value)
    return ast.unparse(ast.fix_missing_locations(tree)), True


def cleanup(source):
    """Minimal native feed for releasing names even from a nearly full heap."""
    tree = ast.parse(source)
    names = []
    for statement in tree.body:
        if not isinstance(statement, ast.Delete):
            return None
        for target in statement.targets:
            if not isinstance(target, ast.Name) or target.id.startswith('_library_'):
                return None
            names.append(target.id)
    if not names:
        return None
    lines = ['try:', '    _library_deleted', 'except NameError:',
             '    _library_had_deleted=False', 'else:', '    _library_had_deleted=True']
    for name in names:
        lines.extend([name, f'if _library_had_deleted and {name} is _library_deleted:',
                      f'    raise NameError("name {name!r} is not defined")', f'{name}=None'])
        lines.extend(['if not _library_had_deleted:', '    class _LibraryDeleted:', '        pass',
                      '    _library_deleted=_LibraryDeleted()', '    _library_had_deleted=True',
                      f'{name}=_library_deleted'])
    return '\n'.join(lines)
