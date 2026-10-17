"""Translate library syntax, leaving ordinary code alone until libraries are used."""
import ast
import builtins
from pathlib import Path

from .registry import MODULES, ALIASES, require

BOOTSTRAP = Path(__file__).with_name('sandbox.py').read_text()
CHECK = 'try:\n    _library_modules\nexcept NameError:\n    _library_ready=None\nelse:\n    _library_ready=list(_library_modules)\n_library_ready'
BUILTINS = frozenset(vars(builtins))


def call(name, *args):
    return ast.Call(ast.Name('_library_' + name, ast.Load()), list(args), [])


def key(node):
    if isinstance(node, ast.Slice):
        return ast.Call(ast.Name('slice', ast.Load()),
                        [node.lower or ast.Constant(None), node.upper or ast.Constant(None),
                         node.step or ast.Constant(None)], [])
    if isinstance(node, ast.Tuple):
        return ast.Tuple([key(item) for item in node.elts], ast.Load())
    return node


def module(name):
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

    def visit_Import(self, node):
        result = []
        for alias in node.names:
            if alias.name.split('.')[0] in MODULES or alias.name.split('.')[0] in ALIASES:
                name = alias.asname or alias.name.split('.')[0]
                result.append(ast.Assign([ast.Name(name, ast.Store())],
                    module(alias.name if alias.asname else name)))
            else:
                result.append(ast.Import([alias]))
        return result

    def visit_ImportFrom(self, node):
        if node.level or not node.module or node.module.split('.')[0] not in (*MODULES, *ALIASES):
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
        if len(node.ops) == 1 and type(node.ops[0]).__name__ in ('Eq', 'NotEq', 'Lt', 'LtE', 'Gt', 'GtE'):
            return call('binary', ast.Constant(type(node.ops[0]).__name__), node.left, node.comparators[0])
        return node

    def visit_Call(self, node):
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
                ast.Call(ast.Name('_library_invoke', ast.Load()),
                         [call('get', target, ast.Constant(node.func.attr)), *args], keywords),
                ast.Call(ast.Attribute(target, node.func.attr, ast.Load()), args, keywords))
            arguments = ast.arguments(posonlyargs=[], args=[ast.arg(name)],
                vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
            return ast.Call(ast.Lambda(arguments, body), [receiver], [])
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id in ('len', 'list', 'tuple', 'print', 'str', 'float', 'int', 'bool'):
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
        isinstance(node, ast.Import) and any(alias.name.split('.')[0] in (*MODULES, *ALIASES) for alias in node.names) or
        isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in (*MODULES, *ALIASES)
        for node in ast.walk(tree))
    if not active:
        return source, False
    tree = LibrarySyntax(tree).visit(tree)
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        tree.body[-1].value = call('result', tree.body[-1].value)
    return ast.unparse(ast.fix_missing_locations(tree)), True
