"""Small defaults and plain-data snapshots shared by optional adapters."""
from .bridge import PRIMITIVES


def is_view(value):
    return False


def view(root, value):
    return None


def iterate(value):
    return iter(value)


def assign(value, name, item):
    raise AttributeError('Unsupported attribute assignment')


def freeze(value):
    """Native arrays/frames in recipes become safe descriptors, never pickle."""
    if isinstance(value, PRIMITIVES):
        # NumPy scalar subclasses cross Monty as builtin primitives. Canonicalize
        # before signing, so native values and restored values authenticate alike.
        for kind in (bool, int, float, complex, str, bytes):
            if isinstance(value, kind):
                return kind(value)
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(freeze(item) for item in value)
    if isinstance(value, dict):
        return {'__recipe_dict__': [[freeze(key), freeze(item)] for key, item in value.items()]}
    from .bridge import backend
    from .registry import available
    for name in ('numpy', 'pandas'):
        if name in available():
            api = backend(name)
            if name == 'numpy' and isinstance(value, api.np.ndarray) and value.dtype.hasobject:
                api.checked_size(value.shape, 'int64')
                if any(not isinstance(item, PRIMITIVES) for item in value.flat):
                    raise TypeError('Object arrays may contain only primitive category labels')
                return {'__recipe_labels__': [value.shape, value.tolist()]}
            spec = api.pack(value)
            if spec is not None:
                return {'__recipe_value__': spec}
    raise TypeError(f'Unsupported recipe data: {type(value).__name__}')


def thaw(value):
    if isinstance(value, dict):
        if set(value) == {'__recipe_dict__'}:
            return {thaw(key): thaw(item) for key, item in value['__recipe_dict__']}
        if set(value) == {'__recipe_labels__'}:
            from .numpy import np, checked_size
            shape, items = value['__recipe_labels__']
            checked_size(shape, 'int64')
            return np.array(items, dtype=object).reshape(shape)
        if set(value) == {'__recipe_value__'}:
            from .bridge import backend
            spec = value['__recipe_value__']
            if spec['module'] not in ('numpy', 'pandas'):
                raise ValueError('Recipes may contain only numeric/frame descriptors')
            return backend(spec['module']).restore(spec['kind'], spec['data'])
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(thaw(item) for item in value)
    return value


def record(commands, name, args, kwargs):
    if len(commands) >= 1000:
        raise ValueError('Documents/figures are limited to 1,000 recorded operations')
    commands.append([name, freeze(args), freeze(kwargs)])


class Record:
    """Named result (e.g. scipy's statistic/pvalue), persisted without native handles."""
    def __init__(self, fields):
        self.fields = fields

    def __repr__(self):
        return repr(self.fields)
