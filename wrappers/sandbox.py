"""Executed INSIDE Monty. These plain classes/closures survive session dumps."""


class _LibraryValue:
    def __init__(self, root, path, auth=None):
        self.root = root
        self.path = path
        self.auth = auth


class _LibraryRender:
    def __init__(self, text):
        self.text = text


_library_modules = {}


def _library_is_value(value):
    return isinstance(value, _LibraryValue)


def _library_pack(value):
    if isinstance(value, _LibraryValue):
        return {'__library_value__': value.root, 'path': value.path, 'auth': value.auth}
    if isinstance(value, slice):
        return {'__library_slice__': [value.start, value.stop, value.step]}
    if isinstance(value, list):
        return [_library_pack(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_library_pack(item) for item in value)
    if isinstance(value, dict):
        return {key: _library_pack(item) for key, item in value.items()}
    if value is float:
        return 'float64'
    if value is int:
        return 'int64'
    if value is bool:
        return 'bool'
    return value


def _library_unpack(value, root):
    if isinstance(value, dict):
        if '__library_dict__' in value:
            return {key: _library_unpack(item, root) for key, item in value['__library_dict__']}
        if '__library_object__' in value:
            spec = value['__library_object__']
            return _LibraryValue(spec, [])
        if '__library_view__' in value:
            return _LibraryValue(root, value['__library_view__'], value['auth'])
        return {key: _library_unpack(item, root) for key, item in value.items()}
    if isinstance(value, list):
        return [_library_unpack(item, root) for item in value]
    if isinstance(value, tuple):
        return tuple(_library_unpack(item, root) for item in value)
    return value


def _library_request(_lib_receiver, _lib_action, *args, **kwargs):
    value, action = _lib_receiver, _lib_action
    response = _library_rpc(action, value.root, value.path, _library_pack(args), _library_pack(kwargs), value.auth)
    if 'state' in response:
        state = response['state']
        value.root.update(state)
    if response.get('updates'):
        _library_update(args, response['updates'])
        _library_update(kwargs, response['updates'])
    if 'argument_result' in response:
        spec = response['argument_result']
        original = _library_find((args, kwargs), spec)
        if original is not None:
            if original.path == spec['path']:
                return original
            return _LibraryValue(original.root, spec['path'], spec['auth'])
    return _library_unpack(response.get('result'), value.root)


def _library_find(value, spec):
    if isinstance(value, _LibraryValue):
        if value.root['module'] == spec['module'] and value.root['identity'] == spec['identity']:
            return value
    elif isinstance(value, (list, tuple, dict)):
        items = value.values() if isinstance(value, dict) else value
        for item in items:
            found = _library_find(item, spec)
            if found is not None:
                return found
    return None


def _library_update(value, updates):
    if isinstance(value, _LibraryValue):
        for update in updates:
            if value.root['module'] == update['module'] and value.root['identity'] == update['identity']:
                value.root.update(update['state'])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _library_update(item, updates)
    elif isinstance(value, dict):
        for item in value.values():
            _library_update(item, updates)


def _library_invoke(function, *args, **kwargs):
    if isinstance(function, _LibraryValue):
        return _library_request(function, 'call', *args, **kwargs)
    return function(*args, **kwargs)


def _library_import(module):
    if module not in _library_modules:
        _library_modules[module] = _library_rpc('import', module)
    return _LibraryValue(_library_modules[module], [])


def _library_get(value, name):
    if not isinstance(value, _LibraryValue):
        return getattr(value, name)
    if value.root['module'] == 'importlib' and not value.path and name == 'import_module':
        return _library_import_module
    response = _library_rpc('get', value.root, value.path, [name], {}, value.auth)
    if 'state' in response:
        value.root.update(response['state'])
    if response.get('call'):
        target = _LibraryValue(value.root, response['path'], response['auth'])
        def invoke(*args, **kwargs):
            return _library_request(target, 'call', *args, **kwargs)
        return invoke
    return _library_unpack(response['result'], value.root)


def _library_item(value, key):
    if isinstance(value, _LibraryGlobals):
        return value._values[key]
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'item', key)
    return value[key]


def _library_setitem(value, key, item):
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'setitem', key, item)
    value[key] = item


def _library_setattr(value, name, item):
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'setattr', name, item)
    setattr(value, name, item)


def _library_len(value):
    if isinstance(value, _LibraryGlobals):
        return len(value._values)
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'len')
    return len(value)


def _library_iter(value):
    if isinstance(value, _LibraryGlobals):
        return value._values
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'iterate')
    return value


def _library_display(value):
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'repr')
    return value


def _library_str(value=''):
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'repr')
    return str(value)


def _library_float(value=0.0):
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'convert', 'float')
    return float(value)


def _library_int(value=0, base=None):
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'convert', 'int')
    return int(value) if base is None else int(value, base)


def _library_bool(value=False):
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'convert', 'bool')
    return bool(value)


def _library_result(value):
    if isinstance(value, _LibraryValue):
        return _LibraryRender(_library_request(value, 'repr'))
    return value


def _library_print(*args, **kwargs):
    print(*[_library_display(value) for value in args], **kwargs)


def _library_list(value=()):
    return list(_library_iter(value))


def _library_tuple(value=()):
    return tuple(_library_iter(value))


def _library_binary(name, left, right):
    if isinstance(left, _LibraryValue):
        return _library_request(left, 'binary', name, right, False)
    if isinstance(right, _LibraryValue):
        return _library_request(right, 'binary', name, left, True)
    operations = {'Add': lambda: left + right, 'Sub': lambda: left - right,
                  'Mult': lambda: left * right, 'Div': lambda: left / right,
                  'FloorDiv': lambda: left // right, 'Mod': lambda: left % right,
                  'Pow': lambda: left ** right, 'MatMult': lambda: left @ right,
                  'BitAnd': lambda: left & right, 'BitOr': lambda: left | right,
                  'BitXor': lambda: left ^ right, 'LShift': lambda: left << right,
                  'RShift': lambda: left >> right, 'Eq': lambda: left == right,
                  'NotEq': lambda: left != right, 'Lt': lambda: left < right,
                  'LtE': lambda: left <= right, 'Gt': lambda: left > right,
                  'GtE': lambda: left >= right}
    return operations[name]()


def _library_unary(name, value):
    if isinstance(value, _LibraryValue):
        return _library_request(value, 'unary', name)
    operations = {'USub': lambda: -value, 'UAdd': lambda: +value, 'Invert': lambda: ~value}
    return operations[name]()


def _library_inplace(name, left, right):
    if isinstance(left, _LibraryValue):
        return _library_request(left, 'inplace', name, right)
    if name == 'Add':
        left += right
    elif name == 'Sub':
        left -= right
    elif name == 'Mult':
        left *= right
    elif name == 'Div':
        left /= right
    else:
        return _library_binary(name, left, right)
    return left


def _library_augitem(value, key, name, other):
    _library_setitem(value, key, _library_binary(name, _library_item(value, key), other))


def _library_augattr(value, attr, name, other):
    _library_setattr(value, attr, _library_binary(name, _library_get(value, attr), other))
