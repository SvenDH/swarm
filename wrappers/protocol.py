"""Authenticate untrusted Monty state before any native restore or path replay.

This is a capability boundary, not an attempt to hide Python helper names.
Keys never enter Monty; descriptors and view paths can be inspected but not forged.
"""
import hashlib
import hmac
import secrets
from datetime import date, time, timedelta
from pathlib import PurePath

from .bridge import dispatch, backend
from .registry import require
from .limits import memory_scope


def plain(value, depth=0):
    """No Monty class/function proxies reach native duck-typed APIs."""
    if depth > 64:
        raise ValueError('Native arguments are nested too deeply')
    if type(value) in (str, bytes, bool, int, float, complex, type(None)) or isinstance(value, (date, time, timedelta)):
        return value
    if isinstance(value, PurePath):
        return str(value)
    if type(value) in (list, tuple):
        return type(value)(plain(item, depth + 1) for item in value)
    if type(value) is dict:
        return {plain(key, depth + 1): plain(item, depth + 1) for key, item in value.items()}
    raise TypeError(f'Native arguments must be plain data or authenticated library references; received {type(value).__name__}')


def fingerprint(value, depth=0):
    if depth > 64:
        raise ValueError('Library state is nested too deeply')
    digest = hashlib.sha256()
    if type(value) in (list, tuple):
        digest.update(b'l' if type(value) is list else b't')
        digest.update(len(value).to_bytes(8, 'big'))
        for item in value:
            digest.update(fingerprint(item, depth + 1))
    elif type(value) is dict:
        digest.update(b'd')
        for key, item in sorted((fingerprint(key, depth + 1), item) for key, item in value.items()):
            digest.update(key)
            digest.update(fingerprint(item, depth + 1))
    else:
        if value is None:
            tag, data = b'n', b''
        elif type(value) is bool:
            tag, data = b'b', bytes([value])
        elif type(value) is int:
            tag, data = b'i', str(value).encode()
        elif type(value) is float:
            tag, data = b'f', value.hex().encode()
        elif type(value) is complex:
            tag, data = b'c', (value.real.hex() + ',' + value.imag.hex()).encode()
        elif type(value) is str:
            tag, data = b's', value.encode()
        elif type(value) is bytes:
            tag, data = b'y', value
        elif isinstance(value, (date, time, timedelta)):
            tag, data = b'z', repr(value).encode()
        else:
            raise TypeError(f'Library state must contain only serializable data, not {type(value).__name__}')
        digest.update(tag)
        digest.update(len(data).to_bytes(8, 'big'))
        digest.update(data)
    return digest.digest()


class Bridge:
    def __init__(self, fs, key, load=backend, memory_budget=None):
        self.fs, self.key, self.load = fs, key, load
        self.memory_budget = memory_budget
        self.loaded = set()

    def initialize(self, module):
        if module not in self.loaded:
            self.load(module)
            self.loaded.add(module)

    def mac(self, domain, value):
        return hmac.digest(self.key, domain + fingerprint(value), 'sha256').hex()

    def seal(self, spec, identity=None):
        root = {key: spec[key] for key in ('module', 'kind', 'data')}
        root['identity'] = identity or secrets.token_hex(16)
        return {**root, 'seal': self.mac(b'root:', root)}

    def check(self, spec):
        if type(spec) is not dict or set(spec) != {'module', 'kind', 'data', 'identity', 'seal'}:
            raise ValueError('Unsigned library state; create a new library session')
        root = {key: spec[key] for key in ('module', 'kind', 'data', 'identity')}
        if type(spec['seal']) is not str or not hmac.compare_digest(spec['seal'], self.mac(b'root:', root)):
            raise ValueError('Invalid library state signature')

    def path_mac(self, spec, path):
        return self.mac(b'path:', [spec['module'], spec['identity'], path])

    def check_path(self, spec, path, auth):
        if not isinstance(path, (list, tuple)) or len(path) > 128:
            raise ValueError('Invalid library path')
        if path and (type(auth) is not str or not hmac.compare_digest(auth, self.path_mac(spec, path))):
            raise ValueError('Invalid library path signature')

    def arguments(self, value, roots):
        if type(value) is dict:
            if '__library_value__' in value:
                if set(value) != {'__library_value__', 'path', 'auth'}:
                    raise ValueError('Invalid library reference')
                spec, path = value['__library_value__'], value['path']
                self.check(spec)
                self.check_path(spec, path, value['auth'])
                roots[(spec['module'], spec['identity'])] = spec
                # The internal decoder only receives already-verified references.
                return {'__library_value__': spec, 'path': path}
            return {plain(key): self.arguments(item, roots) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self.arguments(item, roots) for item in value)
        return plain(value)

    def result(self, value, root):
        if type(value) is dict:
            if '__library_object__' in value:
                return {'__library_object__': self.seal(value['__library_object__'])}
            if '__library_view__' in value:
                path = value['__library_view__']
                return {'__library_view__': path, 'auth': self.path_mac(root, path)}
            return {key: self.result(item, root) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(self.result(item, root) for item in value)
        return value

    def __call__(self, action, spec, path=(), args=(), kwargs=None, auth=None):
        with memory_scope(self.memory_budget):
            return self._call(action, spec, path, args, kwargs, auth)

    def _call(self, action, spec, path, args, kwargs, auth):
        if action == 'import':
            if type(spec) is not str:
                raise ValueError('Expected a library name')
            require(spec)
            self.initialize(spec)
            return self.seal({'module': spec, 'kind': 'module', 'data': None})
        self.check(spec)
        self.check_path(spec, path, auth)
        roots = {(spec['module'], spec['identity']): spec}
        args, kwargs = self.arguments(args, roots), self.arguments(kwargs or {}, roots)
        for module, identity in roots:
            self.initialize(module)
        response = dispatch(self.fs, action, spec, path, args, kwargs)
        if 'state' in response:
            response['state'] = self.seal(response['state'], spec['identity'])
        if response.get('call'):
            call_path = [*path, ['attr', args[0]]]
            response.update(path=call_path, auth=self.path_mac(spec, call_path))
        if 'result' in response:
            response['result'] = self.result(response['result'], spec)
        for update in response.get('updates', ()):
            original = roots[(update['module'], update['identity'])]
            update['state'] = self.seal({**original, 'data': update['data']}, original['identity'])
        if 'argument_result' in response:
            result = response['argument_result']
            original = roots[(result['module'], result['identity'])]
            result['auth'] = self.path_mac(original, result['path'])
        return response
