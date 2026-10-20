"""Session-scoped authenticated RAM handles; native payloads never enter Monty."""
import hmac
import operator
import secrets
import sys
import threading
from collections import OrderedDict
from datetime import date, time, timedelta
from itertools import islice
from pathlib import PurePath
from .bridge import backend, member, TypeRef, PRIMITIVES, OPERATORS, INPLACE
from .registry import require, ALIASES, MODULES
from .limits import memory_scope, memory_errors, memory_bytes


def plain(value, depth=0):
    if depth > 64:
        raise ValueError('Native arguments are nested too deeply')
    if type(value) in (str, bytes, bool, int, float, complex, type(None)) or isinstance(value, (date, time, timedelta)):
        return value
    if isinstance(value, PurePath):
        return str(value)
    if type(value) in (list, tuple):
        return type(value)(plain(item, depth + 1) for item in value)
    if type(value) is dict:
        return {plain(k): plain(v, depth + 1) for k, v in value.items()}
    raise TypeError('Native arguments must be plain data or authenticated library references')


class NativeMemory:
    """Worker-wide LRU. Eviction invalidates references, never serializes data."""
    def __init__(self, budget):
        self.budget, self.entries, self.modules = budget, OrderedDict(), OrderedDict()
        self.identities = {}
        self.lock = threading.RLock()

    def discard(self, identity):
        owner,module,value = self.entries.pop(identity)
        self.identities.pop((owner,module,id(value)),None)

    def clear(self, owner):
        with self.lock:
            for key in [k for k, row in self.entries.items() if row[0] == owner]:
                self.discard(key)
            for key in [k for k in self.modules if k[0] == owner]:
                del self.modules[key]

    def size(self):
        seen = set()
        def measure(value):
            if id(value) in seen:
                return 0
            seen.add(id(value))
            module = type(value).__module__.split('.')[0]
            if module == 'numpy':
                np = backend('numpy').np
                if isinstance(value, np.ndarray):
                    base = value
                    while isinstance(base.base, np.ndarray):
                        base = base.base
                    return 128 + (measure(base) if base is not value else value.nbytes)
            if module == 'PIL':
                from PIL import Image
                if isinstance(value, Image.Image):
                    return 128 + value.width * value.height * max(4, len(value.getbands()))
            if module == 'pandas':
                adapter = backend('pandas')
                pd = adapter.pd
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    usage = value.memory_usage(deep=True)
                    return int(usage.sum() if isinstance(value, pd.DataFrame) else usage)
                if isinstance(value, adapter._LocationIndexer):
                    return measure(value.obj)
                if isinstance(value, (adapter.StringMethods, adapter.DatetimeProperties, adapter.TimedeltaProperties)):
                    return measure(value._parent)
            if type(value).__module__ == 'wrappers.binary':
                from .binary import Buffer
                if isinstance(value, Buffer):
                    data = value.saved if value.stream.closed else value.stream.getvalue()
                    return 128 + len(data) * (4 if value.text else 1)
            if type(value) is dict:
                return sys.getsizeof(value) + sum(measure(k) + measure(v) for k,v in value.items())
            if type(value) in (list, tuple):
                return sys.getsizeof(value) + sum(map(measure, value))
            if type(value).__module__.startswith('wrappers.') and not callable(value):
                return sys.getsizeof(value) + measure(vars(value))
            return sys.getsizeof(value)
        return sum(measure(row[2]) for row in self.entries.values()) + sum(map(measure,self.modules.values()))

    def enforce(self, pinned=(), extra=0, owner=None):
        pinned = set(pinned)
        while self.size() + extra > self.budget:
            victim = next((k for k in self.entries if k not in pinned), None)
            if victim is None:
                module = next((k for k in self.modules if k[0] != owner), None)
                if module is None:
                    raise MemoryError()
                del self.modules[module]
            else:
                self.discard(victim)


class Bridge:
    def __init__(self, fs, key, load=backend, memory_budget=None, memory=None):
        self.fs, self.key, self.load = fs, key, load
        self.memory_budget = memory_bytes() if memory_budget is None else memory_budget
        self.memory = memory if memory is not None else NativeMemory(self.memory_budget)
        self.loaded = set()

    def initialize(self, module):
        if module not in self.loaded:
            self.load(module)
            self.loaded.add(module)

    def mac(self, value):
        return hmac.digest(self.key, repr(value).encode(), 'sha256').hex()

    def seal(self, module, kind, identity):
        return dict(module=module, kind=kind, identity=identity, seal=self.mac((module,kind,identity)))

    def check(self, spec):
        if type(spec) is not dict or set(spec) != {'module','kind','identity','seal'}:
            raise ValueError('Unsigned library reference')
        fields = spec['module'],spec['kind'],spec['identity']
        if any(type(item) is not str for item in (*fields,spec['seal'])) or not hmac.compare_digest(spec['seal'],self.mac(fields)):
            raise ValueError('Invalid library state signature')

    def path_mac(self, spec, path):
        return self.mac((spec['identity'],path))

    def check_path(self, spec, path, auth):
        if not isinstance(path,(list,tuple)) or len(path)>128 or any(
                not isinstance(step,(list,tuple)) or len(step)!=2 or step[0]!='attr' or type(step[1]) is not str for step in path):
            raise ValueError('Invalid library path')
        if path and (type(auth) is not str or not hmac.compare_digest(auth,self.path_mac(spec,path))):
            raise ValueError('Invalid library path signature')

    def resolve(self, spec, path=(), auth=None):
        self.check(spec)
        self.check_path(spec,path,auth)
        module = spec['module']
        self.initialize(module)
        api = backend(module)
        if spec['kind']=='module' and spec['identity']=='module:'+module:
            if module in ('numpy','matplotlib'):
                key = self.key,module
                if key not in self.memory.modules:
                    self.memory.modules[key] = api.Session() if module=='numpy' else api.exports(self.fs)
                self.memory.modules.move_to_end(key)
                value = self.memory.modules[key]
            else:
                value = api.exports(self.fs)
        else:
            try:
                owner,stored_module,value = self.memory.entries[spec['identity']]
            except KeyError:
                raise RuntimeError('Library object is no longer in memory; recompute it or reload it from a saved file') from None
            if owner!=self.key or stored_module!=module:
                raise ValueError('Invalid library reference owner')
            self.memory.entries.move_to_end(spec['identity'])
        for step in path:
            value = member(value,step[1],module,self.fs)
        return value

    def arguments(self, value, refs):
        if type(value) is dict:
            if '__library_value__' in value:
                if set(value)!={'__library_value__','path','auth'}:
                    raise ValueError('Invalid library reference')
                spec = value['__library_value__']
                result = self.resolve(spec,value['path'],value['auth'])
                if callable(result) and not (spec['module']=='numpy' and isinstance(result,type) and result in backend('numpy').SCALARS.values()):
                    raise TypeError('Native library callbacks cannot be passed as data')
                refs.append((result,spec,value['path'],value['auth']))
                return result
            if set(value)=={'__library_slice__'}:
                return slice(*value['__library_slice__'])
            if value=={'__library_ellipsis__':True}:
                return Ellipsis
            return {plain(k):self.arguments(v,refs) for k,v in value.items()}
        if isinstance(value,(list,tuple)):
            return (tuple if isinstance(value,tuple) else list)(self.arguments(item,refs) for item in value)
        return plain(value)

    def encode(self, value, module, pinned):
        if isinstance(value,PRIMITIVES):
            for kind in (bool,str,bytes,int,float,complex):
                if isinstance(value,kind):
                    return kind(value)
            return value
        if isinstance(value,(list,tuple)):
            return (tuple if isinstance(value,tuple) else list)(self.encode(item,module,pinned) for item in value)
        if type(value) is dict and all(isinstance(k,PRIMITIVES) for k in value):
            return {'__library_dict__':[[k,self.encode(v,module,pinned)] for k,v in value.items()]}
        native_module = type(value).__module__.split('.')[0]
        if native_module=='numpy':
            api = backend('numpy')
            if isinstance(value,api.np.generic):
                return self.encode(value.item(),module,pinned)
            if isinstance(value,api.np.ndarray):
                api.checked_size(value.shape,value.dtype)
            module = 'numpy'
        elif native_module=='pandas':
            module = 'pandas'
        elif type(value).__module__.startswith('wrappers.'):
            adapter_name = type(value).__module__.split('.')[1]
            if adapter_name not in ('stdlib','binary'):
                module = next((name for name,adapter in MODULES.items() if adapter==adapter_name),module)
        lookup = self.key,module,id(value)
        identity = self.memory.identities.get(lookup)
        if identity is None:
            identity = secrets.token_hex(16)
            self.memory.entries[identity] = self.key,module,value
            self.memory.identities[lookup] = identity
        pinned.add(identity)
        return {'__library_object__':self.seal(module,'object',identity)}

    @memory_errors
    def __call__(self, action, spec, path=(), args=(), kwargs=None, auth=None):
        with self.memory.lock, memory_scope(self.memory_budget):
            return self._call(action,spec,path,args,kwargs,auth)

    def _call(self, action, spec, path, args, kwargs, auth):
        if action=='import':
            if type(spec) is not str:
                raise ValueError('Expected a library name')
            module = ALIASES.get(spec,spec)
            require(module)
            self.initialize(module)
            return self.seal(module,'module','module:'+module)
        value = self.resolve(spec,path,auth)
        module,api = spec['module'],backend(spec['module'])
        refs = []
        args,kwargs = self.arguments(args,refs),self.arguments(kwargs or {},refs)
        pinned = {spec['identity'],*(ref[1]['identity'] for ref in refs)}
        if action=='get':
            result = member(value,args[0],module,self.fs)
            if (callable(result) and not isinstance(result,TypeRef) and not api.is_view(result)) or (spec['kind']=='module' and (type(result) is dict or result is value)):
                target = [*path,['attr',args[0]]]
                return ({'call':True,'path':target,'auth':self.path_mac(spec,target)} if callable(result) else
                        {'result':{'__library_view__':target,'auth':self.path_mac(spec,target)}})
        elif action in ('call','method'):
            function = member(value,args[0],module,self.fs) if action=='method' else value
            if action=='method':
                args = args[1:]
            result = function(*args,**kwargs)
        elif action=='item':
            result = value[args[0]]
        elif action=='setitem':
            if spec['kind']=='module' and type(value) is dict:
                raise TypeError('Library namespaces are read-only')
            if module=='sympy':
                api.validate(args[1])
            value[args[0]] = args[1]
            result = None
        elif action=='delitem':
            if module!='pandas' or spec['kind']=='module':
                raise TypeError('object does not support item deletion')
            operator.delitem(value,args[0])
            result = None
        elif action=='setattr':
            api.assign(value,args[0],args[1])
            result = None
        elif action=='len':
            result = len(value)
        elif action=='callable':
            result = callable(value)
        elif action=='is_type':
            result = isinstance(value,(TypeRef,type))
        elif action=='isinstance':
            result = isinstance(value,args[0].cls if isinstance(args[0],TypeRef) else args[0])
        elif action=='dir':
            result = sorted(k for k in (value if type(value) is dict else api.members(value,self.fs)) if not k.startswith('_'))
        elif action=='iterate':
            result = list(islice(api.iterate(value),10_001))
            if len(result)>10_000:
                raise ValueError('library iteration is limited to 10,000 items; prefer vectorized operations')
        elif action=='repr':
            result = f"<module '{module}'>" if spec['kind']=='module' and not path else repr(value)
        elif action=='convert':
            result = {'float':float,'int':int,'bool':bool}[args[0]](value)
        elif action in ('binary','inplace'):
            name,other = args[:2]
            reverse = action=='binary' and args[2]
            if module=='numpy':
                api.preflight_binary(name,other if reverse else value,value if reverse else other)
            if module=='sympy':
                api.validate(other)
            function = (INPLACE if action=='inplace' else OPERATORS)[name]
            result = function(other,value) if reverse else function(value,other)
        elif action=='unary':
            result = OPERATORS[args[0]](value)
        else:
            raise ValueError('Unknown library action: '+action)
        for original,original_spec,original_path,original_auth in refs:
            if result is original:
                self.memory.enforce(pinned,owner=self.key)
                return {'argument_result':dict(module=original_spec['module'],identity=original_spec['identity'],path=original_path,auth=original_auth)}
        response = {'result':self.encode(result,module,pinned)}
        self.memory.enforce(pinned,owner=self.key)
        return response
