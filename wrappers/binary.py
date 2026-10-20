"""Plain-data stdlib adapters: no paths, descriptors, pointers or host callbacks."""
import array
import hashlib
import io
import operator
import struct
import zlib

from .bridge import descriptor
from .limits import check_memory, memory_bytes

HASHES = ('md5', 'sha1', 'sha224', 'sha256', 'sha384', 'sha512',
          'sha3_224', 'sha3_256', 'sha3_384', 'sha3_512', 'shake_128', 'shake_256',
          'blake2b', 'blake2s')


def checked_size(size):
    check_memory(operator.index(size))
    return size


class Buffer:
    def __init__(self, text, data=None, newline='\n'):
        self.text, self.newline = text, newline
        data = ('' if text else b'') if data is None else data
        check_memory(len(data) * (4 if text else 1))
        self.stream = io.StringIO(data, newline=newline) if text else io.BytesIO(data)
        self.saved = None

    def close(self):
        if not self.stream.closed:
            self.saved = self.stream.getvalue()
            self.stream.close()

    def write(self, data):
        if self.stream.closed:
            return self.stream.write(data)  # Native closed-file diagnostic.
        length = len(data)
        if self.text and self.newline == '\r\n' and isinstance(data, str):
            length += data.count('\n')
        check_memory(max(len(self.stream.getvalue()), self.stream.tell() + length) * (4 if self.text else 1))
        return self.stream.write(data)

    def writelines(self, lines):
        for line in lines:
            self.write(line)


class Hash:
    def __init__(self, name, data=b'', **options):
        check_memory(len(data))
        hashlib.new(name, data, **options)  # Validate algorithm and options now.
        self.name, self.data, self.options = name, data, options

    def native(self):
        return hashlib.new(self.name, self.data, **self.options)

    def update(self, data):
        check_memory(len(self.data) + len(data))
        # Concatenation also enforces hashlib's bytes-like input requirement.
        self.data += data

    def digest(self, length=None, *, hex=False):
        if length is not None:
            checked_size(operator.index(length) * (2 if hex else 1))
        native = self.native()
        method = native.hexdigest if hex else native.digest
        return method() if length is None else method(length)


class Array:
    def __init__(self, typecode, initializer=()):
        probe = array.array(typecode)
        check_memory(len(initializer) * (1 if isinstance(initializer, bytes) else probe.itemsize))
        self.value = array.array(typecode, initializer)

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        return iter(self.value)

    def __getitem__(self, key):
        value = self.value[key]
        return Array(value.typecode, value) if isinstance(value, array.array) else value

    def __setitem__(self, key, value):
        value = value.value if isinstance(value, Array) else value
        if isinstance(key, slice):
            check_memory((len(self.value) + len(value)) * self.value.itemsize)
        self.value[key] = value

    def __mul__(self, count):
        checked_size(len(self.value) * max(0, operator.index(count)) * self.value.itemsize)
        result = Array(self.value.typecode)
        result.value = self.value * count
        return result

    __rmul__ = __mul__

    def __imul__(self, count):
        checked_size(len(self.value) * max(0, operator.index(count)) * self.value.itemsize)
        self.value *= count
        return self

    def __add__(self, other):
        if not isinstance(other, Array):
            raise TypeError('can only append array to array')
        check_memory((len(self.value) + len(other.value)) * self.value.itemsize)
        result = Array(self.value.typecode)
        result.value = self.value + other.value
        return result

    def __iadd__(self, other):
        if not isinstance(other, Array):
            raise TypeError('can only extend array with array')
        check_memory((len(self.value) + len(other.value)) * self.value.itemsize)
        self.value += other.value
        return self

    def __eq__(self, other):
        return self.value == (other.value if isinstance(other, Array) else other)

    def grow(self, method, items, *args):
        count = 1 if method in ('append', 'insert') else len(items)
        check_memory((len(self.value) + count) * self.value.itemsize)
        return getattr(self.value, method)(*args, items) if method == 'insert' else getattr(self.value, method)(items)


def exports(module, fs):
    if module == 'io':
        return {'BytesIO': lambda initial_bytes=b'': Buffer(False, initial_bytes),
                'StringIO': lambda initial_value='', newline='\n': Buffer(True, initial_value, newline),
                'SEEK_SET': 0, 'SEEK_CUR': 1, 'SEEK_END': 2}
    if module == 'hashlib':
        def constructor(name):
            return lambda data=b'', **options: Hash(name, data, **options)
        def new(name, data=b'', **options):
            if name not in HASHES:
                raise ValueError('unsupported hash type ' + str(name))
            return Hash(name, data, **options)
        return {**{name: constructor(name) for name in HASHES}, 'new': new,
                'algorithms_available': tuple(HASHES), 'algorithms_guaranteed': tuple(HASHES)}
    if module == 'array':
        return {'array': Array, 'typecodes': array.typecodes}
    if module == 'struct':
        def pack(fmt, *values):
            checked_size(struct.calcsize(fmt))
            return struct.pack(fmt, *values)
        def unpack(fmt, data):
            check_memory(struct.calcsize(fmt) * 32)
            return struct.unpack(fmt, data)
        def unpack_from(fmt, data, offset=0):
            check_memory(struct.calcsize(fmt) * 32)
            return struct.unpack_from(fmt, data, offset)
        def make(fmt):
            checked_size(struct.calcsize(fmt))
            return struct.Struct(fmt)
        return {'pack': pack, 'unpack': unpack, 'unpack_from': unpack_from,
                'calcsize': struct.calcsize, 'Struct': make}
    if module == 'zlib':
        def compress(data, level=-1, wbits=zlib.MAX_WBITS):
            check_memory(len(data) + len(data) // 1000 + 64)
            engine = zlib.compressobj(level, wbits=wbits)
            return engine.compress(data) + engine.flush()
        def decompress(data, wbits=zlib.MAX_WBITS, bufsize=zlib.DEF_BUF_SIZE):
            if bufsize <= 0:
                raise ValueError('bufsize must be greater than zero')
            # Never call unbounded decompress/flush on potentially hostile input.
            engine = zlib.decompressobj(wbits)
            result = engine.decompress(data, memory_bytes() + 1)
            check_memory(len(result))
            if not engine.eof:
                raise zlib.error('Error -5 while decompressing data: incomplete or truncated stream')
            return result
        names = ('MAX_WBITS', 'DEFLATED', 'DEF_MEM_LEVEL', 'DEF_BUF_SIZE',
                 'Z_NO_COMPRESSION', 'Z_BEST_SPEED', 'Z_BEST_COMPRESSION', 'Z_DEFAULT_COMPRESSION',
                 'ZLIB_VERSION', 'ZLIB_RUNTIME_VERSION')
        return {**{name: getattr(zlib, name) for name in names}, 'compress': compress,
                'decompress': decompress, 'crc32': zlib.crc32, 'adler32': zlib.adler32}
    raise ImportError(module)


class Adapter:
    def __init__(self, module):
        self.module = module

    def exports(self, fs):
        return exports(self.module, fs)

    def pack(self, value):
        if isinstance(value, Buffer):
            closed = value.stream.closed
            data = [value.text, value.saved if closed else value.stream.getvalue(), value.newline,
                    0 if closed else value.stream.tell(), closed]
            return descriptor('io', 'buffer', data)
        if isinstance(value, Hash):
            return descriptor('hashlib', 'hash', [value.name, value.data, value.options])
        if isinstance(value, Array):
            return descriptor('array', 'array', [value.value.typecode, value.value.tobytes()])
        if isinstance(value, struct.Struct):
            return descriptor('struct', 'struct', value.format)

    def restore(self, kind, data):
        if self.module == 'io' and kind == 'buffer':
            text, content, newline, pos, closed = data
            result = Buffer(text, None if text else content, newline)
            if text:
                check_memory(len(content) * 4)
                # Restore already-normalized text without translating CRLF twice.
                # This is native in-memory state, never pickle or user callbacks.
                result.stream.__setstate__((content, newline, pos, None))
            else:
                result.stream.seek(pos)
            if closed:
                result.close()
            return result
        if self.module == 'hashlib' and kind == 'hash':
            return Hash(data[0], data[1], **data[2])
        if self.module == 'array' and kind == 'array':
            result = Array(data[0])
            check_memory(len(data[1]))
            result.value.frombytes(data[1])
            return result
        if self.module == 'struct' and kind == 'struct':
            return exports('struct', None)['Struct'](data)
        raise ValueError('Invalid library object')

    def members(self, value, fs, name=None):
        if isinstance(value, Buffer):
            stream = value.stream
            def read(size=-1, method='read'):
                # Native read(size) can allocate size even when the input is tiny.
                if size is not None and size >= 0:
                    checked_size(size * (4 if value.text else 1))
                return getattr(stream, method)(size)
            def readlines(hint=-1):
                if not stream.closed:
                    check_memory(len(stream.getvalue()) * 32)
                return stream.readlines(hint)
            return {'read': read, 'readline': lambda size=-1: read(size, 'readline'),
                    'readlines': readlines, 'write': value.write, 'writelines': value.writelines,
                    'seek': stream.seek, 'tell': stream.tell, 'getvalue': stream.getvalue,
                    'truncate': stream.truncate, 'close': value.close, 'closed': stream.closed,
                    'readable': stream.readable, 'writable': stream.writable,
                    'seekable': stream.seekable, 'flush': stream.flush}
        if isinstance(value, Hash):
            native = value.native()
            return {'name': native.name, 'digest_size': native.digest_size, 'block_size': native.block_size,
                    'update': value.update, 'digest': value.digest,
                    'hexdigest': lambda length=None: value.digest(length, hex=True),
                    'copy': lambda: Hash(value.name, value.data, **value.options)}
        if isinstance(value, Array):
            a = value.value
            def tolist():
                check_memory(len(a) * 32)
                return a.tolist()
            def extend(items):
                value.grow('extend', items.value if isinstance(items, Array) else items)
            def frombytes(data):
                check_memory(len(a) * a.itemsize + len(data))
                return a.frombytes(data)
            return {'typecode': a.typecode, 'itemsize': a.itemsize, 'tolist': tolist,
                    'tobytes': a.tobytes, 'append': lambda item: value.grow('append', item),
                    'extend': extend, 'insert': lambda index, item: value.grow('insert', item, index),
                    'fromlist': lambda items: value.grow('fromlist', items), 'frombytes': frombytes,
                    'pop': a.pop, 'remove': a.remove, 'count': a.count, 'index': a.index,
                    'reverse': a.reverse, 'byteswap': a.byteswap}
        if isinstance(value, struct.Struct):
            api = exports('struct', fs)
            return {'format': value.format, 'size': value.size,
                    **{method: (lambda *args, method=method: api[method](value.format, *args))
                       for method in ('pack', 'unpack', 'unpack_from')}}
        return {}

    def is_view(self, value):
        return False

    def view(self, root, value):
        return None

    def iterate(self, value):
        return iter(value)

    def assign(self, value, name, item):
        raise AttributeError('Attribute is read-only')
