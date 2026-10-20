"""Reviewed standard-library operations over a session's filesystem only."""

from .common import Adapter as BaseAdapter
import errno
import posixpath
import stat
import uuid

from .registry import available, ALIASES, NATIVE


class Result:
    def __init__(self, fields, values=None):
        self.fields, self.values = fields, values

    def __getitem__(self, key):
        return self.values[key] if self.values is not None else self.fields[key]

    def __iter__(self):
        return iter(self.values if self.values is not None else self.fields)

    def __len__(self):
        return len(self.values if self.values is not None else self.fields)


def unsupported(*args, **kwargs):
    raise NotImplementedError('Operation is not supported')


def exports(module, fs):
    if module == 'traceback':
        # Exception objects stay in the interpreter. Its facade handles these
        # functions without exposing host frames or native traceback objects.
        return {'format_exception_only': unsupported}
    def status(path, *, follow_symlinks=True):
        item = fs.path_stat(fs.path(path))
        fields = {name: getattr(item, name) for name in (
            'st_mode', 'st_ino', 'st_dev', 'st_nlink', 'st_uid', 'st_gid',
            'st_size', 'st_atime', 'st_mtime', 'st_ctime')}
        return Result(fields, tuple(fields.values()))

    def copyfile(src, dst, *, follow_symlinks=True):
        src, dst = fs.path(src), fs.path(dst)
        if src == dst:
            raise OSError(errno.EINVAL, 'Source and destination are the same file')
        content = fs.path_read_bytes(src)
        fs.write(str(dst), content)
        return str(dst)

    def copy(src, dst, *, follow_symlinks=True):
        target = fs.path(dst)
        if fs.path_is_dir(target):
            target /= fs.path(src).name
        return copyfile(src, str(target), follow_symlinks=follow_symlinks)

    def rmtree(path, ignore_errors=False, onerror=None, *, dir_fd=None, onexc=None):
        if onerror is not None or onexc is not None or dir_fd is not None:
            raise ValueError('Callbacks and file descriptors are not supported')
        path = fs.path(path)
        if str(path) == '/':
            raise PermissionError('Cannot remove the root directory')
        try:
            if not fs.path_is_dir(path):
                raise NotADirectoryError(str(path))
            for entry in reversed(list(fs.walk(str(path)))):
                if fs.path_is_dir(entry):
                    fs.path_rmdir(entry)
                else:
                    fs.path_unlink(entry)
        except OSError:
            if not ignore_errors:
                raise

    def copytree(src, dst, symlinks=False, ignore=None, copy_function=None,
                 ignore_dangling_symlinks=False, dirs_exist_ok=False):
        if symlinks or ignore is not None or copy_function is not None:
            raise ValueError('Symlinks and copy callbacks are not supported')
        src, dst = fs.path(src), fs.path(dst)
        if src == dst or src in dst.parents:
            raise ValueError('Destination must not be inside the source')
        if not fs.path_is_dir(src):
            raise NotADirectoryError(str(src))
        fs.path_mkdir(dst, True, dirs_exist_ok)
        for entry in fs.walk(str(src)):
            target = dst / entry.relative_to(src)
            if fs.path_is_dir(entry):
                fs.path_mkdir(target, True, True)
            else:
                copyfile(str(entry), str(target))
        return str(dst)

    def move(src, dst):
        target = fs.path(dst)
        if fs.path_is_dir(target):
            target /= fs.path(src).name
        fs.path_rename(fs.path(src), target)
        return str(target)

    def mkdtemp(suffix=None, prefix=None, dir=None):
        suffix, prefix = suffix or '', 'tmp' if prefix is None else prefix
        if '/' in prefix or '/' in suffix:
            raise ValueError('prefix and suffix must not contain path separators')
        parent = fs.path('/tmp' if dir is None else dir)
        if not fs.path_is_dir(parent):
            raise FileNotFoundError(str(parent))
        target = parent / (prefix + uuid.uuid4().hex + suffix)
        fs.path_mkdir(target, False, False)
        return str(target)

    def find_spec(name, package=None):
        if not isinstance(name, str):
            raise TypeError('module name must be a string')
        if name.startswith('.'):
            raise ImportError('Relative module discovery is not supported')
        root, *parts = name.split('.')
        root = ALIASES.get(root, root)
        namespaces = {'os.path', 'importlib.util', 'numpy.linalg', 'numpy.random', 'numpy.fft',
                      'PIL.Image', 'PIL.ImageDraw', 'PIL.ImageOps', 'PIL.ImageFilter',
                      'matplotlib.pyplot', 'scipy.ndimage', 'scipy.stats', 'scipy.special'}
        if ((parts and name not in namespaces) or
                root not in available() and root not in NATIVE):
            return None
        return Result({'name': name, 'origin': None, 'loader': None,
                       'submodule_search_locations': None})

    def walk(top, topdown=True, onerror=None, followlinks=False):
        if onerror is not None or followlinks:
            raise ValueError('Callbacks and symbolic link traversal are not supported')
        result = []
        for entry in fs.walk(top):
            if fs.path_is_dir(entry):
                children = sorted(fs.path_iterdir(entry))
                result.append((str(entry), [p.name for p in children if fs.path_is_dir(p)],
                               [p.name for p in children if fs.path_is_file(p)]))
        return result if topdown else list(reversed(result))

    if module == 'os':
        paths = {name: getattr(posixpath, name) for name in (
            'join', 'split', 'splitext', 'splitdrive', 'basename', 'dirname',
            'normpath', 'isabs', 'commonpath', 'commonprefix')}
        paths.update({'abspath': lambda path: str(fs.path(path)),
                      'realpath': lambda path, **kw: str(fs.path(path)),
                      'relpath': lambda path, start=None: posixpath.relpath(str(fs.path(path)), str(fs.path(start or fs.cwd))),
                      'exists': lambda path: fs.path_exists(fs.path(path)),
                      'lexists': lambda path: fs.path_exists(fs.path(path)),
                      'isfile': lambda path: fs.path_is_file(fs.path(path)),
                      'isdir': lambda path: fs.path_is_dir(fs.path(path)),
                      'islink': lambda path: fs.path_is_symlink(fs.path(path)),
                      'getsize': lambda path: fs.path_stat(fs.path(path)).st_size,
                      'expanduser': lambda path: str(path).replace('~/', fs.getenv('HOME') + '/', 1)
                          if str(path).startswith('~/') else fs.getenv('HOME') if str(path) == '~' else str(path)})
        return {'name': 'posix', 'sep': '/', 'pathsep': ':', 'curdir': '.', 'pardir': '..',
                'linesep': '\n', 'devnull': '/dev/null', 'path': paths,
                'getcwd': lambda: fs.cwd, 'getenv': fs.getenv,
                'environ': Result(fs.get_environ()), 'stat': status, 'lstat': status,
                'listdir': lambda path='.': [entry.name for entry in fs.path_iterdir(fs.path(path))],
                'walk': walk,
                'mkdir': lambda path, mode=0o777: fs.path_mkdir(fs.path(path), False, False),
                'makedirs': lambda name, mode=0o777, exist_ok=False: fs.path_mkdir(fs.path(name), True, exist_ok),
                'remove': lambda path: fs.path_unlink(fs.path(path)),
                'unlink': lambda path: fs.path_unlink(fs.path(path)),
                'rmdir': lambda path: fs.path_rmdir(fs.path(path)),
                'rename': lambda src, dst: fs.path_rename(fs.path(src), fs.path(dst)),
                'replace': lambda src, dst: fs.path_rename(fs.path(src), fs.path(dst)),
                'fspath': str, 'fsencode': lambda path: str(path).encode(),
                'fsdecode': lambda path: path.decode() if isinstance(path, bytes) else str(path),
                'chdir': unsupported}
    if module == 'stat':
        return {name: getattr(stat, name) for name in (
            'S_ISDIR', 'S_ISREG', 'S_ISLNK', 'S_ISFIFO', 'S_ISSOCK', 'S_IFMT', 'S_IMODE',
            'S_IFDIR', 'S_IFREG', 'S_IFLNK', 'S_IRUSR', 'S_IWUSR', 'S_IXUSR')}
    if module == 'shutil':
        return {'copyfile': copyfile, 'copy': copy, 'copytree': copytree, 'move': move, 'rmtree': rmtree}
    if module == 'tempfile':
        return {'gettempdir': lambda: '/tmp', 'gettempprefix': lambda: 'tmp', 'mkdtemp': mkdtemp,
                'mkstemp': unsupported}
    if module == 'importlib':
        return {'util': {'find_spec': find_spec}, 'invalidate_caches': lambda: None}
    if module == 'platform':
        return {'system': lambda: 'Linux'}
    if module == 'gc':
        return {'collect': unsupported}
    if module == 'resource':
        def getrlimit(resource):
            if resource != 9:
                raise ValueError('invalid resource specified')
            return (fs.memory_budget, fs.memory_budget)
        def setrlimit(resource, limits):
            getrlimit(resource)
            raise PermissionError('Resource limit changes are not permitted')
        return {'RLIMIT_AS': 9, 'RLIM_INFINITY': -1, 'RUSAGE_SELF': 0,
                'getrlimit': getrlimit, 'setrlimit': setrlimit, 'getrusage': unsupported}
    raise ImportError(module)


class Adapter(BaseAdapter):
    def __init__(self, module):
        self.module = module

    def exports(self, fs):
        return exports(self.module, fs)


    def members(self, value, fs, name=None):
        if isinstance(value, Result):
            return {**value.fields, 'get': value.fields.get,
                    'keys': lambda: list(value.fields), 'items': lambda: list(value.fields.items()),
                    'values': lambda: list(value.fields.values())}
        return {}
