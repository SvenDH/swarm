"""Optional host packages. Discovery never imports the heavy native libraries."""
from importlib.util import find_spec
from functools import lru_cache

STDLIB = {'os', 'stat', 'shutil', 'tempfile', 'importlib', 'platform', 'resource', 'gc', 'traceback'}
NATIVE = ('sys', 'math', 'pathlib', 'datetime', 'json', 'time', 'random', 're',
          'collections', 'itertools', 'functools', 'typing', 'base64', 'binascii')
BINARY = {'zlib', 'struct', 'hashlib', 'array', 'io'}
STDLIB |= BINARY

MODULES = {
    'numpy': 'numpy', 'pandas': 'pandas', 'PIL': 'pil', 'openpyxl': 'openpyxl',
    'pypdf': 'pypdf', 'docx': 'docx', 'matplotlib': 'matplotlib', 'scipy': 'scipy',
    'seaborn': 'seaborn', 'pptx': 'pptx', 'reportlab': 'reportlab',
    'pdfplumber': 'pdfplumber', 'pymupdf': 'pymupdf', 'bs4': 'bs4',
    'sympy': 'sympy', 'sklearn': 'sklearn', 'xlsxwriter': 'xlsxwriter', 'cv2': 'cv2',
}
MODULES.update({name: 'stdlib' for name in STDLIB})
MODULES.update({name: 'binary' for name in BINARY})
ALIASES = {'fitz': 'pymupdf'}
REQUIRES = {'pandas': ('numpy',), 'matplotlib': ('numpy', 'PIL'),
            'seaborn': ('numpy', 'pandas', 'matplotlib'),
            'scipy': ('numpy',), 'sklearn': ('numpy', 'scipy'), 'cv2': ('numpy', 'PIL')}


@lru_cache(maxsize=1)
def available():
    return {name: backend for name, backend in MODULES.items()
            if all(find_spec(dep) is not None for dep in (name, *REQUIRES.get(name, ())))}


def require(name):
    if name not in available():
        raise ImportError(f"No module named '{name}'")
