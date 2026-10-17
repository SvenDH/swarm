"""Optional host packages. Discovery never imports the heavy native libraries."""
from importlib.util import find_spec
from functools import lru_cache

MODULES = {
    'numpy': 'numpy', 'pandas': 'pandas', 'PIL': 'pil', 'openpyxl': 'openpyxl',
    'pypdf': 'pypdf', 'docx': 'docx', 'matplotlib': 'matplotlib', 'scipy': 'scipy',
    'seaborn': 'seaborn', 'pptx': 'pptx', 'reportlab': 'reportlab',
    'pdfplumber': 'pdfplumber', 'pymupdf': 'pymupdf', 'bs4': 'bs4',
    'sympy': 'sympy', 'sklearn': 'sklearn', 'xlsxwriter': 'xlsxwriter',
}
ALIASES = {'fitz': 'pymupdf'}
REQUIRES = {'pandas': ('numpy',), 'matplotlib': ('numpy', 'PIL'),
            'seaborn': ('numpy', 'pandas', 'matplotlib'),
            'scipy': ('numpy',), 'sklearn': ('numpy', 'scipy')}
SUMMARIES = {
    'numpy': 'arrays, tile, slicing/operators, reductions, linalg, random.default_rng, load/save',
    'pandas': 'DataFrame/Series, loc/iloc, groupby, merge, CSV/Excel IO',
    'PIL': 'Image.open/new/fromarray, resize/crop/rotate/convert, pixels, save',
    'openpyxl': 'Workbook/load_workbook, sheets/cells/rows, save',
    'pypdf': 'PdfReader/PdfWriter, pages, extract_text, rotate, append/write',
    'docx': 'Document, paragraphs/runs/tables/pictures, shared.Inches/Cm/Pt, save',
    'matplotlib': 'pyplot.figure/subplots, plot/scatter/bar/hist/imshow, labels/legend, savefig',
    'scipy': 'stats distributions/tests, linalg, signal, fft, interpolate, linprog, distances; no callbacks',
    'seaborn': 'scatterplot/lineplot/histplot/boxplot/barplot/heatmap/violinplot/kdeplot; use ax=',
    'pptx': 'Presentation, slides/layouts/shapes/text/tables/pictures, util.Inches/Pt, save',
    'reportlab': 'pdfgen.canvas.Canvas, text/lines/shapes/images, showPage/save, page sizes/colors',
    'pdfplumber': 'open(path), pages, extract_text/extract_tables, crop, words/chars',
    'pymupdf': 'open(path), pages, get_text/get_pixmap, insert_text, new_page, save; fitz alias',
    'bs4': 'BeautifulSoup(html, "html.parser"), find/find_all/select, text/attrs, tag edits',
    'sympy': 'Symbol/symbols, arithmetic, Eq/solve, expand/factor/simplify, diff/integrate, Matrix',
    'sklearn': 'preprocessing, model_selection.train_test_split, metrics, linear models, KMeans, PCA',
    'xlsxwriter': 'Workbook(path), add_worksheet/add_format/add_chart, write/rows/formulas, close',
}


@lru_cache(maxsize=1)
def available():
    return {name: backend for name, backend in MODULES.items()
            if all(find_spec(dep) is not None for dep in (name, *REQUIRES.get(name, ())))}


def require(name):
    if name not in available():
        raise ImportError(f'{name} is not available: install its host package and dependencies first')


def help_text():
    return 'Available optional libraries (familiar imports; curated APIs only):\n' + '\n'.join(
        f'  {name}: {SUMMARIES[name]}' for name in available())
