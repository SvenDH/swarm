"""PDF-only PyMuPDF with in-memory documents, pixels and virtual IO."""
import pymupdf as pdf
from .bridge import descriptor
from .files import source
from .common import assign
from .limits import check_memory
import math


def exports(fs):
    def open_pdf(filename=None, *, stream=None, filetype='pdf'):
        if filetype != 'pdf':
            raise ValueError('Only PDF documents are exposed')
        if filename is not None:
            stream = source(fs, filename).getvalue()
        return pdf.open() if stream is None else pdf.open(stream=stream, filetype='pdf')
    return {**{name: value for name, value in vars(pdf).items()
               if name.startswith(('PDF_', 'TEXT_', 'LINK_', 'ANNOT_', 'CS_')) and type(value) in (int, float)},
            'open': open_pdf, 'Document': open_pdf, 'Rect': pdf.Rect, 'Point': pdf.Point,
            'Matrix': pdf.Matrix}


def members(value, fs, name=None):
    if isinstance(value, pdf.Document):
        def write(filename, **kwargs):
            if set(kwargs) - {'garbage', 'deflate', 'clean'}:
                raise ValueError('Unsupported PDF save options')
            return fs.write(filename, value.tobytes(**kwargs))
        return {'page_count': value.page_count, 'metadata': value.metadata,
                'load_page': value.load_page, 'new_page': value.new_page,
                'delete_page': value.delete_page, 'insert_pdf': value.insert_pdf,
                'save': write, 'tobytes': value.tobytes, 'close': lambda: None}
    if isinstance(value, pdf.Page):
        def pixmap(*, matrix=None, dpi=None, colorspace=None, clip=None, alpha=False, annots=True):
            rect = pdf.Rect(clip) if clip is not None else value.rect
            transform = matrix or pdf.Matrix(1, 1)
            if dpi is not None:
                transform = pdf.Matrix(dpi / 72, dpi / 72)
            bounds = rect * transform
            check_memory(math.ceil(abs(bounds.width)) * math.ceil(abs(bounds.height)) * 4)
            if colorspace is not None:
                raise ValueError('Custom colorspaces are not exposed')
            return value.get_pixmap(matrix=transform, clip=clip, alpha=alpha, annots=annots)
        def insert_text(point, text, **kwargs):
            if set(kwargs) - {'fontsize', 'fontname', 'color', 'rotate', 'overlay'}:
                raise ValueError('Unsupported text options (host fonts are not exposed)')
            return value.insert_text(point, text, **kwargs)
        return {'number': value.number, 'rect': value.rect, 'get_text': value.get_text,
                'search_for': value.search_for, 'get_links': value.get_links,
                'get_pixmap': pixmap, 'insert_text': insert_text, 'set_rotation': value.set_rotation}
    if isinstance(value, pdf.Pixmap):
        def write(filename):
            return fs.write(filename, value.tobytes('png'))
        return {'width': value.width, 'height': value.height, 'samples': value.samples,
                'tobytes': value.tobytes, 'save': write}
    if isinstance(value, pdf.Rect):
        return {'width': value.width, 'height': value.height, 'x0': value.x0, 'y0': value.y0,
                'x1': value.x1, 'y1': value.y1}
    if isinstance(value, pdf.Point):
        return {'x': value.x, 'y': value.y, 'distance_to': value.distance_to}
    if isinstance(value, pdf.Matrix):
        return {'a': value.a, 'b': value.b, 'c': value.c, 'd': value.d, 'e': value.e, 'f': value.f,
                'is_rectilinear': value.is_rectilinear}
    return {}


def pack(value):
    if isinstance(value, pdf.Document):
        return descriptor('pymupdf', 'document', value.tobytes() if value.page_count else None)
    if isinstance(value, pdf.Pixmap):
        check_memory(value.width * value.height * value.n)
        return descriptor('pymupdf', 'pixmap', value.tobytes('png'))
    if isinstance(value, (pdf.Rect, pdf.Point, pdf.Matrix)):
        return descriptor('pymupdf', type(value).__name__, list(value))


def restore(kind, data):
    if kind == 'document':
        return pdf.open() if data is None else pdf.open(stream=data, filetype='pdf')
    if kind == 'pixmap':
        return pdf.Pixmap(data)
    return {'Rect': pdf.Rect, 'Point': pdf.Point, 'Matrix': pdf.Matrix}[kind](data)


def view(root, value):
    if isinstance(root, pdf.Document) and isinstance(value, pdf.Page):
        return [['item', value.number]]


def is_view(value):
    return isinstance(value, pdf.Page)


def iterate(value):
    return iter(value)
