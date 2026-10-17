import pdfplumber as pdf
from pdfplumber.page import Page, CroppedPage
from .files import source
from .bridge import descriptor
from .common import iterate, assign


def exports(fs):
    def open_pdf(path, **kwargs):
        if set(kwargs) - {'password', 'unicode_norm', 'laparams'}:
            raise ValueError('Unsupported PDF options')
        stream = source(fs, path)
        document = pdf.open(stream, **kwargs)
        document._swarm_source = stream.getvalue(), kwargs
        return document
    return {'open': open_pdf}


def members(value, fs, name=None):
    if isinstance(value, pdf.PDF):
        return {'pages': value.pages, 'metadata': value.metadata, 'close': lambda: None}
    if isinstance(value, Page):
        return {'page_number': value.page_number, 'width': value.width, 'height': value.height,
                'bbox': value.bbox, 'chars': value.chars, 'rects': value.rects,
                'lines': value.lines, 'extract_text': value.extract_text,
                'extract_words': value.extract_words, 'extract_tables': value.extract_tables,
                'extract_table': value.extract_table, 'crop': value.crop,
                'within_bbox': value.within_bbox}
    return {}


def pack(value):
    if isinstance(value, pdf.PDF):
        return descriptor('pdfplumber', 'document', value._swarm_source)
    if isinstance(value, Page):
        return descriptor('pdfplumber', 'page', [value.pdf._swarm_source, value.page_number - 1, list(value.bbox)])


def restore(kind, data):
    original, kwargs = data if kind == 'document' else data[0]
    import io
    document = pdf.open(io.BytesIO(original), **kwargs)
    document._swarm_source = original, kwargs
    if kind == 'document':
        return document
    if kind == 'page':
        page = document.pages[data[1]]
        return page if list(page.bbox) == data[2] else page.crop(data[2])
    raise ValueError('Unsupported PDF object')


def view(root, value):
    if isinstance(root, pdf.PDF) and isinstance(value, Page) and value.pdf is root and not isinstance(value, CroppedPage):
        return [['attr', 'pages'], ['item', value.page_number - 1]]


def is_view(value):
    return isinstance(value, Page)


def iterate(value):
    return iter(value.pages if isinstance(value, pdf.PDF) else value)
