import pdfplumber as pdf
from pdfplumber.page import Page
from .files import source
from .common import iterate, assign


def exports(fs):
    def open_pdf(path, **kwargs):
        if set(kwargs) - {'password', 'unicode_norm', 'laparams'}:
            raise ValueError('Unsupported PDF options')
        stream = source(fs, path)
        return pdf.open(stream, **kwargs)
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


def is_view(value):
    return isinstance(value, Page)


def iterate(value):
    return iter(value.pages if isinstance(value, pdf.PDF) else value)
