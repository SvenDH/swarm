from pypdf import PdfReader, PdfWriter
from pypdf._page import PageObject
import io

from .files import source, save
from .bridge import descriptor


def run(fs, op, path, output=None, pages=None):
    reader = PdfReader(source(fs, path))
    indices = range(len(reader.pages)) if pages is None else pages
    if op == 'read':
        return [reader.pages[i].extract_text() for i in indices]
    if op != 'select':
        raise ValueError('unknown PDF operation')
    writer = PdfWriter()
    for i in indices:
        writer.add_page(reader.pages[i])
    return save(fs, output, writer.write)


def exports(fs):
    def writer(*, clone_from=None):
        if clone_from is None:
            return PdfWriter()
        return PdfWriter(clone_from=clone_from if isinstance(clone_from, PdfReader) else source(fs, clone_from))
    return {'PdfReader': lambda stream, **kwargs: PdfReader(source(fs, stream), **kwargs),
            'PdfWriter': writer}


def members(value, fs, name=None):
    if isinstance(value, PdfWriter):
        def append(fileobj, **kwargs):
            return value.append(fileobj if isinstance(fileobj, PdfReader) else source(fs, fileobj), **kwargs)
        return {'pages': value.pages, 'add_page': value.add_page, 'append': append,
                'add_blank_page': value.add_blank_page, 'add_metadata': value.add_metadata,
                'write': lambda stream: save(fs, stream, value.write), 'close': lambda: None}
    if isinstance(value, PdfReader):
        return {'pages': value.pages, 'metadata': dict(value.metadata or {}),
                'is_encrypted': value.is_encrypted, 'decrypt': value.decrypt}
    if isinstance(value, PageObject):
        return {'extract_text': value.extract_text, 'rotate': value.rotate,
                'scale_to': value.scale_to, 'merge_page': value.merge_page}
    return {}


def pack(value):
    if isinstance(value, (PdfReader, PdfWriter)):
        output = io.BytesIO()
        writer = value if isinstance(value, PdfWriter) else PdfWriter(clone_from=value)
        writer.write(output)
        return descriptor('pypdf', 'writer' if isinstance(value, PdfWriter) else 'reader', output.getvalue())
    if isinstance(value, PageObject):
        writer = PdfWriter()
        writer.add_page(value)
        return pack(writer)  # A detached page is represented as a single-page writer.


def restore(kind, data):
    if kind == 'reader':
        return PdfReader(io.BytesIO(data))
    if kind == 'writer':
        return PdfWriter(clone_from=io.BytesIO(data))
    raise ValueError('Unsupported PDF object')


def view(root, value):
    if isinstance(root, (PdfReader, PdfWriter)) and isinstance(value, PageObject):
        for index, page in enumerate(root.pages):
            if page is value:
                return [['attr', 'pages'], ['item', index]]


def is_view(value):
    return isinstance(value, PageObject)


def iterate(value):
    return iter(value)


def assign(value, name, item):
    raise AttributeError('PDF attributes are read-only; use supported page methods')
