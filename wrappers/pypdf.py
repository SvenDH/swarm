from pypdf import PdfReader, PdfWriter, Transformation
from .common import iterate
from pypdf._page import PageObject

from .files import source, save


def exports(fs):
    def writer(*, clone_from=None):
        if clone_from is None:
            return PdfWriter()
        return PdfWriter(clone_from=clone_from if isinstance(clone_from, PdfReader) else source(fs, clone_from))
    return {'Transformation': Transformation,
            'PdfReader': lambda stream, **kwargs: PdfReader(source(fs, stream), **kwargs),
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
                'rotation': value.rotation, 'mediabox': tuple(value.mediabox), 'cropbox': tuple(value.cropbox),
                'scale': value.scale, 'scale_by': value.scale_by, 'add_transformation': value.add_transformation,
                'scale_to': value.scale_to, 'merge_page': value.merge_page}
    if isinstance(value, Transformation):
        return {'ctm': value.ctm, 'matrix': value.matrix, 'translate': value.translate,
                'scale': value.scale, 'rotate': value.rotate, 'apply_on': value.apply_on}
    return {}


def is_view(value):
    return isinstance(value, PageObject)


def assign(value, name, item):
    raise AttributeError('PDF attributes are read-only; use supported page methods')
