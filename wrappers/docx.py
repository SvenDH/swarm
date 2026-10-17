from docx import Document

from .files import source, save


def run(fs, op, path, paragraphs=None):
    if op not in ('read', 'write', 'append'):
        raise ValueError('unknown document operation')
    document = Document() if op == 'write' else Document(source(fs, path))
    if op == 'read':
        return [p.text for p in document.paragraphs]
    for text in paragraphs or []:
        document.add_paragraph(text)
    return save(fs, path, document.save)
