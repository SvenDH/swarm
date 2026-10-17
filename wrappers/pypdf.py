from pypdf import PdfReader, PdfWriter

from .files import source, save


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
