from openpyxl import Workbook, load_workbook

from .files import source, save


def run(fs, op, path, rows=None, sheet=None):
    if op not in ('read', 'write', 'append'):
        raise ValueError('unknown workbook operation')
    book = Workbook() if op == 'write' else load_workbook(source(fs, path))
    try:
        if op == 'read':
            return {s.title: [list(row) for row in s.values] for s in book.worksheets}
        target = book[sheet] if sheet in book.sheetnames else book.active
        if sheet and sheet not in book.sheetnames:
            target = book.create_sheet(sheet)
        for row in rows or []:
            target.append(row)
        return save(fs, path, book.save)
    finally:
        book.close()
