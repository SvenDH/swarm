from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.worksheet import Worksheet
from openpyxl.cell.cell import Cell, MergedCell
from openpyxl.utils import get_column_letter, column_index_from_string
import io

from .files import source, save
from .bridge import descriptor


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


def exports(fs):
    def load(filename, **kwargs):
        if set(kwargs) - {'data_only'}:
            raise ValueError('Only data_only is supported for load_workbook')
        return load_workbook(source(fs, filename), keep_links=False, **kwargs)
    return {'Workbook': Workbook, 'load_workbook': load,
            'utils': {'get_column_letter': get_column_letter,
                      'column_index_from_string': column_index_from_string}}


def members(value, fs, name=None):
    if isinstance(value, Workbook):
        return {'active': value.active, 'sheetnames': value.sheetnames, 'worksheets': value.worksheets,
                'create_sheet': value.create_sheet, 'remove': value.remove,
                'save': lambda filename: save(fs, filename, value.save), 'close': lambda: None}
    if isinstance(value, Worksheet):
        def rows(*args, **kwargs):
            if (kwargs.get('max_row', value.max_row) - kwargs.get('min_row', 1) + 1) * (
                    kwargs.get('max_col', value.max_column) - kwargs.get('min_col', 1) + 1) > 100_000:
                raise ValueError('iter_rows is limited to 100,000 cells')
            return list(value.iter_rows(*args, **kwargs))
        return {'title': value.title, 'max_row': value.max_row, 'max_column': value.max_column,
                'append': value.append, 'cell': value.cell, 'iter_rows': rows,
                'values': Values(value), 'merge_cells': value.merge_cells,
                'unmerge_cells': value.unmerge_cells, 'delete_rows': value.delete_rows,
                'delete_cols': value.delete_cols}
    if isinstance(value, (Cell, MergedCell)):
        return {'value': value.value, 'row': value.row, 'column': value.column,
                'coordinate': value.coordinate, 'number_format': value.number_format,
                'data_type': value.data_type}
    return {}


def pack(value):
    if isinstance(value, Workbook):
        output = io.BytesIO()
        value.save(output)
        return descriptor('openpyxl', 'workbook', output.getvalue())


def restore(kind, data):
    if kind != 'workbook':
        raise ValueError('Unsupported workbook object')
    return load_workbook(io.BytesIO(data), keep_links=False)


def view(root, value):
    if isinstance(root, Workbook):
        if isinstance(value, Worksheet):
            return [['attr', 'worksheets'], ['item', root.worksheets.index(value)]]
        if isinstance(value, (Cell, MergedCell)):
            return [['attr', 'worksheets'], ['item', root.worksheets.index(value.parent)],
                    ['item', value.coordinate]]


def is_view(value):
    return isinstance(value, (Worksheet, Cell, MergedCell))


class Values:
    def __init__(self, sheet):
        self.sheet = sheet

    def __iter__(self):
        if self.sheet.max_row * self.sheet.max_column > 100_000:
            raise ValueError('worksheet iteration is limited to 100,000 cells')
        return iter(self.sheet.values)


def iterate(value):
    if isinstance(value, Worksheet) and value.max_row * value.max_column > 100_000:
        raise ValueError('worksheet iteration is limited to 100,000 cells')
    return iter(value)


def assign(value, name, item):
    if isinstance(value, Cell) and name == 'value':
        value.value = item
    elif isinstance(value, Cell) and name == 'number_format':
        value.number_format = item
    elif isinstance(value, Worksheet) and name == 'title':
        value.title = item
    else:
        raise AttributeError('Unsupported workbook attribute assignment')
