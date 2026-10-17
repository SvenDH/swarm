"""XlsxWriter recipes use in-memory ZIPs; never native temp/host file paths."""
import io
import xlsxwriter as xlsx
from .bridge import descriptor
from .common import record, thaw, view, iterate, assign
from .files import source


class Workbook:
    def __init__(self, filename, sheets=None, formats=None, charts=None):
        self.filename = filename
        self.sheets, self.formats, self.charts = sheets or [], formats or [], charts or []


class Sheet:
    def __init__(self, book, index):
        self.book, self.index = book, index


class Format:
    def __init__(self, book, index):
        self.book, self.index = book, index


class Chart:
    def __init__(self, book, index):
        self.book, self.index = book, index


def exports(fs):
    def create(filename, options=None):
        if options:
            raise ValueError('Workbook options are fixed: in_memory, no automatic formulas/URLs')
        return Workbook(str(filename))
    return {'Workbook': create}


def ref(value):
    if isinstance(value, Format):
        return {'format': value.index}
    if isinstance(value, Chart):
        return {'chart': value.index}
    if isinstance(value, (list, tuple)):
        return type(value)(ref(item) for item in value)
    if isinstance(value, dict):
        return {key: ref(item) for key, item in value.items()}
    return value


def deref(value, formats, charts):
    if isinstance(value, dict):
        if set(value) == {'format'}:
            return formats[value['format']]
        if set(value) == {'chart'}:
            return charts[value['chart']]
        return {key: deref(item, formats, charts) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(deref(item, formats, charts) for item in value)
    return value


def write(book, fs):
    output = io.BytesIO()
    workbook = xlsx.Workbook(output, {'in_memory': True, 'strings_to_formulas': False, 'strings_to_urls': False})
    formats = [workbook.add_format(properties) for properties in book.formats]
    charts = [workbook.add_chart(options) for options, commands in book.charts]
    for chart, (options, commands) in zip(charts, book.charts):
        for name, args, kwargs in commands:
            chart_functions(chart)[name](*thaw(args), **thaw(kwargs))
    for title, commands in book.sheets:
        sheet = workbook.add_worksheet(title)
        for name, args, kwargs in commands:
            args = deref(thaw(args), formats, charts)
            kwargs = deref(thaw(kwargs), formats, charts)
            if name == 'insert_image':
                cell, pixels, options = args
                sheet.insert_image(cell, 'image.png', {**options, 'image_data': io.BytesIO(pixels)})
            else:
                sheet_functions(sheet)[name](*args, **kwargs)
    workbook.close()
    return fs.write(book.filename, output.getvalue())


def sheet_functions(sheet):
    return {'write': sheet.write, 'write_row': sheet.write_row, 'write_column': sheet.write_column,
            'write_string': sheet.write_string, 'write_number': sheet.write_number,
            'write_formula': sheet.write_formula, 'write_blank': sheet.write_blank,
            'set_column': sheet.set_column, 'set_row': sheet.set_row, 'merge_range': sheet.merge_range,
            'freeze_panes': sheet.freeze_panes, 'autofilter': sheet.autofilter,
            'conditional_format': sheet.conditional_format, 'insert_chart': sheet.insert_chart}


def chart_functions(chart):
    return {'add_series': chart.add_series, 'set_title': chart.set_title,
            'set_x_axis': chart.set_x_axis, 'set_y_axis': chart.set_y_axis,
            'set_legend': chart.set_legend, 'set_style': chart.set_style}


def members(value, fs, name=None):
    if isinstance(value, Workbook):
        def sheet(name=None):
            if len(value.sheets) >= 128:
                raise ValueError('Workbooks are limited to 128 sheets')
            value.sheets.append([name or 'Sheet' + str(len(value.sheets) + 1), []])
            return Sheet(value, len(value.sheets) - 1)
        def fmt(properties=None):
            value.formats.append(dict(properties or {}))
            return Format(value, len(value.formats) - 1)
        def chart(options):
            if set(options) - {'type', 'subtype'}:
                raise ValueError('Unsupported chart options')
            value.charts.append([dict(options), []])
            return Chart(value, len(value.charts) - 1)
        return {'sheets': [Sheet(value, i) for i in range(len(value.sheets))],
                'formats': [Format(value, i) for i in range(len(value.formats))],
                'charts': [Chart(value, i) for i in range(len(value.charts))],
                'add_worksheet': sheet, 'add_format': fmt, 'add_chart': chart,
                'close': lambda: write(value, fs)}
    if isinstance(value, Sheet):
        def add(name, *args, **kwargs):
            record(value.book.sheets[value.index][1], name, ref(args), ref(kwargs))
            return 0
        def insert_image(cell, filename, options=None):
            options = dict(options or {})
            if set(options) - {'x_offset', 'y_offset', 'x_scale', 'y_scale', 'object_position', 'description'}:
                raise ValueError('Unsupported image options')
            return add('insert_image', cell, source(fs, filename).getvalue(), options)
        return {'insert_image': insert_image, **{method: (lambda *a, _method=method, **kw: add(_method, *a, **kw)) for method in (
            'write', 'write_row', 'write_column', 'write_string', 'write_number', 'write_formula',
            'write_blank', 'set_column', 'set_row', 'merge_range', 'freeze_panes', 'autofilter',
            'conditional_format', 'insert_chart')}}
    if isinstance(value, Chart):
        return {method: (lambda *a, _method=method, **kw:
                        record(value.book.charts[value.index][1], _method, a, kw))
                for method in ('add_series', 'set_title', 'set_x_axis', 'set_y_axis', 'set_legend', 'set_style')}
    if isinstance(value, Format):
        return {method: (lambda item=True, _field=field: value.book.formats[value.index].update({_field: item}))
                for method, field in (('set_bold', 'bold'), ('set_italic', 'italic'),
                                      ('set_num_format', 'num_format'), ('set_bg_color', 'bg_color'),
                                      ('set_font_color', 'font_color'), ('set_align', 'align'))}
    return {}


def pack(value):
    if isinstance(value, Workbook):
        return descriptor('xlsxwriter', 'workbook', [value.filename, value.sheets, value.formats, value.charts])


def restore(kind, data):
    if kind != 'workbook':
        raise ValueError('Unsupported workbook object')
    return Workbook(*data)


def view(root, value):
    if isinstance(root, Workbook) and isinstance(value, (Sheet, Format, Chart)) and value.book is root:
        collection = 'sheets' if isinstance(value, Sheet) else 'formats' if isinstance(value, Format) else 'charts'
        return [['attr', collection], ['item', value.index]]


def is_view(value):
    return isinstance(value, (Sheet, Format, Chart))
