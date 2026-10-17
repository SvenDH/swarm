import io
from pptx import Presentation
from pptx.presentation import Presentation as PresentationType
from pptx.slide import Slide, SlideLayout, Slides, SlideLayouts
from pptx.shapes.shapetree import SlideShapes
from pptx.shapes.base import BaseShape
from pptx.shapes.placeholder import SlidePlaceholder
from pptx.shapes.graphfrm import GraphicFrame
from pptx.text.text import TextFrame, _Paragraph, _Run, Font
from pptx.table import Table, _Cell
from pptx.util import Inches, Cm, Pt
from pptx.dml.color import RGBColor, ColorFormat
from .bridge import descriptor
from .files import source, save
from .common import iterate


def exports(fs):
    return {'Presentation': lambda path=None: Presentation(None if path is None else source(fs, path)),
            'util': {'Inches': Inches, 'Cm': Cm, 'Pt': Pt},
            'dml': {'color': {'RGBColor': lambda r, g, b: tuple(RGBColor(r, g, b))}},
            'enum': {'text': {'PP_ALIGN': {'LEFT': 1, 'CENTER': 2, 'RIGHT': 3}},
                     'shapes': {'MSO_SHAPE': {'RECTANGLE': 1, 'OVAL': 9}}}}


def members(value, fs, name=None):
    if isinstance(value, PresentationType):
        return {'slides': value.slides, 'slide_layouts': value.slide_layouts,
                'slide_width': value.slide_width, 'slide_height': value.slide_height,
                'save': lambda path: save(fs, path, value.save)}
    if isinstance(value, (Slide, SlideLayout)):
        return {'shapes': value.shapes, 'placeholders': value.placeholders, 'name': value.name}
    if isinstance(value, Slides):
        return {'add_slide': value.add_slide}
    if isinstance(value, SlideShapes):
        return {'title': value.title, 'add_textbox': value.add_textbox,
                'add_shape': value.add_shape, 'add_table': value.add_table,
                'add_picture': lambda image_file, *a, **kw: value.add_picture(source(fs, image_file), *a, **kw)}
    if isinstance(value, BaseShape):
        result = {'name': value.name, 'left': value.left, 'top': value.top,
                  'width': value.width, 'height': value.height, 'has_text_frame': value.has_text_frame}
        if value.has_text_frame:
            result.update(text=value.text, text_frame=value.text_frame)
        if isinstance(value, GraphicFrame) and value.has_table:
            result['table'] = value.table
        return result
    if isinstance(value, TextFrame):
        return {'text': value.text, 'paragraphs': value.paragraphs,
                'add_paragraph': value.add_paragraph, 'clear': value.clear, 'word_wrap': value.word_wrap}
    if isinstance(value, _Paragraph):
        return {'text': value.text, 'runs': value.runs, 'font': value.font,
                'add_run': value.add_run, 'level': value.level, 'alignment': value.alignment}
    if isinstance(value, _Run):
        return {'text': value.text, 'font': value.font}
    if isinstance(value, Font):
        return {'name': value.name, 'size': value.size, 'bold': value.bold,
                'italic': value.italic, 'color': value.color}
    if isinstance(value, ColorFormat):
        return {'rgb': None if value.type is None else tuple(value.rgb)}
    if isinstance(value, Table):
        return {'cell': value.cell}
    if isinstance(value, _Cell):
        return {'text': value.text, 'text_frame': value.text_frame}
    return {}


def pack(value):
    if isinstance(value, PresentationType):
        output = io.BytesIO()
        value.save(output)
        return descriptor('pptx', 'presentation', output.getvalue())


def restore(kind, data):
    if kind != 'presentation':
        raise ValueError('Unsupported presentation object')
    return Presentation(io.BytesIO(data))


def view(root, value):
    # Other objects use bridge paths (including add_slide/add_textbox call paths).
    if isinstance(root, PresentationType):
        if isinstance(value, Slide):
            for i, slide in enumerate(root.slides):
                if slide.slide_id == value.slide_id:
                    return [['attr', 'slides'], ['item', i]]
        for i, slide in enumerate(root.slides):
            prefix = [['attr', 'slides'], ['item', i], ['attr', 'shapes']]
            if isinstance(value, BaseShape):
                for j, shape in enumerate(slide.shapes):
                    if shape.shape_id == value.shape_id and shape._element is value._element:
                        return [['attr', 'slides'], ['item', i], ['attr', 'shapes'], ['item', j]]
            for j, shape in enumerate(slide.shapes):
                path = [*prefix, ['item', j]]
                if shape.has_text_frame:
                    found = text_view(shape.text_frame, value, [*path, ['attr', 'text_frame']])
                    if found is not None:
                        return found
                if isinstance(shape, GraphicFrame) and shape.has_table:
                    table = shape.table
                    table_path = [*path, ['attr', 'table']]
                    if isinstance(value, Table) and value._tbl is table._tbl:
                        return table_path
                    for r, row in enumerate(table.rows):
                        for c, cell in enumerate(row.cells):
                            cell_path = [*table_path, ['call', 'cell', [r, c], {}]]
                            if isinstance(value, _Cell) and value._tc is cell._tc:
                                return cell_path
                            found = text_view(cell.text_frame, value, [*cell_path, ['attr', 'text_frame']])
                            if found is not None:
                                return found


def text_view(frame, value, path):
    if isinstance(value, TextFrame) and value._txBody is frame._txBody:
        return path
    for i, paragraph in enumerate(frame.paragraphs):
        prefix = [*path, ['attr', 'paragraphs'], ['item', i]]
        if isinstance(value, _Paragraph) and value._p is paragraph._p:
            return prefix
        for j, run in enumerate(paragraph.runs):
            if isinstance(value, _Run) and value._r is run._r:
                return [*prefix, ['attr', 'runs'], ['item', j]]


def is_view(value):
    return isinstance(value, (Slide, SlideLayout, Slides, SlideLayouts, SlideShapes,
                              BaseShape, TextFrame, _Paragraph, _Run, Font, Table, _Cell, ColorFormat))


def assign(value, name, item):
    if isinstance(value, (BaseShape, TextFrame, _Paragraph, _Run, _Cell)) and name == 'text':
        value.text = item
    elif isinstance(value, Font) and name == 'size':
        value.size = item
    elif isinstance(value, Font) and name == 'bold':
        value.bold = item
    elif isinstance(value, Font) and name == 'italic':
        value.italic = item
    elif isinstance(value, Font) and name == 'name':
        value.name = item
    elif isinstance(value, ColorFormat) and name == 'rgb':
        value.rgb = RGBColor(*item)
    elif isinstance(value, _Paragraph) and name == 'alignment':
        value.alignment = item
    elif isinstance(value, _Paragraph) and name == 'level':
        value.level = item
    elif isinstance(value, TextFrame) and name == 'word_wrap':
        value.word_wrap = item
    elif isinstance(value, PresentationType) and name == 'slide_width':
        value.slide_width = item
    elif isinstance(value, PresentationType) and name == 'slide_height':
        value.slide_height = item
    else:
        raise AttributeError('Unsupported presentation attribute')
