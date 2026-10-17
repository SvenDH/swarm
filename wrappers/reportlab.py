"""Checkpointable Canvas drawing recipes, rendered only into virtual-file bytes."""
import io
from reportlab.pdfgen.canvas import Canvas as NativeCanvas
from reportlab.lib import pagesizes, colors
from reportlab.lib.utils import ImageReader
from PIL import Image
from .bridge import descriptor
from .common import record, thaw, view, iterate, assign
from .files import source

FONTS = {'Courier', 'Courier-Bold', 'Courier-Oblique', 'Courier-BoldOblique',
         'Helvetica', 'Helvetica-Bold', 'Helvetica-Oblique', 'Helvetica-BoldOblique',
         'Times-Roman', 'Times-Bold', 'Times-Italic', 'Times-BoldItalic', 'Symbol', 'ZapfDingbats'}


class Canvas:
    def __init__(self, filename, pagesize=pagesizes.A4, commands=None):
        self.filename, self.pagesize, self.commands = filename, pagesize, commands or []


def exports(fs):
    return {'pdfgen': {'canvas': {'Canvas': lambda filename, pagesize=pagesizes.A4: Canvas(str(filename), pagesize)}},
            'lib': {'pagesizes': {'A4': pagesizes.A4, 'letter': pagesizes.letter,
                                  'A3': pagesizes.A3, 'landscape': pagesizes.landscape, 'portrait': pagesizes.portrait},
                    'units': {'inch': 72, 'cm': 72 / 2.54, 'mm': 72 / 25.4},
                    'colors': {name: rgba for name, rgba in (
                        ('black', (0, 0, 0)), ('white', (1, 1, 1)), ('red', (1, 0, 0)),
                        ('green', (0, .5, 0)), ('blue', (0, 0, 1)), ('gray', (.5, .5, .5)))}}}


def functions(canvas):
    return {'drawString': canvas.drawString, 'drawCentredString': canvas.drawCentredString,
            'drawRightString': canvas.drawRightString, 'setFont': canvas.setFont,
            'setFillColorRGB': canvas.setFillColorRGB, 'setStrokeColorRGB': canvas.setStrokeColorRGB,
            'setLineWidth': canvas.setLineWidth, 'line': canvas.line, 'rect': canvas.rect,
            'circle': canvas.circle, 'ellipse': canvas.ellipse, 'showPage': canvas.showPage,
            'translate': canvas.translate, 'scale': canvas.scale, 'rotate': canvas.rotate,
            'saveState': canvas.saveState, 'restoreState': canvas.restoreState,
            'setTitle': canvas.setTitle, 'setAuthor': canvas.setAuthor,
            'setPageSize': canvas.setPageSize}


def draw(value, name, args, kwargs):
    if name == 'setFont' and (args[0] if args else kwargs.get('psfontname')) not in FONTS:
        raise ValueError('Only built-in PDF fonts are exposed')
    record(value.commands, name, args, kwargs)


def write(value, fs):
    output = io.BytesIO()
    canvas = NativeCanvas(output, pagesize=value.pagesize, invariant=1)
    for name, args, kwargs in value.commands:
        args, kwargs = thaw(args), thaw(kwargs)
        if name == 'drawImage':
            image = Image.open(io.BytesIO(args[0]), formats=['PNG', 'JPEG', 'GIF', 'WEBP', 'TIFF', 'BMP', 'ICO'])
            if image.width * image.height > 4_000_000:
                raise ValueError('Images are limited to 4,000,000 pixels')
            image.load()
            canvas.drawImage(ImageReader(image), *args[1:], **kwargs)
        else:
            functions(canvas)[name](*args, **kwargs)
    canvas.save()
    return fs.write(value.filename, output.getvalue())


def members(value, fs, name=None):
    if not isinstance(value, Canvas):
        return {}
    def image(image, x, y, width=None, height=None, **kwargs):
        if set(kwargs) - {'mask', 'preserveAspectRatio', 'anchor'}:
            raise ValueError('Unsupported image options')
        pixels = source(fs, image).getvalue()
        draw(value, 'drawImage', (pixels, x, y), dict(width=width, height=height, **kwargs))
    methods = ('drawString', 'drawCentredString', 'drawRightString', 'setFont',
               'setFillColorRGB', 'setStrokeColorRGB', 'setLineWidth', 'line', 'rect',
               'circle', 'ellipse', 'showPage', 'translate', 'scale', 'rotate',
               'saveState', 'restoreState', 'setTitle', 'setAuthor', 'setPageSize')
    return {**{method: (lambda *a, _method=method, **kw: draw(value, _method, a, kw)) for method in methods},
            'setFillColor': lambda color: draw(value, 'setFillColorRGB', tuple(color), {}),
            'setStrokeColor': lambda color: draw(value, 'setStrokeColorRGB', tuple(color), {}),
            'drawImage': image, 'save': lambda: write(value, fs)}


def pack(value):
    if isinstance(value, Canvas):
        return descriptor('reportlab', 'canvas', [value.filename, value.pagesize, value.commands])


def restore(kind, data):
    if kind != 'canvas':
        raise ValueError('Unsupported Canvas object')
    return Canvas(*data)


def is_view(value):
    return False
