from pathlib import PurePosixPath as Path

from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw, ImageColor, ImageChops, ImageFont
from .api import select, enum_values
import math
from inspect import signature

from .files import source, save
from .bridge import descriptor, TypeRef

FORMATS = {'PNG', 'JPEG', 'GIF', 'WEBP', 'TIFF', 'BMP', 'ICO'}
ENHANCERS = {'Brightness': ImageEnhance.Brightness, 'Contrast': ImageEnhance.Contrast,
             'Color': ImageEnhance.Color, 'Sharpness': ImageEnhance.Sharpness}
FILTERS = {'GaussianBlur': ImageFilter.GaussianBlur, 'BoxBlur': ImageFilter.BoxBlur,
           'UnsharpMask': ImageFilter.UnsharpMask, 'RankFilter': ImageFilter.RankFilter,
           'MedianFilter': ImageFilter.MedianFilter, 'MinFilter': ImageFilter.MinFilter,
           'MaxFilter': ImageFilter.MaxFilter, 'ModeFilter': ImageFilter.ModeFilter,
           'BLUR': ImageFilter.BLUR, 'CONTOUR': ImageFilter.CONTOUR,
           'DETAIL': ImageFilter.DETAIL, 'SHARPEN': ImageFilter.SHARPEN,
           'SMOOTH': ImageFilter.SMOOTH, 'FIND_EDGES': ImageFilter.FIND_EDGES,
           'EDGE_ENHANCE': ImageFilter.EDGE_ENHANCE, 'EDGE_ENHANCE_MORE': ImageFilter.EDGE_ENHANCE_MORE,
           'EMBOSS': ImageFilter.EMBOSS, 'SMOOTH_MORE': ImageFilter.SMOOTH_MORE}
FILTER_PARAMETERS = {'GaussianBlur': 'radius', 'BoxBlur': 'radius', 'MedianFilter': 'size',
                     'MinFilter': 'size', 'MaxFilter': 'size', 'ModeFilter': 'size'}


class PixelAccess:
    """Recreate native pixel access per operation; persist only the image root."""

    def __init__(self, image):
        self.image = image

    def __getitem__(self, key):
        return self.image.load()[key]

    def __setitem__(self, key, value):
        self.image.load()[key] = value


class Drawing:
    def __init__(self, image, mode=None):
        if not isinstance(image, Image.Image):
            raise TypeError('ImageDraw.Draw requires an image')
        ImageDraw.Draw(image, mode)  # Validate without keeping a native drawing handle.
        self.image, self.mode = image, mode

    def methods(self):
        draw = ImageDraw.Draw(self.image, self.mode)
        methods = {'line': draw.line, 'rectangle': draw.rectangle, 'rounded_rectangle': draw.rounded_rectangle,
                'ellipse': draw.ellipse, 'polygon': draw.polygon, 'regular_polygon': draw.regular_polygon,
                'arc': draw.arc, 'pieslice': draw.pieslice, 'chord': draw.chord, 'point': draw.point,
                'text': draw.text, 'multiline_text': draw.multiline_text,
                'textbbox': draw.textbbox, 'multiline_textbbox': draw.multiline_textbbox,
                'textlength': draw.textlength}
        return {name: checked_drawing(method) for name, method in methods.items()}


def checked_drawing(method):
    def invoke(*args, **kwargs):
        bound = signature(method).bind(*args, **kwargs)
        options = {**bound.arguments, **bound.arguments.get('kwargs', {})}
        font = options.get('font')
        if font is not None and not isinstance(font, DefaultFont):
            raise ValueError('Use ImageFont.load_default; host font files are not supported')
        if 'n_sides' in options and (not isinstance(options['n_sides'], int) or not 3 <= options['n_sides'] <= 10000):
            raise ValueError('regular polygons require 3..10,000 sides')
        for name in ('font_size', 'width', 'stroke_width', 'spacing'):
            value = options.get(name)
            if value is not None and (not isinstance(value, (int, float)) or
                                      not math.isfinite(value) or not 0 <= value <= 512):
                raise ValueError('Drawing widths and font sizes must be finite and 0..512')
        if 'text' in options and (not isinstance(options['text'], str) or len(options['text']) > 10000):
            raise ValueError('Drawing text must be a string of at most 10,000 characters')
        if 'text' in options:
            # Pillow renders a text mask before clipping it to the image. Bound
            # that temporary too, not just the destination image's dimensions.
            lines = options['text'].split('\n')
            unit = math.ceil(font.size if font else options.get('font_size') or 10) + 2 * options.get('stroke_width', 0) + 4
            checked_size((max(map(len, lines)) * unit * 2,
                          len(lines) * (unit * 2 + options.get('spacing', 4))))
        if font is not None:
            bound.arguments['font'] = font.native()
        return method(*bound.args, **bound.kwargs)
    return invoke


class Filter:
    def __init__(self, kind, amount=None):
        self.kind, self.amount = kind, amount


class DefaultFont:
    def __init__(self, size=10):
        size = 10 if size is None else size
        if not isinstance(size, (int, float)) or not math.isfinite(size) or not 1 <= size <= 512:
            raise ValueError('font size must be finite and 1..512')
        self.size = size

    def native(self):
        return ImageFont.load_default(size=self.size)


def filter_factory(kind):
    def create(amount=None, **kwargs):
        parameter = FILTER_PARAMETERS[kind]
        if set(kwargs) - {parameter}:
            raise ValueError('Unsupported filter options')
        if amount is not None and parameter in kwargs:
            raise TypeError('filter parameter was supplied twice')
        amount = kwargs.get(parameter, amount if amount is not None else 3 if parameter == 'size' else 2)
        if not isinstance(amount, (int, float)) or not math.isfinite(amount) or not 0 <= amount <= 31:
            raise ValueError('filter radius/size must be finite and 0..31')
        FILTERS[kind](amount)  # Native validation, no image allocation.
        return Filter(kind, amount)
    return create


def unsharp_mask(radius=2, percent=150, threshold=3):
    if (not isinstance(radius, (int, float)) or not math.isfinite(radius) or not 0 <= radius <= 31
            or not isinstance(percent, int) or not 0 <= percent <= 1000
            or not isinstance(threshold, int) or not 0 <= threshold <= 255):
        raise ValueError('UnsharpMask needs radius 0..31, percent 0..1000 and threshold 0..255')
    return Filter('UnsharpMask', dict(radius=radius, percent=percent, threshold=threshold))


def rank_filter(size, rank):
    if not isinstance(size, int) or not 1 <= size <= 31 or size % 2 != 1 or not isinstance(rank, int) or not 0 <= rank < size**2:
        raise ValueError('RankFilter needs odd size 1..31 and rank 0..size**2-1')
    return Filter('RankFilter', dict(size=size, rank=rank))


def checked_size(size, mode='RGBA'):
    from .limits import check_memory
    if len(size) != 2 or min(size) < 0:
        raise ValueError('image dimensions must be nonnegative')
    # Pillow stores RGB internally as four bytes/pixel too. Modes with larger
    # samples are charged conservatively before decoding/allocation.
    sample_bytes = 1 if mode in ('1', 'L', 'P') else 2 if mode.startswith('I;16') else 4
    check_memory(size[0] * size[1] * sample_bytes)


class Enhancer:
    def __init__(self, kind, image):
        self.kind, self.image = kind, image


def enhancer(kind):
    def create(image):
        if not isinstance(image, Image.Image):
            raise TypeError('an image is required')
        return Enhancer(kind, image.copy())
    return create


def sized_operation(function):
    def apply(image, size, *args, **kwargs):
        checked_size(size, image.mode)
        return function(image, tuple(size), *args, **kwargs)
    return apply


def expand(image, border=0, fill=0):
    if isinstance(border, int):
        left = top = right = bottom = border
    elif len(border) == 2:
        left = right = border[0]
        top = bottom = border[1]
    else:
        left, top, right, bottom = border
    if min(left, top, right, bottom) < 0:
        raise ValueError('borders must be nonnegative')
    checked_size((image.width + left + right, image.height + top + bottom), image.mode)
    return ImageOps.expand(image, border=border, fill=fill)


def scale(image, factor, **kwargs):
    if not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
        raise ValueError('scale factor must be finite and positive')
    checked_size((round(image.width * factor), round(image.height * factor)), image.mode)
    return ImageOps.scale(image, factor, **kwargs)


def exports(fs):
    def open_image(fp, **kwargs):
        if set(kwargs) - {'mode', 'formats'}:
            raise ValueError('Unsupported Image.open options')
        formats = kwargs.pop('formats', None) or sorted(FORMATS)
        if set(formats) - FORMATS:
            raise ValueError('Unsupported image format')
        image = Image.open(source(fs, fp), formats=formats, **kwargs)
        checked_size(image.size, image.mode)
        image.load()
        return image
    def new(mode, size, color=0):
        checked_size(size, mode)
        return Image.new(mode, tuple(size), color)
    def frombytes(mode, size, data, *args):
        checked_size(size, mode)
        return Image.frombytes(mode, tuple(size), data, *args)
    def fromarray(obj, mode=None):
        checked_size((obj.shape[1] if obj.ndim > 1 else 1, obj.shape[0]))
        return Image.fromarray(obj, mode=mode)
    def merge(mode, bands):
        bands = tuple(bands)
        for band in bands:
            checked_size(band.size, band.mode)
        return Image.merge(mode, bands)
    def sized(factory):
        def create(size, *args, **kwargs):
            checked_size(size)
            return factory(tuple(size), *args, **kwargs)
        return create
    return {'Image': {'Image': types()['Image'], 'open': open_image, 'new': new,
                      'fromarray': fromarray, 'frombytes': frombytes, 'merge': merge,
                      **{name: int(value) for name, value in Image.Resampling.__members__.items()},
                      'Transpose': enum_values(Image.Transpose), 'Transform': enum_values(Image.Transform),
                      'Dither': enum_values(Image.Dither), 'Palette': enum_values(Image.Palette),
                      'Quantize': enum_values(Image.Quantize),
                      **enum_values(Image.Transpose), **enum_values(Image.Transform),
                      **enum_values(Image.Dither), **enum_values(Image.Quantize),
                      **enum_values(Image.Palette),
                      'blend': Image.blend, 'composite': Image.composite, 'alpha_composite': Image.alpha_composite,
                      'linear_gradient': Image.linear_gradient, 'radial_gradient': Image.radial_gradient,
                      'effect_noise': sized(Image.effect_noise), 'effect_mandelbrot': sized(Image.effect_mandelbrot),
                      'Resampling': {'NEAREST': Image.Resampling.NEAREST, 'BILINEAR': Image.Resampling.BILINEAR,
                                     'BICUBIC': Image.Resampling.BICUBIC, 'LANCZOS': Image.Resampling.LANCZOS}},
            'ImageOps': {'grayscale': ImageOps.grayscale, 'invert': ImageOps.invert,
                         'scale': scale,
                         'mirror': ImageOps.mirror, 'flip': ImageOps.flip,
                         'autocontrast': ImageOps.autocontrast, 'equalize': ImageOps.equalize,
                         'posterize': ImageOps.posterize, 'solarize': ImageOps.solarize,
                         'colorize': ImageOps.colorize, 'exif_transpose': exif_transpose,
                         'fit': sized_operation(ImageOps.fit), 'pad': sized_operation(ImageOps.pad),
                         'contain': sized_operation(ImageOps.contain),
                         'cover': sized_operation(cover), 'expand': expand},
            'ImageEnhance': {name: enhancer(name) for name in ENHANCERS},
            'ImageColor': select(vars(ImageColor), 'getrgb getcolor'),
            'ImageChops': select(vars(ImageChops), '''add add_modulo subtract subtract_modulo
                difference multiply screen overlay soft_light hard_light darker lighter invert
                logical_and logical_or logical_xor offset constant duplicate blend composite'''),
            'ImageFont': {'load_default': DefaultFont},
            'ImageDraw': {'Draw': Drawing},
            'ImageFilter': {**{name: (filter_factory(name) if name in FILTER_PARAMETERS
                                    else Filter(name)) for name in FILTERS if name not in ('UnsharpMask', 'RankFilter')},
                            'UnsharpMask': unsharp_mask, 'RankFilter': rank_filter}}


def exif_transpose(image):
    return ImageOps.exif_transpose(image)  # Never expose inplace mutation of argument roots.


def cover(image, size, method=Image.Resampling.BICUBIC):
    ratio = max(size[0] / image.width, size[1] / image.height)
    checked_size((round(image.width * ratio), round(image.height * ratio)), image.mode)
    return ImageOps.cover(image, size, method=method)


def types():
    return {'Image': TypeRef('Image', Image.Image)}


def members(image, fs, name=None):
    if isinstance(image, DefaultFont):
        native = image.native()
        return {'size': image.size, 'getbbox': native.getbbox, 'getlength': native.getlength}
    if isinstance(image, Drawing):
        return image.methods()
    if isinstance(image, Enhancer):
        return {'enhance': lambda factor: ENHANCERS[image.kind](image.image).enhance(factor)}
    if not isinstance(image, Image.Image):
        return {}
    def write(fp, format=None, **params):
        fmt = format or Image.registered_extensions().get(Path(str(fp)).suffix.lower())
        if fmt is None or fmt.upper() not in FORMATS:
            raise ValueError('Specify an image format or a known extension')
        return save(fs, fp, lambda stream: image.save(stream, format=fmt, **params))
    def resize(size, *args, **kwargs):
        checked_size(size, image.mode)
        return image.resize(tuple(size), *args, **kwargs)
    def crop(box=None):
        if box is not None:
            checked_size((box[2] - box[0], box[3] - box[1]), image.mode)
        return image.crop(box)
    def data():
        if image.width * image.height > 10000:
            raise ValueError('getdata is limited to 10,000 pixels; prefer np.asarray(image)')
        return list(image.getdata())
    return {'size': image.size, 'width': image.width, 'height': image.height,
            'load': lambda: PixelAccess(image),
            'drawing': lambda mode=None: Drawing(image, mode),
            'mode': image.mode, 'format': image.format, 'save': write, 'resize': resize,
            'crop': crop, 'rotate': image.rotate, 'convert': image.convert,
            'copy': image.copy, 'thumbnail': image.thumbnail, 'getpixel': image.getpixel,
            'filter': lambda effect: apply_filter(image, effect),
            'transpose': image.transpose, 'point': image.point, 'getcolors': image.getcolors,
            'getdata': data, 'histogram': image.histogram, 'entropy': image.entropy,
            'putpixel': image.putpixel, 'putalpha': image.putalpha, 'split': image.split,
            'getchannel': image.getchannel, 'getbands': image.getbands,
            'getbbox': image.getbbox, 'getextrema': image.getextrema,
            'paste': image.paste, 'tobytes': image.tobytes, 'close': lambda: None}


def apply_filter(image, effect):
    if not isinstance(effect, Filter):
        raise TypeError('use a supported ImageFilter object')
    factory = FILTERS[effect.kind]
    if isinstance(effect.amount, dict):
        return image.filter(factory(**effect.amount))
    return image.filter(factory() if effect.amount is None else factory(effect.amount))


def pack(image):
    if isinstance(image, DefaultFont):
        return descriptor('PIL', 'default_font', image.size)
    if isinstance(image, Drawing):
        return descriptor('PIL', 'drawing', [pack(image.image)['data'], image.mode])
    if isinstance(image, Filter):
        return descriptor('PIL', 'filter', [image.kind, image.amount])
    if isinstance(image, Enhancer):
        return descriptor('PIL', 'enhancer', [image.kind, pack(image.image)['data']])
    if isinstance(image, Image.Image):
        checked_size(image.size, image.mode)
        return descriptor('PIL', 'image', [image.mode, image.size, image.tobytes(), image.format])


def restore(kind, data):
    if kind == 'default_font':
        return DefaultFont(data)
    if kind == 'drawing':
        return Drawing(restore('image', data[0]), data[1])
    if kind == 'filter':
        if data[0] not in FILTERS:
            raise ValueError('Unsupported image filter')
        return Filter(*data)
    if kind == 'enhancer':
        if data[0] not in ENHANCERS:
            raise ValueError('Unsupported image enhancer')
        return Enhancer(data[0], restore('image', data[1]))
    if kind != 'image':
        raise ValueError('Invalid image')
    mode, size, pixels, fmt = data
    checked_size(size, mode)
    image = Image.frombytes(mode, tuple(size), pixels)
    image.format = fmt
    return image


def is_view(value):
    return isinstance(value, (Drawing, PixelAccess))


def iterate(value):
    return iter(value)


def view(root, value):
    if isinstance(root, Image.Image) and isinstance(value, PixelAccess) and value.image is root:
        return [['call', 'load', [], {}]]
    if isinstance(root, Image.Image) and isinstance(value, Drawing) and value.image is root:
        return [['call', 'drawing', [value.mode], {}]]
    return None


def assign(value, name, item):
    raise AttributeError('Image attributes are read-only')
