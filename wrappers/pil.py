from pathlib import PurePosixPath as Path

from PIL import Image

from .files import source, save
from .bridge import descriptor

FORMATS = {'PNG', 'JPEG', 'GIF', 'WEBP', 'TIFF', 'BMP', 'ICO'}


def run(fs, op, path=None, output=None, size=None, mode='RGB', color='white'):
    if op not in ('info', 'new', 'resize', 'convert'):
        raise ValueError('unknown image operation')
    api = exports(fs)['Image']
    image = api['new'](mode, tuple(size), color) if op == 'new' else api['open'](path)
    try:
        if op == 'info':
            return {'size': list(image.size), 'mode': image.mode, 'format': image.format}
        if op == 'resize':
            image = members(image, fs)['resize'](tuple(size))
        elif op == 'convert':
            image = image.convert(mode)
        return members(image, fs)['save'](output)
    finally:
        image.close()


def exports(fs):
    def open_image(fp, **kwargs):
        if set(kwargs) - {'mode', 'formats'}:
            raise ValueError('Unsupported Image.open options')
        formats = kwargs.pop('formats', None) or sorted(FORMATS)
        if set(formats) - FORMATS:
            raise ValueError('Unsupported image format')
        image = Image.open(source(fs, fp), formats=formats, **kwargs)
        if image.width * image.height > 4_000_000:
            raise ValueError('images are limited to 4,000,000 pixels')
        image.load()
        return image
    def new(mode, size, color=0):
        if len(size) != 2 or size[0] * size[1] > 4_000_000:
            raise ValueError('images are limited to 4,000,000 pixels')
        return Image.new(mode, tuple(size), color)
    def frombytes(mode, size, data, *args):
        if size[0] * size[1] > 4_000_000:
            raise ValueError('images are limited to 4,000,000 pixels')
        return Image.frombytes(mode, tuple(size), data, *args)
    return {'Image': {'open': open_image, 'new': new,
                      'fromarray': Image.fromarray, 'frombytes': frombytes,
                      'Resampling': {'NEAREST': Image.Resampling.NEAREST, 'BILINEAR': Image.Resampling.BILINEAR,
                                     'BICUBIC': Image.Resampling.BICUBIC, 'LANCZOS': Image.Resampling.LANCZOS}}}


def members(image, fs, name=None):
    if not isinstance(image, Image.Image):
        return {}
    def write(fp, format=None, **params):
        fmt = format or Image.registered_extensions().get(Path(str(fp)).suffix.lower())
        if fmt is None or fmt.upper() not in FORMATS:
            raise ValueError('Specify an image format or a known extension')
        return save(fs, fp, lambda stream: image.save(stream, format=fmt, **params))
    def resize(size, *args, **kwargs):
        if size[0] * size[1] > 4_000_000:
            raise ValueError('images are limited to 4,000,000 pixels')
        return image.resize(tuple(size), *args, **kwargs)
    def crop(box=None):
        if box is not None and (box[2] - box[0]) * (box[3] - box[1]) > 4_000_000:
            raise ValueError('images are limited to 4,000,000 pixels')
        return image.crop(box)
    return {'size': image.size, 'width': image.width, 'height': image.height,
            'mode': image.mode, 'format': image.format, 'save': write, 'resize': resize,
            'crop': crop, 'rotate': image.rotate, 'convert': image.convert,
            'copy': image.copy, 'thumbnail': image.thumbnail, 'getpixel': image.getpixel,
            'putpixel': image.putpixel, 'getbbox': image.getbbox, 'getextrema': image.getextrema,
            'paste': image.paste, 'tobytes': image.tobytes, 'close': lambda: None}


def pack(image):
    if isinstance(image, Image.Image):
        if image.width * image.height > 4_000_000:
            raise ValueError('images are limited to 4,000,000 pixels')
        return descriptor('PIL', 'image', [image.mode, image.size, image.tobytes(), image.format])


def restore(kind, data):
    if kind != 'image' or data[1][0] * data[1][1] > 4_000_000:
        raise ValueError('Invalid image')
    mode, size, pixels, fmt = data
    image = Image.frombytes(mode, tuple(size), pixels)
    image.format = fmt
    return image


def is_view(value):
    return False


def iterate(value):
    return iter(value)


def view(root, value):
    return None


def assign(value, name, item):
    raise AttributeError('Image attributes are read-only')
