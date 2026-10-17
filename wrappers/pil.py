from pathlib import PurePosixPath as Path

from PIL import Image

from .files import source, save


def run(fs, op, path=None, output=None, size=None, mode='RGB', color='white'):
    if op not in ('info', 'new', 'resize', 'convert'):
        raise ValueError('unknown image operation')
    image = Image.new(mode, tuple(size), color) if op == 'new' else Image.open(source(fs, path))
    try:
        if op == 'info':
            return {'size': list(image.size), 'mode': image.mode, 'format': image.format}
        if op == 'resize':
            image = image.resize(tuple(size))
        elif op == 'convert':
            image = image.convert(mode)
        fmt = Image.registered_extensions()[Path(output).suffix.lower()]
        return save(fs, output, lambda stream: image.save(stream, format=fmt))
    finally:
        image.close()
