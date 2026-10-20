"""Create vision previews without host paths or executable rich reprs."""
import io
from .limits import memory_scope, check_memory
from .protocol import plain


def preview(fs, bridge, value):
    with memory_scope(fs.memory_budget):
        bridge.initialize('PIL')
        from .pil import exports
        value = plain(value)
        if isinstance(value, str):
            image = exports(fs)['Image']['open'](value)
        elif type(value) is dict and set(value) == {'__library_value__', 'path', 'auth'}:
            spec = value['__library_value__']
            from PIL import Image
            original = bridge.resolve(spec, value['path'], value['auth'])
            if spec['module'] != 'PIL' or not isinstance(original, Image.Image):
                raise TypeError('display expects an image or image file path')
            image = original.copy()
        else:
            raise TypeError('display expects an image or image file path')
        try:
            image.thumbnail((1024, 1024))
            converted = image.convert('RGB')
            try:
                output = io.BytesIO()
                converted.save(output, format='JPEG', quality=85)
                data = output.getvalue()
                check_memory(len(data))
                return data
            finally:
                converted.close()
        finally:
            image.close()
