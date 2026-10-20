"""Library IO stays inside the agent virtual filesystem."""
import io
from .binary import Buffer


def source(fs, path):
    if isinstance(path, Buffer):
        return path.stream
    return io.BytesIO(fs.path_read_bytes(fs.path(path)))

def save(fs, path, writer):
    output = io.BytesIO()
    writer(output)
    if isinstance(path, Buffer):
        path.write(output.getvalue())
        return None
    return fs.write(path, output.getvalue())
