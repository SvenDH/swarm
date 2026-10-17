"""Library IO stays inside the agent virtual filesystem."""
import io


def source(fs, path):
    return io.BytesIO(fs.path_read_bytes(fs.path(path)))

def save(fs, path, writer):
    output = io.BytesIO()
    writer(output)
    return fs.write(path, output.getvalue())
