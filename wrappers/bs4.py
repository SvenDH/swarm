"""HTML-only BeautifulSoup. No URL fetches or external XML entities."""
from bs4 import BeautifulSoup, Tag, NavigableString
from .bridge import descriptor
from .common import iterate


def soup(markup='', features='html.parser', **kwargs):
    if features != 'html.parser' or kwargs:
        raise ValueError('Only the built-in html.parser is exposed')
    if not isinstance(markup, (str, bytes)) or len(markup) > 8 * 1024**2:
        raise ValueError('HTML must be a string or bytes, at most 8 MiB')
    return BeautifulSoup(markup, 'html.parser')


def exports(fs):
    return {'BeautifulSoup': soup}


def members(value, fs, name=None):
    if not isinstance(value, Tag):
        return {}
    return {'name': value.name, 'text': value.get_text(), 'string':
            None if value.string is None else str(value.string), 'attrs': value.attrs,
            'contents': value.contents, 'children': list(value.children),
            'parent': value.parent, 'find': value.find, 'find_all': value.find_all,
            'select': value.select, 'select_one': value.select_one,
            'get': value.get, 'get_text': value.get_text, 'prettify': value.prettify,
            'clear': value.clear, 'append': value.append, 'decompose': value.decompose,
            'extract': value.extract, 'replace_with': value.replace_with}


def pack(value):
    if isinstance(value, NavigableString):
        return descriptor('bs4', 'scalar', str(value))
    if isinstance(value, Tag):
        return descriptor('bs4', 'soup' if isinstance(value, BeautifulSoup) else 'tag', str(value))


def restore(kind, data):
    value = soup(data)
    if kind == 'soup':
        return value
    if kind == 'tag':
        return value.find()
    raise ValueError('Unsupported HTML object')


def view(root, value):
    if root is value:
        return []
    if not isinstance(root, Tag) or not isinstance(value, Tag):
        return None
    path = []
    node = value
    while node is not root and node.parent is not None:
        siblings = node.parent.contents
        index = next(i for i, sibling in enumerate(siblings) if sibling is node)
        path[0:0] = [['attr', 'contents'], ['item', index]]
        node = node.parent
    return path if node is root else None


def is_view(value):
    return isinstance(value, Tag)


def assign(value, name, item):
    if isinstance(value, Tag) and name == 'string':
        value.string = str(item)
    elif isinstance(value, Tag) and name == 'name':
        value.name = str(item)
    else:
        raise AttributeError('Only tag name/string can be assigned')
