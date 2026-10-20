"""HTML-only BeautifulSoup. No URL fetches or external XML entities."""
from bs4 import BeautifulSoup, Tag
from .common import iterate
from itertools import islice
from .limits import check_memory


def nodes(iterator):
    result = list(islice(iterator, 10001))
    if len(result) > 10000:
        raise ValueError('HTML traversal is limited to 10,000 nodes')
    return result


def soup(markup='', features='html.parser', **kwargs):
    if features != 'html.parser' or kwargs:
        raise ValueError('Only the built-in html.parser is exposed')
    if not isinstance(markup, (str, bytes)):
        raise ValueError('HTML must be a string or bytes')
    check_memory(len(markup.encode()) if isinstance(markup, str) else len(markup))
    return BeautifulSoup(markup, 'html.parser')


def exports(fs):
    return {'BeautifulSoup': soup}


def members(value, fs, name=None):
    if not isinstance(value, Tag):
        return {}
    traversals = {'descendants': lambda: value.descendants, 'parents': lambda: value.parents,
                  'next_siblings': lambda: value.next_siblings, 'previous_siblings': lambda: value.previous_siblings,
                  'strings': lambda: value.strings, 'stripped_strings': lambda: value.stripped_strings}
    if name in traversals:
        return {name: nodes(traversals[name]())}
    return {'name': value.name, 'text': value.get_text(), 'string':
            None if value.string is None else str(value.string), 'attrs': value.attrs,
            'contents': value.contents, 'children': list(value.children),
            'parent': value.parent, 'find': value.find, 'find_all': value.find_all,
            'select': value.select, 'select_one': value.select_one,
            'find_next': value.find_next, 'find_previous': value.find_previous,
            'find_next_sibling': value.find_next_sibling, 'find_previous_sibling': value.find_previous_sibling,
            'find_next_siblings': value.find_next_siblings, 'find_previous_siblings': value.find_previous_siblings,
            'find_parent': value.find_parent, 'find_parents': value.find_parents,
            'has_attr': value.has_attr,
            'get': value.get, 'get_text': value.get_text, 'prettify': value.prettify,
            'clear': value.clear, 'append': value.append, 'decompose': value.decompose,
            'extract': value.extract, 'replace_with': value.replace_with}


def is_view(value):
    return isinstance(value, Tag)


def assign(value, name, item):
    if isinstance(value, Tag) and name == 'string':
        value.string = str(item)
    elif isinstance(value, Tag) and name == 'name':
        value.name = str(item)
    else:
        raise AttributeError('Only tag name/string can be assigned')
