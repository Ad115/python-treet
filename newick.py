from typing import (Optional, List, Dict, 
                    Any, Iterable, Sequence, 
                    Tuple, TypeVar, Callable)
from functools import partial

def _find_closing(string: str, start=1, pair='()') -> int:
    """Find the first closing unmatched parenthesis from the start index.
    The `pair` argument allows to specify diferent opening/closing types
    of parenthesis.
    """
    opening = pair[0]
    closing = pair[1]

    next_closing = string.index(closing, start)

    has_next_opening = (string.find(opening, start) != -1)
    if not has_next_opening:
        return next_closing
    
    next_opening = string.index(opening, start)
    if next_closing < next_opening:
        return next_closing

    skip = _find_closing(string, next_opening+1, pair)
    return _find_closing(string, skip+1, pair)
# ---

_find_closing_parenthesis = partial(_find_closing, pair='()')
_find_closing_brackets = partial(_find_closing, pair='[]')
# ---

def _parts_of_subtree(newick) -> Tuple[str, str, str, str]:
    """
    A subtree consists of:
        children, label, branch length/support, comments/features
    """

    children = ''
    if newick.startswith('('):
        # Node has children
        children_end = _find_closing_parenthesis(newick)
        children = newick[1:children_end]
        rest = newick[children_end+1:]
    else:
        rest = newick

    comment = ''
    if '[' in rest: # Extract comments from the end
        rest, comment = rest.split('[', maxsplit=1)
        comment = comment[:-1] # remove ']' from the end

    if ':' in rest: # Extract branch length
        label, length = rest.split(':', maxsplit=1)
    else:
        label = rest
        length = ''

    return children, label.strip(), length, comment
# ---

def _next_node_end(nodes_str: str):
    nodes_str = nodes_str.strip()

    current_end = 0
    # Skip children
    if nodes_str.startswith('('):
        current_end = _find_closing_parenthesis(nodes_str)
    
    # Skip label, distances and comments
    # Strategy: find the next comma that is not surrounded by brackets or
    #    parentheses or the end of the string
    grouping = {'(':'()', '[':'[]'}
    while current_end < len(nodes_str):

        char = nodes_str[current_end]

        if char in grouping:
            current_end = _find_closing(nodes_str, current_end+1, grouping[char])
            continue

        if char == ',':
            # Ah, the lonely comma...
            return current_end-1
        
        current_end += 1
    else:
        return len(nodes_str)-1
# ---

def _split_nodes(nodes_str: str) -> List[str]:

    nodes_str = nodes_str.strip()

    if nodes_str == ',':
        return ['', '']
    if nodes_str == '':
        return []

    next_node_end = _next_node_end(nodes_str)
    if next_node_end != -1:
        node = nodes_str[:next_node_end+1]
        rest = nodes_str[next_node_end+1:]
        if rest.startswith(','): rest = rest[1:]
    else:
        node = ''
        rest = nodes_str[1:]
    
    if rest == ',':
        return [node, '']

    return [node] + _split_nodes(rest)
# ---

def _simple_distance(dist): return float(dist) if dist else None
def _simple_feature(feat): return feat

Distance = TypeVar('Distance')
Feature = TypeVar('Feature')
Tree = TypeVar('Tree')
def parse_newick_subtree(
        newick: str, 
        aggregator: Callable[[str, Sequence[Tree], Distance, Feature], Tree],
        distance_parser: Callable[[str], Distance] = _simple_distance,
        feature_parser: Callable[[str], Feature] = _simple_feature) -> Tree:

    parts = _parts_of_subtree(newick.strip())
    (children_str, label, distance_str, comment_str) = parts

    children = [parse_newick_subtree(child_str, aggregator) 
                for child_str in _split_nodes(children_str)]
    
    features = feature_parser(comment_str)
    distance = distance_parser(distance_str)

    return aggregator(label, children, distance, features)
# ---

def parse_newick(
        newick: str, 
        aggregator: Callable[[str, Sequence[Tree], Distance, Feature], Tree],
        distance_parser: Callable[[str], Distance] = _simple_distance,
        feature_parser: Callable[[str], Feature] = _simple_feature) -> Tree:

    if not newick.endswith(';'):
        raise ValueError("Tree in Newick format must end with ';'")

    root = newick[:-1]
    return parse_newick_subtree(
        root, 
        aggregator, 
        distance_parser=distance_parser, 
        feature_parser=feature_parser
    )
# ---


def test_parse_newick():

    def tree_as_dict(label, children, distance, features):
        return dict(
            label=label, 
            children=children, 
            features=(distance, features)
        )

    def tree_as_list(label, children, distance, features):
        return [label, children]

    nwk = 'A;'
    assert (
        parse_newick(nwk, tree_as_dict) 
        == 
        {
            'label': 'A',
            'children': [], 
            'features': (None, '')
        }
    )

    nwk = "(A, B);"
    assert (
        parse_newick(nwk, tree_as_dict) 
        == 
        {
            'label': '', 'features': (None, ''),
            'children': [
                {'label': 'A', 'children': [], 'features': (None, '')},
                {'label': 'B', 'children': [], 'features': (None, '')}
            ]
        }
    )

    nwk = "(A[{'some_feature': 10}], (B:4[some_other_thing], C)D:122.2)root;"
    assert (
        parse_newick(nwk, tree_as_dict) 
        == 
        {
            'label': 'root','features': (None, ''),
            'children': [
                {
                    'label': 'A', 'features': (None, "{'some_feature': 10}"),
                    'children': []
                },
                {
                    'label': 'D', 'features': (122.2, ''),
                    'children': [
                        {
                            'label': 'B', 'features': (4., 'some_other_thing'),
                            'children': []
                        },
                        {
                            'label': 'C', 'features': (None, ''), 'children': []
                        }
                    ]
                }
            ]
        }
    )

    nwk = "((A, B), (C, D));"
    assert (
        parse_newick(nwk, tree_as_list) 
        == 
        ['', [ 
            ['', [
                ['A', []],
                ['B', []]]],
            [ '', [
                ['C', []],
                ['D', []]]]
        ]]
    )


    nwk = "(, , );"
    assert (
        parse_newick(nwk, tree_as_list) 
        == 
        ['', [
            ['', []],
            ['', []],
            ['', []]
        ]]
    )

    nwk = "((, ), (, ));"
    assert (
        parse_newick(nwk, tree_as_list) 
        == 
        ['', [
            ['', [
                    ['', []],
                    ['', []] ]],
            ['', [
                    ['', []],
                    ['', []] ]]
        ]]
    )