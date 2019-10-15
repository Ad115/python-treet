"""
Generic tree utilities for Python
=================================

Trees are one of the most ubiquitous data structures. It is amazing how often we 
as programmers tend to reimplement the same algorithms for different trees.

This module defines generic tree-traverse and tree-reduce algorithms that can be
used with any tree-like object such as filesystem paths, lists, nested 
dictionaries an expression tree or even specialized tree classes! The only thing 
that must be provided is a function to get child nodes from a parent node.

Examples:

##  Use with any kind of structured tree!
```
    tree = {
        'label':'A', 'children':[
            {'label':'B', 'children':[]},
            {'label':'C', 'children': [
                {'label':'D', 'children':[]}, 
                {'label':'E', 'children':[]}
            ]}
        ]
    }

    def children(node):
        return node['children']
        
    [node['label'] for node in tree_traverse(tree, children, mode='inorder')]

    # Output --> ['B, 'A', 'D', 'C', 'E']
```

##  Even with user-defined classes!
```
    class Tree:
        def __init__(self, label, children=None):
            self.label = label
            self.children = children if children else []

        def is_leaf(self):
            return len(self.children) == 0

    tree = Tree('A', [
            Tree('B'),
            Tree('C',[Tree('D'),Tree('E')])
        ]
    )

    def get_children(node):
        return node.children

    def node_to_newick(node, children):
        if is_leaf(node):
            return node.label
        else:
            return f"({','.join(children)})"


    tree_reduce(tree, get_children, node_to_newick)
                
    # Output --> '(B,(D,E))'
```

## Compose to perform complex algorithms
```
    tree = (('A',('B',('C','D'))),'E')

    def is_leaf(node): 
        return isinstance(node, str)

    def get_children(node): 
        return node if not is_leaf(node) else []

    def induced_subtree(leafs):
        def induced_subtree_generator(node, children):
            if children:
                return tuple(ch for ch in children if not ch is None)
            else:
                return node if node in leafs else None
        return induced_subtree_generator

    leafs = ['B', 'D', 'E']
    induced_subtree = tree_reduce(tree, get_children, induced_subtree(leafs))
    print(induced_subtree)

    # Output --> ((('B',('D',)),),'E')


    def merge_unary_nodes(node, children):
        if is_leaf(node):
            return node
        
        new_children = [
            ch[0] if (len(ch) == 1) else ch
            for ch in children
        ]
        
        return tuple(new_children)

    tree_reduce(induced_subtree, get_children, merge_unary_nodes)

    # Output --> (('B','D'),'E')
```


## Use even with filesystem paths!
```
from pathlib import Path

def enter_folder(path):
    return list(path.iterdir()) if path.is_dir() else []

for item in tree_traverse(Path('/usr'), enter_folder, mode='breadth_first'):
    print(item)

# Output -->
# /
# /proc
# /usr
# ...
# /usr/share
# /usr/bin
# /usr/sbin
# ...
# /usr/bin/jinfo
# /usr/bin/m2400w
# ...
```
"""

from collections import deque
from typing import (Any, List, Callable, Iterable)

TreeLike = Any
Reduced = Any

def tree_reduce(
        tree_node: TreeLike, 
        get_children: Callable[[TreeLike], List[TreeLike]], 
        reduce_fn: Callable[[TreeLike, List[Reduced]], Reduced]
    ) -> Reduced:
    """
    
    Perform a recursive tree reduction for the tree-like object `tree_node` 
    taken as the root. 
    The callable `get_children` returns a list of the descendants of any node of 
    the tree.

    Example:
    ```
        tree = [[['A', 'B'], 'C'], ['D', ['E', 'F'],'G']]

        def is_leaf(node):
            return isinstance(node, str)

        def get_children(node):
            if is_leaf(node):
                return []
            else:
                return node

        def node_to_newick(node, children):
            if is_leaf(node):
                return node
            else:
                return f"({','.join(children)})"

        # <--- Actual reduction
        tree_reduce(tree, get_children, node_to_newick)
            
        # Output --> '(((A,B),C),(D,(E,F),G))'
    ```
    """

    reduced_children = [tree_reduce(child, get_children, reduce_fn) 
                        for child in get_children(tree_node)]
    return reduce_fn(tree_node, reduced_children)
# ---

def tree_traverse(
        root: TreeLike, 
        get_children: Callable[[TreeLike], List[TreeLike]], 
        mode: str = 'depth_first'
    ) -> Iterable[TreeLike]:
    """
    Iterate over the nodes of the given tree-like object. The descendants of 
    each node are obtained by applying the callable `get_children` on the node. 

    The iteration order is depth-first by default and can be changed via the 
    `mode` argument. The available modes are: 'depth first' (same as 
    'preorder'), 'breadth first' (same as 'level order'), 'inorder' and 
    'postorder', the mode is case insensitive and can contain spaces, 
    underscores and dashes.

    Example:
    ```
    tree = [[['A', 'B'], 'C'], ['D', ['E', 'F'],'G']]
     
    children_fn = (lambda node: node if isinstance(node, list) else [])
    is_leaf = (lambda node: isinstance(node, str))

    # <-- Print leaves in order of traversal
    for node in tree_traverse(tree, children_fn, mode='breadth first') 
        if is_leaf(node):
            print(node, end=' ')
    
    # Prints --> C D G A B E F
    ```
    """

    modes = {
        'depthfirst': traverse_depth_first,
        'preorder': traverse_depth_first,
        'breadthfirst': traverse_breadth_first,
        'levelorder': traverse_breadth_first,
        'postorder': traverse_post_order,
        'inorder': traverse_inorder
    }

    mode = (mode
            .replace('-', '')
            .replace('_', '')
            .replace(' ', '')
            .lower()
        )
        
    if mode not in modes:
        raise ValueError(
            f'Tree traversal mode {mode} not valid, '
            f'valid modes are: {", ".join(modes)}'
        )
    
    traversal_fn = modes[mode]
    yield from traversal_fn(root, get_children)
# ---

def traverse_depth_first(
        tree: TreeLike, 
        get_children: Callable[[TreeLike], List[TreeLike]]
    ) -> Iterable[TreeLike]:

    agenda = [tree]
    while agenda:
        current_node = agenda.pop()
        yield current_node
        agenda.extend(reversed(get_children(current_node)))
# ---

def traverse_breadth_first(
        tree: TreeLike, 
        get_children: Callable[[TreeLike], List[TreeLike]]
    ) -> Iterable[TreeLike]:

    agenda = deque([tree])
    while agenda:
        current_node = agenda.popleft()
        yield current_node
        agenda.extend(get_children(current_node))
# ---

def traverse_post_order(
        tree: TreeLike, 
        get_children: Callable[[TreeLike], List[TreeLike]]
    ) -> Iterable[TreeLike]:

    for child in get_children(tree):
        yield from traverse_post_order(child, get_children)
    yield tree
# ---

def traverse_inorder(
        tree: TreeLike, 
        get_children: Callable[[TreeLike], List[TreeLike]]
    ) -> Iterable[TreeLike]:

    children = get_children(tree)
    if children:
        first, *rest = children

        yield from traverse_inorder(first, get_children)
        yield tree
        for child in rest:
            yield from traverse_inorder(child, get_children)
    else:
        yield tree
# ---

"""
--------------------------------------------------------------------------------
Usage and tests
--------------------------------------------------------------------------------
To run tests: `pytest tree_utils.py`
To run static type check: `mypy tree_utils.py`
To run coverage analysis:
    `coverage run --src=. -m pytest tree_utils.py`
    `coverage html`
"""

def test_traverse():
    tree = [[['A', 'B'], 'C'], ['D', ['E', 'F'],'G']]
     
    children_fn = (lambda node: 
        node if isinstance(node, list) else []
    )

    assert (
        [node for node in tree_traverse(tree, children_fn, mode='depth_first')]
         ==
        [node for node in traverse_depth_first(tree, children_fn)]
    )

    assert (
        [node for node in tree_traverse(tree, children_fn, mode='pre order')]
         ==
        [node for node in traverse_depth_first(tree, children_fn)]
    )

    assert (
        [node for node in tree_traverse(tree, children_fn, mode='breadth-first')]
         ==
        [node for node in traverse_breadth_first(tree, children_fn)]
    )

    assert (
        [node for node in tree_traverse(tree, children_fn, mode='LevelOrder')]
         ==
        [node for node in traverse_breadth_first(tree, children_fn)]
    )

    assert (
        [node for node in tree_traverse(tree, children_fn, mode='POSTorder')]
         ==
        [node for node in traverse_post_order(tree, children_fn)]
    )

    assert (
        [node for node in tree_traverse(tree, children_fn, mode='INORDER')]
         ==
        [node for node in traverse_inorder(tree, children_fn)]
    )

def test_breadth_first():
    tree = [[['A', 'B'], 'C'], ['D', ['E', 'F'],'G']]
     
    children_fn = (lambda node: node if isinstance(node, list) else [])
    is_leaf = (lambda node: isinstance(node, str))

    assert (
        [node 
            for node in traverse_breadth_first(tree, children_fn) 
            if is_leaf(node)]
         ==
         'C,D,G,A,B,E,F'.split(',')
     )

def test_depth_first():
    tree = [[['A', 'B'], 'C'], ['D', ['E', 'F'],'G']]
     
    children_fn = (lambda node: node if isinstance(node, list) else [])
    is_leaf = (lambda node: isinstance(node, str))

    assert (
        [node 
            for node in traverse_depth_first(tree, children_fn) 
            if is_leaf(node)]  
         ==
         'A,B,C,D,E,F,G'.split(',')
     )

def test_post_order():
    tree = [[['A', 'B'], 'C'], ['D', ['E', 'F'],'G']]
     
    children_fn = (lambda node: node if isinstance(node, list) else [])
    is_leaf = (lambda node: isinstance(node, str))

    assert (
        [node 
            for node in traverse_post_order(tree, children_fn) 
            if is_leaf(node)]  
        ==
        'A,B,C,D,E,F,G'.split(',')
    )

    tree = {
        'label':'A', 'children':[
            {'label':'B', 'children':[]},
            {'label':'C', 'children': [
                {'label':'D', 'children':[]}, 
                {'label':'E', 'children':[]}
            ]}
    ]
    }

    def children(node):
        return node['children']
    
    assert (
         [node['label']
            for node in traverse_post_order(tree, children)]
         ==
         'B,D,E,C,A'.split(',')
     )

def test_inorder():
    tree = [[['A', 'B'], 'C'], ['D', ['E', 'F'],'G']]
     
    children_fn = (lambda node: node if isinstance(node, list) else [])
    is_leaf = (lambda node: isinstance(node, str))

    assert (
        [node 
            for node in traverse_post_order(tree, children_fn) 
            if is_leaf(node)]  
        ==
        'A,B,C,D,E,F,G'.split(',')
    )

    tree = {
        'label':'A', 'children':[
            {'label':'B', 'children':[]},
            {'label':'C', 'children': [
                {'label':'D', 'children':[]}, 
                {'label':'E', 'children':[]}
            ]}
    ]
    }

    def children(node):
        return node['children']
    
    assert (
         [node['label']
            for node in traverse_inorder(tree, children)]
         ==
         'B,A,D,C,E'.split(',')
     )


def test_tree_reduce_for_newick_output():
    tree = [[['A', 'B'], 'C'], ['D', ['E', 'F'],'G']]

    children_fn = (lambda node: 
        node 
            if isinstance(node, list) 
            else []
     )

    def node_to_newick(node, children):
        print("assembling node to newick form:", node)
        return (
            '(' + ','.join(children) + ')'
            if children
            else node
        )

    assert (
        tree_reduce(tree, children_fn, node_to_newick)
        ==
        '(((A,B),C),(D,(E,F),G))'
    )


def test_tree_reduce_for_induced_subtree():
    tree = (('A',('B',('C','D'))),'E')

    children_fn = (lambda node: 
        node 
            if isinstance(node, tuple) 
            else []
     )

    def induced_subtree(leafs):
        def induced_subtree_generator(node, children):
            print('Processing node:', node)
            if children:
                return tuple(ch for ch in children if not ch is None)
            else:
                return node if node in leafs else None
        return induced_subtree_generator

    leafs = ['B', 'D', 'E']
    induced_subtree = tree_reduce(tree, children_fn, induced_subtree(leafs))
    assert( 
        induced_subtree
        ==
        ((('B',('D',)),),'E')
    )

    def merge_unary_nodes(node, children):
        is_leaf = lambda node: isinstance(node, str)
        if is_leaf(node):
            return node
        
        new_children = [
            ch[0] if (len(ch) == 1) else ch
            for ch in children
        ]
        
        return tuple(new_children)
    
    assert( 
        tree_reduce(induced_subtree, children_fn, merge_unary_nodes)
        ==
        (('B','D'),'E')
    )

"""
Ideas for property-based testing:
    - Different paths, same destination: Different traversals must yield the 
        same nodes.

    - There, and back again: Converting a tree to another format (lists - dicts)
        and back must yield the same tree.
    
    - Some things never change: Maybe check for invariants?

    - The more things change, the more they stay the same: Applying an operation 
        twice yields the same result as applying it once. Maybe filtering or 
        prunning? Traversing twice?

    - Solve a small problem first: Trees should have recursive properties given 
        their nature.

    - Hard to prove, easy to verify: Maybe yield nodes with paths and then 
        verify that following the path the node is found.
"""