"""
Generic recursive tree reduce algorithm
=======================================

Trees are one of the most ubiquitous data structures. It is amazing how often we 
as programmers tend to reimplement the same algorithms for different trees.

This module defines a generic tree-reduce algorithms that can be
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

    def is_leaf(node): return isinstance(node, str)

    def children(node):
        return node['children']

    def all_nodes(node, children):
        nodes = set()
        for child in children:
            if is_leaf(child):
                nodes.add(child)
            else:
                nodes.update(child)
        return nodes | {node['label']}
        
    tree_reduce(tree, children, all_nodes)

    # Output --> {'A','B','C','D','E'}
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
        if node.is_leaf():
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
"""

from typing import (Any, List, Callable)

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

"""
--------------------------------------------------------------------------------
Usage and tests
--------------------------------------------------------------------------------
To run tests: `pytest tree_reduce.py`
To run static type check: `mypy tree_reduce.py`
To run coverage analysis:
    `coverage run --src=. -m pytest tree_reduce.py`
    `coverage html`
"""
def test_tree_reduce_to_list_all_nodes():
    tree = {
        'label':'A', 'children':[
            {'label':'B', 'children':[]},
            {'label':'C', 'children': [
                {'label':'D', 'children':[]}, 
                {'label':'E', 'children':[]}
            ]}
        ]
    }

    def is_leaf(node): return isinstance(node, str)

    def children(node): return node['children']

    def all_nodes(node, children):
        nodes = set()
        for child in children:
            if is_leaf(child):
                nodes.add(child)
            else:
                nodes.update(child)
        return nodes | {node['label']}
        
    assert tree_reduce(tree, children, all_nodes) == {'A','B','C','D','E'}

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
