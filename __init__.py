"""
Generic tree utilities for Python
=================================

Trees are one of the most ubiquitous data structures. It is amazing how often we 
as programmers tend to reimplement the same algorithms for different trees.

This module defines generic tree-traverse and tree-reduce algorithms that can be
used with any tree-like object such as filesystem paths, lists, nested 
dictionaries an expression tree or even specialized tree classes! The only thing 
that must be provided is a function to get child nodes from a parent node.

Also, trees are usually represented in some fields (such as bioinformatics) in 
the newick format, which is nontrivial to parse, so this module includes a 
function to do this.

Examples
--------

Import the basic functions:
```
    import treetools
```

###  Use with any kind of structured tree!
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
        
    [node['label'] 
        for node in treetools.traverse(tree, children, mode='inorder')]

    # Output --> ['B, 'A', 'D', 'C', 'E']
```

###  Even with user-defined classes!
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


    treetools.reduce(tree, get_children, node_to_newick)
                
    # Output --> '(B,(D,E))'
```
### Parse a newick-formatted tree structure
```
    def parse_data(data_string):
        '''
        Example: 
          'data1=xx,data2=yy' 
            -> {'data1':'xx', 'data2': 'yy'}
        '''
        items = feature_string.split(',')
        key_value_pairs = (item.split('=') for item in items)
        return dict(key_value_pairs)

    def parse_branch_length(length_str):
        return float(length_str) if length_str else 0.0

    def tree_builder(label, children, branch_length, node_data):
        return {
            'label': label, 
            'children': children, 
            'length': branch_length,
            'data': node_data}
    
    newick = "(A:0.2:[dat=23,other=45], B:12.4:[dat=122,other=xyz])root[x=y];"
    
    treetools.parse_newick(
        newick,
        aggregator=tree_builder,
        feature_parser=parse_data,
        distance_parser=parse_branch_length
    )

    # Output ->
    {
        'label': 'root', 'length':0, 'data': {'x':'y'},
        'children': [
            {'label': 'A', 'length':0.2, 'data':{'dat':'23','other':'45'},
             'children': []},
            {'label': 'B', 'length':12.4, 'data':{'dat':'122','other':'xyz'},
             'children': []}, 
        ]
    }
```

### Compose to perform complex algorithms
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
    induced = treetools.reduce(tree, get_children, induced_subtree(leafs))
    print(induced)

    # Output --> ((('B',('D',)),),'E')


    def merge_unary_nodes(node, children):
        if is_leaf(node):
            return node
        
        new_children = [
            ch[0] if (len(ch) == 1) else ch
            for ch in children
        ]
        
        return tuple(new_children)

    treetools.reduce(subtree, get_children, merge_unary_nodes)

    # Output --> (('B','D'),'E')
```


### Use even with filesystem paths!
```
from pathlib import Path

def enter_folder(path_str):
    return list(Path(path_str).iterdir()) if path.is_dir() else []

for item in treetools.traverse('/usr', enter_folder, mode='breadth_first'):
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

from .traverse import tree_traverse as traverse
from .reduce import tree_reduce as reduce
from .parse import parse_newick, parse_newick_subtree