"""
This module contains utility functions for manipulating and analyzing tree structures.
"""
from typing import Dict

class kTree():
    def __init__(self, n_children: int, depth: int = 2):
        """
        Initialize a k-ary tree with the specified number of children per node. A k-ary tree is a tree where each internal node has exactly k children.

        Parameters:
            n_children      (int)       Number of children each node has
        """
        self.n_children = n_children
        self.depth

    def internal_nodes(self, depth: int = None) -> int:
        """
        Calculate the number of internal nodes in a k-ary tree. An internal node is a node that has at least one child, i.e., it is not a leaf node.
        """
        if depth:
            (self.n_children ** depth - 1) // (self.n_children - 1)
        else:
            return (self.n_children ** self.depth - 1) // (self.n_children - 1)
    
    def total_nodes(self, depth: int = None) -> int:
        """ Calculate the total number of nodes in a k-ary tree, including both internal and leaf nodes. """
        if depth: 
            return (self.n_children ** (depth + 1) - 1) // (self.n_children - 1)
        else:
            return (self.n_children ** (self.depth + 1) - 1) // (self.n_children - 1)
    
    def find_depth(self, n_nodes: int) -> int: 
        """Find the depth of the tree for a given number of nodes. Assumes strict adherence to k-ary tree structure (each node has exactly k children)."""
        for i in range(10):
            if (self.n_children ** i - 1) / (self.n_children - 1) == n_nodes:
                return i
            
    def leaf_nodes(self, depth: int = None) -> int:
        """Calculate the number of leaf nodes in a k-ary tree. """
        return self.n_children ** depth if depth else self.n_children ** self.depth
            

    def get_indices(self, depth: int = None) -> Dict: 
        """
        Get the indices of the parent and child nodes in a k-ary tree. The indices are calculated based on the depth of the tree.
        """
        indices = {}
        indices['parent_start'] = ((self.n_children ** depth - 1) // (self.n_children - 1)) - 1
        indices['parent_end'] = ((self.n_children ** (depth + 1) - 1) // (self.n_children - 1)) - 1
        indices['child_start'] = ((self.n_children ** (depth + 1) - 1) // (self.n_children - 1)) - 1
        indices['child_end'] = ((self.n_children ** (depth + 2) - 1) // (self.n_children - 1)) - 1
        return indices

# TODO: implement data management (targeted at flatland environment tree observation)