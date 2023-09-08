from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import logging
from typing import Union, Tuple, List, Any, Dict, Callable, Optional

import torch
from torch import Tensor

from .nodes import (
    InternalObject,
    InternalNode,
    LeafObject,
    LeafNode,
    ScalarObject,
    ScalarNode,
    InternalNode,
    is_leaf_obj,
    is_internal_obj,
    is_leaf_node,
    is_internal_node,
    get_internal_node,
    get_leaf_node,
)


def flatten(
    root_obj: Any,
) -> Tuple[Union[InternalNode, LeafNode, ScalarNode], List[Optional[Tensor]]]:
    """Flatten an arbitrary python object into a tree definition and a
    collection of leaves. These can then be repacked by :code:`unflatten` to
    perfectly reconstruct the original python object. The python object is
    recursively unpacked using node representations, which each locally know
    how to unpack themselves.

    :note: The traversal is done in a depth-first way to bias us towards
        finding the left-most leaf node first.

    :param Any root_obj: The python object to flatten.

    :returns: A tree definition of the python object and a list of leaf
        objects (typically Pytorch tensors).
    :rtype: Tuple[Union[InternalNode, LeafNode, ScalarNode], List[Optional[Tensor]]]
    """
    order = []
    leaves = []

    def _dfs(obj):
        # Leaf obj case
        if is_leaf_obj(obj):
            leaf_node = get_leaf_node(obj)(val=obj.shape)
            order.append(leaf_node)
            leaves.append(obj)
            return leaf_node

        # Internal obj recursive case
        elif is_internal_obj(obj):
            # NOTE: Each node needs to know how to flatten its associated type
            #       instance. Ie. BaseModelOutputWithPast needs to be able to
            #       return its attributes in a tuple. They should also be able
            #       to perfectly recreate instances of themselves using a list of
            #       children.
            internal_node = get_internal_node(obj)()
            order.append(internal_node)

            # Internal node knows how to unpack its equivalent internal object
            unvisited_children = internal_node.flatten(obj)

            # Recurse into internal object's children
            for child in unvisited_children:
                internal_node.children.append(_dfs(child))
            return internal_node

        # Scalar obj case
        else:
            # Scalar nodes are just objects
            scalar_node = obj
            order.append(scalar_node)
            return scalar_node

    _dfs(root_obj)
    return order[0], leaves


def unflatten(
    root_node: Union[InternalNode, LeafNode, ScalarNode], leaves: List[Optional[Tensor]]
) -> Any:
    """Repack a tree definition and list of leaves into the original python
    object.

    :param root_node: Root node which defines the tree definition of the python
        object.
    :type root_node: Union[InternalNode, LeafNode, ScalarNode], leaves: List[Optional[Tensor]]
    :param leaves: List of leaf nodes.
    :type leaves: List[Optional[Tensor]]

    :returns: The reconstructed python objects.
    :rtype: Any
    """
    leaves = list(reversed(leaves))

    def _dfs(node):
        # Leaf node case
        if is_leaf_node(node):
            return leaves.pop()

        # Internal node case
        elif is_internal_node(node):
            # Node knows how to pack itself up again into its corresponding obj
            obj = node.unflatten(_dfs(child) for child in node.children)
            return obj

        # Scalar node case
        else:
            return node

    return _dfs(root_node)
