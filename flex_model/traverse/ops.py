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
)


def flatten(
    root_obj: Any,
) -> Tuple[Union[InternalNode, LeafNode, ScalarNode], List[Optional[Tensor]]]:
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
