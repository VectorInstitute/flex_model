from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

# Types which do not have classes
# TODO: Add high-level type for root node tree definitions
InternalObject = Any
LeafObject = Any
ScalarObject = Any
ScalarNode = Any


_INTERNAL_NODE_TYPE_REGISTRY: Dict[type, InternalNode] = {}
_LEAF_NODE_TYPE_REGISTRY: Dict[type, LeafNode] = {}


def register_internal_node_type(internal_node_type: type) -> Callable:
    """Decorator for registering :class:`InternalNode` classes with a
    corresponding :code:`type`.

    :param type internal_node_type: The :code:`type` associated with the node.

    :returns: Inner function which registers the :class:`InternalNode` child
        class with the :code:`type`.
    """

    def _inner(_internal_node_cls: InternalNode) -> InternalNode:
        _INTERNAL_NODE_TYPE_REGISTRY[internal_node_type] = _internal_node_cls
        return _internal_node_cls

    return _inner


def register_leaf_node_type(leaf_node_type: type) -> Callable:
    """Decorator for registering :class:`LeafNode` classes with a
    corresponding :code:`type`.

    :param type internal_node_type: The :code:`type` associated with the node.

    :returns: Inner function which registers the :class:`LeafNode` child
        class with the :code:`type`.
    """

    def _inner(_leaf_node_cls: LeafNode) -> LeafNode:
        _LEAF_NODE_TYPE_REGISTRY[leaf_node_type] = _leaf_node_cls
        return _leaf_node_cls

    return _inner


class InternalNode:
    """Node correponding to unpackable container. These are nodes which we can
    always unpack other python objects from to continue traversal.

    :var children: A list of children nodes.
    :type children: Optional[List[Union[InternalNode, LeafNode, ScalarNode]]]
    """

    def __init__(
        self, children: Optional[List[Union[InternalNode, LeafNode, ScalarNode]]] = None
    ) -> None:
        self.children = children if children is not None else []

    def __eq__(self, other: Any) -> bool:
        """Traverse subtree checking for node equality recursively.

        :param Any other: Other node defining a subtree to check equality
            against.

        :returns: True if the subtrees match, else false.
        :rtype: bool
        """

        def _dfs(
            node1: Union[InternalNode, LeafNode, ScalarNode],
            node2: Union[InternalNode, LeafNode, ScalarNode],
        ) -> bool:
            # Mismatched types
            if type(node1) != type(node2):
                return False

            # Leaf node case
            if is_leaf_node(node1) and is_leaf_node(node2):
                return True

            # Internal node case
            elif is_internal_node(node1) and is_internal_node(node2):
                if len(node1.children) != len(node2.children):
                    return False

                subtrees_equal = []
                for n1, n2 in zip(node1.children, node2.children):
                    subtree = _dfs(n1, n2)
                    subtrees_equal.append(subtree)
                return all(subtrees_equal)

            # Scalar node case
            else:
                return node1 == node2

        return _dfs(self, other)

    def __repr__(self) -> str:
        return f"Node({self.children})"

    def __str__(self) -> str:
        return self.__repr__()

    def flatten(self, instance):
        """Flatten the associated instance by returning its contents.
        :note: Child classes must implement this.

        :note: All flattening functions return a tuple of the unpacked elements.
        """
        raise NotImplementedError

    def unflatten(self, children):
        """Pack the contents (children) back into the associated container.
        :note: Child classes must implement this.
        """
        raise NotImplementedError


@register_internal_node_type(tuple)
class TupleNode(InternalNode):
    """Unpackable node corresponding to tuples."""

    def __repr__(self) -> str:
        return f"TupleNode({self.children})"

    def __str__(self) -> str:
        return self.__repr__()

    def flatten(self, instance: Tuple[Any, ...]) -> Tuple[Any, ...]:
        """Unpack the tuple. Flattening functions always return a tuple of the
        unpacked elements, so this function does nothing.

        :param Tuple[Any, ...] instance: Tuple instance to unpack.

        :returns: A tuple of the unpacked elements.
        :rtype: Tuple[Any, ...]
        """
        return instance

    def unflatten(self, children: List[Any]) -> Tuple[Any, ...]:
        """Re-assemble the tuple.

        :param List[Any] children: List of elements to repack the tuple with.

        :returns: A tuple of the repacked elements.
        :rtype: Tuple[Any, ...]
        """
        return tuple(child for child in children)


@register_internal_node_type(list)
class ListNode(InternalNode):
    """Unpackable node corresponding to lists."""

    def __repr__(self) -> str:
        return f"ListNode({self.children})"

    def __str__(self) -> str:
        return self.__repr__()

    def flatten(self, instance: List[Any]) -> Tuple[Any, ...]:
        """Unpack the list.

        :param List[Any] instance: List instance to unpack.

        :returns: A tuple of the unpacked elements.
        :rtype: Tuple[Any, ...]
        """
        return tuple(instance)

    def unflatten(self, children: List[Any]) -> List[Any]:
        """Re-assemble the list.

        :param List[Any] children: List of elemnets to repack the list with.

        :returns: A list of the repacked elements.
        :rtype: List[Any]
        """
        return list(child for child in children)


@register_internal_node_type(BaseModelOutputWithPast)
class BaseModelOutputWithPastNode(InternalNode):
    """Node corresponding to Huggingface BaseModelOutputWithPast object."""

    def __repr__(self) -> str:
        return f"BaseModelOutputWithPastNode({self.children})"

    def flatten(self, instance: BaseModelOutputWithPast) -> Tuple[Any, Any, Any, Any]:
        """Unpack the :code:`BaseModelOutputWithPast`.

        :param BaseModelOutputWithPast instance: :code:`BaseModelOutputWithPast` to
            unpack.

        :returns: A 4-tuple containing hidden state and other cached values.
        :rtype: Tuple[Any, Any, Any, Any, Any]
        """
        contents = (
            instance.last_hidden_state,
            instance.past_key_values,
            instance.hidden_states,
            instance.attentions,
        )
        return contents

    def unflatten(self, children: List[Any]) -> BaseModelOutputWithPast:
        """Re-assemble the :code:`BaseModelOutputWithPast`.

        :param List[Any] children: List of elements to repack the :code:`BaseModelOutputWithPast`
            with.

        :returns: A :code:`BaseModelOutputWithPast`.
        :rtype: BaseModelOutputWithPast
        """
        return BaseModelOutputWithPast(*children)


class LeafNode:
    """Leaf node, typically corresponding to a tensor.

    :note: Leaf nodes should not hold a ref to the underlying data, only some
        metadata.
    """

    def __init__(self, val: Any = None) -> None:
        self.val = val

    def __eq__(self, other: Any) -> bool:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "LeafNode"

    def __str__(self) -> str:
        return self.__repr__()


@register_leaf_node_type(Tensor)
class TensorNode(LeafNode):
    """Leaf node corresponding to a Pytorch tensor."""

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TensorNode):
            return False

        return self.val == other.val

    def __repr__(self) -> str:
        return f"TensorNode<{self.val}>"


def get_internal_node(internal_obj: InternalObject) -> InternalNode:
    """Retrieve the corresponding :class:`InternalNode` representation of an
    :class:`InternalObject`.

    :param InternalObject internal_obj: Target object.

    :returns: The corresponding node representation.
    :rtype: InternalNode
    """
    return _INTERNAL_NODE_TYPE_REGISTRY[type(internal_obj)]


def get_leaf_node(leaf_obj: LeafObject) -> LeafNode:
    """Retrieve the corresponding :class:`LeafNode` representation of an
    :class:`LeafObject`.

    :param LeafObject leaf_obj: Target object.

    :returns: The corresponding node representation.
    :rtype: LeafNode
    """
    return _LEAF_NODE_TYPE_REGISTRY[type(leaf_obj)]


# TODO: Deprecate in favour of _flatten
def _recursively_find_first_tensor(
    obj: Union[InternalObject, LeafObject, ScalarObject]
) -> Optional[Tensor]:
    if is_leaf_obj(obj):
        return obj

    if not is_internal_obj(obj):
        return

    for ele in obj:
        res = _recursively_find_first_tensor(ele)
        if res is not None:
            return res


def is_leaf_obj(obj: Any) -> bool:
    """Return true if the object corresponds to a leaf object.

    :param Any obj: Object to query against.

    :returns: True if the object has a corresponding leaf object representation.
    :rtype: bool
    """
    return type(obj) in _LEAF_NODE_TYPE_REGISTRY


def is_internal_obj(obj: Any) -> bool:
    """Return true if the object corrsponds to an internal node.
    :param Any obj: Object to query against.

    :returns: True if the object has a corresponding internal object representation.
    :rtype: bool
    """
    return type(obj) in _INTERNAL_NODE_TYPE_REGISTRY


def is_leaf_node(node: Any) -> bool:
    """Return true if the object is a leaf node.
    :param Any obj: Object to query against.

    :returns: True if the object has a corresponding leaf node representation.
    :rtype: bool
    """
    return isinstance(node, LeafNode)


def is_internal_node(node: Any) -> bool:
    """Return true if the object is an internal node.
    :param Any obj: Object to query against.

    :returns: True if the object has a corresponding internal node representation.
    :rtype: bool
    """
    return isinstance(node, InternalNode)
