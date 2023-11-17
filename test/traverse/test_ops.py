import torch

from flex_model.traverse.ops import flatten, unflatten


def test_flatten_and_unflatten():
    layer_output = [
        1,
        2,
        torch.ones((1)).cuda(),
        "zzz",
        (torch.ones((1)).cuda() * 2, torch.ones((1)).cuda() * 3),
    ]
    treedef, leaves = flatten(layer_output)
    for i, leaf_ten in enumerate(leaves):
        assert torch.equal(leaf_ten, torch.ones((1)).cuda() * (i + 1))

    edited_leaves = [leaf_ten * 2 for leaf_ten in leaves]

    result = unflatten(treedef, edited_leaves)
    new_treedef, new_leaves = flatten(result)
    assert new_treedef == treedef
    assert new_leaves == edited_leaves
