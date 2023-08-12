import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from flex_model.traverse.nodes import BaseModelOutputWithPastNode


def test_BaseModelOutputWithPastNode():
    node = BaseModelOutputWithPastNode()
    obj = BaseModelOutputWithPast(
        last_hidden_state=torch.ones((1)),
        past_key_values=torch.ones((1)) * 2,
        hidden_states=torch.ones((1)) * 3,
        attentions=torch.ones((1)) * 4,
    )

    contents = node.flatten(obj)
    for i, c in enumerate(contents):
        assert torch.equal(c, torch.ones((1)) * (i + 1))

    new_obj = node.unflatten(contents)
    assert torch.equal(new_obj.last_hidden_state, obj.last_hidden_state)
    assert torch.equal(new_obj.past_key_values, obj.past_key_values)
    assert torch.equal(new_obj.hidden_states, obj.hidden_states)
    assert torch.equal(new_obj.attentions, obj.attentions)


test_BaseModelOutputWithPastNode()
