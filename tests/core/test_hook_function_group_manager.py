import pytest

from flex_model.core import HookFunction
from flex_model.core.wrapper import _HookFunctionGroupManager
from tests.fixtures import opt_350m_module_names


def test_HookFunctionGroupManager_create(opt_350m_module_names):
    manager = _HookFunctionGroupManager()

    new_hook_fns = manager.create(
        "new_group", "self_attn", (None, None, None), all_names=opt_350m_module_names,
    )
    new_hook_fns = set(new_hook_fns)

    assert "new_group" in manager.groups

    for hook_fn, groups in manager.hook_fn_to_groups_map.items():
        if hook_fn in new_hook_fns:
            assert "new_group" in groups
            assert "self_attn" in hook_fn.module_name
        else:
            assert "new_group" not in groups
            assert "self_attn" not in hook_fn.module_name


def test_HookFunctionGroupManager_update_by_list(opt_350m_module_names):
    manager = _HookFunctionGroupManager()

    original_hf_group = manager.create(
        "new_group", "self_attn", (None, None, None), all_names=opt_350m_module_names,
    )
    new_hf_group = [hf for hf in original_hf_group if "q_proj" in hf.module_name]

    manager.update(
        new_hf_group, group_name="q_proj",
    )

    assert "new_group" in manager.groups
    assert "q_proj" in manager.groups

    original_hf_group = set(original_hf_group)
    new_hf_group = set(new_hf_group)
    for hook_fn, groups in manager.hook_fn_to_groups_map.items():
        if hook_fn in original_hf_group:
            assert "new_group" in groups
            assert "self_attn" in hook_fn.module_name
        else:
            assert "new_group" not in groups
            assert "self_attn" not in hook_fn.module_name

        if hook_fn in new_hf_group:
            assert "q_proj" in groups
            # Don't assert "q_proj" in module name since test does this.
        else:
            assert "q_proj" not in groups


def test_HookFunctionGroupManager_update_by_hook_fn(opt_350m_module_names):
    manager = _HookFunctionGroupManager()

    hook_function = HookFunction("model.decoder.layers.12", (None, None, None),)

    manager.update(hook_function, group_name="new_group")

    assert "new_group" in manager.groups

    for hook_fn, groups in manager.hook_fn_to_groups_map.items():
        if hook_fn is hook_function:
            assert "new_group" in groups
        else:
            assert "new_group" not in groups


def test_HookFunctionGroupManager_update_by_string(opt_350m_module_names):
    manager = _HookFunctionGroupManager()

    new_hf_group = manager.create(
        "new_group", "self_attn", (None, None, None), all_names=opt_350m_module_names,
    )

    manager.update(
        "k_proj", group_name="k_proj_group",
    )

    assert "new_group" in manager.groups
    assert "k_proj_group" in manager.groups

    for hook_fn, groups in manager.hook_fn_to_groups_map.items():
        if hook_fn in new_hf_group:
            assert "new_group" in groups
        else:
            assert "new_group" not in groups

        if "k_proj" in hook_fn.module_name:
            assert "k_proj_group" in groups
        else:
            assert "k_proj_group" not in groups


def test_HookFunctionGroupManager_remove_by_list(opt_350m_module_names):
    manager = _HookFunctionGroupManager()

    new_hf_group = manager.create(
        "new_group", "self_attn", (None, None, None), all_names=opt_350m_module_names,
    )
    other_hf_group = manager.create(
        "other_group", "q_proj", (None, None, None), all_names=opt_350m_module_names,
    )

    assert "new_group" in manager.groups
    assert "other_group" in manager.groups

    manager.remove(new_hf_group, "new_group")

    assert "new_group" not in manager.groups

    for hook_fn, groups in manager.hook_fn_to_groups_map.items():
        if hook_fn in new_hf_group:
            assert "new_group" not in groups
            assert "other_group" not in groups
        else:
            assert "other_group" in groups
            assert "new_group" not in groups


def test_HookFunctionGroupManager_remove_by_hook_fn(opt_350m_module_names):
    manager = _HookFunctionGroupManager()

    new_hf_group = manager.create(
        "new_group", "self_attn", (None, None, None), all_names=opt_350m_module_names,
    )
    hf_to_remove = new_hf_group[0]

    other_hf_group = manager.create(
        "other_group", "q_proj", (None, None, None), all_names=opt_350m_module_names,
    )

    manager.remove(hf_to_remove, "new_group")

    for hook_fn, groups in manager.hook_fn_to_groups_map.items():
        if hook_fn is hf_to_remove:
            assert "new_group" not in groups
            assert "other_group" not in groups
            continue

        if hook_fn in new_hf_group:
            assert "new_group" in groups
            assert "other_group" not in groups
        else:
            assert "other_group" in groups
            assert "new_group" not in groups


def test_HookFunctionGroupManager_remove_by_string(opt_350m_module_names):
    manager = _HookFunctionGroupManager()

    new_hf_group = manager.create(
        "new_group", "self_attn", (None, None, None), all_names=opt_350m_module_names,
    )

    other_hf_group = manager.create(
        "other_group", "q_proj", (None, None, None), all_names=opt_350m_module_names,
    )

    assert "new_group" in manager.groups
    assert "other_group" in manager.groups

    manager.remove("k_proj", "new_group")

    for hook_fn, groups in manager.hook_fn_to_groups_map.items():
        if "k_proj" in hook_fn.module_name:
            assert "new_group" not in groups
            assert "other_group" not in groups
        else:
            if hook_fn in new_hf_group:
                assert "new_group" in groups
                assert "other_group" not in groups
            else:
                assert "new_group" not in groups
                assert "other_group" in groups


def test_HookFunctionGroupManager_bisect(opt_350m_module_names):
    manager = _HookFunctionGroupManager()

    new_hf_group = manager.create(
        "new_group", "self_attn", (None, None, None), all_names=opt_350m_module_names,
    )

    other_hf_group = manager.create(
        "other_group", "fc1", (None, None, None), all_names=opt_350m_module_names,
    )

    active, inactive = manager.bisect("new_group")

    assert active.isdisjoint(inactive)
    assert set(other_hf_group) & inactive == inactive

    active, inactive = manager.bisect(["new_group", "other_group"])

    assert active.isdisjoint(inactive)
    assert inactive == set()
