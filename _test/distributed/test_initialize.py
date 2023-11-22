from flex_model.distributed.distributed_state import GPUDeviceMesh


def test_GPUDeviceMesh():
    cases = {
        (1, 1, 1): [[[0]], [[0]], [[0]]],
        (4, 1, 1): [[[0, 1, 2, 3]], [[0], [1], [2], [3]], [[0], [1], [2], [3]]],
        (1, 4, 1): [[[0], [1], [2], [3]], [[0, 1, 2, 3]], [[0], [1], [2], [3]]],
        (1, 1, 4): [[[0], [1], [2], [3]], [[0], [1], [2], [3]], [[0, 1, 2, 3]]],
        (2, 2, 1): [[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0], [1], [2], [3]]],
        (1, 2, 2): [[[0], [1], [2], [3]], [[0, 2], [1, 3]], [[0, 1], [2, 3]]],
        (2, 2, 2): [
            [[0, 1], [2, 3], [4, 5], [6, 7]],
            [[0, 4], [1, 5], [2, 6], [3, 7]],
            [[0, 2], [1, 3], [4, 6], [5, 7]],
        ],
    }

    for case, solution in cases.items():
        tp = case[0]
        pp = case[1]
        dp = case[2]
        world_size = tp * pp * dp
        gpu_device_mesh = GPUDeviceMesh.build(world_size, tp, pp, dp)
        assert gpu_device_mesh.tp_group_ranks == solution[0], f"{case}"
        assert gpu_device_mesh.pp_group_ranks == solution[1], f"{case}"
        assert gpu_device_mesh.dp_group_ranks == solution[2], f"{case}"
