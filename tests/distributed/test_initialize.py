from flex_model.distributed.backends import GPUDeviceMesh


def test_GPUDeviceMesh():
    cases = [
        (1, 1, 1),
        (4, 1, 1),
        (1, 4, 1),
        (1, 1, 4),
        (4, 4, 1),
        (1, 4, 4),
        (4, 1, 4),
        (2, 2, 2),
        (2, 4, 2),
    ]
    # (tp, pp, dp)
    solutions = [
        [[[0]], [[0]], [[0]],],
        [[[0, 1, 2, 3]], [[0]], [[0]],],
        [[[0], [1], [2], [3]], [[0, 1, 2, 3]], [[0], [1], [2], [3]],],
        [[[0], [1], [2], [3]], [[0]], [[0, 1, 2, 3]],],
        [
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            [[0, 4, 8, 12]],
            [[0], [4], [8], [12]],
        ],
        [
            [[i] for i in range(16)],
            # [[0, 1, 2, 3]],
            [[0, 4, 8, 12]],
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
        ],
        [
            [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
            [[0]],
            [[0, 4, 8, 12]],
        ],
        [[[0, 1], [2, 3], [4, 5], [6, 7]], [[0, 4]], [[0, 2], [4, 6]],],
        [
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
            [[0, 4, 8, 12]],
            [[0, 2], [4, 6], [8, 10], [12, 14]],
        ],
    ]
    for case, solution in zip(cases, solutions):
        tp = case[0]
        pp = case[1]
        dp = case[2]
        world_size = tp * pp * dp
        gpu_device_mesh = GPUDeviceMesh.build(world_size, tp, pp, dp)
        assert gpu_device_mesh.tp_group_ranks == solution[0], f"{case}"
        assert gpu_device_mesh.pp_group_ranks == solution[1], f"{case}"
        assert gpu_device_mesh.dp_group_ranks == solution[2], f"{case}"
