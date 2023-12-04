from .test_fairscale_mpu_vs_ddp import (
    test_fairscale,
    test_forward_hooks_fairscale,
    test_full_backward_hooks_fairscale,
    test_forward_pre_hooks_fairscale,
    test_full_backward_pre_hooks_fairscale,
)
from .test_dp_wrappers_vs_single_gpu import (
    test_forward_hooks_wrapped,
    test_full_backward_hooks_wrapped,
    test_forward_pre_hooks_wrapped,
    test_full_backward_pre_hooks_wrapped,
)
