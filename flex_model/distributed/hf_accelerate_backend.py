from accelerate import Accelerator


def accelerate_is_initialized():
    ps = accelerate.PartialState()
    if (
        ps.distributed_type == accelerate.DistributedType.MULTI_GPU
        or ps.distributed_type == accelerate.DistributedType.FSDP
        or ps.distributed_type == accelerate.DistributedType.MEGATRON_LM
    ):
        return True
    else:
        return False
