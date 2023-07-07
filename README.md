# Installation
Run `pip install -e .` from the root directory.


# Running tests
`cd flex_model/tests`
`python test_single_gpu.py`
`accelerate launch test_distributed.py` <- Make sure this is run on 2 GPUs
