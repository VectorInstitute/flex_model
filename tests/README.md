# Running Tests
You can run tests simply by calling `pytest --ignore=multi_gpu`. These tests
require a single GPU to run. Tests which require multiple gpus to run (see
`multi_gpu` folder) are run using `submitit` instead of `pytest`.
