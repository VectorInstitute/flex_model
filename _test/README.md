# Running Tests
Run tests using the command:
```
torchrun --nnodes 1 --nproc_per_node 1 -m pytest --ignore=_test/multi_gpu _test/
```
These tests require a single GPU to run. Tests which require multiple gpus to
run (see `multi_gpu` folder) are run using `submitit` instead of `pytest`.

# Test Coverage
To generate code coverage reports, run the following command from the top-level
`flex_model` directory (ie. `cd ..`).
```
coverage run -m pytest --ignore=tests/multi_gpu test/
```
And you can read the coverage report by running:
```
coverage report
```
