# Running Tests
You can run tests simply by calling `pytest --ignore=multi_gpu`. These tests
require a single GPU to run. Tests which require multiple gpus to run (see
`multi_gpu` folder) are run using `submitit` instead of `pytest`.

# Test Coverage
You can generate a report on code coverage by running:
```
coverage run --source path/to/flex_model/tests/ -m pytest --ignore=multi_gpu
```
And you can read the coverage report by running:
```
coverage report
```
