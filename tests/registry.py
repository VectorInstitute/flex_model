from functools import wraps
from typing import Dict, Callable

UNIT_TESTS: Dict[str, Callable] = {}


def register_test(test_fn):
    UNIT_TESTS[test_fn.__name__] = test_fn
    return test_fn
