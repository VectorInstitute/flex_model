import logging
import time

import tests.core
import tests.distributed
import tests.traverse
from tests.registry import UNIT_TESTS
from flex_model.utils import setup_logger


logger = logging.getLogger(__name__)


def test():
    setup_logger("info")
    for name, test_fn in UNIT_TESTS.items():
        logger.info(
            f"Running test: {name}..."
        )
        test_fn()
        logger.info(
            f"Test: {name} - Finished successfully!"
        )

        # Wait for any straggler processes
        time.sleep(5)


if __name__ == "__main__":
    test()
