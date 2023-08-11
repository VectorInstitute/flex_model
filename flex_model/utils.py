import logging

def setup_logger(level):
    logging.basicConfig(
        format="%(asctime)s | %(name)s | %(funcName)s | %(message)s",
        level=level.upper(),
    )
