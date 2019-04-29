from datetime import datetime
from logging import FileHandler, Formatter, StreamHandler, getLogger
from os.path import join


def initialize_logger(name):

    logger = getLogger(name)

    logger.setLevel(10)

    file_handler = FileHandler(
        join("/", "tmp", "{}.{:%Y:%m:%d:%H:%M:%S}.log".format(name, datetime.now()))
    )

    file_handler.setFormatter(
        Formatter("%(asctime)s|%(levelname)s: %(message)s\n", "%H%M%S")
    )

    logger.addHandler(file_handler)

    stream_handler = StreamHandler()

    stream_handler.setFormatter(Formatter("%(levelname)s: %(message)s\n"))

    logger.addHandler(stream_handler)

    logger.info("Initialized logger {}.".format(name))

    return logger
