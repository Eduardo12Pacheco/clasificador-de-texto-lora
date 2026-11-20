"""Dataset split names."""

from enum import StrEnum


class Split(StrEnum):
    """The three official MASSIVE partitions."""

    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
