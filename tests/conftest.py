"""Shared pytest and Hypothesis configuration."""

import os

from hypothesis import settings

settings.register_profile(
    "ci",
    max_examples=100,
    derandomize=True,
    deadline=None,
)
settings.register_profile(
    "dev",
    max_examples=25,
    derandomize=False,
    deadline=None,
)

settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))
