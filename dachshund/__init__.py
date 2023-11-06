#!/usr/bin/env python3 -u
# copyright: dachshund, BSD-3-Clause License (see LICENSE file)
"""A little interface to access lots of data.

:mod:`dachshund` provides ightweight interface for accessing ag and econ data.
"""
from typing import List

__version__: str = "0.1.0"

__author__: List[str] = ["RNKuhns"]
__all__: List[str] = [
    "get_default_config",
    "get_config",
    "set_config",
    "reset_config",
    "config_context",
]
