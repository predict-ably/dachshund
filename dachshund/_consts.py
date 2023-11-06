#!/usr/bin/env python3 -u
# copyright: dachshund, BSD-3-Clause License (see LICENSE file)
"""Constants used throughout the  :mod:`dachshund` package.

Include supported types and other constants used throughout the code base.
"""
from typing import List, Literal, get_args

__author__: List[str] = ["RNKuhns"]
__all__: List[str] = [
    "array_container_values",
    "data_container_values",
    "df_container_values",
    "ARRAY_CONTAINERS",
    "DATAFRAME_CONTAINERS",
    "DATA_CONTAINERS",
]


DATA_CONTAINERS = Literal[
    "polars", "polars_eager", "pandas", "modin", "dask", "numpy", "xarray"
]
DATAFRAME_CONTAINERS = Literal["polars", "polars_eagers", "pandas", "modin", "dask"]
ARRAY_CONTAINERS = Literal["numpy", "xarray"]

data_container_values = get_args(DATA_CONTAINERS)
df_container_values = get_args(DATAFRAME_CONTAINERS)
array_container_values = get_args(ARRAY_CONTAINERS)
