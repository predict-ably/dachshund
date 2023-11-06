#!/usr/bin/env python3 -u
# copyright: dachshund, BSD-3-Clause License (see LICENSE file)
# Includes functionality like get_config, set_config, and config_context
# that is similar to scikit-learn and skbase. These elements are copyrighted by
# their respective developers. For conditions see
# https://github.com/scikit-learn/scikit-learn/blob/main/COPYING
# https://github.com/sktime/skbase/blob/main/LICENSE
"""Implement logic for global configuration of dachshund.

Allows users to configure :mod:`dachshund`.
"""
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

from dachshund._config_param_setting import GlobalConfigParamSetting
from dachshund._consts import DATA_CONTAINERS, data_container_values

__author__: List[str] = ["RNKuhns"]
__all__: List[str] = [
    "get_default_config",
    "get_config",
    "set_config",
    "reset_config",
    "config_context",
]


GlobalConfigParam = str

_CONFIG_REGISTRY: Dict[GlobalConfigParam, GlobalConfigParamSetting] = {
    "data_container": GlobalConfigParamSetting(
        name="data_container",
        expected_type=str,
        allowed_values=data_container_values,
        default_value="polars",
    )
}

_GLOBAL_CONFIG_DEFAULT: Dict[GlobalConfigParam, Any] = {
    config_settings.name: config_settings.default_value
    for _, config_settings in _CONFIG_REGISTRY.items()
}

global_config = _GLOBAL_CONFIG_DEFAULT.copy()

_THREAD_LOCAL_DATA = threading.local()


def _get_threadlocal_config() -> Dict[GlobalConfigParam, Any]:
    """Get a threadlocal **mutable** configuration.

    If the configuration does not exist, copy the default global configuration.

    Returns
    -------
    dict
        Threadlocal global config or copy of default global configuration.
    """
    if not hasattr(_THREAD_LOCAL_DATA, "global_config"):
        _THREAD_LOCAL_DATA.global_config = global_config.copy()
    return _THREAD_LOCAL_DATA.global_config  # type: ignore


def get_default_config() -> Dict[GlobalConfigParam, Any]:
    """Retrieve the default global configuration.

    This will always return the default ``dachshund`` global configuration.

    Returns
    -------
    dict
        The default configurable settings (keys) and their default values (values).

    See Also
    --------
    config_context :
        Configuration context manager.
    get_config :
        Retrieve current global configuration values.
    set_config :
        Set global configuration.
    reset_config :
        Reset configuration to ``dachshund`` default.

    Examples
    --------
    >>> from dachshund import get_default_config
    >>> get_default_config()  # doctest: +ELLIPSIS
    {'data_container': 'polars'}
    """
    return _GLOBAL_CONFIG_DEFAULT.copy()


def get_config() -> Dict[GlobalConfigParam, Any]:
    """Retrieve current values for configuration set by :meth:`set_config`.

    Will return the default configuration if know updated configuration has
    been set by :meth:`set_config`.

    Returns
    -------
    dict
        The configurable settings (keys) and their default values (values).

    See Also
    --------
    config_context :
        Configuration context manager.
    get_default_config :
        Retrieve ``dachshund``'s default configuration.
    set_config :
        Set global configuration.
    reset_config :
        Reset configuration to ``dachshund`` default.

    Examples
    --------
    >>> from dachshund import get_config
    >>> get_config()  # doctest: +ELLIPSIS
    {'data_container': 'polars'}
    """
    return _get_threadlocal_config().copy()


def set_config(
    *,
    data_container: Optional[DATA_CONTAINERS] = None,
    local_threadsafe: bool = False,
) -> None:
    """Set global configuration.

    Allows the ``dachshund`` global configuration to be updated.

    Parameters
    ----------
    data_container : {"polars", "polars_eager", "pandas", "modin", "dask", \
    "numpy", "xarray"}, default=None
        The container to use internally in `dachshund`.

        - If "polars", the computations will be done using `polars.LazyFrame`.
        - If "polars_eager", then the data is collected into a `polars.DataFrame`.
        - Otherwise, the data is returned as the specified data container.

    local_threadsafe : bool, default=False
        If False, set the backend as default for all threads.

    Returns
    -------
    None
        No output returned.

    See Also
    --------
    config_context :
        Configuration context manager.
    get_default_config :
        Retrieve ``dachshund``'s default configuration.
    get_config :
        Retrieve current global configuration values.
    reset_config :
        Reset configuration to default.

    Examples
    --------
    >>> from dachshund import get_config, set_config
    >>> get_config()  # doctest: +ELLIPSIS
    {'data_container': 'polars'}
    >>> set_config(data_container='pandas')
    >>> get_config()  # doctest: +ELLIPSIS
    {'data_container': 'pandas'}
    """
    local_config = _get_threadlocal_config()
    msg = "Attempting to set an invalid value for a global configuration.\n"
    msg += "Using current configuration value of parameter as a result.\n"

    def _update_local_config(
        local_config: Dict[GlobalConfigParam, Any],
        param: Any,
        param_name: str,
        msg: str,
    ) -> Dict[GlobalConfigParam, Any]:
        """Update a local config with a parameter value.

        Utility function used to update the local config dictionary.

        Parameters
        ----------
        local_config : dict[GlobalConfigParam, Any]
            The local configuration to update.
        param : Any
            The parameter value to update.
        param_name : str
            The name of the parameter to update.
        msg : str
            Message to pass to `GlobalConfigParam.get_valid_param_or_default`.

        Returns
        -------
        dict
            The updated local configuration.
        """
        config_reg = _CONFIG_REGISTRY.copy()
        value = config_reg[param_name].get_valid_param_or_default(
            param,
            default_value=local_config[param_name],
            msg=msg,
        )
        local_config[param_name] = value
        return local_config

    if data_container is not None:
        local_config = _update_local_config(
            local_config, data_container, "data_container", msg
        )

    if not local_threadsafe:
        global_config.update(local_config)

    return None


def reset_config() -> None:
    """Reset the global configuration to the default.

    Will remove any user updates to the global configuration and reset the values
    back to the ``dachshund`` defaults.

    Returns
    -------
    None
        No output returned.

    See Also
    --------
    config_context :
        Configuration context manager.
    get_default_config :
        Retrieve ``dachshund``'s default configuration.
    get_config :
        Retrieve current global configuration values.
    set_config :
        Set global configuration.

    Examples
    --------
    >>> from dachshund import get_config, set_config, reset_config
    >>> get_config()  # doctest: +ELLIPSIS
    {'data_container': 'polars'}
    >>> set_config(data_container='pandas')
    >>> get_config()  # doctest: +ELLIPSIS
    {'data_container': 'pandas'}
    >>> get_config() == get_default_config()
    False
    >>> reset_config()
    >>> get_config()  # doctest: +ELLIPSIS
    {'data_container': 'polars'}
    >>> get_config() == get_default_config()
    True
    """
    default_config = get_default_config()
    set_config(**default_config)
    return None


@contextmanager
def config_context(
    *,
    data_container: Optional[DATA_CONTAINERS] = None,
    local_threadsafe: bool = False,
) -> Iterator[None]:
    """Context manager for ``dachshund`` global configuration.

    Provides the ability to run code using different configuration without
    having to update the global config.

    Parameters
    ----------
    data_container : {"polars", "polars_eager", "pandas", "modin", "dask", \
    "numpy", "xarray"}, default=None
        The container to use internally in `dachshund`.

        - If "polars", the computations will be done using `polars.LazyFrame`.
        - If "polars_eager", then the data is collected into a `polars.DataFrame`.
        - Otherwise, the data is returned as the specified data container.

    local_threadsafe : bool, default=False
        If False, set the config as default for all threads.

    Yields
    ------
    None
        No output returned.

    See Also
    --------
    get_default_config :
        Retrieve ``dachshund``'s default configuration.
    get_config :
        Retrieve current values of the global configuration.
    set_config :
        Set global configuration.
    reset_config :
        Reset configuration to ``dachshund`` default.

    Notes
    -----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited.

    Examples
    --------
    >>> from dachshund import config_context
    >>> with config_context(data_container='pandas'):
    ...     pass
    """
    old_config = get_config()
    set_config(
        data_container=data_container,
        local_threadsafe=local_threadsafe,
    )

    try:
        yield
    finally:
        set_config(**old_config)
