#!/usr/bin/env python3 -u
# copyright: dachshund, BSD-3-Clause License (see LICENSE file)
# This code reuses some tests in scikit-learn tests/test_config.py
# The code is copyrighted by the respective scikit-learn developers (BSD-3-Clause
# License): https://github.com/scikit-learn/scikit-learn/blob/main/COPYING
"""Test configuration functionality."""
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from joblib import Parallel, delayed

import dachshund as dh
from dachshund._config import _CONFIG_REGISTRY, _GLOBAL_CONFIG_DEFAULT
from dachshund._config_param_setting import GlobalConfigParamSetting
from dachshund._consts import data_container_values

DATA_CONTAINER_VALUES = _CONFIG_REGISTRY["data_container"].get_allowed_values()


@pytest.fixture
def global_config_default():
    """Config registry fixture."""
    return _GLOBAL_CONFIG_DEFAULT


@pytest.mark.parametrize("allowed_values", (None, (), "something", range(1, 8)))
def test_global_config_param_get_allowed_values(allowed_values):
    """Test GlobalConfigParamSetting behavior works as expected."""
    some_config_param = GlobalConfigParamSetting(
        name="some_param",
        expected_type=str,
        default_value="text",
        allowed_values=allowed_values,
    )
    # Verify we always coerce output of get_allowed_values to tuple
    values = some_config_param.get_allowed_values()
    assert isinstance(values, tuple)


@pytest.mark.parametrize("expected_type", (None, Any, (), int, (int, str)))
def test_global_config_param_get_expected_type(expected_type):
    """Test GlobalConfigParamSetting behavior works as expected."""
    some_config_param = GlobalConfigParamSetting(
        name="some_param",
        expected_type=str,
        default_value="text",
        allowed_values=expected_type,
    )
    # Verify we always coerce output of get_allowed_values to tuple
    values = some_config_param.get_expected_type()
    assert isinstance(values, tuple)


@pytest.mark.parametrize("value", (None, (), "wrong_string", "text", range(1, 8)))
def test_global_config_param_is_valid_param_value(value):
    """Test GlobalConfigParamSetting behavior works as expected."""
    some_config_param = GlobalConfigParamSetting(
        name="some_param",
        expected_type=str,
        default_value="text",
        allowed_values=("text", "diagram"),
    )
    # Verify we correctly identify invalid parameters
    if value in ("text", "diagram"):
        expected_valid = True
    else:
        expected_valid = False
    assert some_config_param.is_valid_param_value(value) == expected_valid


def test_global_config_get_valid_or_default_warns() -> None:
    """Test GlobalConfigParamSetting.get_valid_param_or_default warns on invalid."""
    some_config_param = GlobalConfigParamSetting(
        name="some_param",
        expected_type=str,
        default_value="text",
        allowed_values=("text", "diagram"),
    )

    # When calling the method with invalid value it should raise user warning
    with pytest.warns(UserWarning, match=r"When setting global.*"):
        returned_value = some_config_param.get_valid_param_or_default(7)

    # And it should return the default value instead of the passed value
    assert returned_value == some_config_param.default_value

    # When calling the method with invalid value it should raise user warning
    # And the warning should start with `msg` if it is passed
    with pytest.warns(UserWarning, match=r"some message.*"):
        returned_value = some_config_param.get_valid_param_or_default(
            7, msg="some message"
        )
    # And it should return the default value instead of the passed value
    assert returned_value == some_config_param.default_value


def test_get_default_config_always_returns_default(global_config_default):
    """Test get_default_config always returns the default config."""
    assert dh.get_default_config() == global_config_default
    # set config to non-default value to make sure it didn't change
    # what is returned by get_default_config
    dh.set_config(data_container="pandas")
    assert dh.get_default_config() == global_config_default
    dh.reset_config()


@pytest.mark.parametrize("data_container", DATA_CONTAINER_VALUES)
def test_set_config_then_get_config_returns_expected_value(data_container):
    """Verify that get_config returns set config values if set_config run."""
    dh.set_config(data_container=data_container)
    retrieved_default = dh.get_config()
    expected_config = {"data_container": data_container}
    msg = "`get_config` used after `set_config` does not return expected values.\n"
    msg += "After set_config is run, get_config should return the set values.\n "
    msg += f"Expected {expected_config}, but returned {retrieved_default}."
    assert retrieved_default == expected_config, msg
    dh.reset_config()


def test_set_config_with_none_value():
    """Verify that setting config param with None value does not update the config."""
    default_config = dh.get_default_config()
    assert dh.get_config()["data_container"] == default_config["data_container"]
    dh.set_config(data_container=None)
    assert dh.get_config()["data_container"] == default_config["data_container"]

    dh.set_config(data_container="pandas")
    assert dh.get_config()["data_container"] == "pandas"

    dh.set_config(data_container=None)
    assert dh.get_config()["data_container"] == "pandas"
    dh.set_config(data_container=default_config["data_container"])
    assert dh.get_config()["data_container"] == default_config["data_container"]

    dh.reset_config()


def test_config_context_exception():
    """Test config_context followed by exception."""
    dh.reset_config()
    default_config = dh.get_default_config()
    assert dh.get_config()["data_container"] == default_config["data_container"]
    try:
        with dh.config_context(data_container="pandas"):
            assert dh.get_config()["data_container"] == "pandas"
            raise ValueError()
    except ValueError:
        pass
    assert dh.get_config()["data_container"] == default_config["data_container"]

    dh.reset_config()


@pytest.mark.parametrize("data_container", DATA_CONTAINER_VALUES)
def test_reset_config_resets_the_config(data_container, global_config_default):
    """Verify that get_config returns default config if reset_config run."""
    dh.set_config(data_container=data_container)
    dh.reset_config()
    retrieved_config = dh.get_config()

    msg = "`get_config` does not return expected values after `reset_config`.\n"
    msg += "`After reset_config is run, get_config` should return defaults.\n"
    msg += f"Expected {global_config_default}, but returned {retrieved_config}."
    assert retrieved_config == global_config_default, msg
    dh.reset_config()


@pytest.mark.parametrize("data_container", DATA_CONTAINER_VALUES)
def test_config_context(data_container):
    """Verify that config_context affects context but not overall configuration."""
    # Make sure config is reset to default values then retrieve it
    dh.reset_config()
    retrieved_config = dh.get_config()

    # Verify that nothing happens if not used as context manager
    dh.config_context(data_container=data_container)
    assert dh.get_config() == retrieved_config

    # Now lets make sure the config_context is changing the context of those values
    # within the scope of the context manager as expected
    with dh.config_context(data_container=data_container):
        retrieved_context_config = dh.get_config()
    expected_config = {"data_container": data_container}
    msg = "`get_config` does not return expected values within `config_context`.\n"
    msg += "`get_config` should return config defined by `config_context`.\n"
    msg += f"Expected {expected_config}, but returned {retrieved_context_config}."
    assert retrieved_context_config == expected_config, msg

    # Outside of the config_context we should have not affected the retrieved config
    # set by call to reset_config()
    config_post_config_context = dh.get_config()
    msg = "`get_config` does not return expected values after `config_context`a.\n"
    msg += "`config_context` should not affect configuration outside its context.\n"
    msg += f"Expected {config_post_config_context}, but returned {retrieved_config}."
    assert retrieved_config == config_post_config_context, msg

    # positional arguments not allowed
    with pytest.raises(TypeError):
        dh.config_context(True)

    # Unknown arguments raise exception
    with pytest.raises(TypeError):
        dh.config_context(do_something_else=True).__enter__()


def test_nested_config_context():
    """Verify that nested config_context works as expected."""
    default_config = dh.get_default_config()
    # Verify nested attempt to set None value does not work
    with dh.config_context(data_container="pandas"):
        with dh.config_context(data_container="modin"):
            assert dh.get_config()["data_container"] == "modin"

        assert dh.get_config()["data_container"] == "pandas"

        with dh.config_context(data_container="dask"):
            assert dh.get_config()["data_container"] == "dask"

            with dh.config_context(data_container="polars_eager"):
                assert dh.get_config()["data_container"] == "polars_eager"

                # global setting will not be retained outside of context that
                # did not modify this setting
                dh.set_config(data_container="numpy")
                assert dh.get_config()["data_container"] == "numpy"

            assert dh.get_config()["data_container"] == "dask"

        assert dh.get_config()["data_container"] == "pandas"
    assert dh.get_config()["data_container"] == default_config["data_container"]
    dh.reset_config()


def test_set_config_behavior_invalid_value():
    """Test set_config uses default and raises warning when setting invalid value."""
    match = r"Attempting to set an invalid value.*"
    dh.reset_config()
    original_config = dh.get_config()
    with pytest.warns(UserWarning, match=match):
        dh.set_config(data_container="False")
    assert dh.get_config() == original_config

    original_config = dh.get_config()
    with pytest.warns(UserWarning, match=match):
        dh.set_config(data_container=7)
    assert dh.get_config() == original_config

    original_config = dh.get_config()
    with pytest.warns(UserWarning, match=match):
        dh.set_config(data_container=True)
    assert dh.get_config() == original_config

    original_config = dh.get_config()
    with pytest.warns(UserWarning, match=match):
        dh.set_config(data_container=("polars", "pandas"))
    assert dh.get_config() == original_config
    dh.reset_config()


def test_set_config_invalid_keyword_argument():
    """Test set_config behavior when invalid keyword argument passed."""
    with pytest.raises(TypeError):
        dh.set_config(do_something_else=True)

    with pytest.raises(TypeError):
        dh.set_config(True)


def _set_data_container_only(data_container, sleep_duration):
    """Return the value of print_changed_only after waiting `sleep_duration`."""
    with dh.config_context(data_container=data_container):
        time.sleep(sleep_duration)
        return dh.get_config()["data_container"]


def test_config_threadsafe():
    """Test config is actually threadsafe.

    Uses threads directly to test that the global config does not change
    between threads. Same test as `test_config_threadsafe_joblib` but with
    `ThreadPoolExecutor`.
    """

    data_container_vals = data_container_values
    sleep_durations = [0.1] * len(data_container_vals)

    with ThreadPoolExecutor(max_workers=2) as e:
        items = list(
            e.map(_set_data_container_only, data_container_vals, sleep_durations)
        )

    assert items == list(data_container_values)
    dh.reset_config()


@pytest.mark.parametrize("backend", ["loky", "multiprocessing", "threading"])
def test_config_threadsafe_joblib(backend):
    """Test that the global config is threadsafe with all joblib backends.

    Two jobs are spawned and sets assume_finite to two different values.
    When the job with a duration 0.1s completes, the assume_finite value
    should be the same as the value passed to the function. In other words,
    it is not influenced by the other job setting assume_finite to True.
    """
    data_container_vals = data_container_values
    sleep_durations = [0.1] * len(data_container_vals)

    items = Parallel(backend=backend, n_jobs=2)(
        delayed(_set_data_container_only)(data_container, sleep_dur)
        for data_container, sleep_dur in zip(data_container_vals, sleep_durations)
    )

    assert items == list(data_container_values)
    dh.reset_config()
