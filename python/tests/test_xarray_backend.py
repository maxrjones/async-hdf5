"""Tests for the async_hdf5 xarray backend engine."""

import numpy as np
import pytest
import xarray as xr

from .conftest import generated_examples, resolve_folder


@pytest.fixture()
def cf_filepath():
    """Path to the CF-style generated fixture."""
    from .conftest import _generate_fixtures

    _generate_fixtures()
    return str(resolve_folder("tests/data/generated") / "cf_style.h5")


@pytest.fixture()
def multi_dtype_filepath():
    """Path to the multi-dtype generated fixture."""
    from .conftest import _generate_fixtures

    _generate_fixtures()
    return str(resolve_folder("tests/data/generated") / "multi_dtype.h5")


@pytest.fixture()
def nested_groups_filepath():
    """Path to the nested-groups generated fixture."""
    from .conftest import _generate_fixtures

    _generate_fixtures()
    return str(resolve_folder("tests/data/generated") / "nested_groups.h5")


def test_open_dataset(cf_filepath):
    """Basic open_dataset with engine='async_hdf5'."""
    ds = xr.open_dataset(cf_filepath, engine="async_hdf5")
    assert isinstance(ds, xr.Dataset)
    assert "temperature" in ds.data_vars


def test_data_values(cf_filepath):
    """Values loaded via the backend match h5py ground truth."""
    import h5py

    ds = xr.open_dataset(cf_filepath, engine="async_hdf5")

    with h5py.File(cf_filepath, "r") as f:
        expected = f["temperature"][()]
        actual = ds["temperature"].values
        np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_drop_variables(cf_filepath):
    """drop_variables excludes variables from the dataset."""
    ds = xr.open_dataset(
        cf_filepath, engine="async_hdf5", drop_variables=["temperature"]
    )
    assert "temperature" not in ds.data_vars


def test_group(nested_groups_filepath):
    """Opening a specific HDF5 group."""
    ds = xr.open_dataset(nested_groups_filepath, engine="async_hdf5", group="level1")
    assert "data_a" in ds.data_vars


def test_multiple_dtypes(multi_dtype_filepath):
    """Datasets with different dtypes are loaded correctly."""
    import h5py

    ds = xr.open_dataset(multi_dtype_filepath, engine="async_hdf5")

    with h5py.File(multi_dtype_filepath, "r") as f:
        for name in f:
            assert name in ds, f"Missing variable: {name}"
            np.testing.assert_array_equal(ds[name].values, f[name][()])


def test_decode_times_false(cf_filepath):
    """decode_times=False is forwarded to the zarr engine."""
    ds = xr.open_dataset(cf_filepath, engine="async_hdf5", decode_times=False)
    # With decode_times=False, time values should be raw numbers
    assert ds["time"].dtype.kind == "f"


def test_with_explicit_store(cf_filepath):
    """Passing an explicit obstore backend works."""
    from obstore.store import LocalStore

    store = LocalStore()
    ds = xr.open_dataset(cf_filepath, engine="async_hdf5", store=store)
    assert "temperature" in ds.data_vars


# nested_groups.h5 has no datasets at root — data is in subgroups
_no_root_vars = {"nested_groups.h5"}


@pytest.mark.parametrize("filename", generated_examples())
def test_backend_generated_files(filename):
    """All generated fixtures open successfully via the backend."""
    filepath = str(resolve_folder("tests/data/generated") / filename)
    ds = xr.open_dataset(filepath, engine="async_hdf5")
    if filename not in _no_root_vars:
        assert len(ds.data_vars) > 0
