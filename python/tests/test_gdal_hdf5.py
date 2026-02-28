"""Tests against GDAL autotest HDF5/NetCDF4 files.

GDAL's autotest suite includes HDF5 and NetCDF4 files covering various
driver-specific edge cases, geospatial metadata, and data type combinations.

Result summary (GDAL v3.12.2): 73 passed, 13 xfailed.
"""

import pytest

from .conftest import gdal_examples, h5py_comparison, resolve_folder

# ---------------------------------------------------------------------------
# Failure categorization
# ---------------------------------------------------------------------------

# Unsupported datatype class (compound = class 2, etc.)
unsupported_datatype: list[str] = [
    "gdrivers/data/hdf5/complex.h5",
    "gdrivers/data/netcdf/alldatatypes.nc",
]

# Dtype mapping gaps — String mapped to void (|V) instead of (|S)
dtype_string_as_void: list[str] = [
    "gdrivers/data/hdf5/deflate.h5",
    "gdrivers/data/netcdf/nc_mixed_raster_vector.nc",
    "gdrivers/data/netcdf/test_ogr_nc4.nc",
]

# Compound dtype mapped to void instead of complex
dtype_complex_as_void: list[str] = [
    "gdrivers/data/netcdf/complex.nc",
]

# Dtype or value comparison mismatches
dtype_mismatch: list[str] = [
    "gdrivers/data/hdf5/single_char_varname.h5",
    "gdrivers/data/netcdf/bug5291.nc",
    "gdrivers/data/netcdf/int64dim.nc",
]

# Truncated or split files — I/O error: failed to fill whole buffer
truncated_or_split: list[str] = [
    "gdrivers/data/hdf5/test_family_0.h5",
    "gdrivers/data/hdf5/u8be.h5",
    "gdrivers/data/netcdf/byte_truncated.nc",
    "gdrivers/data/netcdf/trmm-nc4.nc",
]

xfail_files = (
    unsupported_datatype
    + dtype_string_as_void
    + dtype_complex_as_void
    + dtype_mismatch
    + truncated_or_split
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel_path", gdal_examples())
async def test_gdal_hdf5_file(rel_path):
    if rel_path in xfail_files:
        pytest.xfail("Known failure")
    filepath = str(resolve_folder("tests/data/gdal") / rel_path)
    await h5py_comparison(filepath)
