from ._async_hdf5 import (
    HDF5File,
    HDF5Group,
    HDF5Dataset,
    ChunkIndex,
    ChunkLocation,
    ___version,  # noqa: F401 # pyright:ignore[reportAttributeAccessIssue]
)
from ._input import ObspecInput

__version__: str = ___version()

__all__ = [
    "HDF5File",
    "HDF5Group",
    "HDF5Dataset",
    "ChunkIndex",
    "ChunkLocation",
    "ObspecInput",
]

# Optional VirtualiZarr integration — only available when virtualizarr is installed.
try:
    from .virtualizarr import open_virtual_hdf5

    __all__ += ["open_virtual_hdf5"]
except ImportError:
    pass
