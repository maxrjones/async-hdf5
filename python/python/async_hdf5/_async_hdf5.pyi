from ._chunk_index import ChunkIndex, ChunkLocation
from ._dataset import HDF5Dataset
from ._file import HDF5File
from ._group import HDF5Group

def ___version() -> str: ...

__all__ = [
    "ChunkIndex",
    "ChunkLocation",
    "HDF5Dataset",
    "HDF5File",
    "HDF5Group",
]
