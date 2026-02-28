"""
VirtualiZarr integration for async-hdf5.

Provides ``open_virtual_hdf5``, an async function that uses async-hdf5 for
HDF5 metadata extraction and returns a VirtualiZarr ``ManifestStore``.  This
lets you open remote HDF5 files with targeted byte-range reads (metadata only)
and then create virtual xarray Datasets or DataTrees via
``ManifestStore.to_virtual_dataset()`` — no libhdf5 or h5netcdf required.

VirtualiZarr is an **optional** dependency.  Importing this module without it
installed will raise ``ImportError``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from async_hdf5 import ChunkIndex, HDF5Dataset, HDF5Group, ObspecInput
    from async_hdf5.store import ObjectStore

try:
    import numpy as np
    from obspec_utils.registry import ObjectStoreRegistry
    from virtualizarr.manifests import (
        ChunkManifest,
        ManifestArray,
        ManifestGroup,
        ManifestStore,
    )
    from virtualizarr.manifests.utils import create_v3_array_metadata
except ImportError as e:
    raise ImportError(
        "virtualizarr and its dependencies are required for async_hdf5.virtualizarr. "
        "Install with: pip install virtualizarr"
    ) from e

from async_hdf5 import HDF5File
from async_hdf5._utils import assign_phony_dims, hdf5_filters_to_zarr_codecs

__all__ = ["open_virtual_hdf5"]


async def open_virtual_hdf5(
    path: str,
    *,
    store: ObjectStore | ObspecInput,
    group: str | None = None,
    url: str | None = None,
    registry: ObjectStoreRegistry | None = None,
    drop_variables: Iterable[str] | None = None,
    block_size: int = 8 * 1024 * 1024,
) -> ManifestStore:
    """Open an HDF5 file as a VirtualiZarr ManifestStore.

    Uses async-hdf5 (a Rust HDF5 binary parser) for metadata extraction and
    returns a ``ManifestStore`` containing chunk manifests with byte offsets
    into the original file.

    Call ``.to_virtual_dataset()`` or ``.to_virtual_datatree()`` on the
    returned store to create an xarray Dataset or DataTree.

    Args:
        path: Path to the HDF5 file within the store (e.g. the filename
            portion of an S3 URL).
        store: An obstore ``ObjectStore`` instance or obspec-compatible
            backend.
        group: HDF5 group to open (e.g.
            ``"science/LSAR/GCOV/grids/frequencyA"``). If ``None``, the
            root group is used.
        url: Full URL of the HDF5 file (e.g.
            ``"s3://bucket/path/file.h5"``). Stored in chunk manifests so
            ManifestStore can resolve the correct store via the registry.
            If ``None``, *path* is used as-is.
        registry: An :class:`ObjectStoreRegistry` mapping URL prefixes to
            store instances. If ``None``, one is created automatically and
            the provided *store* is registered under the scheme/netloc of
            *url*.
        drop_variables: Variable names to exclude from the virtual dataset.
        block_size: Block cache size in bytes. Each unique region of the file
            accessed during metadata parsing triggers a fetch of the aligned
            block containing that region. Default 8 MiB.

    Returns:
        A ManifestStore containing virtual chunk references. Use
        ``.to_virtual_dataset()`` to get an xarray Dataset.
    """
    f = await HDF5File.open(path, store=store, block_size=block_size)
    root = await f.root_group()

    target = (await root.navigate(group)) if group else root

    file_url = url or path
    manifest_group = await _build_manifest_group(file_url, target, drop_variables)

    if registry is None:
        registry = ObjectStoreRegistry()
    _ensure_store_registered(registry, file_url, store)

    return ManifestStore(group=manifest_group, registry=registry)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _build_manifest_group(
    file_url: str,
    group: HDF5Group,
    drop_variables: Iterable[str] | None,
) -> ManifestGroup:
    """Recursively build a ManifestGroup from an async-hdf5 HDF5Group."""
    drop = set(drop_variables or ())

    # First pass: collect datasets and their shapes for dimension assignment.
    datasets: list[tuple[str, HDF5Dataset, ChunkIndex]] = []
    for name in await group.dataset_names():
        if name in drop:
            continue
        ds = await group.dataset(name)
        chunk_idx = await ds.chunk_index()
        datasets.append((name, ds, chunk_idx))

    # Assign phony dimension names grouped by size (like h5netcdf phony_dims="sort").
    dim_names_map = assign_phony_dims(
        [(name, tuple(int(s) for s in ds.shape)) for name, ds, _ in datasets]
    )

    arrays: dict[str, ManifestArray] = {}
    for name, ds, chunk_idx in datasets:
        arrays[name] = _build_manifest_array(
            file_url, ds, chunk_idx, dimension_names=dim_names_map[name]
        )

    groups: dict[str, ManifestGroup] = {}
    for name in await group.group_names():
        if name in drop:
            continue
        child = await group.group(name)
        groups[name] = await _build_manifest_group(file_url, child, drop)

    attrs = await group.attributes()

    return ManifestGroup(arrays=arrays, groups=groups, attributes=attrs)


def _build_manifest_array(
    file_url: str,
    dataset: HDF5Dataset,
    chunk_index: ChunkIndex,
    dimension_names: tuple[str, ...] | None = None,
) -> ManifestArray:
    """Build a ManifestArray from an async-hdf5 dataset and its chunk index."""
    grid_shape = tuple(chunk_index.grid_shape)

    paths = np.full(grid_shape, file_url, dtype=np.dtypes.StringDType())
    offsets = np.empty(grid_shape, dtype=np.uint64)
    lengths = np.empty(grid_shape, dtype=np.uint64)

    for chunk in chunk_index:
        idx = tuple(chunk.indices)
        offsets[idx] = chunk.byte_offset
        lengths[idx] = chunk.byte_length

    manifest = ChunkManifest.from_arrays(paths=paths, offsets=offsets, lengths=lengths)

    codecs = hdf5_filters_to_zarr_codecs(dataset.filters, dataset.element_size)

    shape = tuple(int(s) for s in dataset.shape)
    if dimension_names is None:
        dimension_names = tuple(f"phony_dim_{i}" for i in range(len(shape)))

    metadata = create_v3_array_metadata(
        shape=shape,
        data_type=np.dtype(dataset.numpy_dtype),
        chunk_shape=tuple(int(s) for s in (dataset.chunk_shape or dataset.shape)),
        codecs=codecs,
        dimension_names=dimension_names,
    )

    return ManifestArray(metadata=metadata, chunkmanifest=manifest)


def _ensure_store_registered(
    registry: ObjectStoreRegistry,
    file_url: str,
    store: ObjectStore | ObspecInput,
) -> None:
    """Register *store* in *registry* for the scheme://netloc prefix of *file_url*."""
    parsed = urlparse(file_url)
    if parsed.scheme and parsed.netloc:
        prefix = f"{parsed.scheme}://{parsed.netloc}"
        try:
            registry.resolve(file_url)
        except Exception:
            registry.register(prefix, store)
