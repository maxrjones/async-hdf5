"""xarray backend engine for async-hdf5.

Allows opening HDF5 files directly with xarray::

    ds = xr.open_dataset("file.h5", engine="async_hdf5")

Cloud URLs are auto-detected — no explicit ``store=`` needed::

    ds = xr.open_dataset("s3://bucket/path/file.h5", engine="async_hdf5")

For custom store configuration, pass an ObjectStore via ``store=``::

    from async_hdf5.store import S3Store

    store = S3Store.from_url("s3://my-bucket", skip_signature=True)
    ds = xr.open_dataset("path/to/file.h5", engine="async_hdf5", store=store)
"""

from __future__ import annotations

import asyncio
import posixpath
import threading
from collections.abc import Coroutine, Iterable
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar
from urllib.parse import urlparse

from xarray.backends import BackendEntrypoint

if TYPE_CHECKING:
    import os

    from obstore.store import ObjectStore
    from xarray.core.dataset import Dataset
    from xarray.core.datatree import DataTree

    from async_hdf5._input import ObspecInput

T = TypeVar("T")

_HDF5_EXTENSIONS = {".h5", ".hdf5", ".he5", ".hdf", ".nc", ".nc4"}

# URL schemes that indicate cloud/remote object stores.
_CLOUD_SCHEMES = {"s3", "s3a", "gs", "az", "adl", "azure", "abfs", "abfss"}


def _has_hdf5_extension(path: str) -> bool:
    """Check if a path or URL has an HDF5 file extension."""
    parsed = urlparse(path)
    # For URLs, check the path component; for local paths urlparse puts
    # everything in ``parsed.path`` when there is no scheme.
    _, ext = posixpath.splitext(parsed.path)
    return ext.lower() in _HDF5_EXTENSIONS


def _resolve_store_and_path(
    filename: str,
    store: Any,
) -> tuple[Any, str]:
    """Resolve an object store and path from a filename and optional store.

    When *store* is provided, it is returned as-is alongside the filename.
    When *store* is ``None``, the filename is inspected:

    - Cloud URLs (``s3://``, ``gs://``, ``az://``, …) and HTTP(S) URLs are
      parsed with ``async_hdf5.store.from_url`` to create the appropriate
      store.  The URL is split into a bucket/host-level store and an
      object-key path.
    - Local paths use ``async_hdf5.store.LocalStore``.
    """
    if store is not None:
        return store, filename

    parsed = urlparse(filename)

    if parsed.scheme in _CLOUD_SCHEMES:
        from async_hdf5.store import from_url

        # Build store from the bucket-level URL; the object key is the path.
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        obj_path = parsed.path.lstrip("/")
        return from_url(base_url), obj_path

    if parsed.scheme in ("http", "https"):
        from async_hdf5.store import from_url

        # Split into directory-level store and filename.
        dir_path, _, obj_name = parsed.path.rpartition("/")
        base_url = f"{parsed.scheme}://{parsed.netloc}{dir_path}"
        return from_url(base_url), obj_name

    # Local filesystem path — use the parent directory as prefix so the
    # store path is just the filename.
    from pathlib import Path

    from async_hdf5.store import LocalStore

    resolved = Path(filename).resolve()
    return LocalStore(prefix=resolved.parent), resolved.name


# Module-level persistent event loop for sync→async bridging.
_sync_loop: asyncio.AbstractEventLoop | None = None
_sync_thread: threading.Thread | None = None
_sync_lock = threading.Lock()


def _get_sync_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for sync operations."""
    global _sync_loop, _sync_thread  # noqa: PLW0603

    if _sync_loop is None:
        with _sync_lock:
            if _sync_loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever,
                    name="async_hdf5_sync",
                    daemon=True,
                )
                thread.start()
                _sync_loop = loop
                _sync_thread = thread
    return _sync_loop


def _run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from a synchronous context."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # An event loop is already running (e.g. Jupyter).
    # Submit to a persistent dedicated loop.
    loop = _get_sync_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


class AsyncHDF5BackendEntrypoint(BackendEntrypoint):
    """xarray backend for opening HDF5 files via async-hdf5.

    Parses HDF5 metadata asynchronously in Rust and exposes the file as a
    read-only Zarr v3 store.  Chunk index parsing is deferred until data is
    actually accessed, so opening is fast even for files with many variables.

    Usage::

        import xarray as xr

        ds = xr.open_dataset("file.h5", engine="async_hdf5")
    """

    description: ClassVar[str] = "Open HDF5 files using async-hdf5"
    supports_groups: ClassVar[bool] = True

    open_dataset_parameters: ClassVar[tuple[str, ...]] = (
        "filename_or_obj",
        "drop_variables",
        "mask_and_scale",
        "decode_times",
        "concat_characters",
        "decode_coords",
        "use_cftime",
        "decode_timedelta",
        # async-hdf5 specific
        "store",
        "group",
        "block_size",
        "pre_warm_size",
    )

    def guess_can_open(
        self,
        filename_or_obj: str | os.PathLike[str] | Any,
    ) -> bool:
        if not isinstance(filename_or_obj, str | os.PathLike):
            return False
        return _has_hdf5_extension(str(filename_or_obj))

    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[str] | Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
        mask_and_scale: bool = True,
        decode_times: bool = True,
        concat_characters: bool = True,
        decode_coords: bool = True,
        use_cftime: bool | None = None,
        decode_timedelta: bool | None = None,
        # async-hdf5 specific
        store: ObjectStore | ObspecInput | None = None,
        group: str | None = None,
        block_size: int = 8 * 1024 * 1024,
        pre_warm_size: int | None = None,
    ) -> Dataset:
        import xarray as xr

        from async_hdf5.zarr import open_hdf5

        resolved_store, path = _resolve_store_and_path(str(filename_or_obj), store)

        hdf5_store = _run_sync(
            open_hdf5(
                path=path,
                store=resolved_store,
                group=group,
                drop_variables=drop_variables,
                block_size=block_size,
                pre_warm_size=pre_warm_size,
            )
        )

        return xr.open_dataset(
            hdf5_store,
            engine="zarr",
            consolidated=False,
            zarr_format=3,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

    def open_datatree(
        self,
        filename_or_obj: str | os.PathLike[str] | Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
        mask_and_scale: bool = True,
        decode_times: bool = True,
        concat_characters: bool = True,
        decode_coords: bool = True,
        use_cftime: bool | None = None,
        decode_timedelta: bool | None = None,
        # async-hdf5 specific
        store: ObjectStore | ObspecInput | None = None,
        group: str | None = None,
        block_size: int = 8 * 1024 * 1024,
        pre_warm_size: int | None = None,
    ) -> DataTree:
        from xarray.backends.common import datatree_from_dict_with_io_cleanup

        groups_dict = self.open_groups_as_dict(
            filename_or_obj,
            drop_variables=drop_variables,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
            store=store,
            group=group,
            block_size=block_size,
            pre_warm_size=pre_warm_size,
        )
        return datatree_from_dict_with_io_cleanup(groups_dict)

    def open_groups_as_dict(
        self,
        filename_or_obj: str | os.PathLike[str] | Any,
        *,
        drop_variables: str | Iterable[str] | None = None,
        mask_and_scale: bool = True,
        decode_times: bool = True,
        concat_characters: bool = True,
        decode_coords: bool = True,
        use_cftime: bool | None = None,
        decode_timedelta: bool | None = None,
        # async-hdf5 specific
        store: ObjectStore | ObspecInput | None = None,
        group: str | None = None,
        block_size: int = 8 * 1024 * 1024,
        pre_warm_size: int | None = None,
    ) -> dict[str, Dataset]:
        import xarray as xr

        resolved_store, path = _resolve_store_and_path(str(filename_or_obj), store)

        # Build HDF5Stores for every group in a single async pass so that
        # phony dimension names are assigned with a file-global counter
        # (mirroring h5netcdf behaviour and avoiding conflicts when
        # DataTree checks alignment across parent/child groups).
        group_stores = _run_sync(
            _open_all_groups(
                path=path,
                store=resolved_store,
                group=group,
                drop_variables=drop_variables,
                block_size=block_size,
                pre_warm_size=pre_warm_size,
            )
        )

        groups_dict: dict[str, Dataset] = {}
        for grp_path, hdf5_store in group_stores.items():
            ds = xr.open_dataset(
                hdf5_store,
                engine="zarr",
                consolidated=False,
                zarr_format=3,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
            groups_dict[grp_path] = ds

        return groups_dict


async def _open_all_groups(
    *,
    path: str,
    store: Any,
    group: str | None,
    drop_variables: Iterable[str] | None,
    block_size: int,
    pre_warm_size: int | None,
) -> dict[str, Any]:
    """Walk the HDF5 group tree and build an HDF5Store per group.

    Phony dimension names are assigned with a single file-global counter
    (like h5netcdf) so that different groups never produce conflicting
    dimension sizes under the same name.
    """
    import warnings

    from async_hdf5 import HDF5Dataset, HDF5File, HDF5Group
    from async_hdf5._utils import assign_phony_dims
    from async_hdf5.zarr import HDF5Store, _DatasetInfo

    drop = set(drop_variables or ())

    f = await HDF5File.open(
        path, store=store, block_size=block_size, pre_warm_size=pre_warm_size
    )
    root = await f.root_group()
    target = (await root.navigate(group)) if group else root

    # Phase 1: walk tree, collect datasets from every group.
    collected: dict[str, tuple[HDF5Group, dict[str, HDF5Dataset]]] = {}

    async def _walk(grp: HDF5Group, prefix: str) -> None:
        datasets: dict[str, HDF5Dataset] = {}
        for name in await grp.dataset_names():
            if name not in drop:
                datasets[name] = await grp.dataset(name)
        collected[prefix] = (grp, datasets)
        for name in await grp.group_names():
            child = await grp.group(name)
            child_path = name if prefix == "/" else f"{prefix}/{name}"
            await _walk(child, child_path)

    await _walk(target, "/")

    # Phase 2: assign phony dims with a single global counter across ALL
    # groups, keyed by "group_path/var_name" for uniqueness.
    all_variables: list[tuple[str, tuple[int, ...]]] = []
    for grp_path, (_grp, datasets) in collected.items():
        for name, ds in datasets.items():
            global_key = f"{grp_path}/{name}" if grp_path != "/" else f"/{name}"
            all_variables.append((global_key, tuple(int(s) for s in ds.shape)))

    global_dims = assign_phony_dims(all_variables)

    # Phase 3: build an HDF5Store per group.
    result: dict[str, HDF5Store] = {}
    for grp_path, (grp, datasets) in collected.items():
        dataset_infos: dict[str, _DatasetInfo] = {}
        skipped: list[tuple[str, str]] = []
        for name, ds in datasets.items():
            global_key = f"{grp_path}/{name}" if grp_path != "/" else f"/{name}"
            try:
                dataset_infos[name] = _DatasetInfo(
                    ds, dimension_names=global_dims[global_key]
                )
            except (ValueError, TypeError) as exc:
                skipped.append((name, str(exc)))

        if skipped:
            details = "; ".join(f"{n}: {e}" for n, e in skipped)
            warnings.warn(
                f"Skipped {len(skipped)} dataset(s) in group {grp_path!r} "
                f"with unsupported types: {details}. "
                f"Use drop_variables to suppress this warning.",
                stacklevel=2,
            )

        group_attrs = await grp.attributes()
        result[grp_path] = HDF5Store(
            dataset_infos=dataset_infos,
            group_attrs=group_attrs,
            file_url=path,
        )

    return result
