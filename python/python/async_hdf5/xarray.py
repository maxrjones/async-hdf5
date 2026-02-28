"""xarray backend engine for async-hdf5.

Allows opening HDF5 files directly with xarray::

    ds = xr.open_dataset("file.h5", engine="async_hdf5")

For cloud storage, pass an ObjectStore via ``store=``::

    from async_hdf5.store import S3Store

    store = S3Store(bucket="my-bucket", region="us-east-1")
    ds = xr.open_dataset("path/to/file.h5", engine="async_hdf5", store=store)
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Coroutine, Iterable
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from xarray.backends import BackendEntrypoint

if TYPE_CHECKING:
    from xarray.core.dataset import Dataset

T = TypeVar("T")

_HDF5_EXTENSIONS = {".h5", ".hdf5", ".he5", ".hdf", ".nc", ".nc4"}

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
        if isinstance(filename_or_obj, str | os.PathLike):
            _, ext = os.path.splitext(str(filename_or_obj))
            return ext in _HDF5_EXTENSIONS
        return False

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
        store: Any | None = None,
        group: str | None = None,
        block_size: int = 8 * 1024 * 1024,
        pre_warm_size: int | None = None,
    ) -> Dataset:
        import xarray as xr

        from async_hdf5.zarr import open_hdf5

        if store is None:
            from async_hdf5.store import LocalStore

            store = LocalStore()

        hdf5_store = _run_sync(
            open_hdf5(
                path=str(filename_or_obj),
                store=store,
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
