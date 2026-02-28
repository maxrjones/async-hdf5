import pathlib

import obstore
from async_hdf5 import HDF5File

FIXTURES = pathlib.Path(__file__).parent.parent.parent / "tests" / "fixtures"


async def test_open_and_root_group():
    store = obstore.store.LocalStore()
    path = str(FIXTURES / "attributes.h5")
    f = await HDF5File.open(path, store=store)
    root = await f.root_group()
    assert root.name == "/"


async def test_root_attributes():
    store = obstore.store.LocalStore()
    path = str(FIXTURES / "attributes.h5")
    f = await HDF5File.open(path, store=store)
    root = await f.root_group()
    attrs = await root.attributes()
    assert attrs["title"] == "Test File"
    assert attrs["version"] == 42
    assert abs(attrs["pi"] - 3.14159265358979) < 1e-10


async def test_group_navigation():
    store = obstore.store.LocalStore()
    path = str(FIXTURES / "attributes.h5")
    f = await HDF5File.open(path, store=store)
    root = await f.root_group()

    group = await root.group("mygroup")
    assert group.name == "mygroup"

    attrs = await group.attributes()
    assert attrs["description"] == "A test group"
    assert attrs["count"] == 100


async def test_dataset_metadata():
    store = obstore.store.LocalStore()
    path = str(FIXTURES / "attributes.h5")
    f = await HDF5File.open(path, store=store)
    root = await f.root_group()
    group = await root.group("mygroup")
    ds = await group.dataset("data")

    assert ds.name == "data"
    assert ds.ndim == 2
    assert ds.shape == [3, 4]


async def test_dataset_attributes():
    store = obstore.store.LocalStore()
    path = str(FIXTURES / "attributes.h5")
    f = await HDF5File.open(path, store=store)
    root = await f.root_group()
    group = await root.group("mygroup")
    ds = await group.dataset("data")

    attrs = await ds.attributes()
    assert attrs["units"] == "meters"
    assert abs(attrs["scale_factor"] - 0.01) < 1e-6


async def test_chunk_index():
    store = obstore.store.LocalStore()
    path = str(FIXTURES / "attributes.h5")
    f = await HDF5File.open(path, store=store)
    root = await f.root_group()
    group = await root.group("mygroup")
    ds = await group.dataset("data")

    index = await ds.chunk_index()
    assert len(index) > 0
    assert index.dataset_shape == [3, 4]

    # Iterate over chunks
    for chunk in index:
        assert chunk.byte_offset > 0
        assert chunk.byte_length > 0


async def test_children_and_names():
    store = obstore.store.LocalStore()
    path = str(FIXTURES / "attributes.h5")
    f = await HDF5File.open(path, store=store)
    root = await f.root_group()

    children = await root.children()
    assert "mygroup" in children

    group_names = await root.group_names()
    assert "mygroup" in group_names


async def test_navigate():
    store = obstore.store.LocalStore()
    path = str(FIXTURES / "attributes.h5")
    f = await HDF5File.open(path, store=store)
    root = await f.root_group()

    # Navigate to nested group
    group = await root.navigate("mygroup")
    assert group.name == "mygroup"


async def test_vlen_string_attrs():
    store = obstore.store.LocalStore()
    path = str(FIXTURES / "attributes_vlen.h5")
    f = await HDF5File.open(path, store=store)
    root = await f.root_group()

    attrs = await root.attributes()
    assert attrs["title"] == "Variable Length String"
    assert attrs["version"] == 7


async def test_numpy_dtype():
    store = obstore.store.LocalStore()
    path = str(FIXTURES / "datasets.h5")
    f = await HDF5File.open(path, store=store)
    root = await f.root_group()

    # int32 dataset
    ds = await root.dataset("chunked_1d")
    assert ds.numpy_dtype == "<i4"

    # float32 dataset
    ds = await root.dataset("chunked_2d")
    assert ds.numpy_dtype == "<f4"

    # float64 dataset
    ds = await root.dataset("contiguous_1d")
    assert ds.numpy_dtype == "<f8"

    # complex64 (compound with r/i float32 fields)
    ds = await root.dataset("compound_2d")
    assert ds.numpy_dtype == "<c8"


async def test_filters():
    store = obstore.store.LocalStore()
    path = str(FIXTURES / "datasets.h5")
    f = await HDF5File.open(path, store=store)
    root = await f.root_group()

    # Dataset with shuffle + deflate filters
    ds = await root.dataset("chunked_2d")
    filters = ds.filters
    assert len(filters) == 2

    # Shuffle filter (id=2)
    assert filters[0]["id"] == 2
    assert isinstance(filters[0]["client_data"], list)

    # Deflate filter (id=1)
    assert filters[1]["id"] == 1
    assert filters[1]["client_data"][0] == 1  # level=1

    # Dataset with no filters
    ds = await root.dataset("chunked_1d")
    assert ds.filters == []


async def test_element_size():
    store = obstore.store.LocalStore()
    path = str(FIXTURES / "datasets.h5")
    f = await HDF5File.open(path, store=store)
    root = await f.root_group()

    ds = await root.dataset("chunked_1d")
    assert ds.element_size == 4  # int32

    ds = await root.dataset("contiguous_1d")
    assert ds.element_size == 8  # float64


async def test_fill_value():
    store = obstore.store.LocalStore()
    path = str(FIXTURES / "datasets.h5")
    f = await HDF5File.open(path, store=store)
    root = await f.root_group()

    ds = await root.dataset("chunked_1d")
    fill = ds.fill_value
    # fill_value is either None or raw bytes
    assert fill is None or isinstance(fill, list)
