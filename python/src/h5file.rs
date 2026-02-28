use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyType;
use pyo3_async_runtimes::tokio::future_into_py;

use crate::error::PyAsyncHDF5Result;
use crate::group::PyHDF5Group;
use crate::reader::StoreInput;

#[pyclass(name = "HDF5File", frozen)]
pub(crate) struct PyHDF5File {
    inner: Arc<async_hdf5::HDF5File>,
}

async fn open_file(
    reader: Arc<dyn async_hdf5::AsyncFileReader>,
    prefetch: u64,
    multiplier: f64,
) -> PyAsyncHDF5Result<PyHDF5File> {
    let file = async_hdf5::HDF5File::open_with_options(reader, prefetch, multiplier).await?;
    Ok(PyHDF5File {
        inner: Arc::new(file),
    })
}

#[pymethods]
impl PyHDF5File {
    #[classmethod]
    #[pyo3(signature = (path, *, store, prefetch=65536, multiplier=2.0))]
    fn open<'py>(
        _cls: &'py Bound<PyType>,
        py: Python<'py>,
        path: String,
        store: StoreInput,
        prefetch: u64,
        multiplier: f64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let reader = store.into_async_file_reader(path);
        future_into_py(py, async move {
            Ok(open_file(reader, prefetch, multiplier).await?)
        })
    }

    fn root_group<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let file = self.inner.clone();
        future_into_py(py, async move {
            let group = file.root_group().await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?;
            Ok(PyHDF5Group::new(group))
        })
    }
}
