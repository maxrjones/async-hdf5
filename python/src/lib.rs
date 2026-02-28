mod chunk_index;
mod dataset;
mod error;
mod group;
mod h5file;
mod reader;

use pyo3::prelude::*;

use crate::chunk_index::{PyChunkIndex, PyChunkLocation};
use crate::dataset::PyHDF5Dataset;
use crate::group::PyHDF5Group;
use crate::h5file::PyHDF5File;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[pyfunction]
fn ___version() -> &'static str {
    VERSION
}

/// Raise RuntimeWarning for debug builds
#[pyfunction]
fn check_debug_build(_py: Python) -> PyResult<()> {
    #[cfg(debug_assertions)]
    {
        use pyo3::exceptions::PyRuntimeWarning;
        use pyo3::intern;
        use pyo3::types::PyTuple;

        let warnings_mod = _py.import(intern!(_py, "warnings"))?;
        let warning = PyRuntimeWarning::new_err(
            "async-hdf5 has not been compiled in release mode. Performance will be degraded.",
        );
        let args = PyTuple::new(_py, vec![warning])?;
        warnings_mod.call_method1(intern!(_py, "warn"), args)?;
    }

    Ok(())
}

#[pymodule]
fn _async_hdf5(py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    check_debug_build(py)?;

    m.add_wrapped(wrap_pyfunction!(___version))?;
    m.add_class::<PyHDF5File>()?;
    m.add_class::<PyHDF5Group>()?;
    m.add_class::<PyHDF5Dataset>()?;
    m.add_class::<PyChunkIndex>()?;
    m.add_class::<PyChunkLocation>()?;

    pyo3_object_store::register_store_module(py, m, "async_hdf5", "store")?;
    pyo3_object_store::register_exceptions_module(py, m, "async_hdf5", "exceptions")?;

    Ok(())
}
