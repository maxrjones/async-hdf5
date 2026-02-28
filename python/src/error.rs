use async_hdf5::HDF5Error;
use pyo3::create_exception;
use pyo3::exceptions::PyFileNotFoundError;
use pyo3::prelude::*;

create_exception!(
    async_hdf5,
    AsyncHDF5Exception,
    pyo3::exceptions::PyException,
    "A general error from the underlying Rust async-hdf5 library."
);

#[allow(missing_docs)]
pub enum PyAsyncHDF5Error {
    HDF5Error(HDF5Error),
    PyErr(PyErr),
}

impl From<HDF5Error> for PyAsyncHDF5Error {
    fn from(value: HDF5Error) -> Self {
        Self::HDF5Error(value)
    }
}

impl From<PyErr> for PyAsyncHDF5Error {
    fn from(value: PyErr) -> Self {
        Self::PyErr(value)
    }
}

impl From<PyAsyncHDF5Error> for PyErr {
    fn from(error: PyAsyncHDF5Error) -> Self {
        match error {
            PyAsyncHDF5Error::HDF5Error(err) => match &err {
                HDF5Error::NotFound(_) => PyFileNotFoundError::new_err(err.to_string()),
                _ => AsyncHDF5Exception::new_err(err.to_string()),
            },
            PyAsyncHDF5Error::PyErr(err) => err,
        }
    }
}

pub type PyAsyncHDF5Result<T> = std::result::Result<T, PyAsyncHDF5Error>;
