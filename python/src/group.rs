use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3_async_runtimes::tokio::future_into_py;

use crate::dataset::PyHDF5Dataset;

#[pyclass(name = "HDF5Group", frozen)]
pub(crate) struct PyHDF5Group {
    inner: Arc<async_hdf5::HDF5Group>,
}

impl PyHDF5Group {
    pub(crate) fn new(group: async_hdf5::HDF5Group) -> Self {
        Self {
            inner: Arc::new(group),
        }
    }
}

#[pymethods]
impl PyHDF5Group {
    #[getter]
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn children<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let group = self.inner.clone();
        future_into_py(py, async move {
            let children = group.children().await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?;
            let names: Vec<String> = children.into_iter().map(|c| c.name).collect();
            Ok(names)
        })
    }

    fn group<'py>(&'py self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyAny>> {
        let group = self.inner.clone();
        future_into_py(py, async move {
            let child = group.group(&name).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?;
            Ok(PyHDF5Group::new(child))
        })
    }

    fn dataset<'py>(&'py self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyAny>> {
        let group = self.inner.clone();
        future_into_py(py, async move {
            let ds = group.dataset(&name).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?;
            Ok(PyHDF5Dataset::new(ds))
        })
    }

    fn navigate<'py>(&'py self, py: Python<'py>, path: String) -> PyResult<Bound<'py, PyAny>> {
        let group = self.inner.clone();
        future_into_py(py, async move {
            let child = group.navigate(&path).await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?;
            Ok(PyHDF5Group::new(child))
        })
    }

    fn group_names<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let group = self.inner.clone();
        future_into_py(py, async move {
            let names = group.group_names().await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?;
            Ok(names)
        })
    }

    fn dataset_names<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let group = self.inner.clone();
        future_into_py(py, async move {
            let names = group.dataset_names().await.map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            })?;
            Ok(names)
        })
    }

    fn attributes<'py>(&'py self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let group = self.inner.clone();
        future_into_py(py, async move {
            let attrs = group.attributes().await;
            let dict: HashMap<String, Py<PyAny>> = Python::attach(|py| {
                attrs
                    .into_iter()
                    .map(|a| (a.name, attribute_value_to_py(py, a.value)))
                    .collect()
            });
            Ok(dict)
        })
    }
}

/// Convert an `AttributeValue` to a Python object.
pub(crate) fn attribute_value_to_py(py: Python<'_>, value: async_hdf5::AttributeValue) -> Py<PyAny> {
    use async_hdf5::AttributeValue;

    match value {
        AttributeValue::I8(v) if v.len() == 1 => v[0].into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::I16(v) if v.len() == 1 => v[0].into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::I32(v) if v.len() == 1 => v[0].into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::I64(v) if v.len() == 1 => v[0].into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::U8(v) if v.len() == 1 => v[0].into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::U16(v) if v.len() == 1 => v[0].into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::U32(v) if v.len() == 1 => v[0].into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::U64(v) if v.len() == 1 => v[0].into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::F32(v) if v.len() == 1 => v[0].into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::F64(v) if v.len() == 1 => v[0].into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::I8(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::I16(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::I32(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::I64(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::U8(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::U16(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::U32(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::U64(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::F32(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::F64(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::String(s) => s.into_pyobject(py).unwrap().into_any().unbind(),
        AttributeValue::Raw(v) => v.into_pyobject(py).unwrap().into_any().unbind(),
    }
}
