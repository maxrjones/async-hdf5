use pyo3::prelude::*;

#[pyclass(name = "ChunkLocation", frozen)]
#[derive(Clone)]
pub(crate) struct PyChunkLocation {
    #[pyo3(get)]
    indices: Vec<u64>,
    #[pyo3(get)]
    byte_offset: u64,
    #[pyo3(get)]
    byte_length: u64,
    #[pyo3(get)]
    filter_mask: u32,
}

#[pymethods]
impl PyChunkLocation {
    fn __repr__(&self) -> String {
        format!(
            "ChunkLocation(indices={:?}, byte_offset={}, byte_length={}, filter_mask={})",
            self.indices, self.byte_offset, self.byte_length, self.filter_mask
        )
    }
}

#[pyclass(name = "ChunkIndex", frozen)]
pub(crate) struct PyChunkIndex {
    locations: Vec<PyChunkLocation>,
    grid_shape: Vec<u64>,
    chunk_shape: Vec<u64>,
    dataset_shape: Vec<u64>,
}

impl PyChunkIndex {
    pub(crate) fn new(index: async_hdf5::ChunkIndex) -> Self {
        let grid_shape = index.grid_shape();
        let chunk_shape = index.chunk_shape().to_vec();
        let dataset_shape = index.dataset_shape().to_vec();

        let locations: Vec<PyChunkLocation> = index
            .into_entries()
            .into_iter()
            .map(|c| PyChunkLocation {
                indices: c.indices,
                byte_offset: c.byte_offset,
                byte_length: c.byte_length,
                filter_mask: c.filter_mask,
            })
            .collect();

        Self {
            locations,
            grid_shape,
            chunk_shape,
            dataset_shape,
        }
    }
}

#[pymethods]
impl PyChunkIndex {
    fn __len__(&self) -> usize {
        self.locations.len()
    }

    fn __iter__(&self) -> PyChunkIndexIterator {
        PyChunkIndexIterator {
            locations: self.locations.clone(),
            index: 0,
        }
    }

    #[getter]
    fn grid_shape(&self) -> Vec<u64> {
        self.grid_shape.clone()
    }

    #[getter]
    fn chunk_shape(&self) -> Vec<u64> {
        self.chunk_shape.clone()
    }

    #[getter]
    fn dataset_shape(&self) -> Vec<u64> {
        self.dataset_shape.clone()
    }

    fn get(&self, indices: Vec<u64>) -> Option<PyChunkLocation> {
        self.locations
            .iter()
            .find(|c| c.indices == indices)
            .cloned()
    }

    fn __repr__(&self) -> String {
        format!(
            "ChunkIndex(len={}, grid_shape={:?}, chunk_shape={:?}, dataset_shape={:?})",
            self.locations.len(),
            self.grid_shape,
            self.chunk_shape,
            self.dataset_shape,
        )
    }
}

#[pyclass]
struct PyChunkIndexIterator {
    locations: Vec<PyChunkLocation>,
    index: usize,
}

#[pymethods]
impl PyChunkIndexIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<PyChunkLocation> {
        if self.index < self.locations.len() {
            let item = self.locations[self.index].clone();
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}
