/// The location of a single chunk within an HDF5 file.
#[derive(Debug, Clone)]
pub struct ChunkLocation {
    /// Multi-dimensional chunk index (e.g., [3, 5] for the chunk at row=3, col=5
    /// in the chunk grid).
    pub indices: Vec<u64>,
    /// Byte offset of this chunk in the file.
    pub byte_offset: u64,
    /// Size of this chunk as stored on disk (compressed size).
    pub byte_length: u64,
    /// Filter mask: a bitmask indicating which filters in the pipeline were
    /// applied. Bit N set means filter N was *skipped* for this chunk.
    pub filter_mask: u32,
}

/// Index of all chunks in a dataset.
///
/// For chunked datasets, this is populated by traversing the B-tree.
/// For contiguous datasets, this contains a single entry.
#[derive(Debug, Clone)]
pub struct ChunkIndex {
    entries: Vec<ChunkLocation>,
    /// Chunk dimensions in array elements.
    chunk_shape: Vec<u64>,
    /// Total dataset dimensions.
    dataset_shape: Vec<u64>,
}

impl ChunkIndex {
    /// Create a new ChunkIndex.
    pub fn new(
        entries: Vec<ChunkLocation>,
        chunk_shape: Vec<u64>,
        dataset_shape: Vec<u64>,
    ) -> Self {
        Self {
            entries,
            chunk_shape,
            dataset_shape,
        }
    }

    /// Create a single-entry index for contiguous storage.
    pub fn contiguous(byte_offset: u64, byte_length: u64, dataset_shape: Vec<u64>) -> Self {
        let ndims = dataset_shape.len();
        Self {
            entries: vec![ChunkLocation {
                indices: vec![0; ndims],
                byte_offset,
                byte_length,
                filter_mask: 0,
            }],
            chunk_shape: dataset_shape.clone(),
            dataset_shape,
        }
    }

    /// Number of chunks along each dimension.
    pub fn grid_shape(&self) -> Vec<u64> {
        self.dataset_shape
            .iter()
            .zip(self.chunk_shape.iter())
            .map(|(&ds, &cs)| ds.div_ceil(cs))
            .collect()
    }

    /// Chunk dimensions.
    pub fn chunk_shape(&self) -> &[u64] {
        &self.chunk_shape
    }

    /// Dataset dimensions.
    pub fn dataset_shape(&self) -> &[u64] {
        &self.dataset_shape
    }

    /// Look up a chunk by its multi-dimensional indices.
    pub fn get(&self, indices: &[u64]) -> Option<&ChunkLocation> {
        self.entries.iter().find(|e| e.indices == indices)
    }

    /// Iterate all chunk locations.
    pub fn iter(&self) -> impl Iterator<Item = &ChunkLocation> {
        self.entries.iter()
    }

    /// Total number of chunks.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Consume into the underlying entries.
    pub fn into_entries(self) -> Vec<ChunkLocation> {
        self.entries
    }
}
