use thiserror::Error;

/// Result type for async-hdf5 operations.
pub type Result<T> = std::result::Result<T, HDF5Error>;

/// Errors that can occur when reading HDF5 files.
#[derive(Debug, Error)]
pub enum HDF5Error {
    #[error("Not an HDF5 file: {hint}")]
    InvalidSignature {
        /// Byte offset that was checked.
        offset: u64,
        /// Human-readable hint about what was found instead.
        hint: String,
    },

    #[error("Unsupported superblock version: {0}")]
    UnsupportedSuperblockVersion(u8),

    #[error("Unsupported object header version: {0}")]
    UnsupportedObjectHeaderVersion(u8),

    #[error("Unsupported data layout version: {0}")]
    UnsupportedDataLayoutVersion(u8),

    #[error("Unsupported datatype class: {0}")]
    UnsupportedDatatypeClass(u8),

    #[error("Unsupported filter pipeline version: {0}")]
    UnsupportedFilterPipelineVersion(u8),

    #[error("Unsupported chunk indexing type: {0}")]
    UnsupportedChunkIndexingType(u8),

    #[error("Unsupported B-tree version: {0}")]
    UnsupportedBTreeVersion(u8),

    #[error("Invalid B-tree signature: expected {expected}, got {got}")]
    InvalidBTreeSignature { expected: String, got: String },

    #[error("Unsupported heap version: {0}")]
    UnsupportedHeapVersion(u8),

    #[error("Invalid heap signature: expected {expected}, got {got}")]
    InvalidHeapSignature { expected: String, got: String },

    #[error("Group member not found: {0}")]
    NotFound(String),

    #[error("Expected group at path: {0}")]
    NotAGroup(String),

    #[error("Expected dataset at path: {0}")]
    NotADataset(String),

    #[error("Unexpected end of data: needed {needed} bytes, had {available}")]
    UnexpectedEof { needed: usize, available: usize },

    #[error("Undefined address encountered (unallocated storage)")]
    UndefinedAddress,

    #[error("Invalid object header signature: expected OHDR")]
    InvalidObjectHeaderSignature,

    #[error("Unsupported link type: {0}")]
    UnsupportedLinkType(u8),

    #[error("Unsupported message type: {0:#06x}")]
    UnsupportedMessageType(u16),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("{0}")]
    General(String),

    #[cfg(feature = "object_store")]
    #[error("Object store error: {0}")]
    ObjectStore(#[from] object_store::Error),

    #[cfg(feature = "reqwest")]
    #[error("HTTP error: {0}")]
    Reqwest(#[from] reqwest::Error),
}
