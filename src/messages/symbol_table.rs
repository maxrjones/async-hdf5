use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::Result;

/// Symbol table message — used by v1 groups to reference their B-tree and local heap.
///
/// Message type 0x0011.
#[derive(Debug, Clone)]
pub struct SymbolTableMessage {
    /// Address of the B-tree v1 (type 0) for group members.
    pub btree_address: u64,
    /// Address of the local heap for link names.
    pub local_heap_address: u64,
}

impl SymbolTableMessage {
    /// Parse from the raw message bytes.
    pub fn parse(data: &Bytes, size_of_offsets: u8, size_of_lengths: u8) -> Result<Self> {
        let mut r = HDF5Reader::with_sizes(data.clone(), size_of_offsets, size_of_lengths);

        let btree_address = r.read_offset()?;
        let local_heap_address = r.read_offset()?;

        Ok(Self {
            btree_address,
            local_heap_address,
        })
    }
}
