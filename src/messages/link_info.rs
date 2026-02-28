use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::Result;

/// Link info message — points to fractal heap + B-tree v2 for dense link storage.
///
/// Message type 0x0002. Present in v2 groups that have enough links to overflow
/// inline link messages.
#[derive(Debug, Clone)]
pub struct LinkInfoMessage {
    /// Address of fractal heap for link names.
    pub fractal_heap_address: u64,
    /// Address of B-tree v2 for name-indexed link lookup.
    pub name_btree_address: u64,
    /// Address of B-tree v2 for creation-order-indexed lookup (if tracked).
    pub creation_order_btree_address: Option<u64>,
    /// Maximum creation order index value.
    pub max_creation_index: Option<u64>,
}

impl LinkInfoMessage {
    /// Parse from the raw message bytes.
    pub fn parse(data: &Bytes, size_of_offsets: u8, size_of_lengths: u8) -> Result<Self> {
        let mut r = HDF5Reader::with_sizes(data.clone(), size_of_offsets, size_of_lengths);

        let _version = r.read_u8()?; // version 0
        let flags = r.read_u8()?;

        let max_creation_index = if flags & 0x01 != 0 {
            Some(r.read_u64()?)
        } else {
            None
        };

        let fractal_heap_address = r.read_offset()?;
        let name_btree_address = r.read_offset()?;

        let creation_order_btree_address = if flags & 0x01 != 0 {
            Some(r.read_offset()?)
        } else {
            None
        };

        Ok(Self {
            fractal_heap_address,
            name_btree_address,
            creation_order_btree_address,
            max_creation_index,
        })
    }
}
