use std::sync::Arc;

use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};
use crate::reader::AsyncFileReader;

/// Parsed local heap header.
///
/// Local heaps store small strings (link names) for v1 groups.
/// Symbol table entries reference strings by byte offset into the heap's data segment.
///
/// Binary layout (signature "HEAP"):
///   - Signature (4 bytes): "HEAP"
///   - Version (1 byte): 0
///   - Reserved (3 bytes)
///   - Data Segment Size (L bytes)
///   - Offset to Head of Free-list (L bytes)
///   - Address of Data Segment (O bytes)
#[derive(Debug, Clone)]
pub struct LocalHeap {
    /// The heap's data segment, fetched from the file.
    data_segment: Bytes,
}

impl LocalHeap {
    /// Parse the local heap header and fetch the data segment.
    pub async fn read(
        reader: &Arc<dyn AsyncFileReader>,
        address: u64,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        // Fetch enough for the header: 4 + 1 + 3 + 2*L + O
        let header_size = 8 + 2 * size_of_lengths as u64 + size_of_offsets as u64;
        let header_data = reader.get_bytes(address..address + header_size).await?;

        let mut r = HDF5Reader::with_sizes(header_data, size_of_offsets, size_of_lengths);

        // Verify signature
        let sig = r.read_bytes(4)?;
        if &sig != b"HEAP" {
            return Err(HDF5Error::InvalidHeapSignature {
                expected: "HEAP".into(),
                got: String::from_utf8_lossy(&sig).into(),
            });
        }

        let version = r.read_u8()?;
        if version != 0 {
            return Err(HDF5Error::UnsupportedHeapVersion(version));
        }
        r.skip(3); // reserved

        let data_segment_size = r.read_length()?;
        let _free_list_offset = r.read_length()?;
        let data_segment_address = r.read_offset()?;

        // Fetch the data segment
        let data_segment = reader
            .get_bytes(data_segment_address..data_segment_address + data_segment_size)
            .await?;

        Ok(Self { data_segment })
    }

    /// Read a null-terminated string at the given offset in the data segment.
    pub fn get_string(&self, offset: u64) -> Result<String> {
        let start = offset as usize;
        if start >= self.data_segment.len() {
            return Err(HDF5Error::UnexpectedEof {
                needed: start + 1,
                available: self.data_segment.len(),
            });
        }

        let bytes = &self.data_segment[start..];
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());

        Ok(String::from_utf8_lossy(&bytes[..end]).into())
    }
}
