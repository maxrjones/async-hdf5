use std::sync::Arc;

use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};
use crate::reader::AsyncFileReader;

/// Global Heap Collection signature.
const GCOL_SIGNATURE: [u8; 4] = [b'G', b'C', b'O', b'L'];

/// Read a single object from a global heap collection.
///
/// The global heap stores variable-length data referenced by attributes and
/// datasets. Each collection starts with a "GCOL" header and contains numbered
/// objects.
///
/// Layout of a global heap collection:
///   - Signature "GCOL" (4 bytes)
///   - Version (1 byte, must be 1)
///   - Reserved (3 bytes)
///   - Collection Size (size_of_lengths bytes, includes header)
///   - Objects...
///
/// Each object:
///   - Heap Object Index (2 bytes, u16)
///   - Reference Count (2 bytes, u16)
///   - Reserved (4 bytes)
///   - Object Size (size_of_lengths bytes)
///   - Object Data (object size bytes, padded to 8-byte boundary)
///
/// Object index 0 marks free space (end of objects).
pub async fn read_global_heap_object(
    reader: &Arc<dyn AsyncFileReader>,
    collection_address: u64,
    object_index: u32,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Bytes> {
    // Read enough for the header + a reasonable number of objects.
    // We start with 4KB and extend if needed.
    let initial_size = 4096u64;
    let data = reader
        .get_bytes(collection_address..collection_address + initial_size)
        .await?;

    let mut r = HDF5Reader::with_sizes(data.clone(), size_of_offsets, size_of_lengths);

    // Parse header
    r.read_signature(&GCOL_SIGNATURE)?;
    let version = r.read_u8()?;
    if version != 1 {
        return Err(HDF5Error::General(format!(
            "unsupported global heap collection version: {version}"
        )));
    }
    r.skip(3); // reserved
    let collection_size = r.read_length()?;

    // If the collection is larger than our initial fetch, get the rest
    let data = if collection_size > initial_size {
        reader
            .get_bytes(collection_address..collection_address + collection_size)
            .await?
    } else {
        data
    };
    let mut r = HDF5Reader::with_sizes(data.clone(), size_of_offsets, size_of_lengths);
    // Skip past header: 4 (sig) + 1 (ver) + 3 (reserved) + size_of_lengths
    r.skip(4 + 1 + 3 + size_of_lengths as u64);

    // Scan objects until we find the one we want or hit index 0
    let end = collection_size.min(data.len() as u64);
    while r.position() + 8 < end {
        let idx = r.read_u16()?;
        if idx == 0 {
            // Free space — no more objects
            break;
        }
        let _ref_count = r.read_u16()?;
        r.skip(4); // reserved
        let obj_size = r.read_length()?;

        if idx as u32 == object_index {
            // Found it — read the object data
            let pos = r.position() as usize;
            let obj_end = pos + obj_size as usize;
            if obj_end <= data.len() {
                return Ok(data.slice(pos..obj_end));
            } else {
                return Err(HDF5Error::General(format!(
                    "global heap object {object_index} extends beyond collection"
                )));
            }
        }

        // Skip this object's data (padded to 8-byte boundary)
        let padded = (obj_size + 7) & !7;
        r.skip(padded);
    }

    Err(HDF5Error::General(format!(
        "global heap object {object_index} not found in collection at 0x{collection_address:x}"
    )))
}
