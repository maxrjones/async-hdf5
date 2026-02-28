use std::sync::Arc;

use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::Result;
use crate::group::HDF5Group;
use crate::object_header::ObjectHeader;
use crate::reader::{AsyncFileReader, ReadaheadCache};
use crate::superblock::Superblock;

/// An opened HDF5 file.
///
/// This is the main entry point for reading HDF5 metadata. It holds a
/// reference to the async reader and the parsed superblock.
///
/// # Example
///
/// ```ignore
/// use async_hdf5::HDF5File;
/// use async_hdf5::reader::ObjectReader;
///
/// let reader = ObjectReader::new(store, path);
/// let file = HDF5File::open(reader).await?;
/// let root = file.root_group_header().await?;
/// ```
#[derive(Debug)]
pub struct HDF5File {
    reader: Arc<dyn AsyncFileReader>,
    superblock: Superblock,
}

impl HDF5File {
    /// Open an HDF5 file by parsing its superblock.
    ///
    /// Wraps the given reader in a `ReadaheadCache` to avoid many small
    /// network requests during metadata parsing.
    pub async fn open(reader: impl AsyncFileReader) -> Result<Self> {
        Self::open_with_options(reader, 64 * 1024, 2.0).await
    }

    /// Open with configurable prefetch parameters.
    pub async fn open_with_options(
        reader: impl AsyncFileReader,
        initial_prefetch: u64,
        multiplier: f64,
    ) -> Result<Self> {
        let cached = ReadaheadCache::new(reader)
            .with_initial_size(initial_prefetch)
            .with_multiplier(multiplier);

        let initial_bytes = cached.get_bytes(0..initial_prefetch).await?;
        let (superblock, _offset) = Superblock::parse(&initial_bytes)?;

        Ok(Self {
            reader: Arc::new(cached),
            superblock,
        })
    }

    /// Open with an already-configured reader (e.g., a pre-built ReadaheadCache).
    pub async fn open_raw(reader: Arc<dyn AsyncFileReader>) -> Result<Self> {
        let initial_bytes = reader.get_bytes(0..64 * 1024).await?;
        let (superblock, _offset) = Superblock::parse(&initial_bytes)?;

        Ok(Self {
            reader,
            superblock,
        })
    }

    /// Access the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Access the underlying async reader.
    pub fn reader(&self) -> &Arc<dyn AsyncFileReader> {
        &self.reader
    }

    /// Read and parse the object header at the given file address.
    pub async fn read_object_header(&self, address: u64) -> Result<ObjectHeader> {
        read_object_header(
            &self.reader,
            address,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await
    }

    /// Read the root group's object header.
    pub async fn root_group_header(&self) -> Result<ObjectHeader> {
        self.read_object_header(self.superblock.root_group_address)
            .await
    }

    /// Get the root group as an `HDF5Group` for navigation.
    pub async fn root_group(&self) -> Result<HDF5Group> {
        let header = self.root_group_header().await?;
        Ok(HDF5Group::new(
            "/".to_string(),
            header,
            Arc::clone(&self.reader),
            Arc::new(self.superblock.clone()),
        ))
    }
}

/// Read and parse an object header, following any continuation messages.
pub(crate) async fn read_object_header(
    reader: &Arc<dyn AsyncFileReader>,
    address: u64,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<ObjectHeader> {
    // Initial fetch — 4 KB is usually enough for one object header.
    // The ReadaheadCache will grow if needed.
    let initial_size = 4096u64;
    let data = reader.get_bytes(address..address + initial_size).await?;

    let mut header = ObjectHeader::parse(&data, size_of_offsets, size_of_lengths)?;

    // Follow continuation messages iteratively — continuation chunks can
    // themselves contain further continuation messages (nested chains).
    let mut pending = header.continuation_addresses(size_of_offsets, size_of_lengths)?;
    while let Some((cont_addr, cont_len)) = pending.pop() {
        let cont_data = reader.get_bytes(cont_addr..cont_addr + cont_len).await?;
        let new_messages = parse_continuation_chunk(
            &cont_data,
            size_of_offsets,
            size_of_lengths,
            header.version,
        )?;

        // Check new messages for further continuations before adding them
        for msg in &new_messages {
            if msg.msg_type == crate::object_header::msg_types::HEADER_CONTINUATION {
                let mut r = HDF5Reader::with_sizes(
                    msg.data.clone(),
                    size_of_offsets,
                    size_of_lengths,
                );
                let address = r.read_offset()?;
                let length = r.read_length()?;
                pending.push((address, length));
            }
        }

        header.messages.extend(new_messages);
    }

    Ok(header)
}

/// Parse messages from a continuation chunk (OCHK in v2, raw messages in v1).
fn parse_continuation_chunk(
    data: &Bytes,
    size_of_offsets: u8,
    size_of_lengths: u8,
    header_version: u8,
) -> Result<Vec<crate::object_header::HeaderMessage>> {
    use crate::object_header::msg_types;

    let mut r = HDF5Reader::with_sizes(data.clone(), size_of_offsets, size_of_lengths);
    let mut messages = Vec::new();

    if header_version == 2 {
        // v2 continuation: starts with OCHK signature
        r.read_signature(&[b'O', b'C', b'H', b'K'])?;

        let end = data.len() as u64 - 4; // minus checksum
        while r.position() < end {
            let msg_type = r.read_u8()? as u16;
            let msg_size = r.read_u16()? as usize;
            let flags = r.read_u8()?;

            // NIL message (type 0) signals start of gap/padding to end of chunk.
            if msg_type == msg_types::NIL {
                break;
            }

            let msg_data = if msg_size > 0 {
                let d = r.slice_from_position(msg_size)?;
                r.skip(msg_size as u64);
                d
            } else {
                Bytes::new()
            };

            messages.push(crate::object_header::HeaderMessage {
                msg_type,
                data: msg_data,
                flags,
            });
        }
    } else {
        // v1 continuation: raw messages with 8-byte alignment
        let end = data.len() as u64;
        while r.position() + 8 <= end {
            let msg_type = r.read_u16()?;
            let msg_size = r.read_u16()? as usize;
            let flags = r.read_u8()?;
            r.skip(3); // reserved

            if msg_size == 0 && msg_type == msg_types::NIL {
                let pad = (8 - (r.position() % 8)) % 8;
                r.skip(pad);
                continue;
            }

            let msg_data = if msg_size > 0 {
                let d = r.slice_from_position(msg_size)?;
                r.skip(msg_size as u64);
                d
            } else {
                Bytes::new()
            };

            let pad = (8 - (r.position() % 8)) % 8;
            r.skip(pad);

            messages.push(crate::object_header::HeaderMessage {
                msg_type,
                data: msg_data,
                flags,
            });
        }
    }

    Ok(messages)
}
