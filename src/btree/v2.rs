use std::sync::Arc;

use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};
use crate::reader::AsyncFileReader;

/// Parsed B-tree v2 header (BTHD signature).
#[derive(Debug)]
pub struct BTreeV2Header {
    /// Record type (5=link name, 6=creation order, 10=chunks no filter, 11=chunks filtered, etc.).
    pub record_type: u8,
    /// Size of each tree node in bytes.
    pub node_size: u32,
    /// Size of each record in bytes.
    pub record_size: u16,
    /// Depth of the tree (0 = root is a leaf).
    pub depth: u16,
    /// Split percent.
    pub split_percent: u8,
    /// Merge percent.
    pub merge_percent: u8,
    /// Address of the root node.
    pub root_node_address: u64,
    /// Number of records in the root node.
    pub num_records_in_root: u64,
    /// Total number of records in the entire tree.
    pub total_records: u64,
}

impl BTreeV2Header {
    /// Parse a B-tree v2 header from file.
    pub async fn read(
        reader: &Arc<dyn AsyncFileReader>,
        address: u64,
        size_of_offsets: u8,
        size_of_lengths: u8,
    ) -> Result<Self> {
        // Header: 4 (sig) + 1 (ver) + 1 (type) + 4 (node_size) + 2 (rec_size) + 2 (depth)
        //       + 1 (split) + 1 (merge) + O (root_addr) + L (num_root_rec) + L (total) + 4 (checksum)
        let header_size =
            16 + size_of_offsets as u64 + 2 * size_of_lengths as u64 + 4;
        let data = reader.get_bytes(address..address + header_size).await?;
        let mut r = HDF5Reader::with_sizes(data, size_of_offsets, size_of_lengths);

        r.read_signature(b"BTHD")?;

        let version = r.read_u8()?;
        if version != 0 {
            return Err(HDF5Error::UnsupportedBTreeVersion(version));
        }

        let record_type = r.read_u8()?;
        let node_size = r.read_u32()?;
        let record_size = r.read_u16()?;
        let depth = r.read_u16()?;
        let split_percent = r.read_u8()?;
        let merge_percent = r.read_u8()?;
        let root_node_address = r.read_offset()?;
        let num_records_in_root = r.read_length()?;
        let total_records = r.read_length()?;

        Ok(Self {
            record_type,
            node_size,
            record_size,
            depth,
            split_percent,
            merge_percent,
            root_node_address,
            num_records_in_root,
            total_records,
        })
    }
}

/// A link record from a B-tree v2 type 5 (link name) or type 6 (creation order).
#[derive(Debug, Clone)]
pub struct LinkRecord {
    /// For type 5: Jenkins hash of link name. For type 6: unused.
    pub hash: Option<u32>,
    /// For type 6: creation order value. For type 5: unused.
    pub creation_order: Option<u64>,
    /// The 7-byte fractal heap ID referencing the link message.
    pub heap_id: Vec<u8>,
}

/// Traverse a B-tree v2 and collect all records as raw bytes.
///
/// Returns each record as raw bytes of `record_size` length. Callers
/// can parse the specific record type.
pub async fn collect_all_records(
    reader: &Arc<dyn AsyncFileReader>,
    header: &BTreeV2Header,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<Bytes>> {
    let mut records = Vec::with_capacity(header.total_records as usize);

    if header.total_records == 0 || HDF5Reader::is_undef_addr(header.root_node_address, size_of_offsets) {
        return Ok(records);
    }

    collect_records_from_node(
        reader,
        header.root_node_address,
        header,
        header.depth,
        header.num_records_in_root as usize,
        size_of_offsets,
        size_of_lengths,
        &mut records,
    )
    .await?;

    Ok(records)
}

/// Recursively collect records from a B-tree v2 node.
async fn collect_records_from_node(
    reader: &Arc<dyn AsyncFileReader>,
    node_address: u64,
    header: &BTreeV2Header,
    depth: u16,
    num_records: usize,
    size_of_offsets: u8,
    size_of_lengths: u8,
    records: &mut Vec<Bytes>,
) -> Result<()> {
    let data = reader
        .get_bytes(node_address..node_address + header.node_size as u64)
        .await?;
    let mut r = HDF5Reader::with_sizes(data.clone(), size_of_offsets, size_of_lengths);

    if depth == 0 {
        // Leaf node (BTLF)
        r.read_signature(b"BTLF")?;
        let _version = r.read_u8()?;
        let _type = r.read_u8()?;

        for _ in 0..num_records {
            let record_data = r.slice_from_position(header.record_size as usize)?;
            r.skip(header.record_size as u64);
            records.push(record_data);
        }
    } else {
        // Internal node (BTIN)
        r.read_signature(b"BTIN")?;
        let _version = r.read_u8()?;
        let _type = r.read_u8()?;

        // Records come first, then child pointers.
        // Read all records for this node.
        let mut node_records = Vec::with_capacity(num_records);
        for _ in 0..num_records {
            let record_data = r.slice_from_position(header.record_size as usize)?;
            r.skip(header.record_size as u64);
            node_records.push(record_data);
        }

        // Child pointers: (num_records + 1) entries.
        // Each: address(O) + num_records_in_child(variable) + [total_records_in_child(variable)]
        let max_records_per_node = max_records_for_node(header, depth == 1);
        let num_records_width = bytes_needed(max_records_per_node);
        let total_records_width = if depth > 1 {
            // For non-twig internal nodes, total records field is present
            bytes_needed(header.total_records)
        } else {
            0
        };

        let num_children = num_records + 1;
        let mut child_addrs = Vec::with_capacity(num_children);
        let mut child_num_records = Vec::with_capacity(num_children);

        for _ in 0..num_children {
            let addr = r.read_offset()?;
            let child_recs = read_var_uint_from_reader(&mut r, num_records_width)?;
            if total_records_width > 0 {
                let _total = read_var_uint_from_reader(&mut r, total_records_width)?;
            }
            child_addrs.push(addr);
            child_num_records.push(child_recs as usize);
        }

        // Recursively collect: child[0], record[0], child[1], record[1], ..., child[N]
        for i in 0..num_children {
            Box::pin(collect_records_from_node(
                reader,
                child_addrs[i],
                header,
                depth - 1,
                child_num_records[i],
                size_of_offsets,
                size_of_lengths,
                records,
            ))
            .await?;

            if i < num_records {
                records.push(node_records[i].clone());
            }
        }
    }

    Ok(())
}

/// Parse link records from raw B-tree v2 records.
///
/// For type 5: [hash(4)] [heap_id(7)]  — 11 bytes total
/// For type 6: [creation_order(8)] [heap_id(7)] — 15 bytes total
pub fn parse_link_records(raw_records: &[Bytes], record_type: u8) -> Result<Vec<LinkRecord>> {
    let mut links = Vec::with_capacity(raw_records.len());

    for record in raw_records {
        let mut r = HDF5Reader::new(record.clone());

        match record_type {
            5 => {
                // Link Name for Indexed Group
                let hash = r.read_u32()?;
                let heap_id = r.read_bytes(record.len() - 4)?;
                links.push(LinkRecord {
                    hash: Some(hash),
                    creation_order: None,
                    heap_id,
                });
            }
            6 => {
                // Creation Order for Indexed Group
                let creation_order = r.read_u64()?;
                let heap_id = r.read_bytes(record.len() - 8)?;
                links.push(LinkRecord {
                    hash: None,
                    creation_order: Some(creation_order),
                    heap_id,
                });
            }
            _ => {
                return Err(HDF5Error::General(format!(
                    "unsupported B-tree v2 record type for links: {record_type}"
                )));
            }
        }
    }

    Ok(links)
}

/// Chunk record from B-tree v2 type 11 (filtered chunks).
#[derive(Debug, Clone)]
pub struct ChunkRecordFiltered {
    /// File address of the chunk data.
    pub address: u64,
    /// Compressed/on-disk chunk size.
    pub chunk_size: u64,
    /// Filter mask — bit N set means filter N was *not* applied.
    pub filter_mask: u32,
    /// Chunk offsets (scaled by chunk dimensions).
    pub scaled_offsets: Vec<u64>,
}

/// Chunk record from B-tree v2 type 10 (non-filtered chunks).
#[derive(Debug, Clone)]
pub struct ChunkRecordNonFiltered {
    /// File address of the chunk data.
    pub address: u64,
    /// Chunk offsets (scaled by chunk dimensions).
    pub scaled_offsets: Vec<u64>,
}

/// Parse chunk records from raw B-tree v2 records (type 10 or 11).
///
/// `ndims` is the dataset dimensionality. For filtered chunks (type 11), each
/// record is: address(O) + chunk_size(L) + filter_mask(4) + scaled_offsets(8*ndims).
/// For non-filtered (type 10): address(O) + scaled_offsets(8*ndims).
pub fn parse_chunk_records_filtered(
    raw_records: &[Bytes],
    ndims: usize,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Result<Vec<ChunkRecordFiltered>> {
    let mut chunks = Vec::with_capacity(raw_records.len());

    for record in raw_records {
        let mut r = HDF5Reader::with_sizes(record.clone(), size_of_offsets, size_of_lengths);
        let address = r.read_offset()?;
        let chunk_size = r.read_length()?;
        let filter_mask = r.read_u32()?;

        let mut scaled_offsets = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            scaled_offsets.push(r.read_u64()?);
        }

        chunks.push(ChunkRecordFiltered {
            address,
            chunk_size,
            filter_mask,
            scaled_offsets,
        });
    }

    Ok(chunks)
}

/// Parse non-filtered chunk records from raw B-tree v2 records (type 10).
pub fn parse_chunk_records_non_filtered(
    raw_records: &[Bytes],
    ndims: usize,
    size_of_offsets: u8,
) -> Result<Vec<ChunkRecordNonFiltered>> {
    let mut chunks = Vec::with_capacity(raw_records.len());

    for record in raw_records {
        let mut r = HDF5Reader::with_sizes(record.clone(), size_of_offsets, 8);
        let address = r.read_offset()?;

        let mut scaled_offsets = Vec::with_capacity(ndims);
        for _ in 0..ndims {
            scaled_offsets.push(r.read_u64()?);
        }

        chunks.push(ChunkRecordNonFiltered {
            address,
            scaled_offsets,
        });
    }

    Ok(chunks)
}

/// Calculate maximum records that fit in a node at the given depth.
fn max_records_for_node(header: &BTreeV2Header, is_leaf: bool) -> u64 {
    if is_leaf {
        // Leaf: (node_size - 6 - 4) / record_size
        // 6 = sig(4) + version(1) + type(1), 4 = checksum
        ((header.node_size - 10) / header.record_size as u32) as u64
    } else {
        // Very rough upper bound — exact calculation requires knowing child pointer sizes
        // which depend on the max records. Use a conservative estimate.
        ((header.node_size - 10) / (header.record_size as u32 + 8)) as u64
    }
}

/// Number of bytes needed to represent the given value.
fn bytes_needed(value: u64) -> usize {
    if value == 0 {
        return 1;
    }
    let bits = 64 - value.leading_zeros();
    ((bits + 7) / 8) as usize
}

/// Read a variable-width unsigned integer from the HDF5Reader.
fn read_var_uint_from_reader(r: &mut HDF5Reader, width: usize) -> Result<u64> {
    let bytes = r.read_bytes(width)?;
    Ok(read_var_uint(&bytes))
}

/// Read a variable-width little-endian unsigned integer from a byte slice.
fn read_var_uint(bytes: &[u8]) -> u64 {
    let mut value = 0u64;
    for (i, &b) in bytes.iter().enumerate() {
        value |= (b as u64) << (i * 8);
    }
    value
}
