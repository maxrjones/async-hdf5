use bytes::Bytes;

use crate::endian::{HDF5Reader, UNDEF_ADDR};
use crate::error::{HDF5Error, Result};

/// HDF5 file format signature: `\211HDF\r\n\032\n`
pub const HDF5_SIGNATURE: [u8; 8] = [0x89, 0x48, 0x44, 0x46, 0x0d, 0x0a, 0x1a, 0x0a];

/// The HDF5 superblock, parsed from the beginning of the file.
///
/// Contains file-level parameters that control interpretation of all other
/// structures: the sizes of offset and length fields, and the address of
/// the root group.
#[derive(Debug, Clone)]
pub struct Superblock {
    /// Superblock version (0, 1, 2, or 3).
    pub version: u8,
    /// Number of bytes used for addresses (offsets) throughout the file.
    pub size_of_offsets: u8,
    /// Number of bytes used for sizes (lengths) throughout the file.
    pub size_of_lengths: u8,
    /// File base address (usually 0).
    pub base_address: u64,
    /// Address of the root group's object header.
    pub root_group_address: u64,
    /// End-of-file address.
    pub end_of_file_address: u64,
    /// Address of the superblock extension object header (v2/v3 only, may be UNDEF).
    pub extension_address: u64,
}

impl Superblock {
    /// Parse a superblock from the initial bytes of an HDF5 file.
    ///
    /// The HDF5 signature may appear at offset 0, 512, 1024, 2048, or any
    /// power-of-two multiple of 512. We search the first few candidates.
    pub fn parse(data: &Bytes) -> Result<(Self, u64)> {
        // Search for the signature at standard offsets
        let offsets = [0u64, 512, 1024, 2048, 4096];
        for &offset in &offsets {
            if offset as usize + 8 > data.len() {
                break;
            }
            let slice = &data[offset as usize..offset as usize + 8];
            if slice == HDF5_SIGNATURE {
                let sb = Self::parse_at(data, offset)?;
                return Ok((sb, offset));
            }
        }

        // No HDF5 signature found — try to identify the actual format.
        let hint = identify_format(data);
        Err(HDF5Error::InvalidSignature { offset: 0, hint })
    }

    fn parse_at(data: &Bytes, offset: u64) -> Result<Self> {
        let mut r = HDF5Reader::new(data.clone());
        r.set_position(offset);

        // Skip the 8-byte signature
        r.skip(8);

        let version = r.read_u8()?;

        match version {
            0 | 1 => Self::parse_v0_v1(&mut r, version),
            2 | 3 => Self::parse_v2_v3(&mut r, version),
            _ => Err(HDF5Error::UnsupportedSuperblockVersion(version)),
        }
    }

    /// Parse superblock version 0 or 1.
    ///
    /// Layout after version byte:
    ///   - Version # of File's Free Space Storage (1 byte)
    ///   - Version # of Root Group Symbol Table Entry (1 byte)
    ///   - Reserved (1 byte)
    ///   - Version # of Shared Header Message Format (1 byte)
    ///   - Size of Offsets (1 byte)
    ///   - Size of Lengths (1 byte)
    ///   - Reserved (1 byte)
    ///   - Group Leaf Node K (2 bytes)
    ///   - Group Internal Node K (2 bytes)
    ///   - File Consistency Flags (4 bytes)
    ///   - [v1 only] Indexed Storage Internal Node K (2 bytes) + Reserved (2 bytes)
    ///   - Base Address (O bytes)
    ///   - Address of File Free-space Info (O bytes)
    ///   - End of File Address (O bytes)
    ///   - Driver Information Block Address (O bytes)
    ///   - Root Group Symbol Table Entry (variable)
    fn parse_v0_v1(r: &mut HDF5Reader, version: u8) -> Result<Self> {
        let _free_space_version = r.read_u8()?;
        let _root_group_version = r.read_u8()?;
        let _reserved1 = r.read_u8()?;
        let _shared_header_version = r.read_u8()?;
        let size_of_offsets = r.read_u8()?;
        let size_of_lengths = r.read_u8()?;
        let _reserved2 = r.read_u8()?;

        // Now we know field sizes — update the reader
        *r = HDF5Reader::with_sizes(r.get_ref().clone(), size_of_offsets, size_of_lengths);
        r.set_position(
            8 // signature
            + 1 // version
            + 1 + 1 + 1 + 1 // sub-versions
            + 1 + 1 + 1, // sizes + reserved
        );

        let _group_leaf_k = r.read_u16()?;
        let _group_internal_k = r.read_u16()?;
        let _consistency_flags = r.read_u32()?;

        if version == 1 {
            let _indexed_storage_k = r.read_u16()?;
            let _reserved3 = r.read_u16()?;
        }

        let base_address = r.read_offset()?;
        let _free_space_address = r.read_offset()?;
        let end_of_file_address = r.read_offset()?;
        let _driver_info_address = r.read_offset()?;

        // Root Group Symbol Table Entry
        // Link Name Offset (O bytes) — offset into local heap
        let _link_name_offset = r.read_offset()?;
        // Object Header Address (O bytes) — this is the root group address
        let root_group_address = r.read_offset()?;
        // Cache Type (4 bytes) + Reserved (4 bytes) + Scratch-pad (16 bytes) — skip
        // Total symbol table entry scratch: 4 + 4 + 16 = 24 bytes

        Ok(Self {
            version,
            size_of_offsets,
            size_of_lengths,
            base_address,
            root_group_address,
            end_of_file_address,
            extension_address: UNDEF_ADDR,
        })
    }

    /// Parse superblock version 2 or 3.
    ///
    /// Layout after version byte:
    ///   - Size of Offsets (1 byte)
    ///   - Size of Lengths (1 byte)
    ///   - File Consistency Flags (1 byte)
    ///   - Base Address (O bytes)
    ///   - Superblock Extension Address (O bytes)
    ///   - End of File Address (O bytes)
    ///   - Root Group Object Header Address (O bytes)
    ///   - Superblock Checksum (4 bytes)
    fn parse_v2_v3(r: &mut HDF5Reader, version: u8) -> Result<Self> {
        let size_of_offsets = r.read_u8()?;
        let size_of_lengths = r.read_u8()?;
        let _consistency_flags = r.read_u8()?;

        // Recreate reader with correct sizes
        let pos = r.position();
        *r = HDF5Reader::with_sizes(r.get_ref().clone(), size_of_offsets, size_of_lengths);
        r.set_position(pos);

        let base_address = r.read_offset()?;
        let extension_address = r.read_offset()?;
        let end_of_file_address = r.read_offset()?;
        let root_group_address = r.read_offset()?;
        let _checksum = r.read_u32()?;

        Ok(Self {
            version,
            size_of_offsets,
            size_of_lengths,
            base_address,
            root_group_address,
            end_of_file_address,
            extension_address,
        })
    }
}

/// Inspect the first bytes of data and return a human-readable description
/// of the file format when it is not HDF5.
fn identify_format(data: &Bytes) -> String {
    if data.len() < 4 {
        return format!(
            "file is too small ({} bytes) to contain an HDF5 superblock",
            data.len()
        );
    }

    let head = &data[..std::cmp::min(data.len(), 8)];

    // NetCDF classic / 64-bit offset / CDF-5
    if head.starts_with(b"CDF") && data.len() >= 4 {
        let version_byte = data[3];
        let variant = match version_byte {
            1 => "NetCDF3 classic (CDF-1)",
            2 => "NetCDF3 64-bit offset (CDF-2)",
            5 => "NetCDF3 64-bit data (CDF-5)",
            _ => "NetCDF3 (unknown variant)",
        };
        return format!(
            "file appears to be {} format, not HDF5. \
             NetCDF4 (which uses HDF5) starts with \\x89HDF, \
             but this file starts with CDF\\x{:02x}",
            variant, version_byte
        );
    }

    // HDF4
    if head.len() >= 4 && head[0] == 0x0e && head[1] == 0x03 && head[2] == 0x13 && head[3] == 0x01
    {
        return "file appears to be HDF4 format, not HDF5. \
                async-hdf5 only supports HDF5 (and NetCDF4, which is HDF5-based)"
            .to_string();
    }

    // TIFF (little-endian or big-endian)
    if head.len() >= 4
        && ((head[0] == b'I' && head[1] == b'I' && head[2] == 42 && head[3] == 0)
            || (head[0] == b'M' && head[1] == b'M' && head[2] == 0 && head[3] == 42))
    {
        return "file appears to be TIFF format, not HDF5".to_string();
    }

    // Generic: show first 8 bytes as hex
    let hex: Vec<String> = head.iter().map(|b| format!("{:02x}", b)).collect();
    format!(
        "expected HDF5 signature (\\x89HDF\\r\\n\\x1a\\n) but found [{}]",
        hex.join(" ")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdf5_signature() {
        assert_eq!(HDF5_SIGNATURE[1], b'H');
        assert_eq!(HDF5_SIGNATURE[2], b'D');
        assert_eq!(HDF5_SIGNATURE[3], b'F');
    }

    #[test]
    fn test_superblock_v2_minimal() {
        // Construct a minimal valid superblock v2
        let mut data = Vec::new();
        // Signature
        data.extend_from_slice(&HDF5_SIGNATURE);
        // Version
        data.push(2);
        // Size of Offsets = 8, Size of Lengths = 8
        data.push(8);
        data.push(8);
        // Consistency flags
        data.push(0);
        // Base Address (8 bytes LE) = 0
        data.extend_from_slice(&0u64.to_le_bytes());
        // Extension Address (8 bytes LE) = UNDEF
        data.extend_from_slice(&u64::MAX.to_le_bytes());
        // End of File Address = 4096
        data.extend_from_slice(&4096u64.to_le_bytes());
        // Root Group Object Header Address = 48
        data.extend_from_slice(&48u64.to_le_bytes());
        // Checksum (dummy)
        data.extend_from_slice(&0u32.to_le_bytes());

        let bytes = Bytes::from(data);
        let (sb, offset) = Superblock::parse(&bytes).unwrap();
        assert_eq!(offset, 0);
        assert_eq!(sb.version, 2);
        assert_eq!(sb.size_of_offsets, 8);
        assert_eq!(sb.size_of_lengths, 8);
        assert_eq!(sb.base_address, 0);
        assert_eq!(sb.root_group_address, 48);
        assert_eq!(sb.end_of_file_address, 4096);
        assert_eq!(sb.extension_address, UNDEF_ADDR);
    }
}
