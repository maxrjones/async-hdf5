use bytes::Bytes;

use crate::endian::HDF5Reader;
use crate::error::Result;

/// Fill value message — specifies the default value for uninitialized data.
///
/// Message type 0x0005.
#[derive(Debug, Clone)]
pub struct FillValueMessage {
    /// Raw fill value bytes (interpretation depends on the dataset's data type).
    /// None if fill value is undefined.
    pub value: Option<Vec<u8>>,
    /// Fill time: 0=on allocation, 1=never, 2=if set by user.
    pub fill_time: u8,
}

impl FillValueMessage {
    /// Parse from the raw message bytes.
    pub fn parse(data: &Bytes) -> Result<Self> {
        let mut r = HDF5Reader::new(data.clone());

        let version = r.read_u8()?;

        match version {
            1 | 2 => {
                // v1/v2: alloc_time(1) + fill_time(1) + defined(1) + [size(4) + value]
                let _alloc_time = r.read_u8()?;
                let fill_time = r.read_u8()?;
                let fill_value_defined = r.read_u8()?;

                let value = if fill_value_defined != 0 {
                    let size = r.read_u32()? as usize;
                    if size > 0 {
                        Some(r.read_bytes(size)?)
                    } else {
                        None
                    }
                } else {
                    None
                };

                Ok(Self { value, fill_time })
            }
            3 => {
                // v3: flags(1) — bits encode alloc_time, fill_time, defined, undefined
                let flags = r.read_u8()?;
                let fill_time = (flags >> 2) & 0x03;
                let fill_value_defined = (flags >> 4) & 0x01 != 0;
                let fill_value_undefined = (flags >> 5) & 0x01 != 0;

                let value = if fill_value_defined && !fill_value_undefined {
                    let size = r.read_u32()? as usize;
                    if size > 0 {
                        Some(r.read_bytes(size)?)
                    } else {
                        None
                    }
                } else {
                    None
                };

                Ok(Self { value, fill_time })
            }
            _ => {
                // Unknown version — return empty
                Ok(Self {
                    value: None,
                    fill_time: 0,
                })
            }
        }
    }
}
