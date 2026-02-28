use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use tokio::sync::Mutex;

use crate::error::{HDF5Error, Result};

/// Async interface for reading byte ranges from an HDF5 file.
///
/// Modeled after async-tiff's `AsyncFileReader` trait. Implementations exist
/// for `object_store::ObjectStore`, `reqwest`, and `tokio::fs::File`.
#[async_trait]
pub trait AsyncFileReader: Debug + Send + Sync + 'static {
    /// Fetch the bytes in the given range.
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes>;

    /// Fetch multiple byte ranges. The default implementation calls `get_bytes`
    /// sequentially; `ObjectReader` overrides this with `get_ranges()`.
    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        let mut result = Vec::with_capacity(ranges.len());
        for range in ranges {
            let data = self.get_bytes(range).await?;
            result.push(data);
        }
        Ok(result)
    }
}

#[async_trait]
impl AsyncFileReader for Box<dyn AsyncFileReader + '_> {
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes> {
        self.as_ref().get_bytes(range).await
    }

    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        self.as_ref().get_byte_ranges(ranges).await
    }
}

#[async_trait]
impl AsyncFileReader for Arc<dyn AsyncFileReader + '_> {
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes> {
        self.as_ref().get_bytes(range).await
    }

    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        self.as_ref().get_byte_ranges(ranges).await
    }
}

// ── ObjectReader ────────────────────────────────────────────────────────────

/// An AsyncFileReader that reads from an [`ObjectStore`](object_store::ObjectStore).
#[cfg(feature = "object_store")]
#[derive(Clone, Debug)]
pub struct ObjectReader {
    store: Arc<dyn object_store::ObjectStore>,
    path: object_store::path::Path,
}

#[cfg(feature = "object_store")]
impl ObjectReader {
    /// Create a new ObjectReader.
    pub fn new(store: Arc<dyn object_store::ObjectStore>, path: object_store::path::Path) -> Self {
        Self { store, path }
    }
}

#[cfg(feature = "object_store")]
#[async_trait]
impl AsyncFileReader for ObjectReader {
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes> {
        use object_store::ObjectStoreExt;

        self.store
            .get_range(&self.path, range)
            .await
            .map_err(HDF5Error::from)
    }

    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        self.store
            .get_ranges(&self.path, &ranges)
            .await
            .map_err(HDF5Error::from)
    }
}

// ── TokioReader ─────────────────────────────────────────────────────────────

/// An AsyncFileReader that wraps a `tokio::fs::File` or similar async reader.
#[cfg(feature = "tokio")]
#[derive(Debug)]
pub struct TokioReader<T: tokio::io::AsyncRead + tokio::io::AsyncSeek + Unpin + Send + Debug>(
    tokio::sync::Mutex<T>,
);

#[cfg(feature = "tokio")]
impl<T: tokio::io::AsyncRead + tokio::io::AsyncSeek + Unpin + Send + Debug> TokioReader<T> {
    /// Create a new TokioReader.
    pub fn new(inner: T) -> Self {
        Self(tokio::sync::Mutex::new(inner))
    }
}

#[cfg(feature = "tokio")]
#[async_trait]
impl<T: tokio::io::AsyncRead + tokio::io::AsyncSeek + Unpin + Send + Debug + 'static>
    AsyncFileReader for TokioReader<T>
{
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes> {
        use std::io::SeekFrom;
        use tokio::io::{AsyncReadExt, AsyncSeekExt};

        let mut file = self.0.lock().await;
        file.seek(SeekFrom::Start(range.start)).await?;

        let to_read = (range.end - range.start) as usize;
        let mut buffer = vec![0u8; to_read];

        // Use read_buf loop instead of read_exact to handle EOF gracefully.
        // The ReadaheadCache may request ranges past the end of the file.
        let mut total_read = 0;
        while total_read < to_read {
            let n = file.read(&mut buffer[total_read..]).await?;
            if n == 0 {
                break; // EOF
            }
            total_read += n;
        }
        buffer.truncate(total_read);
        Ok(buffer.into())
    }
}

// ── ReqwestReader ───────────────────────────────────────────────────────────

/// An AsyncFileReader that reads from a URL using reqwest HTTP range requests.
#[cfg(feature = "reqwest")]
#[derive(Debug, Clone)]
pub struct ReqwestReader {
    client: reqwest::Client,
    url: reqwest::Url,
}

#[cfg(feature = "reqwest")]
impl ReqwestReader {
    /// Create a new ReqwestReader.
    pub fn new(client: reqwest::Client, url: reqwest::Url) -> Self {
        Self { client, url }
    }
}

#[cfg(feature = "reqwest")]
#[async_trait]
impl AsyncFileReader for ReqwestReader {
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes> {
        let range_header = format!("bytes={}-{}", range.start, range.end - 1);
        let response = self
            .client
            .get(self.url.clone())
            .header("Range", range_header)
            .send()
            .await?
            .error_for_status()?;
        let bytes = response.bytes().await?;
        Ok(bytes)
    }
}

// ── ReadaheadCache ──────────────────────────────────────────────────────────

/// Sequential block cache that stores contiguous data from the start of a file.
#[derive(Debug)]
struct SequentialBlockCache {
    buffers: Vec<Bytes>,
    len: u64,
}

impl SequentialBlockCache {
    fn new() -> Self {
        Self {
            buffers: vec![],
            len: 0,
        }
    }

    fn contains(&self, range: &Range<u64>) -> bool {
        range.end <= self.len
    }

    fn slice(&self, range: Range<u64>) -> Bytes {
        let out_len = (range.end - range.start) as usize;
        let mut remaining = range;
        let mut out_buffers: Vec<Bytes> = vec![];

        for buf in &self.buffers {
            let current_buf_len = buf.len() as u64;

            if remaining.start >= current_buf_len {
                remaining.start -= current_buf_len;
                remaining.end -= current_buf_len;
                continue;
            }

            let start = remaining.start as usize;
            let length =
                (remaining.end - remaining.start).min(current_buf_len - remaining.start) as usize;
            let end = start + length;

            if start == end {
                continue;
            }

            out_buffers.push(buf.slice(start..end));

            remaining.start = 0;
            if remaining.end <= current_buf_len {
                break;
            }
            remaining.end -= current_buf_len;
        }

        if out_buffers.len() == 1 {
            out_buffers.into_iter().next().unwrap()
        } else {
            let mut out = BytesMut::with_capacity(out_len);
            for b in out_buffers {
                out.extend_from_slice(&b);
            }
            out.into()
        }
    }

    fn append(&mut self, buffer: Bytes) {
        self.len += buffer.len() as u64;
        self.buffers.push(buffer);
    }
}

/// A caching wrapper that prefetches data sequentially from the start of a file
/// in exponentially growing chunks.
///
/// Modeled after async-tiff's `ReadaheadMetadataCache`. This should **always** be
/// used when reading metadata to avoid translating the many small internal reads
/// into individual tiny network requests.
#[derive(Debug)]
pub struct ReadaheadCache<F: AsyncFileReader> {
    inner: F,
    cache: Arc<Mutex<SequentialBlockCache>>,
    initial: u64,
    multiplier: f64,
}

impl<F: AsyncFileReader> ReadaheadCache<F> {
    /// Create a new ReadaheadCache wrapping the given reader.
    /// Default: 32 KiB initial prefetch, 2× growth.
    pub fn new(inner: F) -> Self {
        Self {
            inner,
            cache: Arc::new(Mutex::new(SequentialBlockCache::new())),
            initial: 32 * 1024,
            multiplier: 2.0,
        }
    }

    /// Access the inner reader.
    pub fn inner(&self) -> &F {
        &self.inner
    }

    /// Set the initial fetch size in bytes.
    pub fn with_initial_size(mut self, initial: u64) -> Self {
        self.initial = initial;
        self
    }

    /// Set the growth multiplier for subsequent fetches.
    pub fn with_multiplier(mut self, multiplier: f64) -> Self {
        self.multiplier = multiplier;
        self
    }

    fn next_fetch_size(&self, existing_len: u64) -> u64 {
        if existing_len == 0 {
            self.initial
        } else {
            (existing_len as f64 * self.multiplier).round() as u64
        }
    }
}

#[async_trait]
impl<F: AsyncFileReader + Send + Sync> AsyncFileReader for ReadaheadCache<F> {
    async fn get_bytes(&self, range: Range<u64>) -> Result<Bytes> {
        let mut cache = self.cache.lock().await;

        if cache.contains(&range) {
            return Ok(cache.slice(range));
        }

        let start_len = cache.len;
        let needed = range.end.saturating_sub(start_len);
        let fetch_size = self.next_fetch_size(start_len).max(needed);
        let fetch_range = start_len..start_len + fetch_size;

        // Try to extend the cache. The fetch may fail if we're at or past EOF
        // (e.g., object_store rejects start >= file_length). Handle gracefully
        // by serving whatever we already have.
        match self.inner.get_bytes(fetch_range).await {
            Ok(bytes) => cache.append(bytes),
            Err(_) if range.start < cache.len => {
                // Extension failed (likely EOF), but we can serve a partial response
                return Ok(cache.slice(range.start..cache.len.min(range.end)));
            }
            Err(e) => return Err(e),
        }

        // If we hit EOF, the cache may not fully cover the requested range.
        // Return whatever is available (the object header parser handles
        // short buffers via its own bounds checking).
        if cache.contains(&range) {
            Ok(cache.slice(range))
        } else if range.start < cache.len {
            Ok(cache.slice(range.start..cache.len))
        } else {
            // Requested range starts beyond EOF
            Ok(Bytes::new())
        }
    }

    async fn get_byte_ranges(&self, ranges: Vec<Range<u64>>) -> Result<Vec<Bytes>> {
        // For metadata reads, ranges are typically sequential and small,
        // so delegating to get_bytes (which hits the cache) is efficient.
        let mut result = Vec::with_capacity(ranges.len());
        for range in ranges {
            result.push(self.get_bytes(range).await?);
        }
        Ok(result)
    }
}
