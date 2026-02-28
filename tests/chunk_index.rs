use std::path::PathBuf;

use async_hdf5::HDF5File;
use async_hdf5::reader::TokioReader;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join(name)
}

async fn open_fixture(name: &str) -> async_hdf5::Result<HDF5File> {
    let path = fixture_path(name);
    let file = tokio::fs::File::open(&path).await.unwrap();
    let reader = TokioReader::new(file);
    HDF5File::open(reader).await
}

// ── Dataset metadata tests ──────────────────────────────────────────────────

#[tokio::test]
async fn test_v2_chunked_2d_metadata() {
    let file = open_fixture("datasets.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let ds = root.dataset("chunked_2d").await.unwrap();

    assert_eq!(ds.shape(), &[8, 8]);
    assert_eq!(ds.ndim(), 2);
    assert_eq!(ds.chunk_shape(), Some(&[4u64, 4][..]));
    assert_eq!(ds.element_size(), 4); // float32

    // Should have shuffle + gzip filters
    let filters = ds.filters();
    assert_eq!(filters.filters.len(), 2);
    // shuffle comes first in HDF5 pipeline order
    assert_eq!(filters.filters[0].display_name(), "shuffle");
    assert_eq!(filters.filters[1].display_name(), "deflate");
}

#[tokio::test]
async fn test_v2_contiguous_metadata() {
    let file = open_fixture("datasets.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let ds = root.dataset("contiguous_1d").await.unwrap();

    assert_eq!(ds.shape(), &[5]);
    assert_eq!(ds.ndim(), 1);
    assert_eq!(ds.chunk_shape(), None);
    assert_eq!(ds.element_size(), 8); // float64
    assert!(ds.layout().is_contiguous());
    assert!(ds.filters().filters.is_empty());
}

#[tokio::test]
async fn test_v2_chunked_1d_metadata() {
    let file = open_fixture("datasets.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let ds = root.dataset("chunked_1d").await.unwrap();

    assert_eq!(ds.shape(), &[20]);
    assert_eq!(ds.chunk_shape(), Some(&[5u64][..]));
    assert_eq!(ds.element_size(), 4); // int32
    assert!(ds.filters().filters.is_empty()); // no compression
}

#[tokio::test]
async fn test_v2_compound_metadata() {
    let file = open_fixture("datasets.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let ds = root.dataset("compound_2d").await.unwrap();

    assert_eq!(ds.shape(), &[4, 4]);
    assert_eq!(ds.chunk_shape(), Some(&[2u64, 2][..]));
    assert_eq!(ds.element_size(), 8); // cfloat32 = 2 * float32

    // Verify compound type structure
    match ds.dtype() {
        async_hdf5::DataType::Compound { fields, size } => {
            assert_eq!(*size, 8);
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].name, "r");
            assert_eq!(fields[1].name, "i");
        }
        other => panic!("expected Compound type, got {other:?}"),
    }
}

// ── Chunk index tests ───────────────────────────────────────────────────────

#[tokio::test]
async fn test_v2_chunked_2d_chunk_index() {
    let file = open_fixture("datasets.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let ds = root.dataset("chunked_2d").await.unwrap();

    let index = ds.chunk_index().await.unwrap();

    // 8x8 array with 4x4 chunks = 2x2 grid = 4 chunks
    assert_eq!(index.len(), 4);
    assert_eq!(index.grid_shape(), vec![2, 2]);
    assert_eq!(index.chunk_shape(), &[4, 4]);
    assert_eq!(index.dataset_shape(), &[8, 8]);

    // All chunks should have valid offsets and non-zero sizes
    for chunk in index.iter() {
        assert!(chunk.byte_offset > 0);
        assert!(chunk.byte_length > 0);
        // Compressed, so byte_length < uncompressed (4*4*4 = 64 bytes)
        assert!(chunk.byte_length < 64, "expected compressed chunk < 64 bytes, got {}", chunk.byte_length);
    }
}

#[tokio::test]
async fn test_v2_chunked_1d_chunk_index() {
    let file = open_fixture("datasets.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let ds = root.dataset("chunked_1d").await.unwrap();

    let index = ds.chunk_index().await.unwrap();

    // 20 elements with chunk size 5 = 4 chunks
    assert_eq!(index.len(), 4);
    assert_eq!(index.grid_shape(), vec![4]);

    // Uncompressed chunks: each = 5 * 4 bytes = 20 bytes
    for chunk in index.iter() {
        assert_eq!(chunk.byte_length, 20);
        assert_eq!(chunk.filter_mask, 0);
    }
}

#[tokio::test]
async fn test_v2_contiguous_chunk_index() {
    let file = open_fixture("datasets.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let ds = root.dataset("contiguous_1d").await.unwrap();

    let index = ds.chunk_index().await.unwrap();

    // Contiguous = single entry
    assert_eq!(index.len(), 1);

    let chunk = index.get(&[0]).unwrap();
    assert_eq!(chunk.byte_length, 40); // 5 * 8 bytes (float64)
    assert_eq!(chunk.filter_mask, 0);
}

#[tokio::test]
async fn test_v2_compound_chunk_index() {
    let file = open_fixture("datasets.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let ds = root.dataset("compound_2d").await.unwrap();

    let index = ds.chunk_index().await.unwrap();

    // 4x4 with 2x2 chunks = 2x2 grid = 4 chunks
    assert_eq!(index.len(), 4);
    assert_eq!(index.grid_shape(), vec![2, 2]);

    // Uncompressed: each chunk = 2*2 * 8 bytes (cfloat32) = 32 bytes
    for chunk in index.iter() {
        assert_eq!(chunk.byte_length, 32);
    }
}

// ── v1 format chunk index tests ─────────────────────────────────────────────

#[tokio::test]
async fn test_v1_chunked_metadata() {
    let file = open_fixture("datasets_v1.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let ds = root.dataset("chunked").await.unwrap();

    assert_eq!(ds.shape(), &[4, 6]);
    assert_eq!(ds.chunk_shape(), Some(&[2u64, 3][..]));
    assert_eq!(ds.element_size(), 4); // float32

    // Should have gzip filter
    let filters = ds.filters();
    assert_eq!(filters.filters.len(), 1);
    assert_eq!(filters.filters[0].display_name(), "deflate");
}

#[tokio::test]
async fn test_v1_chunked_chunk_index() {
    let file = open_fixture("datasets_v1.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let ds = root.dataset("chunked").await.unwrap();

    let index = ds.chunk_index().await.unwrap();

    // 4x6 with 2x3 chunks = 2x2 grid = 4 chunks
    assert_eq!(index.len(), 4);
    assert_eq!(index.grid_shape(), vec![2, 2]);

    // All chunks should be compressed (smaller than 2*3*4=24 raw bytes is likely)
    for chunk in index.iter() {
        assert!(chunk.byte_offset > 0);
        assert!(chunk.byte_length > 0);
    }
}

#[tokio::test]
async fn test_v1_contiguous_chunk_index() {
    let file = open_fixture("datasets_v1.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let ds = root.dataset("contiguous").await.unwrap();

    let index = ds.chunk_index().await.unwrap();

    // Contiguous = single entry
    assert_eq!(index.len(), 1);

    let chunk = index.get(&[0]).unwrap();
    assert_eq!(chunk.byte_length, 24); // 3 * 8 bytes (int64)
}
