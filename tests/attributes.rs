use std::path::PathBuf;

use async_hdf5::reader::TokioReader;
use async_hdf5::{AttributeValue, HDF5File};

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

// ── v2 root group attributes ─────────────────────────────────────────────

#[tokio::test]
async fn test_v2_root_group_attrs() {
    let file = open_fixture("attributes.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let attrs = root.attributes().await;

    assert_eq!(attrs.len(), 3);

    let title = root.attribute("title").await.unwrap();
    assert_eq!(title.value.as_str(), Some("Test File"));

    let version = root.attribute("version").await.unwrap();
    assert_eq!(version.value.as_i32(), Some(42));

    let pi = root.attribute("pi").await.unwrap();
    let pi_val = pi.value.as_f64().unwrap();
    assert!((pi_val - std::f64::consts::PI).abs() < 1e-10);
}

// ── v2 group attributes ──────────────────────────────────────────────────

#[tokio::test]
async fn test_v2_group_attrs() {
    let file = open_fixture("attributes.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let group = root.group("mygroup").await.unwrap();
    let attrs = group.attributes().await;

    assert_eq!(attrs.len(), 3);

    let desc = group.attribute("description").await.unwrap();
    assert_eq!(desc.value.as_str(), Some("A test group"));

    let count = group.attribute("count").await.unwrap();
    assert_eq!(count.value.as_i64(), Some(100));

    let values = group.attribute("values").await.unwrap();
    assert_eq!(values.value, AttributeValue::F32(vec![1.0, 2.0, 3.0]));
}

// ── v2 dataset attributes ────────────────────────────────────────────────

#[tokio::test]
async fn test_v2_dataset_attrs() {
    let file = open_fixture("attributes.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let ds = root
        .group("mygroup")
        .await
        .unwrap()
        .dataset("data")
        .await
        .unwrap();
    let attrs = ds.attributes().await;

    assert_eq!(attrs.len(), 4);

    let units = ds.attribute("units").await.unwrap();
    assert_eq!(units.value.as_str(), Some("meters"));

    let scale = ds.attribute("scale_factor").await.unwrap();
    let scale_val = scale.value.as_f32().unwrap();
    assert!((scale_val - 0.01).abs() < 1e-6);

    let range = ds.attribute("valid_range").await.unwrap();
    assert_eq!(range.value, AttributeValue::I32(vec![0, 100]));

    // h5py stores booleans as enum — we decode the base type (int8)
    let flag = ds.attribute("flag").await.unwrap();
    match &flag.value {
        AttributeValue::I8(v) => assert_eq!(v, &[1]),
        other => panic!("expected I8 for boolean, got {other:?}"),
    }
}

// ── v1 format attributes ─────────────────────────────────────────────────

#[tokio::test]
async fn test_v1_root_attrs() {
    let file = open_fixture("attributes_v1.h5").await.unwrap();
    let root = file.root_group().await.unwrap();

    let title = root.attribute("title").await.unwrap();
    assert_eq!(title.value.as_str(), Some("V1 Test File"));

    let version = root.attribute("version").await.unwrap();
    assert_eq!(version.value.as_i32(), Some(1));
}

#[tokio::test]
async fn test_v1_group_attrs() {
    let file = open_fixture("attributes_v1.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let group = root.group("subgroup").await.unwrap();

    let note = group.attribute("note").await.unwrap();
    assert_eq!(note.value.as_str(), Some("hello"));
}

#[tokio::test]
async fn test_v1_dataset_attrs() {
    let file = open_fixture("attributes_v1.h5").await.unwrap();
    let root = file.root_group().await.unwrap();
    let ds = root
        .group("subgroup")
        .await
        .unwrap()
        .dataset("array")
        .await
        .unwrap();

    let offset = ds.attribute("offset").await.unwrap();
    assert_eq!(offset.value.as_f64(), Some(1.5));
}

// ── Variable-length string attributes ────────────────────────────────────

#[tokio::test]
async fn test_vlen_string_attrs() {
    let file = open_fixture("attributes_vlen.h5").await.unwrap();
    let root = file.root_group().await.unwrap();

    let title = root.attribute("title").await.unwrap();
    assert_eq!(title.value.as_str(), Some("Variable Length String"));

    let version = root.attribute("version").await.unwrap();
    assert_eq!(version.value.as_i32(), Some(7));
}
