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

#[tokio::test]
async fn test_v1_root_children() {
    let file = open_fixture("groups_v1.h5").await.unwrap();
    let root = file.root_group().await.unwrap();

    let children = root.children().await.unwrap();
    let mut names: Vec<_> = children.iter().map(|c| c.name.as_str()).collect();
    names.sort();

    assert_eq!(names, vec!["alpha", "beta"]);
}

#[tokio::test]
async fn test_v1_navigate_child_group() {
    let file = open_fixture("groups_v1.h5").await.unwrap();
    let root = file.root_group().await.unwrap();

    let alpha = root.group("alpha").await.unwrap();
    assert_eq!(alpha.name(), "alpha");

    let children = alpha.children().await.unwrap();
    let names: Vec<_> = children.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, vec!["data"]);
}

#[tokio::test]
async fn test_v1_dataset_detection() {
    let file = open_fixture("groups_v1.h5").await.unwrap();
    let root = file.root_group().await.unwrap();

    let group_names = root.group_names().await.unwrap();
    let dataset_names = root.dataset_names().await.unwrap();

    // Root has 2 groups, no datasets
    assert_eq!(group_names.len(), 2);
    assert_eq!(dataset_names.len(), 0);

    // "alpha" has 1 dataset
    let alpha = root.group("alpha").await.unwrap();
    let ds_names = alpha.dataset_names().await.unwrap();
    assert_eq!(ds_names, vec!["data"]);
}

#[tokio::test]
async fn test_v2_root_children() {
    let file = open_fixture("groups_v2_latest.h5").await.unwrap();
    let root = file.root_group().await.unwrap();

    let children = root.children().await.unwrap();
    let mut names: Vec<_> = children.iter().map(|c| c.name.as_str()).collect();
    names.sort();

    assert_eq!(names, vec!["metadata", "science"]);
}

#[tokio::test]
async fn test_v2_deep_navigation() {
    let file = open_fixture("groups_v2_latest.h5").await.unwrap();
    let root = file.root_group().await.unwrap();

    // Navigate the deep NISAR-like hierarchy
    let freq_a = root
        .navigate("science/LSAR/GCOV/grids/frequencyA")
        .await
        .unwrap();

    let children = freq_a.children().await.unwrap();
    let mut names: Vec<_> = children.iter().map(|c| c.name.as_str()).collect();
    names.sort();

    assert_eq!(names, vec!["HHHH", "HVHV"]);
}

#[tokio::test]
async fn test_v2_dataset_detection() {
    let file = open_fixture("groups_v2_latest.h5").await.unwrap();
    let root = file.root_group().await.unwrap();

    let freq_a = root
        .navigate("science/LSAR/GCOV/grids/frequencyA")
        .await
        .unwrap();

    let group_names = freq_a.group_names().await.unwrap();
    let mut dataset_names = freq_a.dataset_names().await.unwrap();
    dataset_names.sort();

    // frequencyA has 0 groups, 2 datasets
    assert_eq!(group_names.len(), 0);
    assert_eq!(dataset_names, vec!["HHHH", "HVHV"]);
}

#[tokio::test]
async fn test_not_found_error() {
    let file = open_fixture("groups_v2_latest.h5").await.unwrap();
    let root = file.root_group().await.unwrap();

    let result = root.group("nonexistent").await;
    assert!(result.is_err());
}
