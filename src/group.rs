use std::sync::Arc;

use bytes::Bytes;

use crate::btree;
use crate::dataset::HDF5Dataset;
use crate::endian::HDF5Reader;
use crate::error::{HDF5Error, Result};
use crate::file::read_object_header;
use crate::heap;
use crate::messages::attribute::{Attribute, AttributeMessage};
use crate::messages::link::{LinkMessage, LinkType};
use crate::messages::link_info::LinkInfoMessage;
use crate::messages::symbol_table::SymbolTableMessage;
use crate::object_header::{msg_types, ObjectHeader};
use crate::reader::AsyncFileReader;
use crate::superblock::Superblock;

/// A named link to a child object (group or dataset).
#[derive(Debug, Clone)]
pub struct ChildLink {
    /// Link name.
    pub name: String,
    /// File address of the child's object header.
    pub address: u64,
}

/// An HDF5 group — a container for datasets and other groups.
///
/// Groups can be navigated by name (like a filesystem). Internally, HDF5 uses
/// two completely different mechanisms depending on the object header version:
///
/// - **v1 groups**: Symbol table message → B-tree v1 (type 0) + local heap for names
/// - **v2 groups**: Inline link messages (small groups) or link info message →
///   fractal heap + B-tree v2 (large groups)
#[derive(Debug)]
pub struct HDF5Group {
    name: String,
    header: ObjectHeader,
    reader: Arc<dyn AsyncFileReader>,
    raw_reader: Arc<dyn AsyncFileReader>,
    superblock: Arc<Superblock>,
}

impl HDF5Group {
    /// Create a new group from its parsed object header.
    pub fn new(
        name: String,
        header: ObjectHeader,
        reader: Arc<dyn AsyncFileReader>,
        raw_reader: Arc<dyn AsyncFileReader>,
        superblock: Arc<Superblock>,
    ) -> Self {
        Self {
            name,
            header,
            reader,
            raw_reader,
            superblock,
        }
    }

    /// The group's name (not the full path).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Access the object header.
    pub fn header(&self) -> &ObjectHeader {
        &self.header
    }

    /// List all child links in this group.
    pub async fn children(&self) -> Result<Vec<ChildLink>> {
        // Try v2 first: check for inline link messages
        let link_msgs = self.header.find_messages(msg_types::LINK);
        if !link_msgs.is_empty() {
            return self.children_from_link_messages(&link_msgs);
        }

        // Try v2 dense: check for link info message
        if let Some(link_info_msg) = self.header.find_message(msg_types::LINK_INFO) {
            return self.children_from_link_info(&link_info_msg.data).await;
        }

        // Try v1: check for symbol table message
        if let Some(sym_msg) = self.header.find_message(msg_types::SYMBOL_TABLE) {
            return self.children_from_symbol_table(&sym_msg.data).await;
        }

        // No children
        Ok(vec![])
    }

    /// Get a child group by name.
    pub async fn group(&self, name: &str) -> Result<HDF5Group> {
        let children = self.children().await?;
        let child = children
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| HDF5Error::NotFound(name.to_string()))?;

        let header = read_object_header(
            &self.reader,
            child.address,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await?;

        // Verify it's actually a group (has link messages, link info, or symbol table)
        let is_group = header.find_message(msg_types::LINK).is_some()
            || header.find_message(msg_types::LINK_INFO).is_some()
            || header.find_message(msg_types::SYMBOL_TABLE).is_some()
            // A group may also have no children but have group info
            || header.find_message(msg_types::GROUP_INFO).is_some()
            // An empty v2 group might only have nil messages
            || !header
                .messages
                .iter()
                .any(|m| m.msg_type == msg_types::DATASPACE);

        if !is_group {
            return Err(HDF5Error::NotAGroup(name.to_string()));
        }

        Ok(HDF5Group::new(
            name.to_string(),
            header,
            Arc::clone(&self.reader),
            Arc::clone(&self.raw_reader),
            Arc::clone(&self.superblock),
        ))
    }

    /// Get a child dataset by name.
    pub async fn dataset(&self, name: &str) -> Result<HDF5Dataset> {
        let children = self.children().await?;
        let child = children
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| HDF5Error::NotFound(name.to_string()))?;

        let header = read_object_header(
            &self.reader,
            child.address,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await?;

        // Verify it's a dataset (has dataspace message)
        if header.find_message(msg_types::DATASPACE).is_none() {
            return Err(HDF5Error::NotADataset(name.to_string()));
        }

        HDF5Dataset::new(
            name.to_string(),
            header,
            Arc::clone(&self.reader),
            Arc::clone(&self.raw_reader),
            Arc::clone(&self.superblock),
        )
    }

    /// Get a child's object header by name (returns the header without
    /// assuming whether it's a group or dataset).
    pub async fn child_header(&self, name: &str) -> Result<(u64, ObjectHeader)> {
        let children = self.children().await?;
        let child = children
            .iter()
            .find(|c| c.name == name)
            .ok_or_else(|| HDF5Error::NotFound(name.to_string()))?;

        let header = read_object_header(
            &self.reader,
            child.address,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await?;

        Ok((child.address, header))
    }

    /// Navigate to a group by slash-separated path (e.g., "science/LSAR/GCOV").
    pub async fn navigate(&self, path: &str) -> Result<HDF5Group> {
        let parts: Vec<&str> = path
            .trim_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();

        let mut current = HDF5Group::new(
            self.name.clone(),
            self.header.clone(),
            Arc::clone(&self.reader),
            Arc::clone(&self.raw_reader),
            Arc::clone(&self.superblock),
        );

        for part in parts {
            current = current.group(part).await?;
        }

        Ok(current)
    }

    /// List all child group names.
    pub async fn group_names(&self) -> Result<Vec<String>> {
        let children = self.children().await?;
        let mut group_names = Vec::new();

        for child in &children {
            let header = read_object_header(
                &self.reader,
                child.address,
                self.superblock.size_of_offsets,
                self.superblock.size_of_lengths,
            )
            .await?;

            // Check if this is a group (has group-like messages, does NOT have dataspace)
            let has_dataspace = header.find_message(msg_types::DATASPACE).is_some();
            if !has_dataspace {
                group_names.push(child.name.clone());
            }
        }

        Ok(group_names)
    }

    /// List all child dataset names.
    pub async fn dataset_names(&self) -> Result<Vec<String>> {
        let children = self.children().await?;
        let mut dataset_names = Vec::new();

        for child in &children {
            let header = read_object_header(
                &self.reader,
                child.address,
                self.superblock.size_of_offsets,
                self.superblock.size_of_lengths,
            )
            .await?;

            // Datasets have a dataspace message
            if header.find_message(msg_types::DATASPACE).is_some() {
                dataset_names.push(child.name.clone());
            }
        }

        Ok(dataset_names)
    }

    /// Access the reader.
    pub fn reader(&self) -> &Arc<dyn AsyncFileReader> {
        &self.reader
    }

    /// Access the superblock.
    pub fn superblock(&self) -> &Arc<Superblock> {
        &self.superblock
    }

    /// Get all attributes attached to this group, resolving vlen data.
    pub async fn attributes(&self) -> Vec<Attribute> {
        attributes_from_header(
            &self.header,
            &self.reader,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await
    }

    /// Get a single attribute by name.
    pub async fn attribute(&self, name: &str) -> Option<Attribute> {
        self.attributes().await.into_iter().find(|a| a.name == name)
    }

    // ── Private helpers ────────────────────────────────────────────────────

    /// Extract children from inline v2 link messages.
    fn children_from_link_messages(
        &self,
        link_msgs: &[&crate::object_header::HeaderMessage],
    ) -> Result<Vec<ChildLink>> {
        let mut children = Vec::with_capacity(link_msgs.len());

        for msg in link_msgs {
            let link = LinkMessage::parse(
                &msg.data,
                self.superblock.size_of_offsets,
                self.superblock.size_of_lengths,
            )?;

            if link.link_type == LinkType::Hard {
                if let Some(addr) = link.target_address {
                    children.push(ChildLink {
                        name: link.name,
                        address: addr,
                    });
                }
            }
            // Skip soft/external links for now
        }

        Ok(children)
    }

    /// Extract children from dense link storage (link info → fractal heap + B-tree v2).
    async fn children_from_link_info(&self, data: &Bytes) -> Result<Vec<ChildLink>> {
        let link_info = LinkInfoMessage::parse(
            data,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )?;

        // Check if the fractal heap address is defined
        if HDF5Reader::is_undef_addr(
            link_info.fractal_heap_address,
            self.superblock.size_of_offsets,
        ) {
            return Ok(vec![]);
        }

        // Read the fractal heap
        let fheap = heap::fractal::FractalHeap::read(
            &self.reader,
            link_info.fractal_heap_address,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await?;

        // Read the B-tree v2 header for the name index
        let btree_header = btree::v2::BTreeV2Header::read(
            &self.reader,
            link_info.name_btree_address,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await?;

        // Collect all records from the B-tree
        let raw_records = btree::v2::collect_all_records(
            &self.reader,
            &btree_header,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await?;

        // Parse link records
        let link_records =
            btree::v2::parse_link_records(&raw_records, btree_header.record_type)?;

        // Resolve each link record through the fractal heap
        let mut children = Vec::with_capacity(link_records.len());
        for record in &link_records {
            let link_msg_bytes = fheap.get_object(&record.heap_id).await?;
            let link = LinkMessage::parse(
                &link_msg_bytes,
                self.superblock.size_of_offsets,
                self.superblock.size_of_lengths,
            )?;

            if link.link_type == LinkType::Hard {
                if let Some(addr) = link.target_address {
                    children.push(ChildLink {
                        name: link.name,
                        address: addr,
                    });
                }
            }
        }

        Ok(children)
    }

    /// Extract children from v1 symbol table (B-tree v1 + local heap).
    async fn children_from_symbol_table(&self, data: &Bytes) -> Result<Vec<ChildLink>> {
        let sym_table = SymbolTableMessage::parse(
            data,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )?;

        // Read the local heap
        let local_heap = heap::local::LocalHeap::read(
            &self.reader,
            sym_table.local_heap_address,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await?;

        // Traverse the B-tree v1
        let entries = btree::v1::read_group_btree_v1(
            &self.reader,
            sym_table.btree_address,
            &local_heap,
            self.superblock.size_of_offsets,
            self.superblock.size_of_lengths,
        )
        .await?;

        Ok(entries
            .into_iter()
            .map(|e| ChildLink {
                name: e.name,
                address: e.object_header_address,
            })
            .collect())
    }
}

/// Extract decoded attributes from inline attribute messages in an object header.
///
/// Resolves variable-length data (e.g., vlen strings) via the global heap.
pub(crate) async fn attributes_from_header(
    header: &ObjectHeader,
    reader: &Arc<dyn AsyncFileReader>,
    size_of_offsets: u8,
    size_of_lengths: u8,
) -> Vec<Attribute> {
    let mut attrs = Vec::new();
    for msg in header.find_messages(msg_types::ATTRIBUTE) {
        if let Ok(am) = AttributeMessage::parse(&msg.data, size_of_offsets, size_of_lengths) {
            match am
                .to_attribute_resolved(reader, size_of_offsets, size_of_lengths)
                .await
            {
                Ok(attr) => attrs.push(attr),
                Err(_) => {
                    // Fall back to non-resolved decode
                    attrs.push(am.to_attribute());
                }
            }
        }
    }
    attrs
}
