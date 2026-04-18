/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use bit_set::BitSet;
// use std::fmt::Debug;
use serde::{Deserialize, Serialize};
use std::{
    fmt::Debug,
    fs::File,
    io::{BufReader, BufWriter},
    path::Path,
};

use diskann::{graph::index::QueryLabelProvider, utils::VectorId};
use diskann_benchmark_runner::files::InputFile;
use diskann_label_filter::{
    kv_index::GenericIndex,
    stores::bftree_store::BfTreeStore,
    traits::{
        posting_list_trait::{PostingList, RoaringPostingList},
        query_evaluator::QueryEvaluator,
    },
    ASTExpr, DefaultKeyCodec,
};
use diskann_providers::model::graph::provider::layers::BetaFilter;

use diskann_tools::utils::ground_truth::read_labels_and_compute_bitmap;
use std::sync::Arc;

pub struct QueryBitmapEvaluator {
    pub ast_expr: ASTExpr,
    evaluated_bitmap: RoaringPostingList,
}

impl QueryBitmapEvaluator {
    /// Create a new filter and evaluate the bitmap immediately (existing behavior).
    pub fn new(
        ast_expr: ASTExpr,
        inverted_index: &GenericIndex<BfTreeStore, RoaringPostingList, DefaultKeyCodec>,
    ) -> Self {
        let evaluated_bitmap = inverted_index.evaluate_query(&ast_expr).unwrap();
        Self {
            ast_expr,
            evaluated_bitmap,
        }
    }

    /// Ensure evaluated and return a reference to the bitmap (convenience).
    fn get_bitmap(&self) -> &RoaringPostingList {
        &self.evaluated_bitmap
    }

    /// Number of matching labels in this filter's evaluated bitmap.
    pub fn count(&self) -> usize {
        self.get_bitmap().len()
    }
}

impl Debug for QueryBitmapEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitmapFilter")
            .field("ast_expr", &self.ast_expr)
            .field("evaluated_bitmap", &self.evaluated_bitmap)
            .finish()
    }
}

impl<T> QueryLabelProvider<T> for QueryBitmapEvaluator
where
    T: VectorId,
{
    fn is_match(&self, vec_id: T) -> bool {
        self.get_bitmap().contains(vec_id.into_usize())
    }
}

#[derive(Debug)]
pub struct BitmapFilter(pub BitSet);

impl<T> QueryLabelProvider<T> for BitmapFilter
where
    T: VectorId,
{
    fn is_match(&self, vec_id: T) -> bool {
        self.0.contains(vec_id.into_usize())
    }
}

#[derive(Serialize, Deserialize)]
struct SerializableBitSet(Vec<u8>);

impl From<&BitSet> for SerializableBitSet {
    fn from(value: &BitSet) -> Self {
        Self(value.get_ref().to_bytes())
    }
}

impl From<SerializableBitSet> for BitSet {
    fn from(value: SerializableBitSet) -> Self {
        BitSet::from_bytes(&value.0)
    }
}

pub(crate) fn generate_bitmaps(
    query_predicates: &InputFile,
    data_labels: &InputFile,
    bitmap_path: Option<&str>,
) -> anyhow::Result<Vec<BitSet>> {
    if let Some(bitmap_path) = bitmap_path {
        let bitmap_path = Path::new(bitmap_path);
        println!(
            "Bitmap cache check: query_labels={}, base_labels={}, cache={}",
            query_predicates.display(),
            data_labels.display(),
            bitmap_path.display()
        );

        if bitmap_path.is_file() {
            let reader = BufReader::new(File::open(bitmap_path)?);
            let serialized: Vec<SerializableBitSet> = bincode::deserialize_from(reader)?;
            let loaded: Vec<BitSet> = serialized.into_iter().map(Into::into).collect();
            println!("Loaded bitmap cache from {}", bitmap_path.display());
            return Ok(loaded);
        }
    }

    let bit_maps = read_labels_and_compute_bitmap(
        data_labels.to_str().unwrap(),
        query_predicates.to_str().unwrap(),
    )?;

    if let Some(bitmap_path) = bitmap_path {
        let bitmap_path = Path::new(bitmap_path);
        let writer = BufWriter::new(File::create(bitmap_path)?);
        let serialized: Vec<SerializableBitSet> =
            bit_maps.iter().map(SerializableBitSet::from).collect();
        bincode::serialize_into(writer, &serialized)?;

        println!(
            "Generated bitmaps from query/base labels and wrote cache to {}",
            bitmap_path.display()
        );
    } else {
        println!("Generated bitmaps from query/base labels without writing a cache file");
    }

    Ok(bit_maps)
}

pub(crate) fn setup_filter_strategies<I, S>(
    beta: f32,
    bit_maps: I,
    search_strategy: S,
) -> Vec<BetaFilter<S, u32>>
where
    I: IntoIterator<Item = Arc<dyn QueryLabelProvider<u32>>>,
    S: Clone,
{
    bit_maps
        .into_iter()
        .map(|bit_map| BetaFilter::<S, u32>::new(search_strategy.clone(), bit_map, beta))
        .collect::<Vec<_>>()
}

pub(crate) fn as_query_label_provider(set: BitSet) -> Arc<dyn QueryLabelProvider<u32>> {
    Arc::new(BitmapFilter(set))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitmap_filter_match() {
        let mut bitset = BitSet::new();
        bitset.insert(1);
        bitset.insert(3);
        let filter = BitmapFilter(bitset);

        assert!(filter.is_match(1u32));
        assert!(filter.is_match(3u32));
        assert!(!filter.is_match(2u32));
        assert!(!filter.is_match(0u32));
    }

    #[test]
    fn test_bitmap_filter_empty() {
        let bitset = BitSet::new();
        let filter = BitmapFilter(bitset);

        assert!(!filter.is_match(0u32));
        assert!(!filter.is_match(10u32));
    }

    #[test]
    fn test_bitmap_filter_large_id() {
        let mut bitset = BitSet::new();
        bitset.insert(1000);
        let filter = BitmapFilter(bitset);

        assert!(filter.is_match(1000u32));
        assert!(!filter.is_match(999u32));
    }
}
