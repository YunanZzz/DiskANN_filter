/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::sync::Arc;

use diskann::{
    ANNResult,
    graph::{self, glue},
    provider,
};
use diskann_utils::{future::AsyncFriendly, views::Matrix};

use crate::search::{self, Search, graph::Strategy};

/// Development-only helper for benchmarking filtered k-nearest neighbors search
/// using the multi-hop search method.
#[derive(Debug)]
pub struct MultiHopDev<DP, T, S>
where
    DP: provider::DataProvider,
{
    index: Arc<graph::DiskANNIndex<DP>>,
    queries: Arc<Matrix<T>>,
    strategy: Strategy<S>,
    labels: Arc<[Arc<super::knn::CountingLabelProvider<DP::InternalId>>]>,
}

impl<DP, T, S> MultiHopDev<DP, T, S>
where
    DP: provider::DataProvider,
{
    /// Construct a new [`MultiHopDev`] searcher.
    pub fn new(
        index: Arc<graph::DiskANNIndex<DP>>,
        queries: Arc<Matrix<T>>,
        strategy: Strategy<S>,
        labels: Arc<[Arc<super::knn::CountingLabelProvider<DP::InternalId>>]>,
    ) -> anyhow::Result<Arc<Self>> {
        strategy.length_compatible(queries.nrows())?;

        if labels.len() != queries.nrows() {
            Err(anyhow::anyhow!(
                "Number of label providers ({}) must be equal to the number of queries ({})",
                labels.len(),
                queries.nrows()
            ))
        } else {
            Ok(Arc::new(Self {
                index,
                queries,
                strategy,
                labels,
            }))
        }
    }
}

impl<DP, T, S> Search for MultiHopDev<DP, T, S>
where
    DP: provider::DataProvider<Context: Default, ExternalId: search::Id>,
    S: for<'a> glue::DefaultSearchStrategy<DP, &'a [T], DP::ExternalId> + Clone + AsyncFriendly,
    T: AsyncFriendly + Clone,
{
    type Id = DP::ExternalId;
    type Parameters = graph::search::Knn;
    type Output = super::knn::Metrics;

    fn num_queries(&self) -> usize {
        self.queries.nrows()
    }

    fn id_count(&self, parameters: &Self::Parameters) -> search::IdCount {
        search::IdCount::Fixed(parameters.k_value())
    }

    async fn search<O>(
        &self,
        parameters: &Self::Parameters,
        buffer: &mut O,
        index: usize,
    ) -> ANNResult<Self::Output>
    where
        O: graph::SearchOutputBuffer<DP::ExternalId> + Send,
    {
        let context = DP::Context::default();
        let bitmap_checks_before = self.labels[index].check_count();
        let multihop_search =
            graph::search::MultihopSearchDev::new(*parameters, &*self.labels[index]);
        let stats = self
            .index
            .search(
                multihop_search,
                self.strategy.get(index)?,
                &context,
                self.queries.row(index),
                buffer,
            )
            .await?;

        Ok(super::knn::Metrics {
            comparisons: stats.cmps,
            hops: stats.hops,
            bitmap_checks: self.labels[index].check_count() - bitmap_checks_before,
        })
    }
}
