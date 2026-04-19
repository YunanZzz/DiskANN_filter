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
use diskann_benchmark_runner::utils::{MicroSeconds, percentiles};
use diskann_utils::{future::AsyncFriendly, views::Matrix};

use crate::{
    recall,
    search::{self, Search, graph::Strategy},
    utils,
};

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
    labels: Arc<[Arc<dyn graph::index::QueryLabelProvider<DP::InternalId>>]>,
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
        labels: Arc<[Arc<dyn graph::index::QueryLabelProvider<DP::InternalId>>]>,
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

/// Additional metrics collected during dev multihop search.
#[derive(Debug, Clone)]
pub struct Metrics<I> {
    pub comparisons: u32,
    pub hops: u32,
    pub query_index: usize,
    pub trace: graph::search::QueryTrace<I>,
}

/// Aggregated summary for dev multihop search runs.
#[derive(Debug, Clone)]
pub struct Summary<I> {
    pub setup: search::Setup,
    pub parameters: graph::search::Knn,
    pub end_to_end_latencies: Vec<MicroSeconds>,
    pub mean_latencies: Vec<f64>,
    pub p90_latencies: Vec<MicroSeconds>,
    pub p99_latencies: Vec<MicroSeconds>,
    pub recall: recall::RecallMetrics,
    pub mean_cmps: f64,
    pub mean_hops: f64,
    pub query_traces: Vec<QueryTraceWithIndex<I>>,
}

/// Query trace annotated with the query index.
#[derive(Debug, Clone)]
pub struct QueryTraceWithIndex<I> {
    pub query_index: usize,
    pub trace: graph::search::QueryTrace<I>,
}

pub struct Aggregator<'a, I> {
    groundtruth: &'a dyn crate::recall::Rows<I>,
    recall_k: usize,
    recall_n: usize,
}

impl<'a, I> Aggregator<'a, I> {
    pub fn new(
        groundtruth: &'a dyn crate::recall::Rows<I>,
        recall_k: usize,
        recall_n: usize,
    ) -> Self {
        Self {
            groundtruth,
            recall_k,
            recall_n,
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
    type Output = Metrics<DP::InternalId>;

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
        let multihop_search =
            graph::search::MultihopSearchDev::new(*parameters, &*self.labels[index]);
        let output = self
            .index
            .search(
                multihop_search,
                self.strategy.get(index)?,
                &context,
                self.queries.row(index),
                buffer,
            )
            .await?;

        Ok(Metrics {
            comparisons: output.stats.cmps,
            hops: output.stats.hops,
            query_index: index,
            trace: output.trace,
        })
    }
}

impl<I, J> search::Aggregate<graph::search::Knn, I, Metrics<J>> for Aggregator<'_, I>
where
    I: crate::recall::RecallCompatible,
    J: Clone + Send + Sync + 'static,
{
    type Output = Summary<J>;

    fn aggregate(
        &mut self,
        run: search::Run<graph::search::Knn>,
        mut results: Vec<search::SearchResults<I, Metrics<J>>>,
    ) -> anyhow::Result<Summary<J>> {
        let recall = match results.first() {
            Some(first) => crate::recall::knn(
                self.groundtruth,
                None,
                first.ids().as_rows(),
                self.recall_k,
                self.recall_n,
                true,
            )?,
            None => anyhow::bail!("Results must be non-empty"),
        };

        let query_traces = match results.first() {
            Some(first) => first
                .output()
                .iter()
                .map(|o| QueryTraceWithIndex {
                    query_index: o.query_index,
                    trace: o.trace.clone(),
                })
                .collect(),
            None => Vec::new(),
        };

        let mut mean_latencies = Vec::with_capacity(results.len());
        let mut p90_latencies = Vec::with_capacity(results.len());
        let mut p99_latencies = Vec::with_capacity(results.len());

        results.iter_mut().for_each(|r| {
            match percentiles::compute_percentiles(r.latencies_mut()) {
                Ok(values) => {
                    let percentiles::Percentiles { mean, p90, p99, .. } = values;
                    mean_latencies.push(mean);
                    p90_latencies.push(p90);
                    p99_latencies.push(p99);
                }
                Err(_) => {
                    let zero = MicroSeconds::new(0);
                    mean_latencies.push(0.0);
                    p90_latencies.push(zero);
                    p99_latencies.push(zero);
                }
            }
        });

        Ok(Summary {
            setup: run.setup().clone(),
            parameters: *run.parameters(),
            end_to_end_latencies: results.iter().map(|r| r.end_to_end_latency()).collect(),
            recall,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            mean_cmps: utils::average_all(
                results
                    .iter()
                    .flat_map(|r| r.output().iter().map(|o| o.comparisons)),
            ),
            mean_hops: utils::average_all(
                results.iter().flat_map(|r| r.output().iter().map(|o| o.hops)),
            ),
            query_traces,
        })
    }
}
