/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Development copy of label-filtered search using multi-hop expansion.
//!
//! This file intentionally mirrors `multihop_search.rs` so the dev search path
//! can evolve independently without changing the production multihop behavior.

use diskann_utils::Reborrow;
use diskann_utils::future::SendFuture;
use diskann_vector::PreprocessedDistanceFunction;
use hashbrown::HashSet;

use super::{Knn, Search, record::SearchRecord, scratch::SearchScratch};
use crate::{
    ANNResult,
    error::{ErrorExt, IntoANNResult},
    graph::{
        glue::{
            self, ExpandBeam, HybridPredicate, Predicate, PredicateMut, SearchExt,
            SearchPostProcess, SearchStrategy,
        },
        index::{
            DiskANNIndex, InternalSearchStats, QueryLabelProvider, QueryVisitDecision, SearchStats,
        },
        search::record::NoopSearchRecord,
        search_output_buffer::SearchOutputBuffer,
    },
    neighbor::Neighbor,
    provider::{BuildQueryComputer, DataProvider},
    utils::VectorId,
};

/// Parameters for development-only label-filtered search using multi-hop expansion.
#[derive(Debug)]
pub struct MultihopSearchDev<'q, InternalId> {
    /// Base graph search parameters.
    pub inner: Knn,
    /// Label evaluator for determining node matches.
    pub label_evaluator: &'q dyn QueryLabelProvider<InternalId>,
}

impl<'q, InternalId> MultihopSearchDev<'q, InternalId> {
    /// Create new development multihop search parameters.
    pub fn new(inner: Knn, label_evaluator: &'q dyn QueryLabelProvider<InternalId>) -> Self {
        Self {
            inner,
            label_evaluator,
        }
    }
}

impl<'q, DP, S, T> Search<DP, S, T> for MultihopSearchDev<'q, DP::InternalId>
where
    DP: DataProvider,
    S: SearchStrategy<DP, T>,
    T: Copy + Send + Sync,
{
    type Output = SearchStats;

    fn search<O, PP, OB>(
        self,
        index: &DiskANNIndex<DP>,
        strategy: &S,
        processor: PP,
        context: &DP::Context,
        query: T,
        output: &mut OB,
    ) -> impl SendFuture<ANNResult<Self::Output>>
    where
        O: Send,
        PP: for<'a> SearchPostProcess<S::SearchAccessor<'a>, T, O> + Send + Sync,
        OB: SearchOutputBuffer<O> + Send + ?Sized,
    {
        async move {
            let mut accessor = strategy
                .search_accessor(&index.data_provider, context)
                .into_ann_result()?;
            let computer = accessor.build_query_computer(query).into_ann_result()?;

            let start_ids = accessor.starting_points().await?;

            let mut scratch = index.search_scratch(self.inner.l_value().get(), start_ids.len());

            let stats = multihop_search_internal_dev(
                index.max_degree_with_slack(),
                &self.inner,
                &mut accessor,
                &computer,
                &mut scratch,
                &mut NoopSearchRecord::new(),
                self.label_evaluator,
            )
            .await?;

            let result_count = processor
                .post_process(
                    &mut accessor,
                    query,
                    &computer,
                    scratch.best.iter().take(self.inner.l_value().get()),
                    output,
                )
                .await
                .into_ann_result()?;

            Ok(stats.finish(result_count as u32))
        }
    }
}

/// A predicate that checks if an item is not in the visited set AND matches the label filter.
pub struct NotInMutWithLabelCheckDev<'a, K>
where
    K: VectorId,
{
    visited_set: &'a mut HashSet<K>,
    query_label_evaluator: &'a dyn QueryLabelProvider<K>,
}

impl<'a, K> NotInMutWithLabelCheckDev<'a, K>
where
    K: VectorId,
{
    /// Construct a new `NotInMutWithLabelCheckDev` around `visited_set`.
    pub fn new(
        visited_set: &'a mut HashSet<K>,
        query_label_evaluator: &'a dyn QueryLabelProvider<K>,
    ) -> Self {
        Self {
            visited_set,
            query_label_evaluator,
        }
    }
}

impl<K> Predicate<K> for NotInMutWithLabelCheckDev<'_, K>
where
    K: VectorId,
{
    fn eval(&self, item: &K) -> bool {
        !self.visited_set.contains(item) && self.query_label_evaluator.is_match(*item)
    }
}

impl<K> PredicateMut<K> for NotInMutWithLabelCheckDev<'_, K>
where
    K: VectorId,
{
    fn eval_mut(&mut self, item: &K) -> bool {
        if self.query_label_evaluator.is_match(*item) {
            return self.visited_set.insert(*item);
        }
        false
    }
}

impl<K> HybridPredicate<K> for NotInMutWithLabelCheckDev<'_, K> where K: VectorId {}

/// Development copy of the internal multihop search implementation.
pub(crate) async fn multihop_search_internal_dev<I, A, T, SR>(
    max_degree_with_slack: usize,
    search_params: &Knn,
    accessor: &mut A,
    computer: &A::QueryComputer,
    scratch: &mut SearchScratch<I>,
    search_record: &mut SR,
    query_label_evaluator: &dyn QueryLabelProvider<I>,
) -> ANNResult<InternalSearchStats>
where
    I: VectorId,
    A: ExpandBeam<T, Id = I> + SearchExt,
    SR: SearchRecord<I> + ?Sized,
{
    let beam_width = search_params.beam_width().get();

    let make_stats = |scratch: &SearchScratch<I>| InternalSearchStats {
        cmps: scratch.cmps,
        hops: scratch.hops,
        range_search_second_round: false,
    };

    if scratch.visited.is_empty() {
        let start_ids = accessor.starting_points().await?;

        for id in start_ids {
            scratch.visited.insert(id);
            let element = accessor
                .get_element(id)
                .await
                .escalate("start point retrieval must succeed")?;
            let dist = computer.evaluate_similarity(element.reborrow());
            scratch.best.insert(Neighbor::new(id, dist));
        }
    }

    let mut one_hop_neighbors = Vec::with_capacity(max_degree_with_slack);
    let mut two_hop_neighbors = Vec::with_capacity(max_degree_with_slack);
    let mut candidates_two_hop_expansion = Vec::with_capacity(max_degree_with_slack);

    while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
        scratch.beam_nodes.clear();
        one_hop_neighbors.clear();
        candidates_two_hop_expansion.clear();
        two_hop_neighbors.clear();

        while scratch.beam_nodes.len() < beam_width
            && let Some(closest_node) = scratch.best.closest_notvisited()
        {
            search_record.record(closest_node, scratch.hops, scratch.cmps);
            scratch.beam_nodes.push(closest_node.id);
        }

        accessor
            .expand_beam(
                scratch.beam_nodes.iter().copied(),
                computer,
                glue::NotInMut::new(&mut scratch.visited),
                |distance, id| one_hop_neighbors.push(Neighbor::new(id, distance)),
            )
            .await?;

        for neighbor in one_hop_neighbors.iter().copied() {
            match query_label_evaluator.on_visit(neighbor) {
                QueryVisitDecision::Accept(accepted) => {
                    scratch.best.insert(accepted);
                }
                QueryVisitDecision::Reject => {
                    candidates_two_hop_expansion.push(neighbor);
                }
                QueryVisitDecision::Terminate => {
                    scratch.cmps += one_hop_neighbors.len() as u32;
                    scratch.hops += scratch.beam_nodes.len() as u32;
                    return Ok(make_stats(scratch));
                }
            }
        }

        scratch.cmps += one_hop_neighbors.len() as u32;
        scratch.hops += scratch.beam_nodes.len() as u32;

        candidates_two_hop_expansion.sort_unstable_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        candidates_two_hop_expansion.truncate(max_degree_with_slack / 2);

        let two_hop_expansion_candidate_ids: Vec<I> =
            candidates_two_hop_expansion.iter().map(|n| n.id).collect();

        accessor
            .expand_beam(
                two_hop_expansion_candidate_ids.iter().copied(),
                computer,
                NotInMutWithLabelCheckDev::new(&mut scratch.visited, query_label_evaluator),
                |distance, id| {
                    two_hop_neighbors.push(Neighbor::new(id, distance));
                },
            )
            .await?;

        two_hop_neighbors
            .iter()
            .for_each(|neighbor| scratch.best.insert(*neighbor));

        scratch.cmps += two_hop_neighbors.len() as u32;
        scratch.hops += two_hop_expansion_candidate_ids.len() as u32;
    }

    Ok(make_stats(scratch))
}
