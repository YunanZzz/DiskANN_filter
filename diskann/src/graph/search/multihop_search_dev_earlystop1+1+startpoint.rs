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
    utils::{TryIntoVectorId, VectorId},
};


// Development-only knobs for seeding extra matching start points into the initial queue.
// Set `EXTRA_MATCH_START_POINTS` to zero to disable the extra scan entirely.
const EXTRA_MATCH_START_POINTS: usize = 1;

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

/// Internal multihop search implementation.
///
/// Performs label-filtered search by expanding through non-matching nodes
/// to find matching neighbors within two hops.
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
    let mut has_entered_effective_region = false;
    let mut pending_stop = false;
    let k_value = search_params.k_value().get();

    // Helper to build the final stats from scratch state.
    let make_stats = |scratch: &SearchScratch<I>| InternalSearchStats {
        cmps: scratch.cmps,
        hops: scratch.hops,
        range_search_second_round: false,
    };

    // Initialize search state if not already initialized.
    // This allows paged search to call multihop_search_internal multiple times
    if scratch.visited.is_empty() {
        let start_ids = accessor.starting_points().await?;

        for id in start_ids.iter().copied() {
            scratch.visited.insert(id);
            let element = accessor
                .get_element(id)
                .await
                .escalate("start point retrieval must succeed")?;
            let dist = computer.evaluate_similarity(element.reborrow());
            scratch.best.insert(Neighbor::new(id, dist));
        }
        //early stop
        if EXTRA_MATCH_START_POINTS > 0 {
            let start_max_id = start_ids
                .iter()
                .copied()
                .max()
                .unwrap_or_default()
                .into_usize();
            let mut added = 0usize;

            // Always scan in reverse order from the max starting point id.
            for raw_id in (0..=start_max_id).rev() {
                if added >= EXTRA_MATCH_START_POINTS {
                    break;
                }

                let Ok(id) = raw_id.try_into_vector_id() else {
                    continue;
                };
                if scratch.visited.contains(&id) || !query_label_evaluator.is_match(id) {
                    continue;
                }

                let element = accessor
                    .get_element(id)
                    .await
                    .escalate("extra matching start point retrieval must succeed")?;
                let dist = computer.evaluate_similarity(element.reborrow());
                scratch.cmps += 1;
                scratch.visited.insert(id);
                scratch.best.insert(Neighbor::new(id, dist));
                added += 1;
            }
        }
    }

    // Pre-allocate with good capacity to avoid repeated allocations
    let mut one_hop_neighbors = Vec::with_capacity(max_degree_with_slack);
    let mut two_hop_neighbors = Vec::with_capacity(max_degree_with_slack);
    let mut candidates_two_hop_expansion = Vec::with_capacity(max_degree_with_slack);

    while scratch.best.has_notvisited_node() && !accessor.terminate_early() {
        let mut hop_match_count = 0usize;
        let mut hop_closest_distance = None;

        scratch.beam_nodes.clear();
        one_hop_neighbors.clear();
        candidates_two_hop_expansion.clear();
        two_hop_neighbors.clear();

        // In this loop we are going to find the beam_width number of nodes that are closest to the query.
        // Each of these nodes will be a frontier node.
        while scratch.beam_nodes.len() < beam_width
            && let Some(closest_node) = scratch.best.closest_notvisited()
        {
            if hop_closest_distance.is_none() {
                hop_closest_distance = Some(closest_node.distance);
            }
            search_record.record(closest_node, scratch.hops, scratch.cmps);
            scratch.beam_nodes.push(closest_node.id);
        }

        // compute distances from query to one-hop neighbors, and mark them visited
        accessor
            .expand_beam(
                scratch.beam_nodes.iter().copied(),
                computer,
                glue::NotInMut::new(&mut scratch.visited),
                |distance, id| one_hop_neighbors.push(Neighbor::new(id, distance)),
            )
            .await?;

        // Process one-hop neighbors based on on_visit() decision
        for neighbor in one_hop_neighbors.iter().copied() {
            match query_label_evaluator.on_visit(neighbor) {
                QueryVisitDecision::Accept(accepted) => {
                    scratch.best.insert(accepted);
                    hop_match_count += 1;
                }
                QueryVisitDecision::Reject => {
                    // Rejected nodes: still add to two-hop expansion so we can traverse through them
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

        if hop_match_count * 2 >= one_hop_neighbors.len() {
            has_entered_effective_region = true;
            continue;
        }

        // sort the candidates for two-hop expansion by distance to query point
        candidates_two_hop_expansion.sort_unstable_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // limit the number of two-hop candidates to avoid too many expansions
        candidates_two_hop_expansion.truncate(max_degree_with_slack / 2);

        // Expand each two-hop candidate: if its neighbor is a match, compute its distance
        // to the query and insert into `scratch.visited`
        // If it is not a match, do nothing
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

        // Next, insert the new matches into `scratch.best` and increment stats counters
        hop_match_count += two_hop_neighbors.len();
        two_hop_neighbors
            .iter()
            .for_each(|neighbor| scratch.best.insert(*neighbor));

        scratch.cmps += two_hop_neighbors.len() as u32;
        scratch.hops += two_hop_expansion_candidate_ids.len() as u32;

        if hop_match_count > 0 {
            has_entered_effective_region = true;
            pending_stop = false; // found new matches, cancel any pending stop
        } else if has_entered_effective_region {
            if scratch.best.size() > k_value {
                let kth_distance = scratch.best.get(k_value - 1).distance;
                if let Some(current_hop_distance) = hop_closest_distance
                    && current_hop_distance > kth_distance
                {
                    if pending_stop {
                        break;
                    }
                    pending_stop = true;
                }
            }
        }
    }

    Ok(make_stats(scratch))
}
