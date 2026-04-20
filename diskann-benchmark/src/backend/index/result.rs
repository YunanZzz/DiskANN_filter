/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    fs::{File, create_dir_all},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};

use diskann_benchmark_core as benchmark_core;
use diskann_benchmark_runner::utils::{percentiles, MicroSeconds};
use serde::Serialize;

use crate::{
    backend::index::build::BuildStats,
    utils::{self, DisplayWrapper, MaybeDisplay},
};

//////////////////
// BuildResult  //
//////////////////
#[derive(Debug, Serialize)]
pub(super) struct BuildResult {
    pub(super) build: Option<BuildStats>,
    pub(super) search: AggregatedSearchResults,
}

impl BuildResult {
    pub(super) fn new_topk(build: Option<BuildStats>) -> Self {
        Self {
            build,
            search: AggregatedSearchResults::Topk(Vec::new()),
        }
    }

    pub(super) fn new_range(build: Option<BuildStats>) -> Self {
        Self {
            build,
            search: AggregatedSearchResults::Range(Vec::new()),
        }
    }

    pub(super) fn append(&mut self, search: AggregatedSearchResults) {
        self.search.append(search);
    }
}

impl std::fmt::Display for BuildResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(ref build) = self.build {
            write!(f, "{}", build)?;
        }

        self.search.fmt(f)?;

        Ok(())
    }
}

//////////////////////
// QuantBuildResult //
//////////////////////

#[cfg(any(feature = "product-quantization", feature = "scalar-quantization",))]
#[derive(Debug, Serialize)]
pub(super) struct QuantBuildResult {
    pub(super) quant_training_time: MicroSeconds,
    pub(super) build: BuildResult,
}

#[cfg(any(feature = "product-quantization", feature = "scalar-quantization",))]
impl std::fmt::Display for QuantBuildResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Quant Training Time: {}s",
            self.quant_training_time.as_seconds()
        )?;
        self.build.fmt(f)
    }
}

///////////////////
// SearchResults //
///////////////////

#[derive(Debug, Serialize)]
pub(super) enum AggregatedSearchResults {
    Topk(Vec<SearchResults>),
    Range(Vec<RangeSearchResults>),
}

impl AggregatedSearchResults {
    pub(super) fn append(&mut self, search: AggregatedSearchResults) {
        match (self, search) {
            (Self::Topk(v), AggregatedSearchResults::Topk(s)) => v.extend(s),
            (Self::Range(v), AggregatedSearchResults::Range(s)) => v.extend(s),
            _ => panic!("Mismatched search result types"),
        }
    }
}

impl std::fmt::Display for AggregatedSearchResults {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Topk(v) => write!(f, "{}", DisplayWrapper(v.as_slice()))?,
            Self::Range(v) => write!(f, "{}", DisplayWrapper(v.as_slice()))?,
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub(super) struct SearchResults {
    pub(super) num_tasks: usize,
    pub(super) search_n: usize,
    pub(super) search_l: usize,
    pub(super) qps: Vec<f64>,
    pub(super) search_latencies: Vec<MicroSeconds>,
    pub(super) mean_latencies: Vec<f64>,
    pub(super) p90_latencies: Vec<MicroSeconds>,
    pub(super) p99_latencies: Vec<MicroSeconds>,
    pub(super) recall: utils::recall::RecallMetrics,
    pub(super) mean_cmps: f32,
    pub(super) mean_hops: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) query_traces: Option<Vec<QueryTrace>>,
}

#[derive(Debug, Serialize, Clone)]
pub(super) struct HopTrace {
    pub(super) hop_index: u32,
    pub(super) one_hop_enqueued: Vec<u32>,
    pub(super) two_hop_enqueued: Vec<u32>,
}

#[derive(Debug, Serialize, Clone)]
pub(super) struct QueryTrace {
    pub(super) query_index: usize,
    pub(super) hops: Vec<HopTrace>,
}

#[derive(Debug, Serialize)]
struct QueryTraceRecord<'a> {
    query_file: &'a str,
    context: Option<&'a str>,
    num_tasks: usize,
    search_n: usize,
    search_l: usize,
    query_index: usize,
    hops: &'a [HopTrace],
}

const TRACE_OUTPUT_DIR: &str = "/scratch1/zhan4404/dataset/caselaw/new_query/trace";

impl SearchResults {
    pub fn new(summary: benchmark_core::search::graph::knn::Summary) -> Self {
        let benchmark_core::search::graph::knn::Summary {
            setup,
            parameters,
            end_to_end_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            recall,
            mean_cmps,
            mean_hops,
            ..
        } = summary;

        let qps = end_to_end_latencies
            .iter()
            .map(|latency| recall.num_queries as f64 / latency.as_seconds())
            .collect();

        Self {
            num_tasks: setup.tasks.into(),
            search_n: parameters.k_value().get(),
            search_l: parameters.l_value().get(),
            qps,
            search_latencies: end_to_end_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            recall: (&recall).into(),
            mean_cmps: mean_cmps as f32,
            mean_hops: mean_hops as f32,
            query_traces: None,
        }
    }

    pub fn new_multihop_dev(summary: benchmark_core::search::graph::multihop_dev::Summary<u32>) -> Self {
        let benchmark_core::search::graph::multihop_dev::Summary {
            setup,
            parameters,
            end_to_end_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            recall,
            mean_cmps,
            mean_hops,
            query_traces,
        } = summary;

        let qps = end_to_end_latencies
            .iter()
            .map(|latency| recall.num_queries as f64 / latency.as_seconds())
            .collect();

        Self {
            num_tasks: setup.tasks.into(),
            search_n: parameters.k_value().get(),
            search_l: parameters.l_value().get(),
            qps,
            search_latencies: end_to_end_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            recall: (&recall).into(),
            mean_cmps: mean_cmps as f32,
            mean_hops: mean_hops as f32,
            query_traces: Some(
                query_traces
                    .into_iter()
                    .map(|trace| QueryTrace {
                        query_index: trace.query_index,
                        hops: trace
                            .trace
                            .hops
                            .into_iter()
                            .map(|hop| HopTrace {
                                hop_index: hop.hop_index,
                                one_hop_enqueued: hop.one_hop_enqueued,
                                two_hop_enqueued: hop.two_hop_enqueued,
                            })
                            .collect(),
                    })
                    .collect(),
            ),
        }
    }
}

pub(super) fn write_query_traces_jsonl(
    results: &[SearchResults],
    query_path: &Path,
    context: Option<&str>,
) -> anyhow::Result<Option<PathBuf>> {
    let has_traces = results
        .iter()
        .any(|result| result.query_traces.as_ref().is_some_and(|traces| !traces.is_empty()));
    if !has_traces {
        return Ok(None);
    }

    create_dir_all(TRACE_OUTPUT_DIR)?;

    let file_name = trace_file_name(query_path, context, results);
    let output_path = Path::new(TRACE_OUTPUT_DIR).join(file_name);
    let file = File::create(&output_path)?;
    let mut writer = BufWriter::new(file);
    let query_file = query_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown-query");

    for result in results {
        let Some(query_traces) = result.query_traces.as_ref() else {
            continue;
        };

        for trace in query_traces {
            let record = QueryTraceRecord {
                query_file,
                context,
                num_tasks: result.num_tasks,
                search_n: result.search_n,
                search_l: result.search_l,
                query_index: trace.query_index,
                hops: &trace.hops,
            };
            serde_json::to_writer(&mut writer, &record)?;
            writer.write_all(b"\n")?;
        }
    }

    writer.flush()?;
    Ok(Some(output_path))
}

fn trace_file_name(query_path: &Path, context: Option<&str>, results: &[SearchResults]) -> String {
    let base = query_path
        .file_stem()
        .and_then(|name| name.to_str())
        .or_else(|| query_path.file_name().and_then(|name| name.to_str()))
        .unwrap_or("query");
    let l_suffix = search_l_suffix(results);

    match context {
        Some(context) if !context.is_empty() => {
            format!(
                "{base}_{}_{}.trace.jsonl",
                l_suffix,
                sanitize_file_component(context)
            )
        }
        _ => format!("{base}_{}.trace.jsonl", l_suffix),
    }
}

fn search_l_suffix(results: &[SearchResults]) -> String {
    let mut values: Vec<usize> = results.iter().map(|result| result.search_l).collect();
    values.sort_unstable();
    values.dedup();

    match values.as_slice() {
        [] => "lunknown".to_string(),
        [single] => format!("l{single}"),
        many => format!(
            "ls{}",
            many.iter()
                .map(|value| value.to_string())
                .collect::<Vec<_>>()
                .join("-")
        ),
    }
}

fn sanitize_file_component(value: &str) -> String {
    value.chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' => ch,
            _ => '_',
        })
        .collect()
}

fn format_search_results_table<F>(
    f: &mut std::fmt::Formatter<'_>,
    results: &[SearchResults],
    batch_formatter: Option<F>,
) -> std::fmt::Result
where
    F: Fn(usize) -> String,
{
    if results.is_empty() {
        return Ok(());
    }

    let has_batch = batch_formatter.is_some();
    let headers: &[&str] = if has_batch {
        &[
            "Batch",
            "Ls",
            "KNN",
            "Avg cmps",
            "Avg hops",
            "QPS - mean(max)",
            "Avg Latency",
            "p99 Latency",
            "Recall",
            "Threads",
        ]
    } else {
        &[
            "Ls",
            "KNN",
            "Avg cmps",
            "Avg hops",
            "QPS - mean(max)",
            "Avg Latency",
            "p99 Latency",
            "Recall",
            "Threads",
        ]
    };

    let mut table = diskann_benchmark_runner::utils::fmt::Table::new(headers, results.len());
    results.iter().enumerate().for_each(|(i, r)| {
        let mut row = table.row(i);
        let mut col_idx = 0;

        if let Some(ref formatter) = batch_formatter {
            row.insert(formatter(i), col_idx);
            col_idx += 1;
        }

        row.insert(r.search_l, col_idx);
        row.insert(r.search_n, col_idx + 1);
        row.insert(r.mean_cmps, col_idx + 2);
        row.insert(r.mean_hops, col_idx + 3);
        row.insert(
            format!(
                "{:.1} ({:.1})",
                MaybeDisplay(percentiles::mean(&r.qps), "missing"),
                MaybeDisplay(percentiles::max_f64(&r.qps), "missing"),
            ),
            col_idx + 4,
        );
        row.insert(
            format!(
                "{:.1}us ({:.1}us)",
                MaybeDisplay(percentiles::mean(&r.mean_latencies), "missing"),
                MaybeDisplay(percentiles::max_f64(&r.mean_latencies), "missing"),
            ),
            col_idx + 5,
        );
        row.insert(
            format!(
                "{:.1}us ({:.1})",
                MaybeDisplay(percentiles::mean(&r.p99_latencies), "missing"),
                MaybeDisplay(r.p99_latencies.iter().max(), "missing"),
            ),
            col_idx + 6,
        );
        row.insert(format!("{:3}", r.recall.average), col_idx + 7);
        row.insert(r.num_tasks, col_idx + 8);
    });

    write!(f, "{}", table)
}

impl std::fmt::Display for DisplayWrapper<'_, [SearchResults]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format_search_results_table(f, self, None::<fn(usize) -> String>)
    }
}

////////////////////////
// RangeSearchResults //
////////////////////////

#[derive(Debug, Serialize)]
pub(super) struct RangeSearchResults {
    pub(super) num_tasks: usize,
    pub(super) initial_l: usize,
    pub(super) qps: Vec<f64>,
    pub(super) search_latencies: Vec<MicroSeconds>,
    pub(super) mean_latencies: Vec<f64>,
    pub(super) p90_latencies: Vec<MicroSeconds>,
    pub(super) p99_latencies: Vec<MicroSeconds>,
    pub(super) average_precision: utils::recall::AveragePrecisionMetrics,
}

impl RangeSearchResults {
    pub fn new(summary: benchmark_core::search::graph::range::Summary) -> Self {
        let benchmark_core::search::graph::range::Summary {
            setup,
            parameters,
            end_to_end_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            average_precision,
            ..
        } = summary;

        let qps = end_to_end_latencies
            .iter()
            .map(|latency| average_precision.num_queries as f64 / latency.as_seconds())
            .collect();

        Self {
            num_tasks: setup.tasks.into(),
            initial_l: parameters.starting_l(),
            qps,
            search_latencies: end_to_end_latencies,
            mean_latencies,
            p90_latencies,
            p99_latencies,
            average_precision: (&average_precision).into(),
        }
    }
}

fn format_range_search_results_table<F>(
    f: &mut std::fmt::Formatter<'_>,
    results: &[RangeSearchResults],
    batch_formatter: Option<F>,
) -> std::fmt::Result
where
    F: Fn(usize) -> String,
{
    if results.is_empty() {
        return Ok(());
    }

    let has_batch = batch_formatter.is_some();
    let headers: &[_] = if has_batch {
        &[
            "Batch",
            "initial Ls",
            "QPS - mean(max)",
            "Avg Latency",
            "p99 Latency",
            "Average Precision",
            "Threads",
        ]
    } else {
        &[
            "initial Ls",
            "QPS - mean(max)",
            "Avg Latency",
            "p99 Latency",
            "Average Precision",
            "Threads",
        ]
    };

    let mut table = diskann_benchmark_runner::utils::fmt::Table::new(headers, results.len());
    results.iter().enumerate().for_each(|(i, r)| {
        let mut row = table.row(i);
        let mut col_idx = 0;

        if let Some(ref formatter) = batch_formatter {
            row.insert(formatter(i), col_idx);
            col_idx += 1;
        }

        row.insert(r.initial_l, col_idx);
        row.insert(
            format!(
                "{:.1} ({:.1})",
                MaybeDisplay(percentiles::mean(&r.qps), "missing"),
                MaybeDisplay(percentiles::max_f64(&r.qps), "missing"),
            ),
            col_idx + 1,
        );
        row.insert(
            format!(
                "{:.1}us ({:.1}us)",
                MaybeDisplay(percentiles::mean(&r.mean_latencies), "missing"),
                MaybeDisplay(percentiles::max_f64(&r.mean_latencies), "missing"),
            ),
            col_idx + 2,
        );
        row.insert(
            format!(
                "{:.1}us ({:.1})",
                MaybeDisplay(percentiles::mean(&r.p99_latencies), "missing"),
                MaybeDisplay(r.p99_latencies.iter().max(), "missing"),
            ),
            col_idx + 3,
        );
        row.insert(
            format!("{:3}", r.average_precision.average_precision),
            col_idx + 4,
        );
        row.insert(r.num_tasks, col_idx + 5);
    });

    write!(f, "{}", table)
}

impl std::fmt::Display for DisplayWrapper<'_, [RangeSearchResults]> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format_range_search_results_table(f, self, None::<fn(usize) -> String>)
    }
}
