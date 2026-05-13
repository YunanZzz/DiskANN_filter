# Run Dev Filter Search

## Overview

The dev filter-search algorithm is integrated into the current `diskann-benchmark` workflow.

A new filter search type has been added:

```text
topk-multihop-filter-dev
```

The implementation is in:

```text
path/to/DiskANN/diskann/src/graph/search/multihop_search_dev.rs
```

Usage is the same as the existing filter-search modes, including:

```text
topk-multihop-filter
topk-beta-filter
```

In practice, select `topk-multihop-filter-dev` in the benchmark input JSON wherever the previous filter-search type would be configured.

## Additional Changes

### 1. Bitmap Cache Path in Input JSON

The benchmark input JSON can provide a bitmap cache file path.

At search startup:

1. If the bitmap cache file exists, it will be loaded into memory.
2. If the bitmap cache file does not exist, the benchmark will follow the default bitmap generation flow.
3. If the input JSON provides a bitmap cache path, the generated bitmap cache will be written to disk at that path.

This avoids rebuilding the bitmap cache repeatedly across runs.

### 2. Global Selectivity From Bitmap

Global selectivity is computed from the bitmap and can be used during search.

This value is available to the dev filter-search implementation and is intended to help tune search behavior based on the filter selectivity.

### 3. Bitmap Check Counter

Bitmap-check counting has been added to filter search.

This allows benchmark output to report how many bitmap checks were performed during filtered search, which is useful for analyzing the cost of different filter-search strategies.

## Example Input File

Use the existing async caselaw multihop filter-search example as a reference:

```text
path/to/DiskANN/diskann-benchmark/example/async-caselaw-multihop-search-filter.json
```

Set the search type in the input JSON to:

```json
"search-type": "topk-multihop-filter-dev"
```

And you can also set the bitmap file path:

```json
"bitmap": "path/to/caselaw_query_single_vector_filters_100_sel_0.005_0.01.bitmap_cache.bin"
```

The rest of the configuration follows the same structure as `topk-multihop-filter` and `topk-beta-filter`.

## Example Command

From the DiskANN repository root:

```bash
cargo run --release --package diskann-benchmark -- run \
  --input-file ./diskann-benchmark/example/async-caselaw-multihop-search-filter.json \
  --output-file caselaw-search-multihop.json
```
