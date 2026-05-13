#!/usr/bin/env python3
"""
Regenerate query embeddings, query predicates, and filtered groundtruth by
selecting queries whose selectivity falls within a given range.

Supported inputs:
1) Query embedding .bin (DiskANN style):
   [uint32 num_queries][uint32 dim][num_queries * dim elements]
2) Query predicates .jsonl:
   one JSON object per line, assumed to be in query order.
3) Range groundtruth .bin (variable length):
   [uint32 num_queries][uint32 total_results]
   [uint32 gt_count_per_query repeated num_queries]
   [uint32 flat_gt_ids repeated total_results]
4) Summary .csv:
   must contain at least query_id and selectivity columns.
"""

from __future__ import annotations

import argparse
import csv
import json
import struct
from pathlib import Path
from typing import List, Tuple

import numpy as np


DTYPE_MAP = {
    "float32": np.float32,
    "float16": np.float16,
    "uint8": np.uint8,
    "int8": np.int8,
}


def read_diskann_matrix(path: Path, dtype: np.dtype) -> np.ndarray:
    with path.open("rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError(f"{path}: invalid matrix header")
        nrows, dim = struct.unpack("<II", header)
        data = np.fromfile(f, dtype=dtype, count=nrows * dim)

    if data.size != nrows * dim:
        raise ValueError(f"{path}: expected {nrows * dim} values, got {data.size}")
    return data.reshape(nrows, dim)


def write_diskann_matrix(path: Path, matrix: np.ndarray, dtype: np.dtype) -> None:
    matrix = np.asarray(matrix, dtype=dtype, order="C")
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2D")
    nrows, dim = matrix.shape
    with path.open("wb") as f:
        f.write(struct.pack("<II", nrows, dim))
        matrix.tofile(f)


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: JSON error at line {ln}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: List[dict], remap_query_id: bool) -> None:
    with path.open("w", encoding="utf-8") as f:
        for i, obj in enumerate(rows):
            if remap_query_id and isinstance(obj, dict) and "query_id" in obj:
                obj = dict(obj)
                obj["query_id"] = i
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_variable_length_gt(path: Path) -> Tuple[np.ndarray, List[np.ndarray]]:
    raw = np.fromfile(path, dtype=np.uint32)
    if raw.size < 2:
        raise ValueError(f"{path}: file too small")

    nq = int(raw[0])
    total = int(raw[1])
    need = 2 + nq + total
    if raw.size < need:
        raise ValueError(
            f"{path}: malformed GT file, expected at least {need} uint32, got {raw.size}"
        )

    counts = raw[2 : 2 + nq].astype(np.int64)
    ids = raw[2 + nq : 2 + nq + total]

    if counts.sum() != total:
        raise ValueError(
            f"{path}: inconsistent GT counts sum={int(counts.sum())}, total={total}"
        )

    per_query_ids: List[np.ndarray] = []
    offset = 0
    for c in counts:
        c_i = int(c)
        per_query_ids.append(ids[offset : offset + c_i].copy())
        offset += c_i

    return counts.astype(np.int32), per_query_ids


def write_variable_length_gt(path: Path, per_query_ids: List[np.ndarray]) -> None:
    nq = len(per_query_ids)
    counts = np.array([len(x) for x in per_query_ids], dtype=np.uint32)
    total = int(counts.sum())
    flat = (
        np.concatenate(per_query_ids).astype(np.uint32)
        if total > 0
        else np.array([], dtype=np.uint32)
    )

    with path.open("wb") as f:
        np.array([nq, total], dtype=np.uint32).tofile(f)
        counts.tofile(f)
        flat.tofile(f)


def read_summary_selection(
    path: Path,
    min_selectivity: float,
    max_selectivity: float,
    num_queries: int,
) -> Tuple[List[int], List[float]]:
    selected_ids: List[int] = []
    selected_selectivities: List[float] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty CSV")
        required = {"query_id", "selectivity"}
        missing = required.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing required columns: {sorted(missing)}")

        for row_num, row in enumerate(reader, start=2):
            try:
                query_id = int(row["query_id"])
                selectivity = float(row["selectivity"])
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{path}: invalid query_id/selectivity at line {row_num}"
                ) from e

            if min_selectivity <= selectivity <= max_selectivity:
                selected_ids.append(query_id)
                selected_selectivities.append(selectivity)

    matched = len(selected_ids)
    if matched == 0:
        raise ValueError(
            f"no query found for selectivity range [{min_selectivity}, {max_selectivity}]"
        )

    if matched < num_queries:
        print(
            f"only {matched} queries matched range "
            f"[{min_selectivity}, {max_selectivity}]; using all matched queries"
        )
        return selected_ids, selected_selectivities

    return selected_ids[:num_queries], selected_selectivities[:num_queries]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter queries by selectivity range and regenerate aligned files."
    )
    ap.add_argument("--query-bin", type=Path, required=True)
    ap.add_argument(
        "--query-dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default="float32",
        help="dtype used in query .bin payload",
    )
    ap.add_argument("--query-filters", type=Path, required=True)
    ap.add_argument("--gt-bin", type=Path, required=True)
    ap.add_argument("--summary-csv", type=Path, required=True)
    ap.add_argument("--min-selectivity", type=float, required=True)
    ap.add_argument("--max-selectivity", type=float, required=True)
    ap.add_argument(
        "--num-queries",
        type=int,
        required=True,
        help="number of queries to keep; if fewer match, keep all matches",
    )
    ap.add_argument("--out-query-bin", type=Path, required=True)
    ap.add_argument("--out-query-filters", type=Path, required=True)
    ap.add_argument("--out-gt-bin", type=Path, required=True)
    ap.add_argument(
        "--no-remap-query-id",
        action="store_true",
        help="do not rewrite query_id in JSONL rows",
    )

    args = ap.parse_args()

    if not (0.0 <= args.min_selectivity <= 1.0):
        raise ValueError("--min-selectivity must be in [0, 1]")
    if not (0.0 <= args.max_selectivity <= 1.0):
        raise ValueError("--max-selectivity must be in [0, 1]")
    if args.min_selectivity > args.max_selectivity:
        raise ValueError("--min-selectivity must be <= --max-selectivity")
    if args.num_queries <= 0:
        raise ValueError("--num-queries must be positive")

    query_dtype = DTYPE_MAP[args.query_dtype]
    queries = read_diskann_matrix(args.query_bin, query_dtype)
    filters = read_jsonl(args.query_filters)
    counts, gt_ids = read_variable_length_gt(args.gt_bin)

    nq = queries.shape[0]
    if len(filters) != nq or len(gt_ids) != nq or len(counts) != nq:
        raise ValueError(
            "length mismatch:\n"
            f"  queries rows      = {nq}\n"
            f"  query filters rows= {len(filters)}\n"
            f"  groundtruth rows  = {len(gt_ids)}\n"
            f"  gt counts rows    = {len(counts)}"
        )

    selected_ids, selected_selectivities = read_summary_selection(
        args.summary_csv,
        args.min_selectivity,
        args.max_selectivity,
        args.num_queries,
    )

    out_of_range = [qid for qid in selected_ids if qid < 0 or qid >= nq]
    if out_of_range:
        raise ValueError(
            f"summary contains query_id outside valid range [0, {nq - 1}]: "
            f"{out_of_range[:10]}"
        )

    keep_idx = np.array(selected_ids, dtype=np.int64)
    new_queries = queries[keep_idx]
    new_filters = [filters[i] for i in keep_idx.tolist()]
    new_gt_ids = [gt_ids[i] for i in keep_idx.tolist()]

    args.out_query_bin.parent.mkdir(parents=True, exist_ok=True)
    args.out_query_filters.parent.mkdir(parents=True, exist_ok=True)
    args.out_gt_bin.parent.mkdir(parents=True, exist_ok=True)

    write_diskann_matrix(args.out_query_bin, new_queries, query_dtype)
    write_jsonl(
        args.out_query_filters,
        new_filters,
        remap_query_id=not args.no_remap_query_id,
    )
    write_variable_length_gt(args.out_gt_bin, new_gt_ids)

    new_counts = np.array([len(x) for x in new_gt_ids], dtype=np.int64)
    selected_selectivities_np = np.array(selected_selectivities, dtype=np.float64)
    print(f"input queries               : {nq}")
    print(
        f"selectivity range           : "
        f"[{args.min_selectivity}, {args.max_selectivity}]"
    )
    print(f"requested queries           : {args.num_queries}")
    print(f"kept queries                : {keep_idx.size}")
    print(
        f"generated selectivity stats : "
        f"min={selected_selectivities_np.min():.10f}, "
        f"max={selected_selectivities_np.max():.10f}, "
        f"mean={selected_selectivities_np.mean():.10f}, "
        f"median={np.median(selected_selectivities_np):.10f}"
    )
    print(
        f"new min/mean/max gt         : "
        f"{int(new_counts.min())}/{new_counts.mean():.2f}/{int(new_counts.max())}"
    )
    print(f"output query bin            : {args.out_query_bin}")
    print(f"output filter jsonl         : {args.out_query_filters}")
    print(f"output gt bin               : {args.out_gt_bin}")


if __name__ == "__main__":
    main()
