"""Reproduce a frozen-label JailbreakBench case study, without LLM queries.

Default: process the compact metadata shipped in jbb_results/.
--download: re-fetch pinned public artifacts, verify SHA256, retain label metadata
only, and exclude the Privacy category before writing any row-level output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from math import comb
from pathlib import Path
from urllib.request import urlopen

BASE = Path(__file__).resolve().parent / "jbb_results"
COMMIT = "909e68c01d94222b8ad2e397a017e2e12e2adb73"
FIELDS = ["method", "access", "model", "index", "category",
          "number_of_queries", "queries_to_jailbreak", "jailbroken",
          "jailbroken_llama_guard1"]


def boolean(value):
    if value in (True, "True", "true", 1, "1"):
        return True
    if value in (False, "False", "false", 0, "0"):
        return False
    raise ValueError(f"Invalid binary annotation: {value!r}")


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def download(manifest):
    rows = []
    for source in manifest["files"]:
        with urlopen(source["url"], timeout=90) as response:
            payload = response.read()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != source["sha256"]:
            raise ValueError(f"Source hash mismatch: {source['path']}")
        artifact = json.loads(payload)
        parts = source["path"].split("/")
        for item in artifact["jailbreaks"]:
            if item["category"] == "Privacy":
                continue
            row = {field: item.get(field) for field in FIELDS}
            row.update(method=parts[1], access=parts[2], model=parts[-1][:-5])
            rows.append(row)
        # Full prompt/response objects are not written to disk.
    write_csv(BASE / "label_metadata.csv", rows, FIELDS)


def summarize(rows):
    grouped = defaultdict(dict)
    category_by_index = {}
    for row in rows:
        row["index"] = int(row["index"])
        row["jailbroken"] = boolean(row["jailbroken"])
        assert row["category"] != "Privacy"
        key = (row["model"], row["method"])
        assert row["index"] not in grouped[key], "Duplicate method/model/index"
        grouped[key][row["index"]] = row
        previous = category_by_index.setdefault(row["index"], row["category"])
        assert previous == row["category"], "Behavior category mismatch"
    models = sorted({key[0] for key in grouped})
    methods_by_model = {m: sorted(a for mm, a in grouped if mm == m) for m in models}
    common_methods = sorted(set.intersection(*(set(v) for v in methods_by_model.values())))
    reports, histograms, matrix_rows, method_rows = [], [], [], []
    for model in models:
        suites = {"common_methods": common_methods,
                  "all_available_methods": methods_by_model[model]}
        for suite, methods in suites.items():
            index_sets = [set(grouped[(model, a)]) for a in methods]
            indices = sorted(set.intersection(*index_sets))
            union_indices = set.union(*index_sets)
            assert indices, "No complete cases"
            d, n = len(methods), len(indices)
            counts = []
            for i in indices:
                values = [int(grouped[(model, a)][i]["jailbroken"]) for a in methods]
                count = sum(values)
                counts.append(count)
                record = {"suite": suite, "model": model, "index": i,
                          "category": category_by_index[i], "S": count}
                record.update(dict(zip(methods, values)))
                matrix_rows.append(record)
            hist = Counter(counts)
            risk_num = n - hist[0]
            method_counts = {a: sum(int(grouped[(model, a)][i]["jailbroken"])
                                    for i in indices) for a in methods}
            for a, num in method_counts.items():
                method_rows.append({"suite": suite, "model": model, "method": a,
                                    "successes": num, "contexts": n,
                                    "fraction": str(Fraction(num, n))})
            row = dict(suite=suite, model=model, methods=methods, d=d, contexts=n,
                       excluded_incomplete_cases=len(union_indices)-n,
                       union_successes=risk_num, union_fraction=str(Fraction(risk_num, n)),
                       union_rate=risk_num/n, strongest_single_successes=max(method_counts.values()),
                       S_histogram={str(s): hist[s] for s in range(d+1)})
            reports.append(row)
            for s in range(d+1):
                histograms.append(dict(suite=suite, model=model, S=s, count=hist[s], contexts=n))
            # Exact finite-population distribution of C under k uniform distinct
            # strategy queries. This is an oracle reference derived from full labels.
            for k in range(1, d+1):
                probabilities = []
                for c in range(k+1):
                    prob = Fraction(0, 1)
                    for s, freq in hist.items():
                        if 0 <= c <= s and 0 <= k-c <= d-s:
                            prob += Fraction(freq * comb(s, c)*comb(d-s, k-c), n*comb(d, k))
                    probabilities.append(str(prob))
                row.setdefault("oracle_hypergeom_counts", {})[str(k)] = probabilities
    all_methods = sorted({row["method"] for row in rows})
    for row in matrix_rows:
        for a in all_methods:
            row.setdefault(a, "")  # Blank means outside suite; never a negative label.
    write_csv(BASE / "matrices.csv", matrix_rows,
              ["suite", "model", "index", "category", "S"] + all_methods)
    write_csv(BASE / "histograms.csv", histograms,
              ["suite", "model", "S", "count", "contexts"])
    write_csv(BASE / "method_counts.csv", method_rows,
              ["suite", "model", "method", "successes", "contexts", "fraction"])
    (BASE / "summary.json").write_text(json.dumps({
        "repository_commit": COMMIT,
        "annotation": "jailbroken (as stored in the pinned artifact)",
        "not_used_as_annotation": "jailbroken_llama_guard1",
        "excluded_category": "Privacy",
        "retained_categories": sorted(set(category_by_index.values())),
        "common_methods": common_methods,
        "case_study_scope": "Finite historical matrix of published complete-strategy outcomes; not a new model evaluation, a fresh-response probability estimate, or a claim about deployed models.",
        "reports": reports,
    }, indent=2) + "\n")
    for r in reports:
        print(r["suite"], r["model"], "d=", r["d"],
              "union=", f"{r['union_successes']}/{r['contexts']}",
              "hist=", r["S_histogram"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    BASE.mkdir(exist_ok=True)
    manifest = json.loads((BASE / "sources.json").read_text())
    assert manifest["repository_commit"] == COMMIT
    if args.download:
        download(manifest)
    with (BASE / "label_metadata.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summarize(rows)


if __name__ == "__main__":
    main()
