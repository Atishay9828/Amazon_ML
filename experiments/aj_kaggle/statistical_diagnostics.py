"""Query-level stratified bootstrap for retrieval ceilings on a frozen sample."""

from pathlib import Path

import numpy as np

from diagnostics import CAPS, part_name, quote


def query_level_retrieval_rows(report, output):
    """Return in-memory aggregate arrays; never persist or expose sampled IDs."""
    import duckdb

    output = Path(output)
    db = duckdb.connect(str(output / "scratch.duckdb"), read_only=True)
    result = {}
    try:
        policies = sorted({row["policy"] for row in report["metrics"]})
        for sample in report["sample"]:
            country = sample["country"]
            truth_sql = f"""SELECT g.source1_entity_id AS qid,
              CASE WHEN g.matched_entity_ids IS NULL OR g.matched_entity_ids=''
                   THEN 0 ELSE len(str_split(g.matched_entity_ids,',')) END AS true_links
              FROM sampled_gt g JOIN sample s ON g.source1_entity_id=s.entity_id
              WHERE s.country={quote(country)}"""
            for policy in policies:
                paths = [output / f"stages_{part_name(country, source, policy)}.parquet"
                         for source in ("source2", "source3")]
                stage_paths = "[" + ",".join(quote(path) for path in paths) + "]"
                cap_sql = ",\n".join(
                    f"count(*) FILTER(WHERE rank<={int(cap)})::BIGINT AS retained_{int(cap)}"
                    for cap in CAPS
                )
                rows = db.execute(f"""WITH truth AS ({truth_sql}), retrieved AS (
                    SELECT qid,{cap_sql} FROM read_parquet({stage_paths}) GROUP BY qid)
                    SELECT t.qid,t.true_links,{','.join('coalesce(r.retained_'+str(int(c))+',0)::BIGINT' for c in CAPS)}
                    FROM truth t LEFT JOIN retrieved r USING(qid) ORDER BY t.qid""").fetchall()
                if len(rows) != int(sample["queries"]):
                    raise ValueError(f"Sample query count changed for {country}/{policy}")
                values = np.asarray([row[1:] for row in rows], dtype=np.int64)
                if np.any(values[:, 1:] > values[:, :1]):
                    raise ValueError(f"Retrieved true-link count exceeds truth for {country}/{policy}")
                if any(np.any(values[:, j + 1] < values[:, j]) for j in range(len(CAPS) - 1)):
                    raise ValueError(f"True-link retention is not monotone across caps for {country}/{policy}")
                result[(country, policy)] = values
        return result
    finally:
        db.close()


def _bootstrap_arrays(values, indices):
    truth = values[:, 0]
    retained = values[:, 1:]
    positive = truth > 0
    f05 = np.ones_like(retained, dtype=np.float64)
    f05[positive, :] = 1.25 * retained[positive, :] / (
        0.25 * truth[positive, None] + retained[positive, :]
    )
    sampled_f05 = f05[indices].mean(axis=1)
    sampled_truth = truth[indices].sum(axis=1)
    sampled_retained = retained[indices].sum(axis=1)
    sampled_recall = np.divide(
        sampled_retained,
        sampled_truth[:, None],
        out=np.full(sampled_retained.shape, np.nan, dtype=np.float64),
        where=sampled_truth[:, None] > 0,
    )
    return sampled_f05, sampled_recall, sampled_retained, sampled_truth


def stratified_bootstrap(query_rows, caps=CAPS, replicates=2000, seed=20260926):
    """Paired Source-1 bootstrap, stratified by country, with singleton F0.5.

    CIs describe uncertainty from resampling the selected development queries.
    They do not represent leaderboard/private-test uncertainty or country-shift
    uncertainty. Per-entity F0.5 values use the optimistic perfect-classifier
    ceiling: all reachable true links retained, all candidate false links dropped.
    """
    if replicates < 200:
        raise ValueError("Use at least 200 bootstrap replicates")
    if not query_rows:
        raise ValueError("No country/policy query arrays supplied")
    countries = sorted({country for country, _ in query_rows})
    policies = sorted({policy for _, policy in query_rows})
    if "legacy" not in policies:
        raise ValueError("A legacy policy is required for paired comparisons")
    caps = tuple(int(cap) for cap in caps)
    if any(values.shape[1] != len(caps) + 1 for values in query_rows.values()):
        raise ValueError("Query arrays must have truth followed by one retained count per cap")
    rng = np.random.default_rng(seed)
    sampled = {}
    for country in countries:
        legacy = query_rows[(country, "legacy")]
        if any(query_rows[(country, policy)].shape[0] != legacy.shape[0] for policy in policies):
            raise ValueError("Paired policies must use the same Source 1 sample size")
        indices = rng.integers(0, legacy.shape[0], size=(replicates, legacy.shape[0]))
        for policy in policies:
            values = query_rows[(country, policy)]
            sampled[(country, policy)] = _bootstrap_arrays(values, indices)

    records = []
    scopes = [(country, [country]) for country in countries]
    if len(countries) > 1:
        scopes.append(("ALL_COUNTRIES_STRATIFIED", countries))
    for scope, strata in scopes:
        weights = np.asarray([
            query_rows[(country, "legacy")].shape[0] for country in strata
        ], dtype=np.float64)
        weights /= weights.sum()
        for policy in policies:
            for cap_index, cap in enumerate(caps):
                f_samples = np.sum(np.stack([
                    sampled[(country, policy)][0][:, cap_index] for country in strata
                ]) * weights[:, None], axis=0)
                retained = np.sum(np.stack([
                    sampled[(country, policy)][2][:, cap_index] for country in strata
                ]), axis=0)
                truth = np.sum(np.stack([
                    sampled[(country, policy)][3] for country in strata
                ]), axis=0)
                recall_samples = np.divide(
                    retained, truth,
                    out=np.full(retained.shape, np.nan, dtype=np.float64),
                    where=truth > 0,
                )
                baseline_f = np.sum(np.stack([
                    sampled[(country, "legacy")][0][:, cap_index] for country in strata
                ]) * weights[:, None], axis=0)
                baseline_retained = np.sum(np.stack([
                    sampled[(country, "legacy")][2][:, cap_index] for country in strata
                ]), axis=0)
                baseline_truth = np.sum(np.stack([
                    sampled[(country, "legacy")][3] for country in strata
                ]), axis=0)
                delta_f = f_samples - baseline_f
                baseline_recall_samples = np.divide(
                    baseline_retained, baseline_truth,
                    out=np.full(baseline_retained.shape, np.nan, dtype=np.float64),
                    where=baseline_truth > 0,
                )
                delta_recall = recall_samples - baseline_recall_samples
                point_values = [query_rows[(country, policy)] for country in strata]
                true_total = sum(int(arr[:, 0].sum()) for arr in point_values)
                retained_total = sum(int(arr[:, cap_index + 1].sum()) for arr in point_values)
                point_recall = retained_total / true_total if true_total else None
                f_points = []
                for arr in point_values:
                    t, r = arr[:, 0], arr[:, cap_index + 1]
                    values = np.ones(t.shape, dtype=np.float64)
                    pos = t > 0
                    values[pos] = 1.25 * r[pos] / (0.25 * t[pos] + r[pos])
                    f_points.append(float(values.mean()))
                point_f = float(np.dot(np.asarray(f_points), weights))
                point_delta_f = None
                point_delta_recall = None
                if policy != "legacy":
                    legacy_f_points = []
                    legacy_retained = 0
                    legacy_truth = 0
                    for country in strata:
                        arr = query_rows[(country, "legacy")]
                        t, r = arr[:, 0], arr[:, cap_index + 1]
                        values = np.ones(t.shape, dtype=np.float64)
                        pos = t > 0
                        values[pos] = 1.25 * r[pos] / (0.25 * t[pos] + r[pos])
                        legacy_f_points.append(float(values.mean()))
                        legacy_retained += int(r.sum())
                        legacy_truth += int(t.sum())
                    point_delta_f = point_f - float(np.dot(np.asarray(legacy_f_points), weights))
                    point_delta_recall = point_recall - legacy_retained / legacy_truth
                q_f = np.quantile(f_samples, [0.025, 0.975])
                q_r = (np.nanquantile(recall_samples, [0.025, 0.975])
                       if np.isfinite(recall_samples).any() else [float("nan"), float("nan")])
                q_df = np.quantile(delta_f, [0.025, 0.975])
                q_dr = (np.nanquantile(delta_recall, [0.025, 0.975])
                        if np.isfinite(delta_recall).any() else [float("nan"), float("nan")])
                records.append({
                    "scope": scope, "policy": policy, "cap": cap,
                    "queries_per_country": [int(query_rows[(country, policy)].shape[0]) for country in strata],
                    "true_links": true_total, "retained_true_links": retained_total,
                    "oracle_macro_f05": point_f, "oracle_macro_f05_ci_low": float(q_f[0]),
                    "oracle_macro_f05_ci_high": float(q_f[1]),
                    "micro_true_link_recall": point_recall,
                    "micro_recall_ci_low": float(q_r[0]), "micro_recall_ci_high": float(q_r[1]),
                    "valid_micro_recall_bootstrap_replicates": int(np.isfinite(recall_samples).sum()),
                    "delta_oracle_f05_vs_legacy": point_delta_f,
                    "delta_oracle_f05_ci_low": float(q_df[0]) if policy != "legacy" else None,
                    "delta_oracle_f05_ci_high": float(q_df[1]) if policy != "legacy" else None,
                    "delta_micro_recall_vs_legacy": point_delta_recall,
                    "delta_micro_recall_ci_low": float(q_dr[0]) if policy != "legacy" else None,
                    "delta_micro_recall_ci_high": float(q_dr[1]) if policy != "legacy" else None,
                    "replicates": replicates, "seed": seed,
                    "interpretation": "Paired Source-1 bootstrap within selected training-development queries; oracle F0.5 is an optimistic ceiling and CIs do not estimate leaderboard uncertainty.",
                })
    return records


def run_bootstrap_diagnostics(report, output, replicates=2000, seed=20260926):
    query_rows = query_level_retrieval_rows(report, output)
    records = stratified_bootstrap(query_rows, replicates=replicates, seed=seed)
    return records
