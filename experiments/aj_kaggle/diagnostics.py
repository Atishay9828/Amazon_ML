"""Preprocessing and retrieval experiments on frozen development queries.

Labels are joined AFTER each candidate set has been selected and ranked.
All target records in a source/country contribute to document frequencies.
Private row-level artifacts go under work/, never into the public repository.
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import duckdb

from preprocessing import PREPROCESS_VERSION, normalized_sql


DIAGNOSTICS_VERSION = "diagnostics-v3-controlled-ablation"
# Execution-only changes preserve the v3 sample and retrieval mathematics.
CHECKPOINT_VERSION = "diagnostics-resume-v1"
REFERENCE_DUCKDB = "1.3.2"
POLICIES = (
    {"name": "legacy", "view": "legacy", "token_df": 64, "keys": 3},
    {"name": "preprocessed", "view": "new", "token_df": 64, "keys": 3},
    {"name": "preprocessed_df256_k3", "view": "new", "token_df": 256, "keys": 3},
    {"name": "preprocessed_df64_k6", "view": "new", "token_df": 64, "keys": 6},
    {"name": "preprocessed_wider", "view": "new", "token_df": 256, "keys": 6},
)
CAPS = (40, 80, 160, 500)
TARGET_BATCH_ROWS = 20_000


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def read_tsv(path):
    return f"read_csv({quote(path)}, delim='\\t', header=true, all_varchar=true)"


def log(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(partial, path)


def json_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_copy(audit, relation, path, options):
    partial = path.with_name(path.name + ".partial")
    audit.run(f"COPY ({relation}) TO {quote(partial)} ({options})")
    os.replace(partial, path)


def legacy_name(expression):
    # Exact v3 behavior, including its space-consuming legal-word expression.
    return f"""trim(regexp_replace(regexp_replace(
      ' ' || trim(regexp_replace(lower(strip_accents(coalesce({expression},''))), '[^a-z0-9]+', ' ', 'g')) || ' ',
      ' (incorporated|inc|corporation|corp|limited|ltd|llc|llp|private|pvt) ', ' ', 'g'),
      ' +', ' ', 'g'))"""


def population_sql(relation):
    return f"""SELECT *, {legacy_name('business_name')} AS legacy_nm,
      trim(regexp_replace(lower(strip_accents(coalesce(business_address,''))), '[^a-z0-9]+', ' ', 'g')) AS legacy_ad,
      regexp_extract(coalesce(business_address,''), '[0-9]+') AS num
      FROM ({normalized_sql(relation)})"""


def key_sql(table, unicode_letters=False, inventory=None):
    letter = r"[\p{L}]" if unicode_letters else "[a-z]"
    filtered = f" SEMI JOIN {inventory} USING(key)" if inventory else ""
    return f"""SELECT DISTINCT rid, key FROM (
      SELECT rid, 'N=' || nm AS key FROM {table} WHERE length(nm)>=3
      UNION ALL SELECT rid, 'A=' || ad FROM {table} WHERE length(ad)>=5
      UNION ALL SELECT rid, 'NT=' || tok FROM {table}, UNNEST(str_split(nm,' ')) u(tok)
        WHERE length(tok)>=3 AND regexp_matches(tok,{quote(letter)})
      UNION ALL SELECT rid, 'AT=' || tok FROM {table}, UNNEST(str_split(ad,' ')) u(tok)
        WHERE length(tok)>=4 AND regexp_matches(tok,{quote(letter)})
      UNION ALL SELECT rid, 'AN=' || num || ':' || tok FROM {table}, UNNEST(str_split(ad,' ')) u(tok)
        WHERE length(num)>=2 AND length(tok)>=4 AND regexp_matches(tok,{quote(letter)})
    ) k{filtered}"""


def part_name(country, source, policy):
    # Country labels are open strings; never use them as literal path fragments.
    return f"{hashlib.sha256(country.encode()).hexdigest()[:12]}_{source}_{policy}"


def selected_keys_sql(token_df, key_budget, exact_df=512):
    return f"""SELECT qk.rid, qk.key FROM qk JOIN frequencies USING(key)
      WHERE freq <= CASE WHEN starts_with(key,'N=') OR starts_with(key,'A=') THEN {exact_df} ELSE {token_df} END
      QUALIFY row_number() OVER (PARTITION BY rid,
        CASE WHEN starts_with(key,'N=') THEN 'N=' WHEN starts_with(key,'A=') THEN 'A='
          WHEN starts_with(key,'NT=') THEN 'NT=' WHEN starts_with(key,'AT=') THEN 'AT=' ELSE 'AN=' END
        ORDER BY freq,key) <= CASE WHEN starts_with(key,'N=') OR starts_with(key,'A=') THEN 1 ELSE {key_budget} END"""


def ranked_sql():
    # Same ranking expression, FLOAT casts and tid tie-break as v3. No labels.
    return """WITH features AS (
      SELECT p.*, jaro_winkler_similarity(q.nm,t.nm)::FLOAT AS name_jw,
        CASE WHEN q.ad='' OR t.ad='' THEN 0 ELSE jaro_winkler_similarity(q.ad,t.ad) END::FLOAT AS address_jw,
        CASE WHEN q.num<>'' AND q.num=t.num THEN 1 ELSE 0 END::FLOAT AS number_equal,
        CASE WHEN q.nm<>'' AND q.nm=t.nm THEN 1 ELSE 0 END::FLOAT AS exact_name
      FROM pairs p JOIN q ON q.rid=p.qid JOIN t ON t.rid=p.tid
    ), scored AS (
      SELECT *, 0.38*greatest(name_jw,address_jw)+0.28*name_jw+0.20*address_jw+
        0.07*number_equal+0.05*exact_name+0.04*least(key_hits,3)/3 AS rank_score
      FROM features
    ) SELECT *, row_number() OVER(PARTITION BY qid ORDER BY rank_score DESC,tid) AS rank
      FROM scored"""


class Audit:
    def __init__(self, output, memory_mb=512):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "spill").mkdir(exist_ok=True)
        self.db = duckdb.connect(str(self.output / "scratch.duckdb"))
        self.db.execute(f"SET memory_limit='{int(memory_mb)}MB'")
        self.db.execute("SET threads=1")
        self.db.execute("SET preserve_insertion_order=false")
        self.db.execute(f"SET temp_directory={quote(self.output / 'spill')}")
        self.db.execute("SET max_temp_directory_size='16GB'")
        self.queries = (self.output / "queries.jsonl").open("a", encoding="utf-8")

    def run(self, sql):
        self.queries.write(json.dumps({"sql": sql}) + "\n")
        self.queries.flush()
        return self.db.execute(sql)

    def rows(self, sql):
        cursor = self.run(sql)
        names = [c[0] for c in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]

    def close(self):
        self.queries.close()
        self.db.close()


def normalize_targets_batched(audit, relation, batch_rows=None):
    """Materialize raw rows on disk before evaluating wide normalization SQL.

    A fresh DuckDB table has dense rowids. Filtering those rowids needs no
    window or sort over the wide normalized population. Each normalization
    expression sees at most batch_rows inputs, even if its CTEs materialize.
    The batch assignment is operational only; ranking still breaks ties by ID.
    """
    batch_rows = TARGET_BATCH_ROWS if batch_rows is None else batch_rows
    if not isinstance(batch_rows, int) or batch_rows < 1:
        raise ValueError("normalization batch_rows must be a positive integer")
    audit.run("DROP TABLE IF EXISTS normalized_targets")
    audit.run(f"""CREATE OR REPLACE TABLE raw_targets AS
      SELECT entity_id,business_name,business_address,country FROM {relation}""")
    count, maximum = audit.run("SELECT count(*),coalesce(max(rowid),-1) FROM raw_targets").fetchone()
    if maximum + 1 != count:
        raise RuntimeError("Fresh raw target table does not have dense rowids")
    audit.run("CREATE OR REPLACE TABLE normalization_batch AS SELECT * FROM raw_targets WHERE false")
    audit.run(f"""CREATE TABLE normalized_targets AS SELECT *,0::INTEGER diagnostic_batch
      FROM ({population_sql('normalization_batch')})""")
    for batch, start in enumerate(range(0, count, batch_rows)):
        audit.run(f"""CREATE OR REPLACE TABLE normalization_batch AS SELECT * FROM raw_targets
          WHERE rowid>={start} AND rowid<{start + batch_rows}""")
        audit.run(f"""INSERT INTO normalized_targets SELECT *,{batch}::INTEGER diagnostic_batch
          FROM ({population_sql('normalization_batch')})""")
        if (batch + 1) % 50 == 0:
            audit.run("CHECKPOINT")
            log(f"    normalized {min(start + batch_rows, count):,}/{count:,} target rows")
    if audit.run("SELECT count(*) FROM normalized_targets").fetchone()[0] != count:
        raise RuntimeError("Batched target normalization changed the record count")
    audit.run("DROP TABLE normalization_batch")
    audit.run("DROP TABLE raw_targets")
    audit.run("CHECKPOINT")
    return count


def target_key_batches(audit, unicode_letters, inventory):
    # Correlated UNNEST may allocate outside spillable hash aggregation. Limit
    # its INPUT rows as well as DuckDB's memory budget, while covering ALL rows.
    batches = [row[0] for row in audit.run("SELECT DISTINCT diagnostic_batch FROM normalized_targets ORDER BY diagnostic_batch").fetchall()]
    for position,batch in enumerate(batches):
        audit.run(f"CREATE OR REPLACE VIEW target_batch AS SELECT * FROM t WHERE diagnostic_batch={int(batch)}")
        yield key_sql('target_batch',unicode_letters,inventory)
        if (position+1)%50==0:
            log(f"    processed {position+1}/{len(batches)} target batches")


def full_frequencies(audit, unicode_letters):
    audit.run("CREATE OR REPLACE TABLE frequencies(key VARCHAR,freq BIGINT)")
    for keys in target_key_batches(audit,unicode_letters,'inventory'):
        audit.run(f"CREATE OR REPLACE TABLE batch_frequencies AS SELECT key,count(*) freq FROM ({keys}) GROUP BY key")
        audit.run("""CREATE OR REPLACE TABLE frequencies AS SELECT key,sum(freq)::BIGINT freq
          FROM (SELECT * FROM frequencies UNION ALL SELECT * FROM batch_frequencies) GROUP BY key""")


def full_postings(audit, unicode_letters):
    audit.run("CREATE OR REPLACE TABLE postings(rid VARCHAR,key VARCHAR)")
    for keys in target_key_batches(audit,unicode_letters,'selected_inventory'):
        audit.run(f"INSERT INTO postings {keys}")


def profile(audit, table, split, source, country):
    row = audit.rows(f"""SELECT count(*) AS records,
      count(*) FILTER(WHERE business_name IS NULL OR trim(business_name)='') AS raw_name_empty,
      count(*) FILTER(WHERE business_address IS NULL OR trim(business_address)='') AS raw_address_empty,
      count(*) FILTER(WHERE legacy_nm='') AS legacy_name_empty,
      count(*) FILTER(WHERE name_core='') AS new_core_empty,
      count(*) FILTER(WHERE name_folded<>'') AS preserved_nonempty_names,
      count(*) FILTER(WHERE legacy_nm='' AND name_folded<>'') AS names_recovered_from_legacy_empty,
      count(*) FILTER(WHERE {legacy_name('legacy_nm')}<>legacy_nm) AS legacy_name_not_idempotent,
      count(*) FILTER(WHERE length(address_numbers)>1) AS addresses_with_multiple_numbers,
      count(*) FILTER(WHERE length(num)=1) AS first_numbers_excluded_from_legacy_AN,
      count(*) FILTER(WHERE address_canonical<>address_folded) AS canonical_address_changed
      FROM {table}""")[0]
    return {"split": split, "source": source, "country": country, **row}


def run_policy(audit, policy, country, source):
    name, token_df, budget = policy["name"], policy["token_df"], policy["keys"]
    log(f"  {name}: full-target DF <= {token_df}, {budget} token keys/channel")
    audit.run(f"CREATE OR REPLACE TABLE selected AS {selected_keys_sql(token_df,budget)}")
    # Selected keys determine the retrieval population; truth is absent here.
    audit.run("CREATE OR REPLACE TABLE selected_inventory AS SELECT DISTINCT key FROM selected")
    full_postings(audit,policy['view']=='new')
    audit.run("""CREATE OR REPLACE TABLE pairs AS SELECT s.rid qid,p.rid tid,count(*)::FLOAT key_hits
      FROM selected s JOIN postings p USING(key) GROUP BY s.rid,p.rid""")
    audit.run(f"CREATE OR REPLACE TABLE ranked AS {ranked_sql()}")
    # Labels are consulted only after the complete candidate order is fixed.
    audit.run("""CREATE OR REPLACE TABLE positive_keys AS
      SELECT DISTINCT g.qid,g.tid,qk.key FROM truth g JOIN qk ON g.qid=qk.rid
      JOIN positive_tk pk ON pk.rid=g.tid AND pk.key=qk.key""")
    audit.run(f"""CREATE OR REPLACE TABLE stages AS
      WITH survival AS (
        SELECT pk.qid,pk.tid,true AS shared,
          bool_or(f.freq <= CASE WHEN starts_with(pk.key,'N=') OR starts_with(pk.key,'A=') THEN 512 ELSE {token_df} END) AS usable,
          bool_or(s.key IS NOT NULL) AS selected
        FROM positive_keys pk LEFT JOIN frequencies f USING(key)
          LEFT JOIN selected s ON s.rid=pk.qid AND s.key=pk.key GROUP BY pk.qid,pk.tid
      ) SELECT g.qid,g.tid,coalesce(s.shared,false) shared,coalesce(s.usable,false) usable,
          coalesce(s.selected,false) selected,r.rank
        FROM truth g LEFT JOIN survival s USING(qid,tid) LEFT JOIN ranked r USING(qid,tid)""")
    audit.run("""ALTER TABLE stages ADD COLUMN target_address_empty BOOLEAN""")
    audit.run("""UPDATE stages SET target_address_empty=(t.business_address IS NULL OR trim(t.business_address)='')
      FROM normalized_targets t WHERE stages.tid=t.rid""")
    bad = audit.run("SELECT count(*) FROM stages WHERE selected<>(rank IS NOT NULL) OR (selected AND NOT usable) OR (usable AND NOT shared)").fetchone()[0]
    if bad:
        raise RuntimeError(f"Stage attribution disagrees with actual retrieval on {bad} true links")
    raw_pairs = audit.run("SELECT count(*) FROM ranked").fetchone()[0]
    nq = audit.run("SELECT count(*) FROM q").fetchone()[0]
    records = []
    for cap in CAPS:
        row = audit.rows(f"""SELECT count(*) AS true_links,
          count(*) FILTER(WHERE NOT shared) AS lost_no_shared_key,
          count(*) FILTER(WHERE shared AND NOT usable) AS lost_frequency_gate,
          count(*) FILTER(WHERE usable AND NOT selected) AS lost_query_key_budget,
          count(*) FILTER(WHERE selected AND rank>{cap}) AS lost_rank_cap,
          count(*) FILTER(WHERE rank<={cap}) AS retained_links,
          count(*) FILTER(WHERE target_address_empty) AS true_links_missing_address,
          count(*) FILTER(WHERE target_address_empty AND rank<={cap}) AS retained_missing_address_links
          FROM stages""")[0]
        if row["true_links"] != sum(row[k] for k in ("lost_no_shared_key","lost_frequency_gate","lost_query_key_budget","lost_rank_cap","retained_links")):
            raise RuntimeError("Disjoint loss counts do not reconcile")
        burden = audit.rows(f"""WITH counts AS (SELECT qid,count(*) n FROM ranked WHERE rank<={cap} GROUP BY qid)
          SELECT sum(coalesce(n,0)) AS candidate_pairs,
            count(*) FILTER(WHERE coalesce(n,0)=0) AS queries_without_candidates,
            count(*) FILTER(WHERE n={cap}) AS queries_at_cap,
            quantile_cont(coalesce(n,0),0.5) AS candidates_p50,
            quantile_cont(coalesce(n,0),0.95) AS candidates_p95,
            max(coalesce(n,0)) AS candidates_max FROM q LEFT JOIN counts ON q.rid=counts.qid""")[0]
        row.update(burden)
        row.update(country=country,source=source,policy=name,cap=cap,queries=nq,raw_pairs=raw_pairs)
        row["candidate_link_recall"] = row["retained_links"]/row["true_links"] if row["true_links"] else None
        row["known_nonmatch_candidates"] = row["candidate_pairs"]-row["retained_links"]
        row["candidate_positive_fraction"] = row["retained_links"]/row["candidate_pairs"] if row["candidate_pairs"] else None
        row["missing_address_link_recall"] = row["retained_missing_address_links"]/row["true_links_missing_address"] if row["true_links_missing_address"] else None
        records.append(row)
    stage_file = audit.output / f"stages_{part_name(country,source,name)}.parquet"
    atomic_copy(audit,"SELECT * FROM stages",stage_file,"FORMAT parquet,COMPRESSION zstd")
    atomic_json(audit.output/f"metrics_{part_name(country,source,name)}.json",records)
    # Small private inspection sample; only aggregates are displayed/published.
    examples = audit.output / f"misses_{part_name(country,source,name)}.csv"
    atomic_copy(audit,f"""SELECT s.*,q.business_name query_name,q.business_address query_address,
      t.business_name target_name,t.business_address target_address
      FROM stages s JOIN normalized_queries q ON s.qid=q.rid JOIN normalized_targets t ON s.tid=t.rid
      WHERE rank IS NULL OR rank>40 ORDER BY s.qid,s.tid LIMIT 40""",examples,"HEADER true")
    recall = records[0]['candidate_link_recall']
    recall_text = f"{recall:.4%}" if recall is not None else "n/a (no true links in this slice)"
    log(f"    retained@40={recall_text}; candidate pairs={records[0]['candidate_pairs']:,}")
    return records


def compare_stages(audit, country, source, policy):
    base = audit.output / f"stages_{part_name(country,source,'legacy')}.parquet"
    alternative = audit.output / f"stages_{part_name(country,source,policy)}.parquet"
    return audit.rows(f"""SELECT {quote(country)} AS country,{quote(source)} AS source,{quote(policy)} AS "policy",
      count(*) FILTER(WHERE coalesce(b.rank<=40,false)=false AND a.rank<=40) AS recovered_true_links_at40,
      count(*) FILTER(WHERE b.rank<=40 AND coalesce(a.rank<=40,false)=false) AS lost_true_links_at40
      FROM read_parquet({quote(base)}) b JOIN read_parquet({quote(alternative)}) a USING(qid,tid)""")[0]


def semantics_sha256():
    # Version labels alone cannot catch an accidental unversioned SQL edit.
    return json_sha256({"population":population_sql("input_rows"),
        "keys":[key_sql("rows",unicode_letters) for unicode_letters in (False,True)],
        "selection":[selected_keys_sql(p['token_df'],p['keys']) for p in POLICIES],
        "ranking":ranked_sql(),"policies":POLICIES,"caps":CAPS})


def partition_artifacts(output, completed):
    artifacts = {}
    for country,source in sorted(completed):
        for policy in POLICIES:
            name = part_name(country,source,policy['name'])
            for filename in (f"stages_{name}.parquet",f"metrics_{name}.json",f"misses_{name}.csv"):
                path = output/filename
                if not path.is_file():
                    raise RuntimeError(f"Completed partition is missing artifact: {filename}")
                artifacts[filename] = {"bytes":path.stat().st_size,"sha256":sha256_file(path)}
    return artifacts


def save_progress(audit, manifest, sample_sha, profiles, metrics, comparisons):
    completed = {(r['country'],r['source']) for r in metrics}
    payload = {"checkpoint_version":CHECKPOINT_VERSION,
      "manifest_sha256":json_sha256(manifest),"sample_sha256":sample_sha,
      "semantics_sha256":semantics_sha256(),
      "completed_partitions":[list(p) for p in sorted(completed)],
      "artifacts":partition_artifacts(audit.output,completed),
      "profiles":profiles,"metrics":metrics,"comparisons":comparisons}
    atomic_json(audit.output/"progress.json",payload)


def load_progress(audit, manifest, sample_sha, membership):
    """Validate completion evidence before reusing any partition.

    Original v2 progress files have no hashes. They are accepted only with the
    exact v2 input/config manifest, recomputed frozen sample, complete policy/
    cap inventory, matching metrics files, and stages reconciled against the
    selected truth. The caller then binds these verified artifacts with hashes.
    Files from a partition absent from progress are never completion evidence.
    """
    path = audit.output/"progress.json"
    if not path.exists():
        return [],[],[],set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload,dict):
        raise RuntimeError("Invalid progress checkpoint")
    modern = 'checkpoint_version' in payload
    if modern:
        expected = {"checkpoint_version":CHECKPOINT_VERSION,
          "manifest_sha256":json_sha256(manifest),"sample_sha256":sample_sha,
          "semantics_sha256":semantics_sha256()}
        if any(payload.get(key)!=value for key,value in expected.items()):
            raise RuntimeError("Progress checkpoint version, manifest, sample or semantics mismatch")
    elif set(payload)!={'profiles','metrics','comparisons'}:
        raise RuntimeError("Unrecognized legacy progress checkpoint")
    profiles,metrics,comparisons = (payload.get(k) for k in ('profiles','metrics','comparisons'))
    if any(not isinstance(rows,list) for rows in (profiles,metrics,comparisons)):
        raise RuntimeError("Invalid progress record lists")
    queries = {r['country']:r['queries'] for r in membership}
    try:
        completed = {(r['country'],r['source']) for r in metrics}
        if any(country not in queries or source not in ('source2','source3') for country,source in completed):
            raise ValueError("unknown partition")
        metric_keys = [(r['country'],r['source'],r['policy'],r['cap']) for r in metrics]
        expected_keys = {(c,s,p['name'],cap) for c,s in completed for p in POLICIES for cap in CAPS}
        comparison_keys = [(r['country'],r['source'],r['policy']) for r in comparisons]
        expected_comparisons = {(c,s,p['name']) for c,s in completed for p in POLICIES if p['name']!='legacy'}
        profile_keys = [(r['country'],r['source']) for r in profiles]
        expected_profiles = completed | {(c,'source1') for c,_ in completed}
        if (len(metric_keys)!=len(set(metric_keys)) or set(metric_keys)!=expected_keys or
            len(comparison_keys)!=len(set(comparison_keys)) or set(comparison_keys)!=expected_comparisons or
            len(profile_keys)!=len(set(profile_keys)) or set(profile_keys)!=expected_profiles):
            raise ValueError("incomplete or duplicate policy/cap/profile records")
        if modern and payload.get('completed_partitions')!=[list(p) for p in sorted(completed)]:
            raise ValueError("completion inventory disagrees")
        if any(r['queries']!=queries[r['country']] for r in metrics):
            raise ValueError("query population disagrees")
        for row in profiles:
            if row['split']!=('train-dev' if row['source']=='source1' else 'train-full'):
                raise ValueError("profile split disagrees")
            if row['source']=='source1' and row['records']!=queries[row['country']]:
                raise ValueError("query profile population disagrees")
    except (KeyError,TypeError,ValueError) as exc:
        raise RuntimeError(f"Invalid completed-partition records: {exc}") from exc
    fingerprints = partition_artifacts(audit.output,completed)
    if modern and payload.get('artifacts')!=fingerprints:
        raise RuntimeError("Completed partition artifact hash mismatch")
    for country,source in sorted(completed):
        prefix = 'S2-' if source=='source2' else 'S3-'
        audit.run(f"""CREATE OR REPLACE TABLE checkpoint_truth AS SELECT * FROM (
          SELECT source1_entity_id qid,unnest(str_split(matched_entity_ids,',')) tid
          FROM sampled_gt g JOIN sample s ON g.source1_entity_id=s.entity_id
          WHERE s.country={quote(country)} AND matched_entity_ids IS NOT NULL AND matched_entity_ids<>'')
          WHERE starts_with(tid,{quote(prefix)})""")
        for policy in POLICIES:
            name = policy['name']
            selected_metrics = [r for r in metrics if (r['country'],r['source'],r['policy'])==(country,source,name)]
            metric_file = audit.output/f"metrics_{part_name(country,source,name)}.json"
            if json_sha256(json.loads(metric_file.read_text(encoding='utf-8')))!=json_sha256(selected_metrics):
                raise RuntimeError(f"Completed metrics file disagrees with progress: {metric_file.name}")
            stage_file = audit.output/f"stages_{part_name(country,source,name)}.parquet"
            audit.run(f"CREATE OR REPLACE TABLE checkpoint_stages AS SELECT * FROM read_parquet({quote(stage_file)})")
            disagreement = audit.run("""SELECT count(*) FROM (
              (SELECT qid,tid FROM checkpoint_stages EXCEPT ALL SELECT * FROM checkpoint_truth)
              UNION ALL (SELECT * FROM checkpoint_truth EXCEPT ALL SELECT qid,tid FROM checkpoint_stages))""").fetchone()[0]
            invalid = audit.run("""SELECT count(*) FROM checkpoint_stages WHERE shared IS NULL OR usable IS NULL
              OR selected IS NULL OR target_address_empty IS NULL OR (usable AND NOT shared)
              OR (selected AND NOT usable) OR selected<>(rank IS NOT NULL) OR rank<1""").fetchone()[0]
            if disagreement or invalid:
                raise RuntimeError(f"Completed stages disagree with selected truth or stage invariants: {stage_file.name}")
            for record in selected_metrics:
                cap = record['cap']
                derived = audit.rows(f"""SELECT count(*) true_links,
                  count(*) FILTER(WHERE NOT shared) lost_no_shared_key,
                  count(*) FILTER(WHERE shared AND NOT usable) lost_frequency_gate,
                  count(*) FILTER(WHERE usable AND NOT selected) lost_query_key_budget,
                  count(*) FILTER(WHERE selected AND rank>{cap}) lost_rank_cap,
                  count(*) FILTER(WHERE rank<={cap}) retained_links,
                  count(*) FILTER(WHERE target_address_empty) true_links_missing_address,
                  count(*) FILTER(WHERE target_address_empty AND rank<={cap}) retained_missing_address_links
                  FROM checkpoint_stages""")[0]
                if any(record.get(k)!=v for k,v in derived.items()):
                    raise RuntimeError(f"Completed stage counts disagree with metrics: {stage_file.name}")
                if not 0<=record['retained_links']<=record['candidate_pairs']<=record['raw_pairs']:
                    raise RuntimeError("Completed candidate counts do not reconcile")
        for name in (p['name'] for p in POLICIES if p['name']!='legacy'):
            stored = next(r for r in comparisons if (r['country'],r['source'],r['policy'])==(country,source,name))
            if stored!=compare_stages(audit,country,source,name):
                raise RuntimeError("Completed comparison disagrees with stage files")
    return profiles,metrics,comparisons,completed


def locate_data():
    paths = []
    for root in (Path("/kaggle/input"),Path.cwd()):
        if root.exists():
            paths.extend(root.glob("**/dataset/train/train_ground_truth.tsv"))
    roots = sorted({p.parent.parent.resolve() for p in paths})
    if len(roots)!=1:
        raise RuntimeError(f"Set data_root explicitly; found {len(roots)} official dataset copies")
    return roots[0]


def run_diagnostics(data_root=None, output=None, per_country=1000, memory_mb=512):
    if duckdb.__version__ != REFERENCE_DUCKDB:
        raise RuntimeError("This diagnostic preserves the original DuckDB 1.3.2 development/validation split. "
                           "Use the existing Kaggle kernel with that version or the documented isolated local runtime. "
                           "Do not downgrade a loaded notebook package in place.")
    if per_country<1 or memory_mb<128:
        raise ValueError("per_country must be positive and memory_mb must be >=128")
    data = Path(data_root) if data_root else locate_data()
    default = Path("/kaggle/working/aj_preprocessing_diagnostics") if Path("/kaggle/working").exists() else Path.cwd()/"work"/"aj_preprocessing_diagnostics"
    output = Path(output) if output else default
    required = [data/"train"/f"train_source{i}.tsv" for i in (1,2,3)]+[data/"train"/"train_ground_truth.tsv"]
    manifest = {"diagnostics_version": DIAGNOSTICS_VERSION,"preprocessing_version": PREPROCESS_VERSION,
      "duckdb": duckdb.__version__,"per_country":per_country,"memory_mb":memory_mb,
      "sample_rule":"5 <= DuckDB 1.3.2 hash(entity_id)%100 < 15; order by sha256('aj-diagnostic-v1:' || entity_id), entity_id within country",
      "sources":[{"file":p.name,"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in required],
      "policies":POLICIES,"caps":CAPS,"label_use":"after candidate ranking only"}
    manifest_path = output/"input_manifest.json"
    if manifest_path.exists():
        found = json.loads(manifest_path.read_text(encoding="utf-8"))
        if json_sha256(found)!=json_sha256(manifest):
            raise RuntimeError("Input manifest mismatch: data, sample, configuration or version changed; use a fresh output directory")
    else:
        if output.exists() and any(output.iterdir()):
            raise RuntimeError("Nonempty output directory has no input manifest; use a fresh output directory")
        output.mkdir(parents=True,exist_ok=True)
        atomic_json(manifest_path,manifest)
    audit = Audit(output,memory_mb)
    start = time.time()
    try:
        log(f"Diagnostic dev queries: up to {per_country:,}/country; full target-country populations; DuckDB {duckdb.__version__}")
        audit.run(f"""CREATE OR REPLACE TABLE sample AS SELECT *,hash(entity_id)%100 AS original_bucket
          FROM {read_tsv(required[0])} WHERE hash(entity_id)%100>=5 AND hash(entity_id)%100<15
          QUALIFY row_number() OVER(PARTITION BY country ORDER BY sha256('aj-diagnostic-v1:' || entity_id),entity_id)<={int(per_country)}""")
        frozen = output/'frozen_queries.parquet'
        if frozen.exists():
            try:
                audit.run(f"CREATE OR REPLACE VIEW frozen_checkpoint AS SELECT * FROM read_parquet({quote(frozen)})")
                schema = audit.run("DESCRIBE sample").fetchall()
                frozen_schema = audit.run("DESCRIBE frozen_checkpoint").fetchall()
                difference = audit.run("""SELECT count(*) FROM (
                  (SELECT * FROM sample EXCEPT ALL SELECT * FROM frozen_checkpoint)
                  UNION ALL (SELECT * FROM frozen_checkpoint EXCEPT ALL SELECT * FROM sample))""").fetchone()[0]
                if schema!=frozen_schema or difference:
                    raise RuntimeError("Frozen sample differs from the recomputed development selection")
            except duckdb.Error as exc:
                raise RuntimeError("Frozen sample is unreadable or has an incompatible schema") from exc
        else:
            if (output/'progress.json').exists():
                raise RuntimeError("Progress exists but its frozen sample is missing")
            atomic_copy(audit,"SELECT * FROM sample",frozen,"FORMAT parquet,COMPRESSION zstd")
        sample_sha = sha256_file(frozen)
        membership = audit.rows("SELECT country,count(*) queries,min(original_bucket) min_bucket,max(original_bucket) max_bucket FROM sample GROUP BY country ORDER BY country")
        if not membership or any(r['min_bucket']<5 or r['max_bucket']>=15 for r in membership):
            raise RuntimeError("Empty or contaminated development sample")
        audit.run(f"""CREATE OR REPLACE TABLE sampled_gt AS SELECT g.* FROM {read_tsv(required[3])} g
          JOIN sample s ON g.source1_entity_id=s.entity_id""")
        if audit.run("SELECT count(*) FROM sampled_gt").fetchone()[0] != sum(r['queries'] for r in membership):
            raise RuntimeError("Ground truth coverage differs from sampled Source 1 membership")
        profiles,metrics,comparisons,completed = load_progress(audit,manifest,sample_sha,membership)
        resumed = [list(p) for p in sorted(completed)]
        for entry in membership:
            country = entry['country']
            audit.run(f"CREATE OR REPLACE TABLE normalized_queries AS {population_sql(f'(SELECT * FROM sample WHERE country={quote(country)})')}")
            query_profile = profile(audit,"normalized_queries","train-dev","source1",country)
            previous = next((r for r in profiles if r['country']==country and r['source']=='source1'),None)
            if previous is not None:
                if previous!=query_profile:
                    raise RuntimeError("Completed query profile differs from the frozen development sample")
            else:
                profiles.append(query_profile)
            for source,prefix in (("source2","S2-"),("source3","S3-")):
                if (country,source) in completed:
                    log(f"Reuse verified completed partition: {country} {source}")
                    continue
                log(f"Normalize full train {country} {source}")
                target_relation = f"(SELECT * FROM {read_tsv(data/'train'/f'train_{source}.tsv')} WHERE country={quote(country)})"
                normalize_targets_batched(audit,target_relation)
                profiles.append(profile(audit,"normalized_targets","train-full",source,country))
                audit.run(f"""CREATE OR REPLACE TABLE truth AS SELECT source1_entity_id qid,unnest(str_split(matched_entity_ids,',')) tid
                  FROM sampled_gt g JOIN normalized_queries q ON g.source1_entity_id=q.rid
                  WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids<>''""")
                audit.run(f"DELETE FROM truth WHERE NOT starts_with(tid,{quote(prefix)})")
                orphan = audit.run("SELECT count(*) FROM truth ANTI JOIN normalized_targets ON truth.tid=normalized_targets.rid").fetchone()[0]
                if orphan:
                    raise RuntimeError(f"{orphan} positive targets absent from full source-country population")
                for view in ("legacy","new"):
                    nm,ad = ("legacy_nm","legacy_ad") if view=="legacy" else ("name_core","address_canonical")
                    for alias,table in (("q","normalized_queries"),("t","normalized_targets")):
                        batch_col = ',diagnostic_batch' if alias=='t' else ''
                        audit.run(f"CREATE OR REPLACE VIEW {alias} AS SELECT rid,{nm} nm,{ad} ad,num{batch_col} FROM {table}")
                    audit.run(f"CREATE OR REPLACE TABLE qk AS {key_sql('q',view=='new')}")
                    audit.run("CREATE OR REPLACE TABLE inventory AS SELECT DISTINCT key FROM qk")
                    log(f"  {view}: count key frequencies across ALL targets")
                    full_frequencies(audit,view=='new')
                    audit.run("CREATE OR REPLACE VIEW positive_targets AS SELECT t.* FROM t SEMI JOIN truth ON t.rid=truth.tid")
                    audit.run(f"CREATE OR REPLACE TABLE positive_tk AS {key_sql('positive_targets',view=='new')}")
                    for policy in [p for p in POLICIES if p['view']==view]:
                        metrics.extend(run_policy(audit,policy,country,source))
                comparisons.extend(compare_stages(audit,country,source,p['name'])
                                   for p in POLICIES if p['name']!='legacy')
                save_progress(audit,manifest,sample_sha,profiles,metrics,comparisons)
                audit.run("CHECKPOINT")
        save_progress(audit,manifest,sample_sha,profiles,metrics,comparisons)
        report = {"manifest":manifest,"sample":membership,"profiles":profiles,"metrics":metrics,"comparisons":comparisons,
          "resumed_partitions":resumed,"checkpoint_version":CHECKPOINT_VERSION,
          "semantics_sha256":semantics_sha256(),"sample_sha256":sample_sha,
          "elapsed_seconds":time.time()-start,"scope":"Balanced exploratory development sample. No classifier fitting or test predictions. France has no labeled evaluation.",
          "validation":"PASS: development split exclusion, truth coverage, stage monotonicity and loss reconciliation"}
        atomic_json(output/"report.json",report)
        log(f"DIAGNOSTICS COMPLETE: {output/'report.json'}")
        return report
    finally:
        audit.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root')
    parser.add_argument('--output')
    parser.add_argument('--per-country',type=int,default=1000)
    parser.add_argument('--memory-mb',type=int,default=512)
    args=parser.parse_args()
    if args.per_country<1 or args.memory_mb<128:
        parser.error('per-country must be positive and memory-mb must be >=128')
    run_diagnostics(args.data_root,args.output,args.per_country,args.memory_mb)


if __name__ == '__main__':
    main()
