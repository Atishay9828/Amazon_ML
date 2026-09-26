"""Conservative matching views for the four-column challenge record schema.

The raw columns are returned unchanged.  Derived text is normalized to NFC,
lowercased, and spaced at punctuation while retaining Unicode letters, marks,
and numbers.  Accent folding is conditional: after punctuation cleanup it is
applied only to Latin-only text.  Mixed-script text and scripts such as
Devanagari keep their combining marks.

``name_core`` removes a sequence of known legal suffix tokens only at the end.
``address_canonical`` is an additional token-abbreviation view, not an address
parser.  ``address_numbers`` contains every maximal Unicode number run in input
order, including runs inside alphanumeric tokens, with repeats and leading
zeros preserved.  It makes no claim about which number is a house number,
unit, floor, or postcode.

This module imports no third-party package and performs no I/O.  The returned
SQL targets DuckDB 1.3.2 and later.  The caller supplies a trusted relation SQL
fragment with VARCHAR columns entity_id, business_name, business_address, and
country; it may be a quoted table name, table function, or parenthesized query.
Do not pass untrusted user text as the relation fragment.
"""

from __future__ import annotations


PREPROCESS_VERSION = "aj-preprocess-v1-unicode-latin-fold"

# These are whole tokens, and only a trailing sequence is removed.  In
# particular, generic words such as services, group, trade, and company remain.
_LEGAL_SUFFIXES = (
    "incorporated", "inc", "corporation", "corp", "limited", "ltd",
    "llc", "llp", "private", "pvt", "plc",
)
_SUFFIX_ALTERNATION = "|".join(_LEGAL_SUFFIXES)
_TRAILING_SUFFIX_PATTERN = (
    rf"(^| )({_SUFFIX_ALTERNATION})( ({_SUFFIX_ALTERNATION}))*$"
)


def normalized_sql(input_relation: str) -> str:
    """Return a SELECT adding separate, explicitly named matching views.

    The original fields, including NULLs and country labels, are not changed.
    NULL or punctuation-only names/addresses become empty derived strings.
    ``name_is_empty`` and ``address_is_empty`` refer to those folded strings;
    ``name_core_is_empty`` separately identifies a suffix-only name whose
    folded text still exists.  ``rid`` is exactly ``entity_id``.

    The relation argument is trusted SQL, not a path or a SQL value to quote.
    """
    if not isinstance(input_relation, str):
        raise TypeError("input_relation must be a trusted SQL relation string")
    relation = input_relation.strip()
    if not relation:
        raise ValueError("input_relation must not be empty")

    return rf"""
WITH _aj_preprocess_unicode AS (
    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        lower(nfc_normalize(coalesce(business_name, ''))) AS name_unicode,
        lower(nfc_normalize(coalesce(business_address, ''))) AS address_unicode
    FROM {relation}
),
_aj_preprocess_spaced AS (
    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        trim(regexp_replace(name_unicode, '[^\p{{L}}\p{{M}}\p{{N}}]+', ' ', 'g'))
            AS name_spaced,
        trim(regexp_replace(address_unicode, '[^\p{{L}}\p{{M}}\p{{N}}]+', ' ', 'g'))
            AS address_spaced
    FROM _aj_preprocess_unicode
),
_aj_preprocess_folded AS (
    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        CASE WHEN NOT regexp_matches(name_spaced, '[^\p{{Latin}}\p{{N}} ]')
            THEN strip_accents(name_spaced) ELSE name_spaced END AS name_folded,
        CASE WHEN NOT regexp_matches(address_spaced, '[^\p{{Latin}}\p{{N}} ]')
            THEN strip_accents(address_spaced) ELSE address_spaced END AS address_folded
    FROM _aj_preprocess_spaced
),
_aj_preprocess_views AS (
    SELECT
        entity_id,
        business_name,
        business_address,
        country,
        name_folded,
        trim(regexp_replace(name_folded, '{_TRAILING_SUFFIX_PATTERN}', '')) AS name_core,
        address_folded,
        array_to_string(
            list_transform(string_split(address_folded, ' '), tok -> CASE tok
                WHEN 'road' THEN 'rd'
                WHEN 'street' THEN 'st'
                WHEN 'avenue' THEN 'ave'
                WHEN 'boulevard' THEN 'blvd'
                WHEN 'lane' THEN 'ln'
                WHEN 'drive' THEN 'dr'
                ELSE tok END),
            ' '
        ) AS address_canonical,
        regexp_extract_all(address_folded, '\p{{N}}+') AS address_numbers
    FROM _aj_preprocess_folded
)
SELECT
    entity_id,
    business_name,
    business_address,
    country,
    entity_id AS rid,
    name_folded,
    name_core,
    address_folded,
    address_canonical,
    address_numbers,
    name_folded = '' AS name_is_empty,
    address_folded = '' AS address_is_empty,
    name_core = '' AS name_core_is_empty
FROM _aj_preprocess_views
""".strip()


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



"""Aggregate-only views and charts for the preprocessing diagnostic notebook."""

import json
from pathlib import Path



def aggregate_metrics(report):
    fields = ('true_links','retained_links','candidate_pairs','lost_no_shared_key',
              'lost_frequency_gate','lost_query_key_budget','lost_rank_cap',
              'true_links_missing_address','retained_missing_address_links')
    groups = {}
    for row in report['metrics']:
        key = (row['policy'],row['cap'])
        group = groups.setdefault(key,dict(policy=key[0],cap=key[1],**{field:0 for field in fields}))
        for field in fields:
            group[field] += row[field]
    for group in groups.values():
        group['candidate_link_recall'] = group['retained_links']/group['true_links'] if group['true_links'] else None
        group['missing_address_link_recall'] = group['retained_missing_address_links']/group['true_links_missing_address'] if group['true_links_missing_address'] else None
        group['known_nonmatch_candidates'] = group['candidate_pairs']-group['retained_links']
        group['candidate_pair_true_link_share'] = group['retained_links']/group['candidate_pairs'] if group['candidate_pairs'] else None
        group['candidate_pair_false_link_share'] = group['known_nonmatch_candidates']/group['candidate_pairs'] if group['candidate_pairs'] else None
    return list(groups.values())


def entity_f05(true_links, retained_links):
    """Oracle per-Source-1 F0.5 if the classifier keeps only reachable truths.

    These are upper-bound values: the diagnostic uses labels after retrieval
    solely to estimate the best possible result from this candidate set.
    """
    true_links, retained_links = int(true_links), int(retained_links)
    if true_links < 0 or retained_links < 0 or retained_links > true_links:
        raise ValueError("retained_links must be between zero and true_links")
    if true_links == 0:
        return 1.0  # An empty prediction correctly identifies a singleton.
    if retained_links == 0:
        return 0.0
    recall = retained_links / true_links
    precision = 1.0  # Oracle discards every retrieved false candidate.
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def aggregate_entity_metrics(entity_rows):
    """Combine Source 2/3 stage rows to challenge-style per-entity aggregates."""
    if not entity_rows:
        raise ValueError("entity_rows must include every sampled Source 1 query")
    identifiers = [row["qid"] for row in entity_rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("entity_rows must have one combined row per Source 1 query")
    rows, recalls, f05_values = [], [], []
    for row in entity_rows:
        truth, retained = int(row["true_links"]), int(row["retained_links"])
        if truth < 0 or retained < 0 or retained > truth:
            raise ValueError("retrieved true links cannot exceed a query's true-link count")
        rows.append((truth, retained))
        if truth:
            recalls.append(retained / truth)
        f05_values.append(entity_f05(truth, retained))
    singleton_count = sum(truth == 0 for truth, _ in rows)
    positives = [(truth, retained) for truth, retained in rows if truth > 0]
    complete = sum(retained == truth for truth, retained in positives)
    partial = sum(0 < retained < truth for truth, retained in positives)
    zero_retrieved = sum(retained == 0 for truth, retained in positives)
    true_links = sum(truth for truth, _ in rows)
    retained_links = sum(retained for _, retained in rows)
    sorted_recalls = sorted(recalls)
    middle = len(sorted_recalls) // 2
    median_recall = None if not sorted_recalls else (
        sorted_recalls[middle] if len(sorted_recalls) % 2
        else (sorted_recalls[middle - 1] + sorted_recalls[middle]) / 2
    )
    return {
        "queries": len(rows),
        "singletons": singleton_count,
        "positive_queries": len(positives),
        "complete_positive_queries": complete,
        "partial_positive_queries": partial,
        "positive_queries_with_zero_true_links_retrieved": zero_retrieved,
        "micro_true_link_recall": retained_links / true_links if true_links else None,
        "macro_link_recall_positive_queries": sum(recalls) / len(recalls) if recalls else None,
        "median_link_recall_positive_queries": median_recall,
        "complete_positive_query_rate": complete / len(positives) if positives else None,
        "positive_query_recall_coverage": sum(r > 0 for r in recalls) / len(positives) if positives else None,
        "singleton_rate": singleton_count / len(rows),
        "oracle_macro_f05_all_queries": sum(f05_values) / len(rows),
        "scope": "Source 2 and Source 3 combined per sampled Source 1 entity; oracle upper bound, not a fitted matcher score.",
    }


def entity_level_metrics(report, output):
    """Compute singleton-aware entity metrics from persisted truth-stage artifacts."""
    import duckdb

    output = Path(output)
    db = duckdb.connect(str(output / "scratch.duckdb"), read_only=True)
    try:
        results = []
        for sample in report["sample"]:
            country = sample["country"]
            truth_relation = f"""(SELECT g.source1_entity_id AS qid,
              CASE WHEN g.matched_entity_ids IS NULL OR g.matched_entity_ids=''
                   THEN 0 ELSE len(str_split(g.matched_entity_ids,',')) END AS true_links
              FROM sampled_gt g JOIN sample s ON g.source1_entity_id=s.entity_id
              WHERE s.country={quote(country)})"""
            for policy in sorted({row["policy"] for row in report["metrics"]}):
                paths = [output / f"stages_{part_name(country,source,policy)}.parquet"
                         for source in ("source2", "source3")]
                for cap in sorted({row["cap"] for row in report["metrics"]}):
                    stage_paths = "[" + ",".join(quote(path) for path in paths) + "]"
                    entity_rows = [dict(zip(("qid", "true_links", "retained_links"), row))
                        for row in db.execute(f"""WITH truth AS (SELECT * FROM {truth_relation}),
                          retrieved AS (
                            SELECT qid,count(*) FILTER(WHERE rank<={int(cap)})::BIGINT AS retained_links
                            FROM read_parquet({stage_paths}) GROUP BY qid
                          )
                          SELECT t.qid,t.true_links,coalesce(r.retained_links,0)::BIGINT
                          FROM truth t LEFT JOIN retrieved r USING(qid) ORDER BY t.qid""").fetchall()]
                    values = aggregate_entity_metrics(entity_rows)
                    values.update(country=country,policy=policy,cap=cap,
                                  true_links=sum(row["true_links"] for row in entity_rows),
                                  retained_links=sum(row["retained_links"] for row in entity_rows))
                    results.append(values)
        return results
    finally:
        db.close()


def test_data_profile(data_root, output, per_country=1000, memory_mb=512):
    """Unlabeled test shape only; no test performance claim or business lookup."""
    data_root,output = Path(data_root),Path(output)
    audit = Audit(output/'test_profile',memory_mb)
    population, sampled = [],[]
    try:
        for source in ('source1','source2','source3'):
            path = data_root/'test'/f'test_{source}.tsv'
            rows = audit.rows(f"""SELECT country,count(*) records,
              count(*) FILTER(WHERE business_name IS NULL OR trim(business_name)='') raw_name_empty,
              count(*) FILTER(WHERE business_address IS NULL OR trim(business_address)='') raw_address_empty
              FROM {read_tsv(path)} GROUP BY country ORDER BY country""")
            population.extend(dict(source=source,**row) for row in rows)
            relation = f"""(SELECT * FROM {read_tsv(path)} WHERE hash(entity_id)%100<1
              QUALIFY row_number() OVER(PARTITION BY country ORDER BY sha256('aj-unlabeled-profile-v1:' || entity_id),entity_id)<={int(per_country)})"""
            audit.run(f"CREATE OR REPLACE TABLE test_sample AS {population_sql(relation)}")
            countries = [row[0] for row in audit.run('SELECT DISTINCT country FROM test_sample ORDER BY country').fetchall()]
            for country in countries:
                audit.run(f"CREATE OR REPLACE VIEW country_sample AS SELECT * FROM test_sample WHERE country={quote(country)}")
                sampled.append(profile(audit,'country_sample','test-sample',source,country))
        result = {'population':population,'sampled_normalization':sampled,
          'scope':'Full test row counts by source/country; normalization profiles use up to 1,000 deterministic records per source/country. No test labels or performance estimates.'}
        atomic_json(output/'test_profile.json',result)
        return result
    finally:
        audit.close()


def render_report(report, output, show=False):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    output=Path(output)
    aggregated=aggregate_metrics(report)
    entity_metrics = report.get("entity_metrics", [])
    table=pd.DataFrame(aggregated)
    table.to_csv(output/'summary_metrics.csv',index=False)
    pd.DataFrame(report['metrics']).to_csv(output/'metrics_by_country_source.csv',index=False)
    pd.DataFrame(report['profiles']).to_csv(output/'normalization_profile.csv',index=False)
    pd.DataFrame(report['comparisons']).to_csv(output/'paired_comparisons.csv',index=False)
    if entity_metrics:
        pd.DataFrame(entity_metrics).to_csv(output/'entity_metrics.csv',index=False)
    source1_queries=sum(row['queries'] for row in report['sample'])
    pretty={'legacy':'Original v3','preprocessed':'New preprocessing',
            'preprocessed_df256_k3':'DF limit 256; 3 keys',
            'preprocessed_df64_k6':'DF limit 64; 6 keys',
            'preprocessed_wider':'DF limit 256; 6 keys'}
    colors={'legacy':'#555f6a','preprocessed':'#127c87',
            'preprocessed_df256_k3':'#b04a5a','preprocessed_df64_k6':'#5c6bc0',
            'preprocessed_wider':'#d7791f'}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold'})
    fig,axes=plt.subplots(1,3,figsize=(17,4.8),layout='constrained')
    for policy in pretty:
        rows=table[table.policy==policy].sort_values('cap')
        axes[0].plot(rows.cap,rows.candidate_link_recall*100,marker='o',label=pretty[policy],color=colors[policy])
        axes[1].plot(rows.cap,rows.missing_address_link_recall*100,marker='o',label=pretty[policy],color=colors[policy])
        if entity_metrics:
            entity_rows=pd.DataFrame(entity_metrics)
            entity_rows=entity_rows[entity_rows.policy==policy]
            ceiling=(entity_rows.assign(weighted=entity_rows.oracle_macro_f05_all_queries*entity_rows.queries)
                     .groupby('cap').agg(weighted=('weighted','sum'),queries=('queries','sum')))
            axes[2].plot(ceiling.index,ceiling.weighted/ceiling.queries*100,marker='o',label=pretty[policy],color=colors[policy])
    for ax in axes:
        ax.set_ylim(0,100)
        ax.set_xlabel('Candidate cap per target source')
        ax.set_ylabel('Labeled true links retrieved (%)')
        ax.grid(alpha=.2)
    axes[0].set_title('All sampled true links')
    axes[1].set_title('True links with a missing target address')
    axes[0].legend(fontsize=9,loc='lower right')
    if entity_metrics:
        axes[2].set_title('Oracle macro F0.5 ceiling')
        axes[2].set_ylabel('Optimistic upper bound (%)')
        axes[2].legend(fontsize=9,loc='lower right')
    else:
        axes[2].set_visible(False)
    fig.suptitle(f'Retrieval diagnostic: {source1_queries:,} development queries; full target indexes')
    curve=output/'recall_vs_cap.png'
    fig.savefig(curve,dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    rows=table[table.cap==40].set_index('policy').loc[list(pretty)]
    components=[('retained_links','Retrieved','#17816f'),('lost_no_shared_key','No shared eligible key','#684c91'),
                ('lost_frequency_gate','Frequency gate','#c56827'),('lost_query_key_budget','Query key limit','#c34d62'),
                ('lost_rank_cap','Rank below top 40','#708497')]
    fig,ax=plt.subplots(figsize=(12,4.3),layout='constrained')
    left=[0.0]*len(rows)
    for field,label,color in components:
        values=(rows[field]/rows.true_links*100).fillna(0).to_numpy()
        ax.barh([pretty[p] for p in rows.index],values,left=left,label=label,color=color)
        for i,value in enumerate(values):
            if value>=6:
                ax.text(left[i]+value/2,i,f'{value:.1f}%',ha='center',va='center',color='white',fontsize=10)
        left=[a+b for a,b in zip(left,values)]
    ax.set_xlim(0,100)
    ax.set_xlabel('Share of sampled labeled links (%)')
    ax.set_title('First stage that loses each true link — cap 40 per source')
    ax.invert_yaxis()
    ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.2),fontsize=9)
    loss=output/'loss_by_stage.png'
    fig.savefig(loss,dpi=150,bbox_inches='tight')
    if show:
        plt.show()
    plt.close(fig)
    print(table[['policy','cap','candidate_link_recall','missing_address_link_recall','candidate_pairs']].to_string(index=False))
    if entity_metrics:
        entity_table=pd.DataFrame(entity_metrics)
        print(entity_table[['country','policy','cap','micro_true_link_recall',
          'macro_link_recall_positive_queries','complete_positive_query_rate',
          'positive_queries_with_zero_true_links_retrieved','oracle_macro_f05_all_queries']].to_string(index=False))
    print('Candidate-set true/false shares use labels on this diagnostic sample; they are not fitted-model precision. Oracle macro F0.5 is an optimistic upper bound, not a fitted matcher or leaderboard score.')
    return {'aggregated':aggregated,'figures':[str(curve),str(loss)]}



"""Synthetic-only normalization checks; no challenge records are embedded."""


import unittest

import duckdb



class PreprocessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.db = duckdb.connect()
        self.db.execute(
            "CREATE TABLE synthetic_records (entity_id VARCHAR, business_name VARCHAR, "
            "business_address VARCHAR, country VARCHAR)"
        )

    def tearDown(self) -> None:
        self.db.close()

    def normalize(self, name, address="", country="Exampleland", entity_id="S1-synthetic"):
        self.db.execute("DELETE FROM synthetic_records")
        self.db.execute(
            "INSERT INTO synthetic_records VALUES (?, ?, ?, ?)",
            [entity_id, name, address, country],
        )
        cursor = self.db.execute(normalized_sql("synthetic_records"))
        values = cursor.fetchone()
        return dict(zip([column[0] for column in cursor.description], values))

    def test_raw_columns_and_rid_are_retained_exactly(self):
        row = self.normalize("  ACME Ltd.  ", "12 Main Road", "Unknown Region X", " S1-01 ")
        self.assertEqual(row["business_name"], "  ACME Ltd.  ")
        self.assertEqual(row["business_address"], "12 Main Road")
        self.assertEqual(row["entity_id"], " S1-01 ")
        self.assertEqual(row["rid"], row["entity_id"])
        self.assertEqual(row["country"], "Unknown Region X")

    def test_whitespace_is_trimmed_and_collapsed(self):
        row = self.normalize(" \t ACME\n Widgets \r Ltd. \t", " \t 12  Main\nRoad,\r Suite 4 \t")
        self.assertEqual(row["name_folded"], "acme widgets ltd")
        self.assertEqual(row["name_core"], "acme widgets")
        self.assertEqual(row["address_folded"], "12 main road suite 4")
        self.assertEqual(row["address_canonical"], "12 main rd suite 4")

    def test_nulls_keep_raw_nulls_and_produce_empty_views(self):
        row = self.normalize(None, None, None, None)
        for field in ("entity_id", "business_name", "business_address", "country", "rid"):
            self.assertIsNone(row[field])
        for field in ("name_folded", "name_core", "address_folded", "address_canonical"):
            self.assertEqual(row[field], "")
        self.assertEqual(row["address_numbers"], [])
        self.assertTrue(row["name_is_empty"])
        self.assertTrue(row["name_core_is_empty"])
        self.assertTrue(row["address_is_empty"])

    def test_punctuation_only_fields_are_flagged_empty(self):
        row = self.normalize(" & / -- . ", " , # ( ) ")
        self.assertEqual(row["name_folded"], "")
        self.assertEqual(row["address_folded"], "")
        self.assertTrue(row["name_is_empty"])
        self.assertTrue(row["address_is_empty"])

    def test_repeated_legal_suffixes_are_removed_only_at_end(self):
        row = self.normalize("Acme Pvt. Ltd. Limited LLC")
        self.assertEqual(row["name_folded"], "acme pvt ltd limited llc")
        self.assertEqual(row["name_core"], "acme")
        row = self.normalize("The Ltd Studio Pvt Ltd")
        self.assertEqual(row["name_core"], "the ltd studio")
        row = self.normalize("Acme Pvt Ltd Studio")
        self.assertEqual(row["name_core"], "acme pvt ltd studio")

    def test_only_legal_suffixes_do_not_destroy_folded_view(self):
        row = self.normalize("Ltd. Pvt Limited")
        self.assertEqual(row["name_folded"], "ltd pvt limited")
        self.assertEqual(row["name_core"], "")
        self.assertFalse(row["name_is_empty"])
        self.assertTrue(row["name_core_is_empty"])

    def test_suffixes_require_whole_tokens_and_generic_words_remain(self):
        row = self.normalize("Incognito Ltdx Services Group Company")
        self.assertEqual(row["name_core"], "incognito ltdx services group company")
        row = self.normalize("Limited Edition Inc")
        self.assertEqual(row["name_core"], "limited edition")

    def test_ampersands_are_spaced_without_inventing_and(self):
        row = self.normalize("A&B and Sons")
        self.assertEqual(row["name_folded"], "a b and sons")
        self.assertNotEqual(row["name_folded"], "a and b and sons")

    def test_apostrophes_and_punctuation_separate_tokens(self):
        self.assertEqual(self.normalize("O'Reilly's")["name_folded"], "o reilly s")
        row = self.normalize("Alpha/Beta—Gamma, (R&D)")
        self.assertEqual(row["name_folded"], "alpha beta gamma r d")

    def test_address_abbreviations_are_an_additional_view(self):
        address = "1 Road Rd Street St Avenue Ave Boulevard Blvd Lane Ln Drive Dr"
        row = self.normalize("Synthetic", address)
        self.assertEqual(row["business_address"], address)
        self.assertEqual(row["address_folded"], address.lower())
        self.assertEqual(row["address_canonical"], "1 rd rd st st ave ave blvd blvd ln ln dr dr")

    def test_address_abbreviations_match_whole_tokens(self):
        row = self.normalize("Synthetic", "2 Broadway Streetlight Driveway, Rd.")
        self.assertEqual(row["address_canonical"], "2 broadway streetlight driveway rd")

    def test_all_number_runs_are_retained_without_assigning_semantics(self):
        row = self.normalize("Synthetic", "Unit 4B, 12-14 Main Road, Floor 2, 560001")
        self.assertEqual(row["address_numbers"], ["4", "12", "14", "2", "560001"])
        self.assertEqual(row["address_canonical"], "unit 4b 12 14 main rd floor 2 560001")
        self.assertFalse(any("house" in key or "first_number" in key for key in row))

    def test_number_order_repeats_and_leading_zeros_are_preserved(self):
        row = self.normalize("Synthetic", "Suite 2, 2 Main St 00002")
        self.assertEqual(row["address_numbers"], ["2", "2", "00002"])

    def test_single_digits_fraction_runs_and_postcode_leading_order(self):
        row = self.normalize("Synthetic", "00002, Unit 7, 1/2 Main Road")
        self.assertEqual(row["address_numbers"], ["00002", "7", "1", "2"])
        self.assertEqual(row["business_address"], "00002, Unit 7, 1/2 Main Road")
        self.assertEqual(row["address_canonical"], "00002 unit 7 1 2 main rd")

    def test_reordered_addresses_keep_their_order(self):
        first = self.normalize("Synthetic", "Unit 4, 12 Main Road, 00002")
        second = self.normalize("Synthetic", "00002, Main Road 12, Unit 4")
        self.assertEqual(first["address_numbers"], ["4", "12", "00002"])
        self.assertEqual(second["address_numbers"], ["00002", "12", "4"])
        self.assertNotEqual(first["address_canonical"], second["address_canonical"])

    def test_unicode_number_runs_are_preserved(self):
        row = self.normalize("Synthetic", "１２ Rue ٣، Apt ４B")
        self.assertEqual(row["address_numbers"], ["１２", "٣", "４"])

    def test_latin_accents_are_folded_after_nfc_normalization(self):
        row = self.normalize("CAFÉ Société Été Ltd", "12 Allée Émile Road")
        self.assertEqual(row["name_folded"], "cafe societe ete ltd")
        self.assertEqual(row["name_core"], "cafe societe ete")
        self.assertEqual(row["address_canonical"], "12 allee emile rd")
        row = self.normalize("Cafe\u0301 Ltd")
        self.assertEqual(row["name_core"], "cafe")

    def test_cyrillic_cjk_and_arabic_letters_survive(self):
        row = self.normalize("北京 Москва شركة", "東京 12 شارع")
        self.assertEqual(row["name_folded"], "北京 москва شركة")
        self.assertEqual(row["address_folded"], "東京 12 شارع")
        self.assertFalse(row["name_is_empty"])

    def test_devanagari_vowel_marks_survive(self):
        name = "श्री गणेश प्राइवेट लिमिटेड"
        address = "१२ महात्मा रोड, यूनिट ३"
        row = self.normalize(name, address)
        self.assertEqual(row["name_folded"], name)
        self.assertEqual(row["name_core"], name)
        self.assertEqual(row["address_folded"], "१२ महात्मा रोड यूनिट ३")
        self.assertEqual(row["address_numbers"], ["१२", "३"])

    def test_mixed_script_text_keeps_accents_and_combining_marks(self):
        row = self.normalize("Café श्री Industries Pvt Ltd", "12 Café सड़क Road")
        self.assertEqual(row["name_folded"], "café श्री industries pvt ltd")
        self.assertEqual(row["name_core"], "café श्री industries")
        self.assertEqual(row["address_canonical"], "12 café सड़क rd")

    def test_country_is_an_open_set_and_is_not_normalized(self):
        for country in ("France", "日本", "Côte d’Ivoire", "  Region-Ω  ", ""):
            with self.subTest(country=country):
                self.assertEqual(self.normalize("Synthetic", country=country)["country"], country)

    def test_folded_views_are_idempotent(self):
        for name, address in (
            (" Café ☕ Ltd. ", "12 Café Road, Unit 4"),
            ("Café श्री Pvt Ltd", "१२ सड़क Road"),
            ("A&B", " \t 2 -- Main Avenue . "),
            (None, None),
        ):
            with self.subTest(name=name):
                first = self.normalize(name, address)
                second = self.normalize(first["name_folded"], first["address_folded"])
                self.assertEqual(second["name_folded"], first["name_folded"])
                self.assertEqual(second["address_folded"], first["address_folded"])

    def test_core_and_canonical_views_are_idempotent(self):
        for name, address in (
            ("Acme Pvt Ltd", "Unit 4B, 12 Main Road 00002"),
            ("Café श्री Pvt Ltd", "१२ सड़क Road"),
            ("Ltd Pvt Limited", "1 Road Rd Street St"),
        ):
            with self.subTest(name=name):
                first = self.normalize(name, address)
                second = self.normalize(first["name_core"], first["address_canonical"])
                self.assertEqual(second["name_core"], first["name_core"])
                self.assertEqual(second["address_canonical"], first["address_canonical"])
                self.assertEqual(second["address_numbers"], first["address_numbers"])

    def test_version_and_relation_contract(self):
        self.assertIsInstance(PREPROCESS_VERSION, str)
        self.assertTrue(PREPROCESS_VERSION)
        with self.assertRaises(ValueError):
            normalized_sql(" \t ")
        with self.assertRaises(TypeError):
            normalized_sql(None)



"""Query-level stratified bootstrap for retrieval ceilings on a frozen sample."""

from pathlib import Path

import numpy as np



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



# ------------------------- EDITABLE SETTINGS -------------------------
DATA_ROOT = locate_data()
DIAGNOSTIC_OUTPUT = (Path('/kaggle/working/aj_preprocessing_ablations')
                     if Path('/kaggle/working').exists()
                     else Path.cwd()/'work'/'aj_preprocessing_ablations')
QUERIES_PER_COUNTRY = 1000
MEMORY_MB = 512
REUSE_COMPLETED_REPORT = True
# ---------------------------------------------------------------------

suite=unittest.defaultTestLoader.loadTestsFromTestCase(PreprocessingTests)
test_result=unittest.TextTestRunner(verbosity=1).run(suite)
if not test_result.wasSuccessful():
    raise RuntimeError('Preprocessing tests failed; diagnostic stopped')

report_path=DIAGNOSTIC_OUTPUT/'report.json'
report=None
if REUSE_COMPLETED_REPORT and report_path.is_file():
    saved=json.loads(report_path.read_text(encoding='utf-8'))
    manifest=saved.get('manifest',{})
    compatible=(manifest.get('diagnostics_version')==DIAGNOSTICS_VERSION
                and manifest.get('preprocessing_version')==PREPROCESS_VERSION
                and manifest.get('duckdb')==duckdb.__version__
                and manifest.get('per_country')==QUERIES_PER_COUNTRY
                and manifest.get('memory_mb')==MEMORY_MB
                and manifest.get('caps')==list(CAPS)
                and manifest.get('policies')==list(POLICIES)
                and saved.get('semantics_sha256')==semantics_sha256())
    if compatible:
        for source in manifest['sources']:
            path=DATA_ROOT/'train'/source['file']
            if not path.is_file() or sha256_file(path)!=source['sha256']:
                compatible=False
                break
    if compatible:
        report=saved
        print('Reusing completed diagnostic with matching input hashes and configuration.')
if report is None:
    report=run_diagnostics(DATA_ROOT,DIAGNOSTIC_OUTPUT,QUERIES_PER_COUNTRY,MEMORY_MB)

test_profile=test_data_profile(DATA_ROOT,DIAGNOSTIC_OUTPUT,QUERIES_PER_COUNTRY,MEMORY_MB)
figures=render_report(report,DIAGNOSTIC_OUTPUT,show=True)
print('Tests passed:',test_result.testsRun)
print('Report:',report_path)
print('Unlabeled test profile:',DIAGNOSTIC_OUTPUT/'test_profile.json')
print('Private examples and query membership stay in the working folder.')
