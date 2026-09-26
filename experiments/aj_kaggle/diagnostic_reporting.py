"""Aggregate-only views and charts for the preprocessing diagnostic notebook."""

import json
from pathlib import Path

from diagnostics import Audit, atomic_json, part_name, population_sql, profile, quote, read_tsv


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


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--report',required=True)
    parser.add_argument('--data-root')
    args=parser.parse_args()
    report_path=Path(args.report)
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if args.data_root:
        test_data_profile(args.data_root,report_path.parent)
    render_report(report,report_path.parent)
