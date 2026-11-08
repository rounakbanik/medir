# Data contracts

Each JSONL line is one UTF-8 JSON object. Extra fields are rejected to expose schema drift.
Source adapters retain raw files separately, and `drugir run` creates normalized files under
`data/processed/`. Those generated files should not be edited directly.

## Local entities: data/raw/local/entities.jsonl

```json
{"id":"drug:abacavir","kind":"drug","name":"Abacavir","aliases":[],"identifiers":{"drugbank":["DB01048"]},"sources":["manual:reviewed"]}
{"id":"disease:alzheimer","kind":"disease","name":"Alzheimer Disease","aliases":["Alzheimer's disease"],"cns":true}
```

`kind` is `drug` or `disease`. `approved` is true, false, or null; `approved_since` is an ISO date
or null. `bbb` is `positive`, `negative`, `conflicting`, or null. Do not infer these values from
co-occurrence or missing data. When manually specifying metadata, identify its source in `sources`.
Existing IDs merge aliases/identifiers; contradictory BBB observations become `conflicting`.
Identical aliases belonging to multiple entities remain ambiguous and are not text-linked.

## Local documents: data/raw/local/documents.jsonl

```json
{"id":"local:study-001","title":"Your study title","abstract":"The actual evidence text.","publication_date":"2020-05-31","date_precision":"month","entity_ids":["drug:abacavir","disease:alzheimer"],"source":"manual","url":"https://example.org/study"}
```

`id` and nonempty `abstract` are required. Optional fields: `title`, `publication_date`,
`date_precision`, `mesh` (list of names), `mesh_ids` (name to UI), `chemicals` (list of names),
`chemical_ids` (name to UI), `doi`, `source`, `url`, and `entity_ids`.
Explicit entity IDs must already exist in the registry. Dictionary linking supplements them.
Dates are ISO dates; when only a month/year is known, supply the end of that interval and its
precision. Missing dates are allowed for ordinary retrieval and excluded by temporal filtering.

Put PubMed efetch XML/XML.gz directly under `data/raw/pubmed/`. The adapter reads structured
abstract sections, mixed XML text, chemicals, MeSH terms/IDs, DOI, and publication dates. Retracted
publication records and records without abstracts are skipped. Duplicate IDs, DOI, or normalized
text are collapsed deterministically by file order. The first record wins; keep conflicting
versions outside the input folders until resolved.

## Alias crosswalk: data/raw/local/aliases.csv

```csv
entity_id,alias
drug:abacavir,An independently verified synonym
```

Use verified mappings for salts and brands; the system does not automatically equate them. Drug
IDs are generated from normalized ingredient names unless a preexisting exact alias resolves them.
Product RxNorm IDs are not assigned to ingredients because they are different entity levels.
External identifier lookup is supported as `mesh:D...`, `umls:C...`, `drugbank:DB...`, etc.

The openFDA v1 adapter takes only single-ingredient products with an original AP submission.
Combination products are counted and skipped. Original application dates describe ingredient
availability, not the approval date of a particular indication. Absence from openFDA means unknown
approval, not evidence of rejection. A historical approval is not a current marketing/safety claim.

## Judgments: data/raw/local/judgments.jsonl

```json
{"drug_id":"drug:abacavir","disease_id":"disease:alzheimer","relevance":0,"source":"synthetic-format-example-only","status":"example"}
```

The preceding line illustrates formatting only and is not an assertion about the pair. Do not
import format examples as scientific labels. `relevance` is 0 or 1; both entity IDs must exist.
`source` must identify the judgment evidence. Optional `status` retains the source's original label.

For verified temporal studies, add `confirmed_date`, `prior_indication_date`, and `prior_indication`.
An eligible outcome must occur after the cutoff, and the different indication must predate it.
Dates must refer to actual events, not dataset download, trial registration, or model training dates.
These fields require source review; v1 never invents them from repoDB status.

repoDB CSV/TSV requires `drug_name`, `drug_id`, `ind_name`, `status`; `ind_id` is retained when present.
Statuses Approved/Terminated/Suspended/Withdrawn become 1/0/0/0 benchmark judgments. Exact scoped
disease names/aliases are matched; subtypes are not automatically promoted to a parent disease.

## Derived artifacts

- `entities.jsonl`, `documents.jsonl`, `judgments.jsonl`: canonical stage contracts.
- `ingestion-report.json`: input checksums, mapping/coverage counts, duplicate counts, limitations.
- Index `manifest.json`: corpus/entity fingerprints, date cutoff, excluded-document count.
- `dense.json`, `embeddings.npy`: model settings/revision and aligned normalized embedding rows.
- Run `queries.jsonl`: fixed positive/negative judgments and candidate/corpus coverage per query.
- Run `rankings.jsonl`: method variant, ranked entities, scores, and supporting evidence.

Downloader manifests exclude API keys and contact emails. Acquisition-managed PubMed XML files
not in the latest successful `active-files.json` are excluded from ingestion; manually supplied
files without download manifests are still included.
