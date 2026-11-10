# Adding data

Add source files under `data/raw/`, then rebuild. All commands below run from the project root
with the virtual environment activated. These paths use `configs/default.yaml`; another config
may set a different `paths.raw`.

For the sources in [the project proposal](<../Project Proposal.pdf>), start with
[Importing the proposal datasets](#importing-the-proposal-datasets). PubMed, openFDA, B3DB,
and repoDB exports have dedicated loaders. The steps below cover manually collected papers,
reviewed labels, and entity/identifier mappings that supplement those datasets.

## 1. Choose the right folder and filename

For papers collected manually, use `data/raw/local/`. Create the folder if needed:

```bash
mkdir -p data/raw/local
```

You can keep separate files for each collection; there is no need to append to someone else's file.

| Content | Example path | Format |
|---|---|---|
| Papers or abstracts | `data/raw/local/lit-review.documents.jsonl` | One document record per line; schema below. `.documents.jsonl.gz` also works. |
| New drugs/diseases or metadata | `data/raw/local/lit-review.entities.jsonl` | One entity record per line. |
| Additional names for existing entities | `data/raw/local/lit-review.aliases.csv` | CSV with `entity_id,alias` columns. |
| Reviewed evaluation labels | `data/raw/local/lit-review.judgments.jsonl` | One drug/disease judgment per line. |

**The filename ending matters.** `lit-review.documents.jsonl` is accepted; `lit-review.jsonl`
and `notes.jsonl` are ignored. Plain `documents.jsonl`, `entities.jsonl`, `aliases.csv`, and
`judgments.jsonl` also work. Subfolders under `local/` are scanned too.

JSONL means one complete JSON object on each line, saved as UTF-8. Use double quotes, JSON
`true`/`false`/`null`, and no comments or trailing commas. Do not wrap the records in a JSON
array or spread one record over several lines. Unknown fields are rejected.

PDF, Word, Excel, plain-text files, and generic article CSVs are not imported directly.
Extract their article text and convert it to the document schema below. The reference code's
Wikipedia `docid`/`text` format is an internal indexing format; local uploads use `id`/`abstract`.

## 2. Add the papers

Save records in `data/raw/local/lit-review.documents.jsonl`. This is a fictional formatting
example; replace it with the actual paper and metadata before importing:

```jsonl
{"id":"local:team-study-001","title":"Example study title","abstract":"Example Drug and Alzheimer disease are mentioned in this fictional formatting example.","publication_date":"2021-06-30","date_precision":"month","entity_ids":["drug:example-drug","disease:alzheimer"],"source":"format-example","url":"https://example.org/papers/team-study-001"}
```

| Field | Required? | What to put here |
|---|---|---|
| `id` | Yes | A unique, stable string, such as `local:team-study-001`. Reuse the same ID when correcting that paper. |
| `abstract` | Yes | Nonempty evidence text from the source. Escape line breaks inside the string as `\n`. |
| `title` | No | The paper title; defaults to an empty string. |
| `publication_date` | No | `YYYY-MM-DD`, or `null` if unknown. |
| `date_precision` | No | `day`, `month`, `year`, or `unknown`; defaults to `unknown`. |
| `entity_ids` | No | List of existing drug/disease IDs. Omit it or use `[]` to rely on name matching. |
| `source`, `url`, `doi` | No | Source name, paper URL, and DOI. Include them when available so results can be traced back. |
| `mesh`, `chemicals` | No | Lists of MeSH terms and chemical names, such as `["Alzheimer Disease"]`. |
| `mesh_ids`, `chemical_ids` | No | Name-to-ID objects, such as `{"Alzheimer Disease":"D000544"}`. |

For a month-only date, use the month's last day and `date_precision: "month"`. For a year-only
date, use December 31 and `"year"`. Leave unknown dates as `null`; those papers are excluded
when a time cutoff is enabled.

The system finds known names/aliases in the title, abstract, MeSH terms, and chemical names.
Explicit `entity_ids` supplement those matches and must refer to entities in the catalogue.
They identify mentions, not proof that a drug treats a disease. Ordinary disease/drug retrieval
needs both the query entity and candidate entity linked to a supporting paper.

## 3. Add missing entities and aliases

Check `data/processed/entities.jsonl` for existing IDs, names, and aliases first. Reuse an
existing ID for the same entity. For a new entity, use a stable ID such as `drug:example-drug`
or `disease:example-disease` and add it to `data/raw/local/lit-review.entities.jsonl`:

```jsonl
{"id":"drug:example-drug","kind":"drug","name":"Example Drug","aliases":["Example compound X"],"approved":null,"bbb":null,"sources":["format-example"]}
```

Only `id`, `kind`, and `name` are required. `kind` must be `drug` or `disease`. Optional fields are:

| Field | Values |
|---|---|
| `aliases` | List of verified alternative names. |
| `identifiers` | Namespace-to-list object, such as `{"pubchem":["12345"]}`. |
| `cns` | `true` for a CNS disease that should use BBB reranking; otherwise `false`. |
| `approved`, `approved_since` | `true`/`false`/`null`, and an approval date in `YYYY-MM-DD` or `null`. |
| `bbb` | `"positive"`, `"negative"`, `"conflicting"`, or `null`. |
| `sources` | List of source references for the entity and any supplied metadata. |

Use `null` for unknown approval/BBB information. The default `approved_only: true` setting
excludes drugs without confirmed approval from search results, even if their papers were loaded.
Supply approval metadata only when supported by a source; openFDA can also supply it during import.

The document example above refers to `drug:example-drug`, so that entity record must be present
in the same import. `disease:alzheimer` already exists in the default config. Entities load before
documents, regardless of their filenames. For a new disease to join automatic PubMed acquisition
and the disease coverage report, also append its entity record under `diseases` in the config.

To add a name to an existing entity without writing another entity record, use
`data/raw/local/lit-review.aliases.csv`:

```csv
entity_id,alias
drug:example-drug,Example compound Y
```

These are also fictional names. Use reviewed synonyms for real records. Alias matching does not
infer equivalent salts or brands, and aliases shared by multiple IDs remain ambiguous. Local
entity records load before the external catalogues; put an alias there if it must reconcile a
source name during catalogue import. The alias CSV loads later and does not merge existing IDs
or transfer their approval/BBB metadata.

## 4. Add evaluation labels, if available

Papers can be searched without adding judgments. Judgments are separate, reviewed labels used
to score retrieval; they do not become searchable evidence. Put them in
`data/raw/local/lit-review.judgments.jsonl`:

```jsonl
{"drug_id":"drug:example-drug","disease_id":"disease:alzheimer","relevance":1,"source":"synthetic-format-example-only","status":"example"}
```

This is a fictional label, not a claim about a treatment. For real labels, `drug_id`,
`disease_id`, `relevance`, and `source` are required. Both IDs must exist with the corresponding
entity types. `relevance` is `1` for a reviewed positive or `0` for a reviewed negative;
`source` should identify the supporting dataset or evidence. `status` is optional.
Leave unreviewed pairs out. Queries need at least one positive judgment to enter evaluation.

Temporal evaluations also need verified `confirmed_date`, `prior_indication_date`, and
`prior_indication`; see [data-contracts.md](data-contracts.md) for those rules.

## 5. Import, rebuild, and check

Run preparation first so schema/ID errors are caught before building embeddings:

```bash
drugir prepare
```

Fix any reported errors in the raw files and rerun. Check `data/processed/ingestion-report.json`:
`inputs` should include the new file, document counts should make sense, and `counts` and
`ambiguous_aliases` show duplicates, missing dates, and linking issues. The prepared
`documents.jsonl` contains the final `entity_ids` for each paper.

Then build and try a search:

```bash
drugir index --dense
drugir doctor
drugir search "Alzheimer's" --method hybrid --k 5
```

Use the disease or drug you added for the search. Check the returned evidence IDs/titles;
a successfully imported paper may still rank outside the candidate pool or be filtered by
approval, dates, or missing entity links. `drugir index` without `--dense` is enough for BM25,
but dense/hybrid searches need embeddings rebuilt for the changed corpus.

For later imports, `drugir run --dense` combines preparation and indexing. Add `--download`
only when you also want to fetch external datasets. Corpus changes currently rebuild the full
dense matrix. Restart a running API after rebuilding, and rerun evaluation after adding labels:

```bash
drugir evaluate --methods bm25,hybrid
```

Edit or remove records in the raw files, then repeat the import. Duplicate paper IDs, DOIs, or
normalized title/abstract text keep the first encountered record; adding another copy does not
update the original. Do not edit `data/processed/` or `artifacts/index/` directly. Files are not
watched in the background. Raw data is git-ignored, so share collected files separately when
another teammate needs the same corpus.

## Importing the proposal datasets

The proposal's Data section names PubMed, DrugBank **or** openFDA, B3DB, repoDB, and identifiers
from RxNorm/DrugBank and MeSH/MONDO. We use the openFDA alternative for v1 because we do not
currently have DrugBank access.

| Proposal component | Input used now | What it supplies |
|---|---|---|
| Literature | PubMed abstracts and MeSH metadata | Searchable papers and drug/disease mentions. |
| Drug information | openFDA **Drugs@FDA** | Ingredient/brand names and original approval evidence. |
| BBB labels | B3DB **classification** table | Positive/negative BBB labels for matched drugs. |
| Repurposing cases | Archived repoDB CSV | Drug vocabulary, indication outcomes, and evaluation labels. |
| Entity IDs | DrugBank IDs from repoDB, MeSH IDs from PubMed, plus reviewed local mappings | Consistent IDs and query lookup; RxNorm and MONDO mappings use the same local entity schema. |

To download the four public datasets through the existing adapters:

```bash
drugir fetch --source pubmed
drugir fetch --source openfda
drugir fetch --source b3db
drugir fetch --source repodb
drugir run --dense
```

`drugir fetch` without `--source` fetches all four. PubMed selection uses the diseases, MeSH
queries, date range, and sample limits in the config. For files collected separately, copy
them to the folders below, preserving their original fields, then run `drugir prepare` and
`drugir index --dense`.

| Source and folder | Accepted files | Expected export |
|---|---|---|
| PubMed: `data/raw/pubmed/` | `.xml`, `.xml.gz`, `.jsonl`, `.jsonl.gz` | PubMed **efetch XML**, or the document JSONL schema above. A PMID list, search-history file, or ESearch JSON is not an abstract export. |
| openFDA: `data/raw/openfda/` | `.json`, `.json.zip` | **Drugs@FDA application** records. Plain JSON can be a `results` envelope or an application list; ZIP JSON members need a `results` envelope. |
| B3DB: `data/raw/b3db/` | `.tsv`, `.tsv.gz` | Classification table with `compound_name` and `BBB+/BBB-` columns. Labels must be `BBB+` or `BBB-`; optional `CID` is retained as a PubChem ID for matched drugs. |
| repoDB: `data/raw/repodb/` | `.csv`, `.tsv` | Original archive columns `drug_name`, `drug_id`, `ind_name`, `status`; optional `ind_id`. Additional archive columns are accepted. |

PubMed records without abstracts and retracted records are skipped. MeSH and chemical names/IDs
are extracted from the XML. Manually added exports can use a new filename; leave downloader
manifests and `active-files.json` alone, since they track the downloaded selection.

Drugs@FDA records need `application_number`, `products[].active_ingredients[].name`, and
`submissions`. Approval comes from an `ORIG` submission with status `AP`; dates use
`submission_status_date` in `YYYYMMDD` form. Only single-ingredient products are imported.
Other openFDA datasets, such as adverse events, NDC listings, or drug labels, need their own
adapters. **The current Drugs@FDA loader does not supply structured targets or mechanisms.**
Mechanism queries currently search the PubMed text.

B3DB joins against known drug names and aliases. It does not add every B3DB compound to the
approved-drug catalogue. Use the classification export; a regression-only file with `logBB`
but no `BBB+/BBB-` column cannot supply the labels this reranker expects.

repoDB uses the DrugBank ID in `drug_id` and the indication ID in `ind_id` when present.
Indication names must match a known disease name/alias. `Approved` becomes a positive benchmark
label; `Terminated`, `Suspended`, and `Withdrawn` become negatives. Unknown outcomes remain
unjudged. These labels do not establish independent drug approval or a dated repurposing event;
the proposal's time-split test still needs the reviewed facts described under [Time splits](methodology.md#time-splits).

## RxNorm, DrugBank, MeSH, and MONDO identifiers

Add reviewed mappings in `data/raw/local/crosswalk.entities.jsonl`, reusing the existing
canonical `id` for each entity. The following are templates; replace the names/IDs and every
`REPLACE_...` value with a verified mapping before importing:

```jsonl
{"id":"drug:example-drug","kind":"drug","name":"Example Drug","identifiers":{"rxnorm":["REPLACE_WITH_RXCUI"],"drugbank":["REPLACE_WITH_DRUGBANK_ID"]},"sources":["REPLACE_WITH_MAPPING_SOURCE"]}
{"id":"disease:example-disease","kind":"disease","name":"Example Disease","identifiers":{"mesh":["REPLACE_WITH_MESH_UI"],"mondo":["REPLACE_WITH_MONDO_LOCAL_ID"]},"sources":["REPLACE_WITH_MAPPING_SOURCE"]}
```

Each namespace maps to a **list of strings**, even for numeric IDs. Use lowercase namespace
keys `rxnorm`, `drugbank`, `mesh`, and `mondo`. Store an RxCUI as its numeric string, a DrugBank ID
as `DB...`, a MeSH UI as `D...`, and a MONDO ID's numeric part with its leading zeros. For example,
an ID of the form `MONDO:1234567` is stored as `"mondo":["1234567"]` and queried as
`mondo:1234567`. The number here illustrates the format, not an asserted disease mapping.

External IDs are available as queries after rebuilding. These two work with the current corpus:

```bash
drugir search "drugbank:DB01048" --direction drug-to-disease --method bm25
drugir search "mesh:D000544" --direction disease-to-drug --method bm25
```

The same lookup works for a supplied `rxnorm:<RXCUI>` or `mondo:<number>` mapping. Mappings
enrich an entity with the same canonical ID; they do not merge two different canonical IDs.
Use ingredient-level drug mappings for ingredient records. Product/strength-specific RxNorm IDs
should not be copied onto an ingredient just because a product contains it.

**Raw DrugBank XML/ZIP, RxNorm release files, and MONDO ontology files are not drop-in inputs.**
Export their reviewed names/aliases/identifier mappings to the entity JSONL schema above.
There is no automatic download or full parser for these alternative releases yet. DrugBank
approval facts need a source that supports the FDA approval claim; a DrugBank ID alone does
not set `approved: true`. Structured DrugBank targets/mechanisms need a separate adapter and
schema extension; the current entity schema rejects those extra fields.

The knowledge-graph sources named under Future Scope in the proposal (Hetionet and Open Targets)
are future extensions. Their graph exports also need adapters before they can be used.
The complete current record schemas are in [data-contracts.md](data-contracts.md).
