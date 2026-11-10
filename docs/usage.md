# Using the system

You can search from the terminal, [Python](#python-and-http-api), or the
[browser API page](#try-searches-in-the-browser).
Complete [Setup](../README.md#setup) first; a fresh clone does not include the collected data or indexes.

## Search from the terminal

With an existing dataset and index, start here:

```bash
source .venv/bin/activate
drugir doctor
drugir search "Alzheimer's" --method bm25 --k 5
```

`doctor` shows saved dataset/index counts and metadata. If no index exists yet but the raw files
are present, run `drugir run` first. BM25 needs only the lexical index. For dense or hybrid
search, build embeddings with `drugir index --dense`.

Other searches:

```bash
# Disease → drugs
drugir search "Alzheimer's" --method hybrid --k 5

# Drug → diseases
drugir search abacavir --direction drug-to-disease --method hybrid

# Free-text mechanism query
drugir search "reverse transcriptase inhibition" --direction mechanism --method hybrid

# Apply the BBB reranker
drugir search "Alzheimer's" --method hybrid --feasibility --k 5
```

Search prints JSON. Add `--output runs/search.json` to save it. Use `drugir search --help`
for the full options. If the virtual environment is not activated, call `.venv/bin/drugir`.

Entity queries must match a known name, alias, or ID. For example, `disease:alzheimer` works
as well as the disease name. Use `--direction mechanism` for free text.

Each result includes:

- The drug or disease, its identifiers, and available approval/BBB metadata.
- `retrieval_score`: the entity score before BBB reranking.
- `feasibility_adjustment`: the BBB contribution.
- `score`: the final ranking score.
- `evidence_count`: matching papers in the retrieved pool.
- `evidence`: up to three paper references and abstract excerpts by default.

`results` is the ranked list. Read each result's `entity.name`, then inspect the supporting
paper titles, URLs, and `snippet` text under `evidence`. The top-level `warnings` list records
missing literature coverage and qualifications on BBB reranking.

Scores are relative to the current query, not probabilities. A BBB bonus can push a score
above one. The excerpts are selected from the abstracts; they are not generated explanations.

## Python and HTTP API

Use the engine directly from Python:

```python
from drug_repurpose_ir.config import load_config
from drug_repurpose_ir.engine import SearchEngine

engine = SearchEngine(load_config("configs/default.yaml"))
response = engine.search("Alzheimer's", method="bm25", k=5)
for result in response.results:
    print(result.entity.name, result.score)
    for paper in result.evidence:
        print("  ", paper.title, paper.url)
```

Reuse an engine across queries to reuse loaded models and caches. Create a new one after
changing its data or settings.

## Try searches in the browser

After setup and indexing, start the backend and leave that terminal running:

```bash
drugir serve --host 127.0.0.1 --port 8000
```

If you installed only the core dependencies, first run `pip install -e '.[api]'`.
Open [localhost:8000/docs](http://127.0.0.1:8000/docs). This page lets you run requests
and inspect the JSON responses directly:

1. Expand **GET /health**, click **Try it out**, then **Execute** to check the loaded document count.
2. Expand **GET /entities** and click **Try it out**. Set `kind` to `drug` and `q` to `abacavir`,
   then **Execute**. The response gives the matching catalogue names and IDs. Use `kind: disease`
   to look up diseases. This lookup searches preferred names, not aliases.
3. Expand **POST /search**, click **Try it out**, and replace the request body with:

```json
{
  "query": "abacavir",
  "direction": "drug-to-disease",
  "method": "bm25",
  "k": 5,
  "feasibility": false
}
```

Click **Execute** and read `results`, `evidence`, and `warnings` in the response. To search
for drugs instead, set `query` to `Alzheimer's` and `direction` to `disease-to-drug`.
Stop the server with Ctrl+C. Restart it after changing the data/index or configuration.

The same API works from another program or terminal:

```bash
curl http://127.0.0.1:8000/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"Alzheimer disease","method":"bm25","k":5}'
```

## Try the demo

The demo uses fictional data and separate output folders. Once dependencies are installed,
these commands need no network access:

```bash
drugir run --config configs/demo.yaml
drugir search "Brain disease" --config configs/demo.yaml
drugir evaluate --config configs/demo.yaml --methods random,bm25,bm25+feasibility
```
