"""Explicit, cached acquisition of the four public source datasets.

The CLI invokes this module only for fetch or run --download. Downloaded
inputs remain under data/raw with per-file manifests recording checksums,
source URLs, and safe query metadata. PubMed sampling and extra anchor
queries are controlled by configuration rather than hard-coded study scope.
"""

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from .config import Settings
from .io import atomic_text, digest, write_json


class Downloader:
    """Reusable HTTP client with conservative pacing, retries, and file-cache checks.

    Cache reuse requires matching URL, non-secret parameters, and file checksum.
    refresh bypasses reuse; it does not mean upstream data are historically fixed.
    The object owns an HTTP client, which the caller must close after use.
    """

    def __init__(self, raw: Path, refresh: bool = False):
        """Initialize HTTP state and cache policy for one acquisition operation.

        Args:
            raw: Root input directory associated with this downloader.
            refresh: Download again even when an existing file passes cache checks.

        A monotonic request timestamp supports pacing without depending on wall-clock
        changes. A descriptive User-Agent identifies this application to services.
        """
        self.raw = raw
        self.refresh = refresh
        self.client = httpx.Client(
            timeout=120, follow_redirects=True, headers={"User-Agent": "DrugRePurposeIR/0.1"}
        )
        self.last_request = 0.0

    def close(self):
        """Release connections held by the HTTP client after acquisition finishes."""
        self.client.close()

    def get(self, url: str, params=None):
        """Request a URL with at most five attempts and conservative request spacing.

        Args:
            url: Public source endpoint, without credentials embedded in the URL.
            params: Optional query parameters; secrets are not persisted by save.

        Returns:
            A successful/non-error HTTP response for the caller to decode.

        HTTP 429 and server errors retry with exponential backoff. Transport errors
        retry too; other HTTP errors fail immediately with a sanitized URL message.
        Rate limits are per Downloader instance, not coordinated across processes.
        """
        # Retry only a bounded number of times so an unavailable source eventually fails.
        for attempt in range(5):
            # All sources, including NCBI, remain below three requests/second.
            time.sleep(max(0, 0.4 - (time.monotonic() - self.last_request)))
            self.last_request = time.monotonic()
            try:
                response = self.client.get(url, params=params)
                # 429 indicates throttling; transient server errors may also recover after a delay.
                if response.status_code == 429 or response.status_code >= 500:
                    time.sleep(min(2**attempt, 16))
                    continue
                if response.status_code >= 400:
                    raise RuntimeError(f"HTTP {response.status_code}: {url} (credentials omitted)")
                return response
            except httpx.TransportError:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)
        raise RuntimeError(f"Download failed after retries: {url} (credentials omitted)")

    def save(self, path: Path, url: str, params=None, metadata=None):
        """Reuse a verified local download or fetch it and write its provenance manifest.

        Args:
            path: Raw output filename; parents are created as needed.
            url: Source endpoint or binary-download URL.
            params: Request parameters; api_key and email are removed from manifests.
            metadata: Additional safe source/version/query context for provenance.

        Returns:
            The local Path, whether the file was downloaded or reused.

        The response is buffered in memory, written as .part, and then replaces the
        destination. ZIP payloads are saved intact; ingestion reads members later.
        """
        manifest = path.with_name(path.name + ".manifest.json")
        # Credentials/contact details affect the request but are not part of persisted provenance.
        safe_params = {k: v for k, v in (params or {}).items() if k not in ("api_key", "email")}
        # A file alone is insufficient: its request identity and original bytes must still match.
        if path.exists() and manifest.exists() and not self.refresh:
            previous = json.loads(manifest.read_text())
            if (
                previous.get("url") == url
                and previous.get("parameters") == safe_params
                and previous.get("sha256") == digest(path)
            ):
                return path
        # Only contact the dataset URL when reuse is disabled or validation fails.
        response = self.get(url, params)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Binary downloads never extract files into the filesystem.
        temp = path.with_name(path.name + ".part")
        temp.write_bytes(response.content)
        temp.replace(path)
        # Write provenance after the payload is complete so it describes the saved bytes.
        write_json(
            manifest,
            {
                "url": url,
                "parameters": safe_params,
                "sha256": digest(path),
                "bytes": path.stat().st_size,
                "retrieved_at": datetime.now(UTC).isoformat(),
                "metadata": metadata or {},
            },
        )
        print(f"Downloaded {path.name}: {path.stat().st_size:,} bytes", flush=True)
        return path


def fetch(settings: Settings, source: str = "all", refresh=False):
    """Acquire selected public sources and record which local files are ready.

    Args:
        settings: Output paths and source/query sampling configuration.
        source: all, pubmed, openfda, b3db, or repodb.
        refresh: Bypass validated local file caches when true.

    Returns:
        Paths of corpus/dataset files made ready, including reused downloads.

    RepoDB resolves an archived Figshare CSV; B3DB uses the configured pinned
    URL; openFDA downloads every Drugs@FDA bulk partition; PubMed searches IDs
    then fetches XML in batches. PubMed active-files.json is committed only after
    all configured jobs finish, preventing old managed batches from being mixed
    into a successful new selection. HTTP connections close even after failure.
    """
    download = Downloader(settings.paths.raw, refresh)
    completed = []
    try:
        # Source 1: resolve the archived CSV through official release metadata.
        if source in ("all", "repodb"):
            article_id = settings.acquisition.repodb_article_id
            metadata = download.get(f"https://api.figshare.com/v2/articles/{article_id}").json()
            files = [f for f in metadata["files"] if f["name"].lower().endswith(".csv")]
            if len(files) != 1:
                raise ValueError("Expected one repoDB CSV; inspect the release metadata")
            item = files[0]
            completed.append(
                download.save(
                    settings.paths.raw / "repodb" / "repodb.csv",
                    item["download_url"],
                    metadata={
                        "article_id": article_id,
                        "version": metadata["version"],
                        "original_filename": item["name"],
                    },
                )
            )
        # Source 2: save the pinned BBB classification table exactly as supplied.
        if source in ("all", "b3db"):
            completed.append(
                download.save(
                    settings.paths.raw / "b3db" / "classification.tsv",
                    settings.acquisition.b3db_url,
                )
            )
        # Source 3: use the full bulk export instead of an API page/sample of approvals.
        if source in ("all", "openfda"):
            metadata = download.get("https://api.fda.gov/download.json").json()
            dataset = metadata["results"]["drug"]["drugsfda"]
            # Process every partition, even if the upstream export later grows beyond one ZIP.
            for i, part in enumerate(dataset["partitions"]):
                completed.append(
                    download.save(
                        settings.paths.raw / "openfda" / f"drugsfda-{i:03}.json.zip",
                        part["file"],
                        metadata={
                            "export_date": dataset.get("export_date"),
                            "total_records": dataset.get("total_records"),
                        },
                    )
                )
        # Source 4: search PubMed IDs first, then download the selected records in batches.
        if source in ("all", "pubmed"):
            a = settings.acquisition
            common = {"db": "pubmed", "tool": "DrugRePurposeIR"}
            for env, param in [("NCBI_EMAIL", "email"), ("NCBI_API_KEY", "api_key")]:
                if os.getenv(env):
                    common[param] = os.environ[env]
            # Use one job structure for configured diseases and extra anchor-case searches.
            jobs = []
            for disease in settings.diseases:
                term = (
                    f'"{disease.name}"[MeSH Terms] AND hasabstract[text] AND '
                    f"{a.pubmed_start_year}:{a.pubmed_end_year}[pdat]"
                )
                if a.pubmed_extra_query:
                    term += f" AND ({a.pubmed_extra_query})"
                key = disease.id.split(":")[-1]
                jobs.append((key, disease.id, term))
            # Safe, unique keys become local filenames; the query text remains configurable.
            for key, query in a.pubmed_queries.items():
                if not key or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in key):
                    raise ValueError(
                        "PubMed query keys must contain lowercase letters, digits, - or _"
                    )
                term = (
                    f"({query}) AND hasabstract[text] AND "
                    f"{a.pubmed_start_year}:{a.pubmed_end_year}[pdat]"
                )
                jobs.append((key, key, term))
            if len({key for key, _, _ in jobs}) != len(jobs):
                raise ValueError("PubMed query keys must not collide with disease keys")
            active = []
            for key, entity_id, term in jobs:
                search_path = settings.paths.raw / "pubmed" / f"{key}.search.json"
                download.save(
                    search_path,
                    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                    {
                        **common,
                        "term": term,
                        "retmode": "json",
                        "sort": a.pubmed_sort,
                        "retmax": a.pubmed_per_disease,
                    },
                )
                data = json.loads(search_path.read_text())
                if "esearchresult" not in data or "ERROR" in data.get("esearchresult", {}):
                    raise ValueError(f"PubMed returned an error for {key}")
                result = data["esearchresult"]
                # These are the bounded sampled PMIDs, not every match counted by PubMed.
                ids = result["idlist"]
                if not ids:
                    print(f"WARNING: no PubMed records for {key}", flush=True)
                for i in range(0, len(ids), a.pubmed_batch_size):
                    batch = ids[i : i + a.pubmed_batch_size]
                    batch_path = settings.paths.raw / "pubmed" / f"{key}-{i:05}.xml"
                    completed.append(
                        download.save(
                            batch_path,
                            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                            {**common, "id": ",".join(batch), "retmode": "xml"},
                            metadata={
                                "query_entity": entity_id,
                                "matched_count": result["count"],
                                "sampled": len(ids),
                                "query": term,
                            },
                        )
                    )
                    active.append(batch_path.name)
            # Commit the active managed selection only after all PubMed jobs succeed.
            write_json(settings.paths.raw / "pubmed" / "active-files.json", active)
        if source not in {"all", "pubmed", "openfda", "b3db", "repodb"}:
            raise ValueError(f"Unknown source: {source}")
    # Release the client even if one source fails partway through acquisition.
    finally:
        download.close()
    # Each individual file has its own immutable-input provenance manifest.
    atomic_text(settings.paths.raw / "LAST_FETCH.txt", "\n".join(str(p) for p in completed) + "\n")
    return completed
