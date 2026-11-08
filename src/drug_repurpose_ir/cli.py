"""User-facing terminal commands for the complete research pipeline.

The drugir console entry point in pyproject.toml points to app below. Typer
turns command parameters into --options and uses command docstrings as help.
Keep orchestration here and reusable algorithms in their respective modules.
Only fetch and run --download explicitly acquire source datasets; first use
of neural functionality can separately download pretrained model weights.
"""

import json
from pathlib import Path

import typer
from dotenv import load_dotenv

from . import __version__
from .acquisition import fetch as acquire
from .config import load_config
from .engine import SearchEngine
from .evaluation import evaluate as run_evaluation
from .indexing import build_index
from .ingestion import ingest

# Register this Typer application as the drugir console command.
app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    help="DrugRePurposeIR: bidirectional literature evidence retrieval.",
)


def settings_from(config):
    """Load the config-adjacent project .env file, then validated YAML settings.

    Existing shell variables take precedence over dotenv values by default.
    Config/data path resolution is delegated to config.load_config.
    """
    path = Path(config).resolve()
    # Read optional project credentials without overwriting existing shell environment values.
    load_dotenv(path.parent.parent / ".env")
    return load_config(path)


@app.command()
def fetch(config: Path = Path("configs/default.yaml"), source: str = "all", refresh: bool = False):
    """Download or reuse PubMed, openFDA, B3DB, and repoDB source files.

    Select one source with --source or all by default. --refresh intentionally
    bypasses file-cache reuse. This command acquires raw inputs only; follow it
    with prepare/index, or use run --download for the combined workflow.
    """
    paths = acquire(settings_from(config), source, refresh)
    typer.echo(f"Ready: {len(paths)} dataset files")


@app.command()
def prepare(config: Path = Path("configs/default.yaml")):
    """Normalize local source exports and print their coverage summary.

    Writes canonical entities/documents/judgments and the complete ingestion
    report. It neither downloads raw sources nor updates the search index.
    """
    report = ingest(settings_from(config))
    typer.echo(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("inputs", "ambiguous_aliases")}, indent=2
        )
    )


@app.command()
def index(config: Path = Path("configs/default.yaml"), dense: bool = False):
    """Build or reuse a lexical snapshot and optional dense document embeddings.

    --dense enables model loading and document encoding. Without it this command
    remains usable with the base installation. Prepared data must already exist.
    """
    typer.echo(json.dumps(build_index(settings_from(config), dense), indent=2))


@app.command()
def run(config: Path = Path("configs/default.yaml"), download: bool = False, dense: bool = False):
    """Prepare all local inputs and build indexes, optionally downloading first.

    This is the everyday command after adding data to the documented folders.
    --download runs acquisition before preparation; --dense also validates or
    builds neural embeddings. The default performs no source acquisition.
    """
    settings = settings_from(config)
    if download:
        acquire(settings)
    # Preparation always reads local raw files; acquisition is opt-in via --download.
    report = ingest(settings)
    result = build_index(settings, dense)
    typer.echo(f"Ready: {report['documents']} documents, {report['entities']} entities")
    typer.echo(json.dumps(result, indent=2))


@app.command()
def search(
    query: str,
    config: Path = Path("configs/default.yaml"),
    direction: str = "disease-to-drug",
    method: str = "bm25",
    k: int = 10,
    feasibility: bool = False,
    output: Path | None = None,
):
    """Search the current index and print or save a structured JSON response.

    Entity directions accept names, aliases, or IDs; --direction mechanism
    accepts free text. --k counts final entities, while paper depth is configured
    separately. --output writes the same JSON otherwise printed to the terminal.
    """
    response = SearchEngine(settings_from(config)).search(query, direction, method, k, feasibility)
    value = response.model_dump_json(indent=2)
    if output:
        from .io import atomic_text

        atomic_text(output, value + "\n")
    else:
        typer.echo(value)


@app.command()
def evaluate(
    config: Path = Path("configs/default.yaml"),
    methods: str = "random,bm25,bm25+feasibility,abc",
    directions: str = "both",
    output: Path | None = None,
    max_queries: int | None = None,
):
    """Run selected ranking variants and save metrics, uncertainty, and evidence.

    --methods accepts comma-separated names, optionally suffixed +feasibility.
    --directions accepts both or comma-separated entity directions. --max-queries
    takes a stable-sorted prefix for smoke tests; omit it for the full query set.
    Use a fresh --output directory to preserve earlier experiment artifacts.
    """
    if max_queries is not None and max_queries < 1:
        raise typer.BadParameter("max-queries must be positive")
    # Translate convenient comma-separated terminal options into reusable Python arguments.
    selected = None if directions == "both" else [d.strip() for d in directions.split(",")]
    result = run_evaluation(
        settings_from(config),
        [m.strip() for m in methods.split(",")],
        selected,
        output,
        max_queries,
    )
    typer.echo(f"Evaluation saved: {result}")


@app.command()
def doctor(config: Path = Path("configs/default.yaml")):
    """Print local data/index/model-manifest readiness without source requests.

    This reads manifests and ingestion counts; it is an inventory, not a full
    corruption scan or proof that every model/backend is currently available.
    It does not load model weights or rebuild a stale index.
    """
    settings = settings_from(config)
    status = {"version": __version__, "paths": settings.paths.model_dump(mode="json")}
    for name, path in [
        ("ingestion", settings.paths.processed / "ingestion-report.json"),
        ("index", settings.paths.index / "manifest.json"),
        ("dense", settings.paths.index / "dense.json"),
    ]:
        status[name] = json.loads(path.read_text()) if path.exists() else "missing"
    if isinstance(status["ingestion"], dict):
        status["ingestion"].pop("inputs", None)
        status["ingestion"].pop("ambiguous_aliases", None)
    typer.echo(json.dumps(status, indent=2))


@app.command()
def serve(config: Path = Path("configs/default.yaml"), host: str = "127.0.0.1", port: int = 8000):
    """Start the optional FastAPI service and interactive /docs API interface.

    Defaults bind to 127.0.0.1:8000 for local development. The service loads the
    prepared index at startup and reuses one engine; restart after rebuilding.
    Install the api extra before using this command.
    """
    try:
        import uvicorn

        from .api import create_app
    except ImportError as error:
        raise RuntimeError("Install the API with: pip install -e '.[api]'") from error
    # Serve the same engine/API implementation rather than duplicating retrieval in a frontend.
    uvicorn.run(create_app(settings_from(config)), host=host, port=port)


if __name__ == "__main__":
    app()
