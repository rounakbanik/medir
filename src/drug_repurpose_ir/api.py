"""Small optional HTTP adapter around the same engine used by the CLI.

The API adds request validation and HTTP errors, not a separate retrieval
implementation. One engine is created at startup and reused for requests.
This local development backend is ready for a future frontend integration.
"""

from threading import Lock

from fastapi import FastAPI, HTTPException

from . import __version__
from .engine import SearchEngine
from .models import SearchRequest, SearchResponse


def create_app(settings):
    """Construct a FastAPI application bound to one validated index/settings snapshot.

    Args:
        settings: Configuration used to initialize the shared SearchEngine.

    Returns:
        FastAPI app exposing /health, /entities, /search, and generated /docs.

    Engine initialization fails early for a missing/stale index. A lock serializes
    searches because lazy neural initialization and per-engine caches are mutable.
    """
    # Fail startup immediately if the required prepared index is missing or stale.
    engine = SearchEngine(settings)
    # Neural backends initialize lazily; serialize access for a local single-process v1.
    lock = Lock()
    app = FastAPI(
        title="DrugRePurposeIR",
        version=__version__,
        description="Research literature retrieval and candidate evidence.",
    )

    @app.get("/health")
    def health():
        """Report the loaded document count and snapshot hash without running retrieval."""
        return {
            "status": "ok",
            "documents": len(engine.documents),
            "index_hash": engine.manifest["index_hash"],
        }

    @app.post("/search", response_model=SearchResponse)
    def search(request: SearchRequest):
        """Validate a search request, call the shared engine, and translate expected errors.

        Schema errors are handled by FastAPI/Pydantic. Engine ValueError becomes
        HTTP 422, and RuntimeError becomes HTTP 503. The lock protects lazy model
        initialization and caches across simultaneous local requests.
        """
        try:
            # Keep mutable caches and first model initialization safe across request threads.
            with lock:
                # Pydantic field names match engine keyword arguments, keeping one request contract.
                return engine.search(**request.model_dump())
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.get("/entities")
    def entities(kind: str = "disease", q: str = "", limit: int = 20):
        """Return a bounded catalogue-name search for query discovery/autocomplete.

        kind restricts drug versus disease. q performs a case-insensitive substring
        search of preferred names only, not every alias. limit is restricted to
        1..100. Results are catalogue records, not ranked therapeutic suggestions.
        """
        if kind not in {"drug", "disease"} or not 1 <= limit <= 100:
            raise HTTPException(status_code=422, detail="Invalid kind or limit (1-100)")
        return [
            e.model_dump(mode="json")
            for e in engine.registry.entities.values()
            if e.kind == kind and q.lower() in e.name.lower()
        ][:limit]

    return app
