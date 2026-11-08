"""DrugRePurposeIR's package identity and version.

Start with models.py to understand shared records, cli.py for terminal workflows,
and engine.py for retrieval. This lightweight package import does not initialize
an index, download datasets, or load neural models. The version is included in
API/download metadata to identify the implementation used for an experiment.
"""

# Keep this value aligned with the package version declared in pyproject.toml.
__version__ = "0.1.0"
