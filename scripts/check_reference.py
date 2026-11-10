"""Run the supplied non-neural IR tests against the packaged reference modules.

Run from the repository root with .venv/bin/python scripts/check_reference.py.
Fixtures are copied to a temporary directory so save/load tests cannot modify
the reference folder. Model-downloading tests are not part of this check.
"""

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

from drug_repurpose_ir.ir import document_preprocessor, indexing, ranker, relevance


def main():
    """Load the original tests with their expected module names in this process only."""
    root = Path(__file__).resolve().parents[1]
    source = root / "reference/starter-code/tests"
    if not source.exists():
        raise SystemExit("Place the supplied assignment tests under reference/starter-code/tests.")
    # The original tests use flat imports. These aliases are confined to this test process.
    sys.modules.update(
        {
            "document_preprocessor": document_preprocessor,
            "indexing": indexing,
            "ranker": ranker,
            "relevance": relevance,
        }
    )
    names = [
        "test_indexing_public",
        "test_scorer_public",
        "test_ranker_public",
        "test_relevance_public",
        "test_document_preprocessor_public",
    ]
    with tempfile.TemporaryDirectory(prefix="drugir-reference-tests-") as temp:
        folder = Path(temp) / "tests"
        shutil.copytree(source, folder, ignore=shutil.ignore_patterns("__pycache__"))
        suite = unittest.TestSuite()
        for name in names:
            spec = importlib.util.spec_from_file_location(name, folder / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if name == "test_document_preprocessor_public":
                # Doc2query downloads a T5 model; RegexTokenizer is the default path being reused.
                suite.addTests(
                    unittest.defaultTestLoader.loadTestsFromTestCase(module.TestRegexTokenizer)
                )
            else:
                tests = unittest.defaultTestLoader.loadTestsFromModule(module)
                for group in tests:
                    for test in group:
                        if test._testMethodName != "test_cross_encoder_query":
                            suite.addTest(test)
        # Original fixture filenames are relative to the current working directory.
        import os

        os.chdir(folder)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
