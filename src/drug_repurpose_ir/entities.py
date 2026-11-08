"""Canonical entity lookup, dictionary linking, and normalized text hashes.

This v1 uses explicit names/aliases, not a trained biomedical NER model.
Normalization removes case, punctuation, and accents but does not infer
equivalent salt forms, chemical structures, subtypes, or treatment relations.
Ambiguous aliases are retained for review and skipped during mention linking.
"""

import hashlib
import re
import unicodedata
from collections import defaultdict

import ahocorasick

from .models import Entity


def normalize(text: str) -> str:
    """Convert text into lowercase ASCII word tokens separated by single spaces.

    The same transformation is used for names and documents so punctuation and
    accent differences do not break exact matches. For example, a hyphen becomes
    a word boundary. This operation is deliberately lossy and is not an ontology.
    """
    # Decompose accents before ASCII conversion; non-ASCII-only symbols may be lost.
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return " ".join(re.findall(r"[a-z0-9]+", text))


def drug_id(name: str) -> str:
    """Create a stable local drug ID from a normalized ingredient name.

    Example: "Example Drug" becomes "drug:example-drug". This is a local join
    key, not an RxNorm/DrugBank identifier or a chemical equivalence assertion.
    """
    return "drug:" + normalize(name).replace(" ", "-")


class Registry:
    """In-memory catalogue supporting ID/name resolution and text mention lookup.

    entities stores canonical records by ID. _lookup maps each normalized alias
    to a set of possible IDs so ambiguity is explicit. _automaton is a lazily
    rebuilt Aho-Corasick matcher, allowing many aliases to be scanned in one pass.
    """

    def __init__(self, entities=()):
        """Create an empty catalogue and merge an optional iterable of entities.

        Args:
            entities: Existing canonical Entity records, usually loaded from JSONL.
        """
        self.entities: dict[str, Entity] = {}
        # Keep a set, not one ID: an alias may legitimately refer to several entities.
        self._lookup: dict[str, set[str]] = defaultdict(set)
        # Any alias change invalidates the lazily compiled mention matcher.
        self._automaton = None
        for entity in entities:
            self.add(entity)

    def add(self, entity: Entity):
        """Insert or enrich one entity and invalidate the text-matching cache.

        Existing IDs merge aliases, provenance, and external identifiers. An observed
        approval remains true if a later record has no/negative approval metadata;
        the earliest available approval date is retained. Disagreeing BBB labels
        become conflicting rather than one source silently replacing another.

        The supplied/canonical Entity can be mutated while merging. Make a deep copy
        first when the caller needs an unchanged original. Different IDs are not
        automatically merged, even when names look chemically related.

        Raises:
            ValueError: The same ID is supplied as both a drug and a disease.
        """
        # Merging is by canonical ID only; similar names do not merge salt forms.
        old = self.entities.get(entity.id)
        if old:
            if old.kind != entity.kind:
                raise ValueError(f"Conflicting entity kinds for {entity.id}")
            # Sorting makes merged outputs deterministic when the same sources are rerun.
            old.aliases = sorted(set(old.aliases + entity.aliases + [entity.name]) - {old.name})
            old.sources = sorted(set(old.sources + entity.sources))
            for key, values in entity.identifiers.items():
                old.identifiers[key] = sorted(set(old.identifiers.get(key, []) + values))
            # Missing metadata must not erase an approval observed in another source.
            if entity.approved is not None:
                old.approved = old.approved is True or entity.approved
            dates = [d for d in [old.approved_since, entity.approved_since] if d]
            old.approved_since = min(dates) if dates else None
            # Once measurements disagree, conflicting remains an explicit neutral state.
            if entity.bbb:
                old.bbb = entity.bbb if old.bbb in (None, entity.bbb) else "conflicting"
            old.cns = old.cns or entity.cns
            entity = old
        else:
            self.entities[entity.id] = entity
        # Namespaced IDs can be entered directly as queries, e.g. mesh:D012345.
        external_ids = [
            f"{key}:{value}" for key, values in entity.identifiers.items() for value in values
        ]
        for alias in [entity.name, entity.id, *entity.aliases, *external_ids]:
            key = normalize(alias)
            if key:
                self._lookup[key].add(entity.id)
        # Any alias change invalidates the lazily compiled mention matcher.
        self._automaton = None

    def resolve(self, name: str, kind: str | None = None) -> Entity | None:
        """Return exactly one entity matching a name, alias, or namespaced ID.

        Args:
            name: Text to normalize and look up, including canonical IDs if desired.
            kind: Optional drug/disease restriction before deciding uniqueness.

        Returns:
            The canonical Entity, or None for no match or an ambiguous match.
        """
        candidates = self._lookup.get(normalize(name), set())
        # Type restriction can disambiguate an otherwise shared drug/disease alias.
        candidates = [i for i in candidates if not kind or self.entities[i].kind == kind]
        return self.entities[candidates[0]] if len(candidates) == 1 else None

    def mentions(self, text: str) -> set[str]:
        """Find unambiguous canonical mentions in text using whole token sequences.

        Build the matcher only when needed. Aliases shorter than three characters
        and aliases that look like local IDs are excluded from free-text scanning.
        Surrounding spaces prevent "alpha" matching inside "alphabet". Matches are
        returned as a set because repeated mentions do not create extra entities.
        """
        if self._automaton is None:
            # Compile many exact aliases once instead of scanning the text separately per name.
            auto = ahocorasick.Automaton()
            for alias, ids in self._lookup.items():
                if (
                    len(ids) == 1
                    and len(alias) >= 3
                    and not alias.startswith(("drug ", "disease "))
                ):
                    # Padding creates token boundaries and prevents substring matches inside words.
                    auto.add_word(" " + alias + " ", next(iter(ids)))
            if not len(auto):
                return set()
            auto.make_automaton()
            self._automaton = auto
        return {eid for _, eid in self._automaton.iter(" " + normalize(text) + " ")}

    @property
    def ambiguous_aliases(self):
        """Expose aliases with multiple possible IDs for the ingestion report."""
        return {k: sorted(v) for k, v in self._lookup.items() if len(v) > 1}


def text_hash(text: str) -> str:
    """Hash normalized text for punctuation/case-insensitive duplicate detection.

    Used for literature deduplication, not file-integrity verification. File
    integrity uses io.digest, which hashes the original bytes instead.
    """
    return hashlib.sha256(normalize(text).encode()).hexdigest()
