"""Adapt PubMed XML exports to the common Document schema.

PubMed records contain nested/mixed XML text, structured abstracts, source
identifiers, and dates at different precision levels. This adapter preserves
useful metadata, skips unusable/retracted records, and yields one paper at a
time. It does not infer therapeutic relationships from the article text.
"""

import calendar
import gzip
import re
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

from .models import Document


def content(node) -> str:
    """Flatten all nested XML text and normalize repeated whitespace.

    itertext includes text inside formatting tags such as <i> and <b>, unlike
    node.text alone. Missing nodes are represented by the empty string.
    """
    return " ".join("".join(node.itertext()).split()) if node is not None else ""


def parse_date(node):
    """Return a conservative date and its precision from a PubMed date node.

    Args:
        node: An XML date element, or None if the source has no date.

    Returns:
        (date_or_none, precision), where precision is day/month/year/unknown.
        Month-only dates become the month's last day; year-only dates become
        December 31. A MedlineDate string may supply a usable four-digit year.

    These are upper-bound representations of imprecise dates, not assertions
    that publication happened on that exact day. Invalid dates remain unknown.
    """
    if node is None:
        return None, "unknown"
    value = content(node)
    # MedlineDate can be free text such as "1998 Winter" rather than separate fields.
    year = node.findtext("Year") or (re.search(r"\b(18|19|20)\d{2}\b", value) or [None])[0]
    if not year:
        return None, "unknown"
    try:
        y = int(year)
        month = node.findtext("Month")
        day = node.findtext("Day")
        # Use the END of an imprecise interval to avoid admitting post-cutoff evidence.
        if not month:
            return date(y, 12, 31), "year"
        # PubMed may use a numeric month or an English abbreviation such as Jan.
        if month.isdigit():
            m = int(month)
        else:
            m = list(calendar.month_abbr).index(month[:3].title())
        if not day:
            return date(y, m, calendar.monthrange(y, m)[1]), "month"
        return date(y, m, int(day)), "day"
    # Do not invent a precise date when the source cannot be interpreted safely.
    except (ValueError, IndexError):
        return None, "unknown"


def read_pubmed(path: Path):
    """Yield normalized documents from one PubMed XML or XML.gz export.

    Args:
        path: Local efetch-style PubmedArticleSet export.

    Yields:
        Document records with PMID, title, structured abstract, namespaced source
        IDs, DOI, date/precision, and a stable PubMed evidence URL.

    Records lacking PMID/abstract and records marked Retracted Publication are
    skipped. ArticleDate is preferred when available, then journal PubDate.
    Processed article elements are cleared to release their child XML content.

    Raises:
        ValueError: The XML contains an explicit PubMed API error element.
        xml.etree.ElementTree.ParseError: The input is malformed XML.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        # End events ensure the complete article subtree is available before reading it.
        for _, article in ET.iterparse(stream, events=("end",)):
            if article.tag in {"ERROR", "Error"}:
                raise ValueError(f"{path}: PubMed returned an error: {content(article)}")
            # Ignore child-element events; only complete PubmedArticle nodes become documents.
            if article.tag != "PubmedArticle":
                continue
            citation = article.find("MedlineCitation")
            paper = citation.find("Article") if citation is not None else None
            if paper is None:
                article.clear()
                continue
            pmid = citation.findtext("PMID")
            # Keep structured labels such as BACKGROUND/RESULTS when joining abstract sections.
            abstract = "\n".join(
                ((a.get("Label") + ": ") if a.get("Label") else "") + content(a)
                for a in paper.findall("Abstract/AbstractText")
            ).strip()
            types = {content(t) for t in paper.findall("PublicationTypeList/PublicationType")}
            # A retrievable paper needs an ID and abstract; explicitly retracted records are
            # skipped.
            if not pmid or not abstract or "Retracted Publication" in types:
                article.clear()
                continue
            publication_date, precision = parse_date(paper.find("ArticleDate"))
            if publication_date is None:
                publication_date, precision = parse_date(paper.find("Journal/JournalIssue/PubDate"))
            # Use PubMed's typed ArticleId list to distinguish DOI from other article IDs.
            doi = next(
                (
                    content(a)
                    for a in article.findall("PubmedData/ArticleIdList/ArticleId")
                    if a.get("IdType") == "doi"
                ),
                None,
            )
            # The rest of the project consumes this common schema, not raw XML tags.
            yield Document(
                id=f"pubmed:{pmid}",
                title=content(paper.find("ArticleTitle")),
                abstract=abstract,
                publication_date=publication_date,
                date_precision=precision,
                mesh=[
                    content(m)
                    for m in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
                ],
                mesh_ids={
                    content(m): m.get("UI")
                    for m in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
                    if m.get("UI")
                },
                chemicals=[
                    content(c) for c in citation.findall("ChemicalList/Chemical/NameOfSubstance")
                ],
                chemical_ids={
                    content(c): c.get("UI")
                    for c in citation.findall("ChemicalList/Chemical/NameOfSubstance")
                    if c.get("UI")
                },
                doi=doi,
                source="pubmed",
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )
            article.clear()
