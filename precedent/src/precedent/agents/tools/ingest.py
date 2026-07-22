"""Document parsing, clause segmentation, and metadata extraction.

Read CONTEXT.md first.

Plain Python functions, wrapped as ADK FunctionTools by
:mod:`precedent.agents.intake` — nothing here imports ``google.adk``.

Contract text is untrusted input (see CONTEXT.md): a submitted file might be
mislabeled, oversized, or carry active PDF content designed to run when
opened by a careless viewer. :func:`parse_document` verifies file type from
content rather than trusting the extension, enforces a size ceiling, and
strips JavaScript/auto-run actions from PDFs before any text is extracted.
"""

from __future__ import annotations

import json
import re
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

from docx import Document
from pypdf import ObjectDeletionFlag, PdfReader, PdfWriter

MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB
MAX_DOCX_UNCOMPRESSED_SIZE = 100 * 1024 * 1024  # guards against zip bombs

_MAGIC_PDF = b"%PDF-"
_MAGIC_ZIP = b"PK\x03\x04"


class UnsupportedFileError(ValueError):
    """Raised when a file's actual content doesn't match a supported type."""


def _sniff_file_type(raw: bytes) -> str:
    """Identify a file's real type from its content, never its extension."""

    if raw.startswith(_MAGIC_PDF):
        return "pdf"

    if raw.startswith(_MAGIC_ZIP):
        # DOCX is OOXML packaged as a zip; the zip magic number alone isn't
        # proof of that — any zip (or a spoofed one) starts the same way.
        # Confirm the canonical Word package member is actually present.
        try:
            with zipfile.ZipFile(BytesIO(raw)) as zf:
                if "word/document.xml" in zf.namelist():
                    return "docx"
        except zipfile.BadZipFile:
            pass
        raise UnsupportedFileError("Zip-format file is not a valid DOCX package")

    try:
        json.loads(raw.decode("utf-8"))
        return "json"
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass

    raise UnsupportedFileError("Unrecognized file type: no matching magic bytes")


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sanitize_pdf(reader: PdfReader) -> PdfWriter:
    """Strip JavaScript and other active content from a PDF before any text
    is extracted from it.

    Removes document-level auto-run actions (``/OpenAction``) and named
    JavaScript (``/Names/JavaScript``) from the catalog, plus
    annotation-based actions (links, form fields, launch actions) from every
    page — the mechanisms a PDF uses to execute anything when opened. Text
    is extracted only from this sanitized copy, never the original reader.
    """

    writer = PdfWriter()
    writer.append(reader)

    root = writer.root_object
    if "/OpenAction" in root:
        del root["/OpenAction"]
    names = root.get("/Names")
    if names is not None and "/JavaScript" in names:
        del names["/JavaScript"]

    for page in writer.pages:
        writer.remove_objects_from_page(
            page, [ObjectDeletionFlag.ALL_ANNOTATIONS, ObjectDeletionFlag.LINKS]
        )

    return writer


def _parse_pdf(raw: bytes) -> str:
    reader = PdfReader(BytesIO(raw))
    sanitized = _sanitize_pdf(reader)
    return "\n\n".join(page.extract_text() or "" for page in sanitized.pages)


def _parse_docx(raw: bytes) -> str:
    # Uncompressed-size guard against zip bombs: a small compressed file
    # that expands into an enormous document.
    with zipfile.ZipFile(BytesIO(raw)) as zf:
        total_uncompressed = sum(info.file_size for info in zf.infolist())
        if total_uncompressed > MAX_DOCX_UNCOMPRESSED_SIZE:
            raise ValueError("DOCX contents exceed the maximum allowed uncompressed size")

    document = Document(BytesIO(raw))
    return "\n".join(p.text for p in document.paragraphs)


def _parse_json(raw: bytes) -> str:
    data = json.loads(raw.decode("utf-8"))
    if "full_text" not in data:
        raise ValueError("JSON contract file is missing the 'full_text' field")
    return data["full_text"]


def parse_document(path: str) -> str:
    """Parse a contract document into normalized text.

    Supports PDF, DOCX, and the synthetic incoming-contract JSON format (see
    data/synthetic/incoming/). File type is determined from content (magic
    bytes / package structure), never from the file extension, since a
    contract submission is untrusted input and an extension is just a claim
    about what the bytes are.
    """

    file_path = Path(path)
    if file_path.stat().st_size > MAX_FILE_SIZE:
        raise ValueError(f"File exceeds the maximum allowed size of {MAX_FILE_SIZE} bytes")

    raw = file_path.read_bytes()
    file_type = _sniff_file_type(raw)

    if file_type == "pdf":
        text = _parse_pdf(raw)
    elif file_type == "docx":
        text = _parse_docx(raw)
    else:
        text = _parse_json(raw)

    return _normalize_whitespace(text)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

_HEADING_PATTERN = re.compile(r"^[ \t]*(\d+(?:\.\d+)*)\.[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def segment_clauses(text: str) -> list[dict[str, Any]]:
    """Split contract text into clause segments.

    Primary strategy: split on numbered clause headings (``1. Indemnity``,
    ``2.1 Sub-clause``, ...). Falls back to blank-line paragraph grouping if
    no numbered headings are found at all. Each segment gets a stable,
    0-based ``index`` reflecting document order.
    """

    matches = list(_HEADING_PATTERN.finditer(text))

    if matches:
        segments = []
        for i, m in enumerate(matches):
            body_start = m.end()
            body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            segments.append(
                {
                    "index": i,
                    "heading": m.group(2).strip(),
                    "text": text[body_start:body_end].strip(),
                }
            )
        return segments

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [{"index": i, "heading": None, "text": p} for i, p in enumerate(paragraphs)]


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

_PARTIES_PATTERN = re.compile(
    r"(?:by and between|between)\s+(.+?)\s+(?:\([^)]*\)\s+)?and\s+(.+?)(?=[,.]|\s+\(|\s*$)",
    re.IGNORECASE,
)
_EFFECTIVE_DATE_PATTERN = re.compile(
    r"effective(?:\s+as\s+of|\s+date\s*(?:is|:)?)\s+"
    r"([A-Z][a-z]+\s+\d{1,2},?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})",
    re.IGNORECASE,
)
_GOVERNING_LAW_PATTERN = re.compile(
    r"govern(?:ed|ing)(?:\s+by(?:\s+and\s+construed\s+in\s+accordance\s+with)?)?\s+"
    r"(?:the\s+)?laws?\s+of\s+([A-Z][A-Za-z .,'()-]*?)(?=[.,;\n]|$)",
    re.IGNORECASE,
)
_CONTRACT_TYPE_PATTERN = re.compile(r"^[ \t]*([A-Z][A-Z0-9 &,'-]{3,80}AGREEMENT)[ \t]*$", re.MULTILINE)

METADATA_FIELDS = ("parties", "effective_date", "governing_law", "contract_type")


def extract_metadata(text: str) -> dict[str, Any]:
    """Extract parties, effective date, governing law, and contract type.

    Regex-first: each field is only populated when a confident pattern
    match is found. Any field regex could not resolve is listed under
    ``needs_model_fallback`` instead of being guessed at — callers with
    model access (the intake agent) fill those in; this function never
    calls a model itself.
    """

    result: dict[str, Any] = {field: None for field in METADATA_FIELDS}
    needs_model_fallback: list[str] = []

    parties_match = _PARTIES_PATTERN.search(text)
    if parties_match:
        result["parties"] = [
            parties_match.group(1).strip(" \"'"),
            parties_match.group(2).strip(" \"'"),
        ]
    else:
        needs_model_fallback.append("parties")

    date_match = _EFFECTIVE_DATE_PATTERN.search(text)
    if date_match:
        result["effective_date"] = date_match.group(1).strip()
    else:
        needs_model_fallback.append("effective_date")

    law_match = _GOVERNING_LAW_PATTERN.search(text)
    if law_match:
        result["governing_law"] = law_match.group(1).strip()
    else:
        needs_model_fallback.append("governing_law")

    type_match = _CONTRACT_TYPE_PATTERN.search(text[:1000])
    if type_match:
        result["contract_type"] = type_match.group(1).strip().title()
    else:
        needs_model_fallback.append("contract_type")

    result["needs_model_fallback"] = needs_model_fallback
    return result
