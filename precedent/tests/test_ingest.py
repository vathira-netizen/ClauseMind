"""Tests for document parsing, segmentation, and metadata extraction.

Read CONTEXT.md first.
"""

from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DictionaryObject, NameObject, TextStringObject

from precedent.agents.tools.ingest import (
    MAX_FILE_SIZE,
    UnsupportedFileError,
    _sanitize_pdf,
    extract_metadata,
    parse_document,
    segment_clauses,
)

INCOMING_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "incoming"
INCOMING_FILES = sorted(INCOMING_DIR.glob("*.json"))


@pytest.mark.parametrize("path", INCOMING_FILES, ids=lambda p: p.name)
def test_parse_document_reads_incoming_json_fixtures(path: Path) -> None:
    text = parse_document(str(path))

    assert text
    # Every synthetic incoming contract starts with a numbered clause heading.
    assert text.startswith("1. ")


@pytest.mark.parametrize("path", INCOMING_FILES, ids=lambda p: p.name)
def test_segment_clauses_splits_incoming_fixtures_on_numbered_headings(path: Path) -> None:
    text = parse_document(str(path))
    segments = segment_clauses(text)

    assert len(segments) >= 3
    assert [s["index"] for s in segments] == list(range(len(segments)))
    assert all(s["heading"] for s in segments)
    assert all(s["text"] for s in segments)


def test_segment_clauses_falls_back_to_paragraphs_without_numbered_headings() -> None:
    text = (
        "This is the first paragraph of an unstructured document.\n\n"
        "This is the second paragraph, with no numbering at all.\n\n"
        "And a third one to round it out."
    )

    segments = segment_clauses(text)

    assert len(segments) == 3
    assert [s["index"] for s in segments] == [0, 1, 2]
    assert all(s["heading"] is None for s in segments)
    assert segments[0]["text"].startswith("This is the first paragraph")


def test_extract_metadata_resolves_all_fields_via_regex_when_present() -> None:
    text = (
        'MASTER SERVICES AGREEMENT\n\n'
        'This Agreement is made by and between Acme Corp. ("Company") and '
        'Widget Industries Inc. ("Vendor"), effective as of January 15, 2025. '
        "This Agreement shall be governed by the laws of the State of Delaware.\n\n"
        "1. Indemnity\n\nSome clause text here."
    )

    metadata = extract_metadata(text)

    assert metadata["parties"] == ["Acme Corp.", "Widget Industries Inc"]
    assert metadata["effective_date"] == "January 15, 2025"
    assert "Delaware" in metadata["governing_law"]
    assert metadata["contract_type"] == "Master Services Agreement"
    assert metadata["needs_model_fallback"] == []


@pytest.mark.parametrize("path", INCOMING_FILES, ids=lambda p: p.name)
def test_extract_metadata_flags_model_fallback_when_regex_finds_nothing(path: Path) -> None:
    # The synthetic incoming fixtures are just numbered clauses with no
    # preamble (no parties/date/governing-law/title sentence), so regex
    # legitimately can't resolve any field here — this proves the
    # regex-first, flag-don't-guess contract holds rather than asserting a
    # result the fixtures don't actually contain.
    text = parse_document(str(path))

    metadata = extract_metadata(text)

    assert metadata["needs_model_fallback"] == [
        "parties",
        "effective_date",
        "governing_law",
        "contract_type",
    ]


def test_parse_document_rejects_extension_spoofed_file(tmp_path: Path) -> None:
    spoofed = tmp_path / "fake.pdf"
    spoofed.write_bytes(b"this is not a pdf at all, just plain text pretending to be one")

    with pytest.raises(UnsupportedFileError):
        parse_document(str(spoofed))


def test_parse_document_rejects_oversized_file(tmp_path: Path) -> None:
    big = tmp_path / "big.json"
    big.write_bytes(b'{"full_text": "' + b"a" * (MAX_FILE_SIZE + 1) + b'"}')

    with pytest.raises(ValueError, match="maximum allowed size"):
        parse_document(str(big))


def test_parse_document_rejects_non_docx_zip(tmp_path: Path) -> None:
    fake_docx = tmp_path / "fake.docx"
    with zipfile.ZipFile(fake_docx, "w") as zf:
        zf.writestr("not_word_content.txt", "hello")

    with pytest.raises(UnsupportedFileError):
        parse_document(str(fake_docx))


def test_parse_document_rejects_json_without_full_text(tmp_path: Path) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_bytes(b'{"contract_id": "abc"}')

    with pytest.raises(ValueError, match="full_text"):
        parse_document(str(bad_json))


def _pdf_with_javascript_and_open_action() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_js("app.alert('malicious');")

    open_action = DictionaryObject(
        {
            NameObject("/S"): NameObject("/JavaScript"),
            NameObject("/JS"): TextStringObject("app.alert('open-action malicious');"),
        }
    )
    writer.root_object[NameObject("/OpenAction")] = open_action

    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def test_sanitize_pdf_strips_open_action_and_javascript() -> None:
    raw = _pdf_with_javascript_and_open_action()
    reader = PdfReader(BytesIO(raw))

    # Confirm the fixture actually carries what we're about to strip.
    assert "/OpenAction" in reader.trailer["/Root"]
    names = reader.trailer["/Root"].get("/Names")
    assert names is not None and "/JavaScript" in names

    sanitized = _sanitize_pdf(reader)

    assert "/OpenAction" not in sanitized.root_object
    sanitized_names = sanitized.root_object.get("/Names")
    assert sanitized_names is None or "/JavaScript" not in sanitized_names


def test_parse_document_handles_malicious_pdf_end_to_end(tmp_path: Path) -> None:
    raw = _pdf_with_javascript_and_open_action()
    path = tmp_path / "malicious.pdf"
    path.write_bytes(raw)

    # Must not raise, and must not surface any script content as text.
    text = parse_document(str(path))

    assert "malicious" not in text
