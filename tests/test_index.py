from __future__ import annotations

import json
from pathlib import Path

import pymupdf
import pytest

import atlasrag.index as pdf_index
from atlasrag.index import (
    INDEX_VERSION,
    PDFIndex,
    index_name_for,
    manifest_path_for,
)


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text(
        (72, 72),
        "Introduction",
        fontsize=18,
    )
    page.insert_text(
        (72, 110),
        "AtlasRAG indexes text-based PDF documents "
        "for grounded question answering.",
        fontsize=11,
    )

    path = tmp_path / "0123abcd.pdf"
    document.save(path)
    document.close()
    return path


@pytest.fixture
def isolated_data_dir(tmp_path, monkeypatch) -> Path:
    data_dir = tmp_path / "data"
    monkeypatch.setattr(pdf_index, "DATA_DIR", data_dir)
    return data_dir


def test_index_name_uses_current_version():
    name = index_name_for("abc", 250)

    assert name == (
        f"pdf_{INDEX_VERSION.replace('-', '_')}_abc_250"
    )


def test_manifest_path_matches_pdf_index(
    sample_pdf,
    isolated_data_dir,
):
    index = PDFIndex(sample_pdf, chunk_size=250)

    expected = (
        isolated_data_dir
        / index_name_for(index.document_id, 250)
        / "manifest.json"
    )

    assert index.manifest_path == expected
    assert manifest_path_for(
        index.document_id,
        250,
    ).name == "manifest.json"


def test_display_name_is_used_in_citations(
    sample_pdf,
    isolated_data_dir,
):
    index = PDFIndex(
        sample_pdf,
        display_name="annual-report.pdf",
    )

    index._create_manifest()

    manifest = json.loads(
        index.manifest_path.read_text(encoding="utf-8")
    )

    assert manifest["filename"] == "annual-report.pdf"
    assert manifest["version"] == INDEX_VERSION
    assert all(
        chunk["filename"] == "annual-report.pdf"
        for chunk in manifest["chunks"]
    )
    assert all(
        "0123abcd" not in chunk["retrieval_text"]
        for chunk in manifest["chunks"]
    )


def test_display_name_drops_directories(sample_pdf):
    index = PDFIndex(
        sample_pdf,
        display_name="../../secret/report.pdf",
    )

    assert index.display_name == "report.pdf"


def test_manifest_round_trip(
    sample_pdf,
    isolated_data_dir,
):
    first = PDFIndex(sample_pdf, display_name="a.pdf")
    first._create_manifest()

    second = PDFIndex(sample_pdf, display_name="a.pdf")

    assert second._load_manifest()
    assert second.metadata.keys() == first.metadata.keys()