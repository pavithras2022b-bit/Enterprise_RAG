"""
Module 1 - Document Ingestion: format-specific text extraction.

Provides extract_pdf, extract_docx, extract_html functions and a
DocumentLoader class that dispatches to the correct extractor by file
extension, per the Phase 4 module contract:

    extract_pdf(path) -> str
    extract_docx(path) -> str
    extract_html(path) -> str

DocumentLoader wraps these into RawDocument objects with provenance
metadata attached.

Reference: Capstone Project Guide, Section 3A/12.2 (pdfplumber,
PyMuPDF, python-docx, BeautifulSoup), Phase 4 Module 1 contract,
Phase 5 file structure (ingestion/loaders.py).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import fitz  # PyMuPDF
import pdfplumber
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

from ingestion.dedupe import compute_checksum
from ingestion.schema import DocumentFormat, RawDocument, SourceSystem

logger = logging.getLogger(__name__)


class DocumentLoaderError(Exception):
    """Raised when a document cannot be extracted by any registered loader."""


class UnsupportedFormatError(DocumentLoaderError):
    """Raised when a file extension has no registered extraction function."""


def extract_pdf(path: str | Path) -> str:
    """
    Extract raw text from a PDF file.

    Uses pdfplumber as the primary extractor (stronger table/layout
    handling) and falls back to PyMuPDF (fitz) if pdfplumber raises or
    returns no text, since PyMuPDF is more robust against certain
    malformed PDF structures. This dual-library strategy matches the
    project guide's chosen stack (Section 12.2).

    Args:
        path: Filesystem path to the .pdf file.

    Returns:
        The concatenated raw text of every page, separated by
        newlines.

    Raises:
        DocumentLoaderError: If both pdfplumber and PyMuPDF fail to
            extract any text from the file (e.g. a scanned, image-only
            PDF with no OCR layer -- OCR is explicitly out of scope
            for Module 1 per Section 3B of the project guide).
    """
    path = Path(path)
    text_parts: list[str] = []

    try:
        with pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
                else:
                    logger.debug(
                        "pdfplumber found no text on page %d of %s",
                        page_number,
                        path,
                    )
        if text_parts:
            return "\n".join(text_parts)
        logger.warning(
            "pdfplumber extracted no text from %s; falling back to PyMuPDF",
            path,
        )
    except Exception as exc:  # noqa: BLE001 - extraction is best-effort by design
        logger.warning(
            "pdfplumber failed on %s (%s); falling back to PyMuPDF", path, exc
        )

    try:
        doc = fitz.open(path)
        try:
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    text_parts.append(page_text)
        finally:
            doc.close()
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoaderError(
            f"Failed to extract text from PDF {path} using both pdfplumber "
            f"and PyMuPDF: {exc}"
        ) from exc

    if not text_parts:
        raise DocumentLoaderError(
            f"No extractable text found in PDF {path} (possibly a scanned, "
            f"image-only document requiring OCR, which is out of scope for "
            f"Module 1 per Section 3B of the project guide)."
        )
    return "\n".join(text_parts)


def extract_docx(path: str | Path) -> str:
    """
    Extract raw text from a Word (.docx) file.

    Extracts paragraph text and table cell text, in document order,
    since HR/policy DOCX files frequently place key facts (e.g. leave
    entitlements) inside tables rather than plain paragraphs.

    Args:
        path: Filesystem path to the .docx file.

    Returns:
        The concatenated raw text of all paragraphs and tables.

    Raises:
        DocumentLoaderError: If the file cannot be opened/parsed as a
            valid DOCX, or contains no extractable text.
    """
    path = Path(path)
    try:
        document = DocxDocument(str(path))
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoaderError(f"Failed to open DOCX {path}: {exc}") from exc

    text_parts: list[str] = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)

    for table in document.tables:
        for row in table.rows:
            cell_texts = [
                cell.text.strip() for cell in row.cells if cell.text.strip()
            ]
            if cell_texts:
                text_parts.append(" | ".join(cell_texts))

    if not text_parts:
        raise DocumentLoaderError(
            f"DOCX {path} was parsed successfully but yielded no extractable "
            f"text (empty document?)."
        )
    return "\n".join(text_parts)


def extract_html(path: str | Path) -> str:
    """
    Extract raw visible text from an HTML file.

    Strips <script>, <style>, and <noscript> tags before extraction
    (their contents are never human-readable prose and would corrupt
    downstream chunking/embedding), then joins block-level text with
    newlines to loosely preserve document structure.

    Args:
        path: Filesystem path to the .html/.htm file.

    Returns:
        The extracted visible text content, one line per non-empty
        block-level text node.

    Raises:
        DocumentLoaderError: If the file cannot be read/parsed, or
            yields no visible text.
    """
    path = Path(path)
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise DocumentLoaderError(f"Failed to read HTML file {path}: {exc}") from exc

    try:
        soup = BeautifulSoup(raw_bytes, "html.parser")
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoaderError(f"Failed to parse HTML {path}: {exc}") from exc

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    non_empty_lines = [line for line in lines if line]

    if not non_empty_lines:
        raise DocumentLoaderError(
            f"HTML {path} was parsed successfully but yielded no visible "
            f"text content."
        )
    return "\n".join(non_empty_lines)


class DocumentLoader:
    """
    Dispatches document extraction to the correct format-specific
    function based on file extension, and wraps the result into a
    RawDocument with provenance metadata attached.

    This is the entry point Module 1's callers (the ingestion
    orchestration step / Phase 6 pipeline wiring) should use, rather
    than calling extract_pdf / extract_docx / extract_html directly,
    since it also handles checksum computation and RawDocument
    construction end-to-end.
    """

    _EXTENSION_MAP: dict[str, DocumentFormat] = {
        ".pdf": DocumentFormat.PDF,
        ".docx": DocumentFormat.DOCX,
        ".html": DocumentFormat.HTML,
        ".htm": DocumentFormat.HTML,
    }

    _EXTRACTORS: dict[DocumentFormat, Callable[[Path], str]] = {
        DocumentFormat.PDF: extract_pdf,
        DocumentFormat.DOCX: extract_docx,
        DocumentFormat.HTML: extract_html,
    }

    def __init__(
        self, source_system: SourceSystem = SourceSystem.LOCAL_FILESYSTEM
    ) -> None:
        """
        Args:
            source_system: The SourceSystem tag to attach to every
                RawDocument this loader instance produces. Defaults to
                LOCAL_FILESYSTEM, appropriate for the synthetic
                capstone corpus described in Section 10, Phase 1 of
                the project guide.
        """
        self._source_system = source_system

    def supports(self, path: str | Path) -> bool:
        """Returns True if the file extension has a registered extractor."""
        return Path(path).suffix.lower() in self._EXTENSION_MAP

    def load(self, path: str | Path) -> RawDocument:
        """
        Extract a single file into a RawDocument.

        Args:
            path: Filesystem path to the document.

        Returns:
            A populated RawDocument with checksum, extracted text, and
            provenance metadata.

        Raises:
            FileNotFoundError: If path does not exist.
            UnsupportedFormatError: If the file extension has no
                registered extractor.
            DocumentLoaderError: If extraction fails for a supported
                format.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Document path does not exist: {path}")

        extension = path.suffix.lower()
        document_format = self._EXTENSION_MAP.get(extension)
        if document_format is None:
            raise UnsupportedFormatError(
                f"No extractor registered for extension '{extension}' "
                f"(file: {path}). Supported extensions: "
                f"{sorted(self._EXTENSION_MAP)}"
            )

        extractor = self._EXTRACTORS[document_format]
        logger.info("Extracting %s using %s", path, extractor.__name__)
        text = extractor(path)

        checksum = compute_checksum(path.read_bytes())
        document_id = f"{document_format.value}_{checksum[:16]}"

        return RawDocument(
            document_id=document_id,
            text=text,
            source_path=str(path),
            source_system=self._source_system,
            document_format=document_format,
            collected_at=datetime.now(timezone.utc),
            checksum=checksum,
        )

    def load_directory(
        self, directory: str | Path, recursive: bool = True
    ) -> list[RawDocument]:
        """
        Extract every supported file under a directory into RawDocuments.

        Files with unsupported extensions are skipped (logged at DEBUG
        level) rather than raising, since a raw document directory
        (Stage 0 of the project guide) legitimately contains a mix of
        supported and irrelevant files. Files that fail extraction are
        logged as errors and skipped, not raised, so a single corrupt
        file doesn't halt ingestion of the whole corpus.

        Args:
            directory: Root directory to scan.
            recursive: If True, scans subdirectories too (glob "**/*"
                vs "*").

        Returns:
            A list of successfully-extracted RawDocuments, in sorted
            path order.

        Raises:
            NotADirectoryError: If directory does not exist or is not
                a directory.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")

        pattern = "**/*" if recursive else "*"
        documents: list[RawDocument] = []

        for file_path in sorted(directory.glob(pattern)):
            if not file_path.is_file():
                continue
            if not self.supports(file_path):
                logger.debug("Skipping unsupported file: %s", file_path)
                continue
            try:
                documents.append(self.load(file_path))
            except DocumentLoaderError as exc:
                logger.error(
                    "Skipping %s due to extraction failure: %s", file_path, exc
                )

        logger.info(
            "Loaded %d documents from %s (recursive=%s)",
            len(documents),
            directory,
            recursive,
        )
        return documents
