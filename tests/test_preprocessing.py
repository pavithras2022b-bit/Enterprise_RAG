"""
Unit tests for Module 2 - Cleaning.

Covers preprocessing/schema.py, preprocessing/cleaner.py, and
preprocessing/metadata_tagger.py.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from ingestion.schema import DocumentFormat, RawDocument, SourceSystem
from preprocessing.cleaner import (
    CleaningError,
    DocumentCleaner,
    dedupe_paragraphs,
    fix_encoding,
    strip_boilerplate,
)
from preprocessing.metadata_tagger import (
    MetadataTagger,
    MetadataTaggingError,
    tag_metadata,
)
from preprocessing.schema import CleanDocument, DocumentType, SchemaValidationError


def _make_raw_document(
    text: str = "hello world",
    source_path: str = "/data/raw/employee_handbook.pdf",
    checksum: str = "checksum-1",
) -> RawDocument:
    return RawDocument(
        document_id="pdf_abc123",
        text=text,
        source_path=source_path,
        source_system=SourceSystem.LOCAL_FILESYSTEM,
        document_format=DocumentFormat.PDF,
        collected_at=datetime.now(timezone.utc),
        checksum=checksum,
    )


# --------------------------------------------------------------------------
# schema.py
# --------------------------------------------------------------------------


class TestCleanDocument:
    def test_from_raw_document_inherits_provenance(self) -> None:
        raw = _make_raw_document()
        clean = CleanDocument.from_raw_document(raw, "cleaned text")

        assert clean.document_id == raw.document_id
        assert clean.checksum == raw.checksum
        assert clean.source_path == raw.source_path
        assert clean.text == "cleaned text"
        assert clean.document_type == DocumentType.UNKNOWN
        assert clean.department is None
        assert clean.effective_date is None

    def test_from_raw_document_with_tags(self) -> None:
        raw = _make_raw_document()
        clean = CleanDocument.from_raw_document(
            raw,
            "cleaned text",
            document_type=DocumentType.POLICY,
            department="Finance",
            effective_date=date(2025, 1, 1),
        )
        assert clean.document_type == DocumentType.POLICY
        assert clean.department == "Finance"
        assert clean.effective_date == date(2025, 1, 1)

    def test_empty_document_id_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            CleanDocument(
                document_id="",
                text="text",
                source_path="/x.pdf",
                source_system=SourceSystem.LOCAL_FILESYSTEM,
                document_format=DocumentFormat.PDF,
                collected_at=datetime.now(timezone.utc),
                checksum="c1",
            )

    def test_empty_text_logs_warning_but_does_not_raise(self, caplog) -> None:
        with caplog.at_level("WARNING"):
            clean = CleanDocument(
                document_id="pdf_x",
                text="   ",
                source_path="/x.pdf",
                source_system=SourceSystem.LOCAL_FILESYSTEM,
                document_format=DocumentFormat.PDF,
                collected_at=datetime.now(timezone.utc),
                checksum="c1",
            )
        assert clean.text.strip() == ""
        assert any("empty or whitespace" in r.message for r in caplog.records)

    def test_is_immutable(self) -> None:
        clean = CleanDocument.from_raw_document(_make_raw_document(), "text")
        with pytest.raises(AttributeError):
            clean.text = "mutated"  # type: ignore[misc]

    def test_to_metadata_dict_includes_tags_when_present(self) -> None:
        raw = _make_raw_document()
        clean = CleanDocument.from_raw_document(
            raw,
            "text",
            document_type=DocumentType.POLICY,
            department="HR",
            effective_date=date(2025, 6, 1),
        )
        metadata = clean.to_metadata_dict()
        assert metadata["document_type"] == "policy"
        assert metadata["department"] == "HR"
        assert metadata["effective_date"] == "2025-06-01"

    def test_to_metadata_dict_omits_absent_optional_tags(self) -> None:
        clean = CleanDocument.from_raw_document(_make_raw_document(), "text")
        metadata = clean.to_metadata_dict()
        assert "department" not in metadata
        assert "effective_date" not in metadata


# --------------------------------------------------------------------------
# cleaner.py - free functions
# --------------------------------------------------------------------------


class TestFixEncoding:
    def test_repairs_mojibake(self) -> None:
        # "café" mis-decoded as latin-1-then-utf8 mojibake
        broken = "cafÃ©"
        fixed = fix_encoding(broken)
        assert fixed == "café"

    def test_empty_string_returns_empty_string(self) -> None:
        assert fix_encoding("") == ""


class TestStripBoilerplate:
    def test_removes_repeated_footer(self) -> None:
        text = (
            "Section 1: Leave Policy\n"
            "Confidential -- Internal Use Only\n"
            "Casual leave is 12 days per year.\n"
            "Confidential -- Internal Use Only\n"
            "Section 2: Sick Leave\n"
            "Confidential -- Internal Use Only\n"
        )
        cleaned = strip_boilerplate(text, min_repeats=3)
        assert "Confidential -- Internal Use Only" not in cleaned
        assert "Casual leave is 12 days per year." in cleaned

    def test_removes_page_number_lines(self) -> None:
        text = "Some content.\nPage 3 of 40\nMore content.\n12\n"
        cleaned = strip_boilerplate(text)
        assert "Page 3 of 40" not in cleaned
        assert "Some content." in cleaned
        assert "More content." in cleaned

    def test_does_not_remove_lines_below_repeat_threshold(self) -> None:
        text = "Repeated twice.\nOther line.\nRepeated twice.\n"
        cleaned = strip_boilerplate(text, min_repeats=3)
        assert cleaned.count("Repeated twice.") == 2

    def test_long_repeated_line_not_stripped_by_line_length_cap(self) -> None:
        long_line = "x" * 250
        text = f"{long_line}\n{long_line}\n{long_line}\n"
        cleaned = strip_boilerplate(text, min_repeats=3, max_line_length=200)
        assert cleaned.count(long_line) == 3

    def test_empty_string_returns_empty_string(self) -> None:
        assert strip_boilerplate("") == ""


class TestDedupeParagraphs:
    def test_removes_near_duplicate_paragraph(self) -> None:
        text = (
            "Employees are entitled to twelve casual leave days per year.\n\n"
            "Something unrelated about VPN access procedures here.\n\n"
            "Employees are entitled to twelve casual leave days per year annually.\n"
        )
        cleaned = dedupe_paragraphs(text, similarity_threshold=0.7, shingle_size=3)
        occurrences = cleaned.count("twelve casual leave days")
        assert occurrences == 1

    def test_keeps_distinct_paragraphs(self) -> None:
        text = "Paragraph about leave policy.\n\nParagraph about VPN setup.\n"
        cleaned = dedupe_paragraphs(text)
        assert "leave policy" in cleaned
        assert "VPN setup" in cleaned

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            dedupe_paragraphs("text", similarity_threshold=1.5)

    def test_empty_string_returns_empty_string(self) -> None:
        assert dedupe_paragraphs("") == ""


# --------------------------------------------------------------------------
# cleaner.py - DocumentCleaner
# --------------------------------------------------------------------------


class TestDocumentCleaner:
    def test_full_pipeline_on_noisy_text(self) -> None:
        noisy_text = (
            "Employee Handbook\n"
            "Confidential -- Internal Use Only\n"
            "\n\n"
            "Section 4.2: Casual leave is 12 days per year.\n"
            "Page 1 of 3\n"
            "Confidential -- Internal Use Only\n"
            "\n\n"
            "Section 4.3: Sick leave is 8 days per year.\n"
            "Page 2 of 3\n"
            "Confidential -- Internal Use Only\n"
            "\n\n\n"
            "Section 4.2: Casual leave is 12 days per year annually.\n"
        )
        cleaner = DocumentCleaner(min_repeats=3, similarity_threshold=0.6, shingle_size=3)
        cleaned = cleaner.clean(noisy_text)

        assert "Confidential -- Internal Use Only" not in cleaned
        assert "Page 1 of 3" not in cleaned
        assert cleaned.count("Casual leave is 12 days per year") == 1
        assert "Sick leave is 8 days per year" in cleaned
        assert "\n\n\n" not in cleaned

    def test_none_input_raises_cleaning_error(self) -> None:
        cleaner = DocumentCleaner()
        with pytest.raises(CleaningError):
            cleaner.clean(None)  # type: ignore[arg-type]

    def test_empty_string_input_returns_empty_string(self, caplog) -> None:
        cleaner = DocumentCleaner()
        with caplog.at_level("WARNING"):
            result = cleaner.clean("")
        assert result == ""

    def test_invalid_min_repeats_raises(self) -> None:
        with pytest.raises(ValueError):
            DocumentCleaner(min_repeats=0)

    def test_invalid_similarity_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            DocumentCleaner(similarity_threshold=-0.1)

    def test_collapses_excess_whitespace(self) -> None:
        cleaner = DocumentCleaner()
        text = "Line one.\n\n\n\n\nLine two.    has   extra   spaces."
        cleaned = cleaner.clean(text)
        assert "\n\n\n" not in cleaned
        assert "   " not in cleaned


# --------------------------------------------------------------------------
# metadata_tagger.py
# --------------------------------------------------------------------------


class TestMetadataTagger:
    def test_infer_document_type_from_filename(self) -> None:
        raw = _make_raw_document(source_path="/data/raw/employee_handbook.pdf")
        tagger = MetadataTagger()
        assert tagger.infer_document_type(raw, "") == DocumentType.HANDBOOK

    def test_infer_document_type_from_content(self) -> None:
        raw = _make_raw_document(source_path="/data/raw/doc_001.pdf")
        tagger = MetadataTagger()
        text = "Frequently Asked Questions about VPN access."
        assert tagger.infer_document_type(raw, text) == DocumentType.FAQ

    def test_infer_document_type_unknown_when_no_match(self) -> None:
        raw = _make_raw_document(source_path="/data/raw/doc_001.pdf")
        tagger = MetadataTagger()
        assert tagger.infer_document_type(raw, "Some unrelated content.") == DocumentType.UNKNOWN

    def test_infer_department_picks_highest_scoring(self) -> None:
        tagger = MetadataTagger()
        text = "This finance policy covers expense reimbursement and payroll for Finance staff."
        assert tagger.infer_department(text) == "Finance"

    def test_infer_department_none_when_no_keywords(self) -> None:
        tagger = MetadataTagger()
        assert tagger.infer_department("Nothing department-specific here.") is None

    def test_infer_effective_date_parses_written_date(self) -> None:
        tagger = MetadataTagger()
        text = "This policy is Effective Date: January 1, 2025 for all staff."
        assert tagger.infer_effective_date(text) == date(2025, 1, 1)

    def test_infer_effective_date_parses_numeric_date(self) -> None:
        tagger = MetadataTagger()
        text = "Effective: 01/15/2025 onward."
        assert tagger.infer_effective_date(text) == date(2025, 1, 15)

    def test_infer_effective_date_none_when_absent(self) -> None:
        tagger = MetadataTagger()
        assert tagger.infer_effective_date("No date phrase here.") is None

    def test_tag_produces_complete_clean_document(self) -> None:
        raw = _make_raw_document(source_path="/data/raw/leave_policy.pdf")
        text = "Leave Policy. Effective Date: March 1, 2025. Applies to all Finance employees."
        tagger = MetadataTagger()
        clean = tagger.tag(raw, text)

        assert isinstance(clean, CleanDocument)
        assert clean.document_type == DocumentType.POLICY
        assert clean.department == "Finance"
        assert clean.effective_date == date(2025, 3, 1)
        assert clean.document_id == raw.document_id

    def test_tag_with_none_raw_raises(self) -> None:
        tagger = MetadataTagger()
        with pytest.raises(MetadataTaggingError):
            tagger.tag(None, "text")  # type: ignore[arg-type]

    def test_invalid_scan_chars_raises(self) -> None:
        with pytest.raises(ValueError):
            MetadataTagger(scan_chars=0)

    def test_module_level_tag_metadata_function(self) -> None:
        raw = _make_raw_document(source_path="/data/raw/it_manual.pdf")
        clean = tag_metadata(raw, "IT Manual covering VPN setup procedures.")
        assert clean.document_type == DocumentType.MANUAL
        assert clean.department == "IT"
