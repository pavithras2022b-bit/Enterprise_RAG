"""
Unit tests for Module 5 - Relationship Extraction.

Covers relation_extraction/schema.py's LabeledRelationExample,
model.py (insert_entity_markers, RelationClassifier), rules.py
(RulePatternExtractor), and train.py (train_re). Model/training tests
use the session-scoped tiny_relation_checkpoint_dir fixture (see
conftest.py) to exercise the real transformers/torch code paths fully
offline.
"""

from __future__ import annotations

import logging

import pytest

from chunking.schema import Chunk
from ner.schema import Entity, EntityType
from relation_extraction.model import (
    OBJECT_END_TOKEN,
    OBJECT_START_TOKEN,
    RelationClassifier,
    RelationModelError,
    SUBJECT_END_TOKEN,
    SUBJECT_START_TOKEN,
    build_relation_label_maps,
    insert_entity_markers,
    predict_relation,
    predict_relations_for_chunk,
)
from relation_extraction.rules import PatternRule, RulePatternExtractor, apply_rule_patterns
from relation_extraction.schema import (
    LabeledRelationExample,
    NO_RELATION_LABEL,
    Relation,
    RelationType,
    SchemaValidationError,
)
from relation_extraction.train import RelationTrainingError, train_re

logging.disable(logging.CRITICAL)


def _entity(text: str, entity_type: EntityType, start: int, end: int) -> Entity:
    return Entity(
        text=text, entity_type=entity_type, start_offset=start, end_offset=end, source_chunk_id="c1"
    )


# --------------------------------------------------------------------------
# schema.py - LabeledRelationExample
# --------------------------------------------------------------------------


class TestLabeledRelationExample:
    def test_valid_construction(self) -> None:
        example = LabeledRelationExample(
            text="Asha Rao belongs to the Finance department.",
            subject_span=(0, 8),
            object_span=(24, 31),
            label="BELONGS_TO",
        )
        assert example.label == "BELONGS_TO"

    def test_no_relation_label_valid(self) -> None:
        example = LabeledRelationExample(
            text="Asha Rao uses SAP FI.", subject_span=(0, 8), object_span=(14, 20), label=NO_RELATION_LABEL
        )
        assert example.label == NO_RELATION_LABEL

    def test_invalid_label_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            LabeledRelationExample(
                text="text here", subject_span=(0, 4), object_span=(5, 9), label="NOT_A_REAL_RELATION"
            )

    def test_span_out_of_bounds_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            LabeledRelationExample(text="short", subject_span=(0, 4), object_span=(3, 20), label=NO_RELATION_LABEL)

    def test_overlapping_spans_raise(self) -> None:
        with pytest.raises(SchemaValidationError):
            LabeledRelationExample(
                text="Asha Rao belongs to Finance", subject_span=(0, 8), object_span=(4, 12), label=NO_RELATION_LABEL
            )

    def test_zero_length_span_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            LabeledRelationExample(text="text", subject_span=(2, 2), object_span=(0, 1), label=NO_RELATION_LABEL)


# --------------------------------------------------------------------------
# model.py - build_relation_label_maps / insert_entity_markers
# --------------------------------------------------------------------------


class TestBuildRelationLabelMaps:
    def test_includes_no_relation_and_all_relation_types(self) -> None:
        id2label, label2id = build_relation_label_maps()
        assert len(id2label) == 1 + len(RelationType)
        assert label2id[NO_RELATION_LABEL] == 0

    def test_maps_are_inverses(self) -> None:
        id2label, label2id = build_relation_label_maps()
        for index, label in id2label.items():
            assert label2id[label] == index


class TestInsertEntityMarkers:
    def test_forward_order(self) -> None:
        text = "Asha Rao belongs to the Finance department."
        marked = insert_entity_markers(text, subject_span=(0, 8), object_span=(24, 31))
        assert SUBJECT_START_TOKEN in marked
        assert marked.index(SUBJECT_START_TOKEN) < marked.index(OBJECT_START_TOKEN)
        assert "Asha Rao" in marked
        assert "Finance" in marked

    def test_reversed_order(self) -> None:
        text = "SAP FI is used by the Payments project."
        # object appears before subject in the text
        marked = insert_entity_markers(text, subject_span=(23, 31), object_span=(0, 6))
        assert marked.index(OBJECT_START_TOKEN) < marked.index(SUBJECT_START_TOKEN)

    def test_overlapping_spans_raise(self) -> None:
        with pytest.raises(ValueError):
            insert_entity_markers("Asha Rao belongs", subject_span=(0, 8), object_span=(4, 12))

    def test_out_of_bounds_span_raises(self) -> None:
        with pytest.raises(ValueError):
            insert_entity_markers("short text", subject_span=(0, 4), object_span=(5, 50))

    def test_all_markers_present(self) -> None:
        marked = insert_entity_markers("A B", subject_span=(0, 1), object_span=(2, 3))
        for token in (SUBJECT_START_TOKEN, SUBJECT_END_TOKEN, OBJECT_START_TOKEN, OBJECT_END_TOKEN):
            assert token in marked


# --------------------------------------------------------------------------
# model.py - RelationClassifier
# --------------------------------------------------------------------------


class TestRelationClassifier:
    def test_loads_from_local_checkpoint(self, tiny_relation_checkpoint_dir: str) -> None:
        model = RelationClassifier(tiny_relation_checkpoint_dir)
        assert len(model.id2label) == len(build_relation_label_maps()[0])

    def test_invalid_checkpoint_path_raises(self, tmp_path) -> None:
        with pytest.raises(RelationModelError):
            RelationClassifier(str(tmp_path / "does_not_exist"))

    def test_predict_relation_runs_without_error(self, tiny_relation_checkpoint_dir: str) -> None:
        model = RelationClassifier(tiny_relation_checkpoint_dir)
        subject = _entity("Asha Rao", EntityType.EMPLOYEE, 0, 8)
        obj = _entity("Finance", EntityType.DEPARTMENT, 24, 31)
        result = model.predict_relation(
            subject, obj, "Asha Rao belongs to the Finance department.", "c1", confidence_threshold=0.0
        )
        assert result is None or isinstance(result, Relation)

    def test_invalid_confidence_threshold_raises(self, tiny_relation_checkpoint_dir: str) -> None:
        model = RelationClassifier(tiny_relation_checkpoint_dir)
        subject = _entity("Asha Rao", EntityType.EMPLOYEE, 0, 8)
        obj = _entity("Finance", EntityType.DEPARTMENT, 24, 31)
        with pytest.raises(ValueError):
            model.predict_relation(subject, obj, "Asha Rao belongs to the Finance department.", "c1", confidence_threshold=1.5)

    def test_high_confidence_threshold_never_returns_below_threshold(
        self, tiny_relation_checkpoint_dir: str
    ) -> None:
        model = RelationClassifier(tiny_relation_checkpoint_dir)
        subject = _entity("Asha Rao", EntityType.EMPLOYEE, 0, 8)
        obj = _entity("Finance", EntityType.DEPARTMENT, 24, 31)
        result = model.predict_relation(
            subject, obj, "Asha Rao belongs to the Finance department.", "c1", confidence_threshold=0.999
        )
        if result is not None:
            assert result.confidence >= 0.999

    def test_predict_relation_module_function(self, tiny_relation_checkpoint_dir: str) -> None:
        model = RelationClassifier(tiny_relation_checkpoint_dir)
        subject = _entity("Asha Rao", EntityType.EMPLOYEE, 0, 8)
        obj = _entity("Finance", EntityType.DEPARTMENT, 24, 31)
        result = predict_relation(
            model, subject, obj, "Asha Rao belongs to the Finance department.", "c1", confidence_threshold=0.0
        )
        assert result is None or isinstance(result, Relation)

    def test_predict_relations_for_chunk_both_directions_tried(self, tiny_relation_checkpoint_dir: str) -> None:
        model = RelationClassifier(tiny_relation_checkpoint_dir)
        chunk = Chunk(
            chunk_id="doc1_chunk_0000",
            text="Asha Rao belongs to the Finance department.",
            source_document_id="doc1",
            position=0,
        )
        entities = [
            _entity("Asha Rao", EntityType.EMPLOYEE, 0, 8),
            _entity("Finance", EntityType.DEPARTMENT, 24, 31),
        ]
        relations = predict_relations_for_chunk(model, chunk, entities, confidence_threshold=0.0)
        assert isinstance(relations, list)
        assert all(r.source_chunk_id == "doc1_chunk_0000" for r in relations)

    def test_predict_relations_for_chunk_respects_max_pair_distance(
        self, tiny_relation_checkpoint_dir: str
    ) -> None:
        model = RelationClassifier(tiny_relation_checkpoint_dir)
        far_text = "Asha Rao " + ("filler word " * 30) + "Finance department."
        chunk = Chunk(chunk_id="doc1_chunk_0000", text=far_text, source_document_id="doc1", position=0)
        entities = [
            _entity("Asha Rao", EntityType.EMPLOYEE, 0, 8),
            _entity("Finance", EntityType.DEPARTMENT, far_text.index("Finance"), far_text.index("Finance") + 7),
        ]
        relations = predict_relations_for_chunk(
            model, chunk, entities, confidence_threshold=0.0, max_pair_distance=20
        )
        assert relations == []


# --------------------------------------------------------------------------
# rules.py
# --------------------------------------------------------------------------


class TestRulePatternExtractor:
    def test_belongs_to_pattern_matches(self) -> None:
        text = "Asha Rao belongs to the Finance department."
        entities = [
            _entity("Asha Rao", EntityType.EMPLOYEE, 0, 8),
            _entity("Finance", EntityType.DEPARTMENT, 24, 31),
        ]
        relations = apply_rule_patterns(text, entities, source_chunk_id="c1")
        assert len(relations) == 1
        assert relations[0].predicate == RelationType.BELONGS_TO

    def test_follows_pattern_matches(self) -> None:
        text = "The Finance department follows the Leave Policy."
        entities = [
            _entity("Finance", EntityType.DEPARTMENT, 4, 11),
            _entity("Leave Policy", EntityType.POLICY, 35, 47),
        ]
        relations = apply_rule_patterns(text, entities, source_chunk_id="c1")
        assert len(relations) == 1
        assert relations[0].predicate == RelationType.FOLLOWS

    def test_uses_pattern_matches(self) -> None:
        text = "The Payments project uses SAP FI."
        entities = [
            _entity("Payments", EntityType.PROJECT, 4, 12),
            _entity("SAP FI", EntityType.SOFTWARE, 26, 32),
        ]
        relations = apply_rule_patterns(text, entities, source_chunk_id="c1")
        assert len(relations) == 1
        assert relations[0].predicate == RelationType.USES

    def test_no_match_when_connector_does_not_match(self) -> None:
        text = "Asha Rao dislikes the Finance department."
        entities = [
            _entity("Asha Rao", EntityType.EMPLOYEE, 0, 8),
            _entity("Finance", EntityType.DEPARTMENT, 22, 29),
        ]
        relations = apply_rule_patterns(text, entities, source_chunk_id="c1")
        assert relations == []

    def test_no_match_when_types_do_not_match_rule(self) -> None:
        text = "Finance belongs to the Payments project."
        entities = [
            _entity("Finance", EntityType.DEPARTMENT, 0, 7),
            _entity("Payments", EntityType.PROJECT, 23, 31),
        ]
        # BELONGS_TO rule requires EMPLOYEE -> DEPARTMENT, not DEPARTMENT -> PROJECT
        relations = apply_rule_patterns(text, entities, source_chunk_id="c1")
        assert relations == []

    def test_connector_too_long_skips_pair(self) -> None:
        extractor = RulePatternExtractor(max_connector_length=5)
        text = "Asha Rao belongs to the Finance department."
        entities = [
            _entity("Asha Rao", EntityType.EMPLOYEE, 0, 8),
            _entity("Finance", EntityType.DEPARTMENT, 24, 31),
        ]
        assert extractor.apply_rule_patterns(text, entities, source_chunk_id="c1") == []

    def test_empty_entities_returns_empty_list(self) -> None:
        assert apply_rule_patterns("some text", []) == []

    def test_custom_rules(self) -> None:
        import re

        custom_rule = PatternRule(
            subject_type=EntityType.EMPLOYEE,
            object_type=EntityType.ROLE,
            connector_pattern=re.compile(r"\bpromoted to\b"),
            relation_type=RelationType.HAS_ROLE,
        )
        extractor = RulePatternExtractor(rules=(custom_rule,))
        text = "Asha Rao promoted to Senior Analyst."
        entities = [
            _entity("Asha Rao", EntityType.EMPLOYEE, 0, 8),
            _entity("Senior Analyst", EntityType.ROLE, 21, 35),
        ]
        relations = extractor.apply_rule_patterns(text, entities, source_chunk_id="c1")
        assert len(relations) == 1
        assert relations[0].predicate == RelationType.HAS_ROLE

    def test_invalid_max_connector_length_raises(self) -> None:
        with pytest.raises(ValueError):
            RulePatternExtractor(max_connector_length=0)

    def test_overlapping_entities_skipped(self) -> None:
        text = "Finance department"
        entities = [
            _entity("Finance department", EntityType.DEPARTMENT, 0, 19),
            _entity("department", EntityType.POLICY, 8, 18),  # overlaps with the first
        ]
        # Should not raise even though spans overlap -- simply produces no match.
        relations = apply_rule_patterns(text, entities, source_chunk_id="c1")
        assert relations == []

    def test_reports_to_creates_reverse_manages_edge(self) -> None:
        text = "Asha Rao reports to Priya Chandran."
        entities = [
            _entity("Asha Rao", EntityType.EMPLOYEE, 0, 8),
            _entity("Priya Chandran", EntityType.EMPLOYEE, 20, 33),
        ]
        relations = apply_rule_patterns(text, entities, source_chunk_id="c1")
        assert len(relations) == 1
        assert relations[0].predicate == RelationType.MANAGES
        assert relations[0].subject_entity.text == "Priya Chandran"
        assert relations[0].object_entity.text == "Asha Rao"

    def test_reports_to_skips_non_nearest_employee_cross_pair(self) -> None:
        text = "Asha Rao and Neha Kapoor report to Priya Chandran."
        entities = [
            _entity("Asha Rao", EntityType.EMPLOYEE, 0, 8),
            _entity("Neha Kapoor", EntityType.EMPLOYEE, 13, 24),
            _entity("Priya Chandran", EntityType.EMPLOYEE, 35, 48),
        ]
        relations = apply_rule_patterns(text, entities, source_chunk_id="c1")
        assert len(relations) == 1
        assert relations[0].object_entity.text == "Neha Kapoor"


# --------------------------------------------------------------------------
# train.py
# --------------------------------------------------------------------------


_TRAINING_EXAMPLES = [
    LabeledRelationExample(
        text="Asha Rao belongs to the Finance department.",
        subject_span=(0, 8), object_span=(24, 31), label="BELONGS_TO",
    ),
    LabeledRelationExample(
        text="The Finance department follows the Leave Policy.",
        subject_span=(4, 11), object_span=(35, 47), label="FOLLOWS",
    ),
    LabeledRelationExample(
        text="The Payments project uses SAP FI.",
        subject_span=(4, 12), object_span=(26, 32), label="USES",
    ),
    LabeledRelationExample(
        text="Asha Rao mentioned SAP FI in passing.",
        subject_span=(0, 8), object_span=(19, 25), label="NO_RELATION",
    ),
]


class TestTrainRe:
    def test_trains_and_returns_usable_model(self, tiny_relation_checkpoint_dir: str, tmp_path) -> None:
        output_dir = str(tmp_path / "trained_re")
        model = train_re(
            _TRAINING_EXAMPLES,
            base_checkpoint=tiny_relation_checkpoint_dir,
            output_dir=output_dir,
            num_epochs=1,
            batch_size=2,
            validation_split=0.0,
        )
        assert isinstance(model, RelationClassifier)
        subject = _entity("Asha Rao", EntityType.EMPLOYEE, 0, 8)
        obj = _entity("Finance", EntityType.DEPARTMENT, 24, 31)
        result = model.predict_relation(
            subject, obj, "Asha Rao belongs to the Finance department.", "c1", confidence_threshold=0.0
        )
        assert result is None or isinstance(result, Relation)

    def test_saves_checkpoint_to_output_dir(self, tiny_relation_checkpoint_dir: str, tmp_path) -> None:
        output_dir = tmp_path / "trained_re_2"
        train_re(
            _TRAINING_EXAMPLES,
            base_checkpoint=tiny_relation_checkpoint_dir,
            output_dir=str(output_dir),
            num_epochs=1,
            batch_size=2,
            validation_split=0.0,
        )
        assert (output_dir / "config.json").exists()

    def test_with_validation_split_runs_evaluation(self, tiny_relation_checkpoint_dir: str, tmp_path) -> None:
        output_dir = str(tmp_path / "trained_re_val")
        model = train_re(
            _TRAINING_EXAMPLES,
            base_checkpoint=tiny_relation_checkpoint_dir,
            output_dir=output_dir,
            num_epochs=1,
            batch_size=2,
            validation_split=0.5,
        )
        assert isinstance(model, RelationClassifier)

    def test_empty_labeled_data_raises(self, tiny_relation_checkpoint_dir: str, tmp_path) -> None:
        with pytest.raises(RelationTrainingError):
            train_re([], base_checkpoint=tiny_relation_checkpoint_dir, output_dir=str(tmp_path / "out"))

    def test_invalid_num_epochs_raises(self, tiny_relation_checkpoint_dir: str, tmp_path) -> None:
        with pytest.raises(ValueError):
            train_re(
                _TRAINING_EXAMPLES,
                base_checkpoint=tiny_relation_checkpoint_dir,
                output_dir=str(tmp_path / "out"),
                num_epochs=0,
            )

    def test_invalid_validation_split_raises(self, tiny_relation_checkpoint_dir: str, tmp_path) -> None:
        with pytest.raises(ValueError):
            train_re(
                _TRAINING_EXAMPLES,
                base_checkpoint=tiny_relation_checkpoint_dir,
                output_dir=str(tmp_path / "out"),
                validation_split=1.0,
            )

    def test_invalid_base_checkpoint_raises(self, tmp_path) -> None:
        with pytest.raises(RelationTrainingError):
            train_re(
                _TRAINING_EXAMPLES,
                base_checkpoint=str(tmp_path / "does_not_exist"),
                output_dir=str(tmp_path / "out"),
            )
