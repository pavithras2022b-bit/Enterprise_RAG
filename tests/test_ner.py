"""
Unit tests for Module 4 - NER.

Covers ner/schema.py, ner/model.py, ner/resolver.py, and ner/train.py.
Model/training tests use the session-scoped tiny_bert_checkpoint_dir
fixture (see conftest.py) to exercise the real transformers/torch code
paths fully offline, without Hugging Face Hub access.
"""

from __future__ import annotations

import logging

import pytest

from ner.extractor import _merge_adjacent_entity_fragments, map_generic_label_to_enterprise_type
from ner.model import NERModel, NERModelError, predict_entities, predict_entities_for_chunk
from ner.resolver import EntityResolver, resolve_entity
from ner.schema import (
    Entity,
    EntityType,
    LabeledExample,
    SchemaValidationError,
    build_bio_label_maps,
)
from ner.train import NERTrainingError, train_ner

logging.disable(logging.CRITICAL)  # keep HF Trainer's per-epoch logging out of test output


# --------------------------------------------------------------------------
# schema.py
# --------------------------------------------------------------------------


class TestEntity:
    def test_construction(self) -> None:
        entity = Entity(
            text="Finance department",
            entity_type=EntityType.DEPARTMENT,
            start_offset=10,
            end_offset=29,
            source_chunk_id="chunk_0001",
        )
        assert entity.span_length == 19
        assert entity.is_resolved is False
        assert entity.text == "Finance department"

    def test_resolved_after_canonical_name_set(self) -> None:
        entity = Entity(
            text="Finance Dept.",
            entity_type=EntityType.DEPARTMENT,
            start_offset=0,
            end_offset=13,
            source_chunk_id="c1",
            canonical_name="Finance",
        )
        assert entity.is_resolved is True

    def test_empty_text_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            Entity(
                text="   ",
                entity_type=EntityType.DEPARTMENT,
                start_offset=0,
                end_offset=5,
                source_chunk_id="c1",
            )

    def test_negative_start_offset_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            Entity(
                text="Finance",
                entity_type=EntityType.DEPARTMENT,
                start_offset=-1,
                end_offset=5,
                source_chunk_id="c1",
            )

    def test_end_before_start_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            Entity(
                text="Finance",
                entity_type=EntityType.DEPARTMENT,
                start_offset=10,
                end_offset=5,
                source_chunk_id="c1",
            )

    def test_confidence_out_of_range_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            Entity(
                text="Finance",
                entity_type=EntityType.DEPARTMENT,
                start_offset=0,
                end_offset=7,
                source_chunk_id="c1",
                confidence=1.5,
            )

    def test_is_immutable(self) -> None:
        entity = Entity(
            text="Finance",
            entity_type=EntityType.DEPARTMENT,
            start_offset=0,
            end_offset=7,
            source_chunk_id="c1",
        )
        with pytest.raises(AttributeError):
            entity.text = "mutated"  # type: ignore[misc]


class TestLabeledExample:
    def test_valid_construction(self) -> None:
        example = LabeledExample(
            tokens=("Finance", "department"),
            tags=("B-DEPARTMENT", "I-DEPARTMENT"),
        )
        assert len(example.tokens) == 2

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            LabeledExample(tokens=("Finance", "department"), tags=("B-DEPARTMENT",))

    def test_empty_tokens_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            LabeledExample(tokens=(), tags=())

    def test_invalid_tag_prefix_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            LabeledExample(tokens=("Finance",), tags=("X-DEPARTMENT",))

    def test_invalid_entity_type_in_tag_raises(self) -> None:
        with pytest.raises(SchemaValidationError):
            LabeledExample(tokens=("Finance",), tags=("B-NOT_A_REAL_TYPE",))

    def test_o_tag_always_valid(self) -> None:
        example = LabeledExample(tokens=("the",), tags=("O",))
        assert example.tags == ("O",)


class TestBuildBioLabelMaps:
    def test_default_covers_all_entity_types(self) -> None:
        id2label, label2id = build_bio_label_maps()
        # 1 "O" + 2 tags (B/I) per entity type
        assert len(id2label) == 1 + 2 * len(EntityType)
        assert id2label[0] == "O"
        assert label2id["O"] == 0

    def test_subset_of_entity_types(self) -> None:
        id2label, label2id = build_bio_label_maps([EntityType.DEPARTMENT])
        assert len(id2label) == 3  # O, B-DEPARTMENT, I-DEPARTMENT
        assert "B-DEPARTMENT" in label2id
        assert "B-EMPLOYEE" not in label2id

    def test_maps_are_inverses(self) -> None:
        id2label, label2id = build_bio_label_maps()
        for index, label in id2label.items():
            assert label2id[label] == index


class TestGenericLabelMapping:
    """The CoNLL compatibility adapter is explicit and deterministic."""

    @pytest.mark.parametrize(
        ("label", "span", "context", "expected"),
        [
            ("PER", "Asha Rao", "Asha Rao works in Finance.", EntityType.EMPLOYEE),
            ("ORG", "Finance", "Finance department follows the policy.", EntityType.DEPARTMENT),
            ("LOC", "Chennai", "The office is in Chennai.", EntityType.NETWORK_RESOURCE),
            ("MISC", "Annual Leave", "The Annual Leave policy applies.", EntityType.LEAVE_TYPE),
            ("MISC", "Project Orion", "Project Orion is an initiative.", EntityType.PROJECT),
            ("MISC", "SAP FI", "Project Orion uses SAP FI for accounting.", EntityType.SOFTWARE),
        ],
    )
    def test_maps_supported_generic_labels(self, label, span, context, expected) -> None:
        assert map_generic_label_to_enterprise_type(label, span, context) == expected

    def test_unknown_generic_label_is_not_mapped(self) -> None:
        assert map_generic_label_to_enterprise_type("EVENT", "Town hall", "Town hall") is None

    def test_merges_adjacent_generic_name_fragments(self) -> None:
        text = "Divya Ramaswamy"
        entities = [
            Entity("Di", EntityType.EMPLOYEE, 0, 2, "c1", confidence=0.8),
            Entity("vya Ramaswamy", EntityType.EMPLOYEE, 2, 15, "c1", confidence=0.9),
        ]
        merged = _merge_adjacent_entity_fragments(entities, text)
        assert len(merged) == 1
        assert merged[0].text == "Divya Ramaswamy"


# --------------------------------------------------------------------------
# model.py
# --------------------------------------------------------------------------


class TestNERModel:
    def test_loads_from_local_checkpoint(self, tiny_bert_checkpoint_dir: str) -> None:
        model = NERModel(tiny_bert_checkpoint_dir)
        assert len(model.id2label) == len(build_bio_label_maps()[0])

    def test_invalid_checkpoint_path_raises(self, tmp_path) -> None:
        with pytest.raises(NERModelError):
            NERModel(str(tmp_path / "does_not_exist"))

    def test_predict_entities_empty_text_returns_empty_list(
        self, tiny_bert_checkpoint_dir: str
    ) -> None:
        model = NERModel(tiny_bert_checkpoint_dir)
        assert model.predict_entities("") == []
        assert model.predict_entities("   ") == []

    def test_predict_entities_runs_without_error(self, tiny_bert_checkpoint_dir: str) -> None:
        model = NERModel(tiny_bert_checkpoint_dir)
        entities = model.predict_entities(
            "Asha Rao belongs to the Finance department.",
            source_chunk_id="chunk_0001",
            confidence_threshold=0.0,
        )
        assert isinstance(entities, list)
        for entity in entities:
            assert isinstance(entity, Entity)
            assert entity.source_chunk_id == "chunk_0001"

    def test_invalid_confidence_threshold_raises(self, tiny_bert_checkpoint_dir: str) -> None:
        model = NERModel(tiny_bert_checkpoint_dir)
        with pytest.raises(ValueError):
            model.predict_entities("text", confidence_threshold=1.5)

    def test_high_confidence_threshold_yields_fewer_or_equal_entities(
        self, tiny_bert_checkpoint_dir: str
    ) -> None:
        model = NERModel(tiny_bert_checkpoint_dir)
        text = "Asha Rao belongs to the Finance department."
        low = model.predict_entities(text, confidence_threshold=0.0)
        high = model.predict_entities(text, confidence_threshold=0.99)
        assert len(high) <= len(low)

    def test_predict_entities_module_function(self, tiny_bert_checkpoint_dir: str) -> None:
        model = NERModel(tiny_bert_checkpoint_dir)
        entities = predict_entities(model, "Finance department.", confidence_threshold=0.0)
        assert isinstance(entities, list)

    def test_predict_entities_for_chunk(self, tiny_bert_checkpoint_dir: str) -> None:
        from chunking.schema import Chunk

        model = NERModel(tiny_bert_checkpoint_dir)
        chunk = Chunk(
            chunk_id="doc1_chunk_0000",
            text="Finance department follows the Leave Policy.",
            source_document_id="doc1",
            position=0,
        )
        entities = predict_entities_for_chunk(model, chunk, confidence_threshold=0.0)
        assert all(e.source_chunk_id == "doc1_chunk_0000" for e in entities)


# --------------------------------------------------------------------------
# resolver.py
# --------------------------------------------------------------------------


def _entity(text: str, entity_type: EntityType = EntityType.DEPARTMENT) -> Entity:
    return Entity(
        text=text, entity_type=entity_type, start_offset=0, end_offset=len(text), source_chunk_id="c1"
    )


class TestResolveEntity:
    def test_first_mention_becomes_its_own_canonical_name(self) -> None:
        resolved = resolve_entity(_entity("Finance department"), existing_nodes=[])
        assert resolved.canonical_name == "Finance department"

    def test_merges_paraphrase_variant_above_threshold(self) -> None:
        existing = resolve_entity(_entity("Finance department"), existing_nodes=[])
        new_mention = resolve_entity(
            _entity("the Finance team"), existing_nodes=[existing], similarity_threshold=65.0
        )
        assert new_mention.canonical_name == existing.canonical_name

    def test_does_not_merge_across_entity_types(self) -> None:
        finance_dept = resolve_entity(_entity("Finance", EntityType.DEPARTMENT), existing_nodes=[])
        finance_role = resolve_entity(
            _entity("Finance", EntityType.ROLE),
            existing_nodes=[finance_dept],
            similarity_threshold=50.0,
        )
        # Same text, different type -> must NOT merge with the department node
        assert finance_role.entity_type != finance_dept.entity_type
        assert finance_role.canonical_name == "Finance"

    def test_distinct_departments_stay_distinct(self) -> None:
        finance = resolve_entity(_entity("Finance department"), existing_nodes=[])
        it_dept = resolve_entity(
            _entity("IT department"), existing_nodes=[finance], similarity_threshold=85.0
        )
        assert it_dept.canonical_name != finance.canonical_name

    def test_known_limitation_abbreviation_not_merged_at_default_threshold(self) -> None:
        # Documents a real, honest limitation (see configs/ner.yaml):
        # character-level fuzzy matching does not bridge "department"
        # vs "Dept." at the default threshold.
        full = resolve_entity(_entity("Finance department"), existing_nodes=[])
        abbreviated = resolve_entity(
            _entity("Finance Dept."), existing_nodes=[full], similarity_threshold=85.0
        )
        assert abbreviated.canonical_name != full.canonical_name

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            resolve_entity(_entity("Finance"), existing_nodes=[], similarity_threshold=150.0)


class TestEntityResolver:
    def test_resolve_all_grows_registry_across_calls(self) -> None:
        resolver = EntityResolver(similarity_threshold=65.0)
        entities = [_entity("Finance department"), _entity("the Finance team"), _entity("Legal team")]
        resolved = resolver.resolve_all(entities)

        assert resolved[0].canonical_name == resolved[1].canonical_name
        assert resolved[2].canonical_name != resolved[0].canonical_name
        assert resolver.resolved_count == 3

    def test_canonical_names_grouped_by_type(self) -> None:
        resolver = EntityResolver(similarity_threshold=65.0)
        resolver.resolve(_entity("Finance department", EntityType.DEPARTMENT))
        resolver.resolve(_entity("Finance Analyst", EntityType.ROLE))
        names = resolver.canonical_names()
        assert "DEPARTMENT" in names
        assert "ROLE" in names

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            EntityResolver(similarity_threshold=-1.0)


# --------------------------------------------------------------------------
# train.py
# --------------------------------------------------------------------------


_TRAINING_EXAMPLES = [
    LabeledExample(
        tokens=("Asha", "Rao", "belongs", "to", "the", "Finance", "department", "."),
        tags=("B-EMPLOYEE", "I-EMPLOYEE", "O", "O", "O", "B-DEPARTMENT", "I-DEPARTMENT", "O"),
    ),
    LabeledExample(
        tokens=("The", "Finance", "department", "follows", "the", "Leave", "Policy", "."),
        tags=("O", "B-DEPARTMENT", "I-DEPARTMENT", "O", "O", "B-POLICY", "I-POLICY", "O"),
    ),
    LabeledExample(
        tokens=("Employees", "get", "twelve", "casual", "leave", "days", "."),
        tags=("O", "O", "O", "B-LEAVE_TYPE", "I-LEAVE_TYPE", "O", "O"),
    ),
    LabeledExample(
        tokens=("The", "Payments", "project", "uses", "SAP", "FI", "."),
        tags=("O", "B-PROJECT", "I-PROJECT", "O", "B-SOFTWARE", "I-SOFTWARE", "O"),
    ),
]


class TestTrainNer:
    def test_trains_and_returns_usable_model(self, tiny_bert_checkpoint_dir: str, tmp_path) -> None:
        output_dir = str(tmp_path / "trained_ner")
        model = train_ner(
            _TRAINING_EXAMPLES,
            base_checkpoint=tiny_bert_checkpoint_dir,
            output_dir=output_dir,
            num_epochs=1,
            batch_size=2,
            validation_split=0.0,
        )
        assert isinstance(model, NERModel)
        entities = model.predict_entities(
            "Asha Rao belongs to the Finance department.", confidence_threshold=0.0
        )
        assert isinstance(entities, list)

    def test_saves_checkpoint_to_output_dir(self, tiny_bert_checkpoint_dir: str, tmp_path) -> None:
        output_dir = tmp_path / "trained_ner_2"
        train_ner(
            _TRAINING_EXAMPLES,
            base_checkpoint=tiny_bert_checkpoint_dir,
            output_dir=str(output_dir),
            num_epochs=1,
            batch_size=2,
            validation_split=0.0,
        )
        assert (output_dir / "config.json").exists()

    def test_with_validation_split_runs_evaluation(self, tiny_bert_checkpoint_dir: str, tmp_path) -> None:
        output_dir = str(tmp_path / "trained_ner_val")
        model = train_ner(
            _TRAINING_EXAMPLES,
            base_checkpoint=tiny_bert_checkpoint_dir,
            output_dir=output_dir,
            num_epochs=1,
            batch_size=2,
            validation_split=0.5,
        )
        assert isinstance(model, NERModel)

    def test_empty_labeled_data_raises(self, tiny_bert_checkpoint_dir: str, tmp_path) -> None:
        with pytest.raises(NERTrainingError):
            train_ner([], base_checkpoint=tiny_bert_checkpoint_dir, output_dir=str(tmp_path / "out"))

    def test_invalid_num_epochs_raises(self, tiny_bert_checkpoint_dir: str, tmp_path) -> None:
        with pytest.raises(ValueError):
            train_ner(
                _TRAINING_EXAMPLES,
                base_checkpoint=tiny_bert_checkpoint_dir,
                output_dir=str(tmp_path / "out"),
                num_epochs=0,
            )

    def test_invalid_validation_split_raises(self, tiny_bert_checkpoint_dir: str, tmp_path) -> None:
        with pytest.raises(ValueError):
            train_ner(
                _TRAINING_EXAMPLES,
                base_checkpoint=tiny_bert_checkpoint_dir,
                output_dir=str(tmp_path / "out"),
                validation_split=1.0,
            )

    def test_invalid_base_checkpoint_raises(self, tmp_path) -> None:
        with pytest.raises(NERTrainingError):
            train_ner(
                _TRAINING_EXAMPLES,
                base_checkpoint=str(tmp_path / "does_not_exist"),
                output_dir=str(tmp_path / "out"),
            )
