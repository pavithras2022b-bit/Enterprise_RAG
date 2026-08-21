"""
Unit tests for Module 11 - Cross-Encoder Re-ranking.

Covers reranking/schema.py, model.py, and train.py. Model/training
tests use the session-scoped tiny_cross_encoder_dir fixture (see
conftest.py) to exercise the real sentence-transformers/torch code
paths fully offline.
"""

from __future__ import annotations

import logging

import pytest

from reranking.model import CrossEncoderReranker, CrossEncoderRerankerError, rerank, score_pair
from reranking.schema import RankedCandidate, SchemaValidationError
from reranking.train import RelevanceTriple, RerankerTrainingError, fine_tune_cross_encoder
from reranking.train import SchemaValidationError as TrainSchemaValidationError
from retrieval.schema import CandidateSource, FusedCandidate

logging.disable(logging.CRITICAL)


def _candidate(candidate_id: str, text: str, fused_score: float = 0.5) -> FusedCandidate:
    return FusedCandidate(
        candidate_id=candidate_id,
        source=CandidateSource.VECTOR,
        vector_score=fused_score,
        graph_score=None,
        fused_score=fused_score,
        text=text,
    )


# --------------------------------------------------------------------------
# schema.py
# --------------------------------------------------------------------------


class TestRankedCandidate:
    def test_valid_construction(self) -> None:
        original = _candidate("c1", "Casual leave is 12 days per year.")
        ranked = RankedCandidate(
            candidate_id="c1", text=original.text, rerank_score=0.87,
            original_candidate=original, rank=1,
        )
        assert ranked.rank == 1
        assert ranked.original_candidate is original

    def test_empty_candidate_id_raises(self) -> None:
        original = _candidate("c1", "text")
        with pytest.raises(SchemaValidationError):
            RankedCandidate(candidate_id="", text="text", rerank_score=0.5, original_candidate=original, rank=1)

    def test_empty_text_raises(self) -> None:
        original = _candidate("c1", "text")
        with pytest.raises(SchemaValidationError):
            RankedCandidate(candidate_id="c1", text="", rerank_score=0.5, original_candidate=original, rank=1)

    def test_non_finite_score_raises(self) -> None:
        original = _candidate("c1", "text")
        with pytest.raises(SchemaValidationError):
            RankedCandidate(
                candidate_id="c1", text="text", rerank_score=float("nan"),
                original_candidate=original, rank=1,
            )

    def test_rank_below_one_raises(self) -> None:
        original = _candidate("c1", "text")
        with pytest.raises(SchemaValidationError):
            RankedCandidate(candidate_id="c1", text="text", rerank_score=0.5, original_candidate=original, rank=0)


# --------------------------------------------------------------------------
# model.py
# --------------------------------------------------------------------------


class TestCrossEncoderReranker:
    def test_loads_from_local_checkpoint(self, tiny_cross_encoder_dir: str) -> None:
        reranker = CrossEncoderReranker(tiny_cross_encoder_dir)
        assert reranker is not None

    def test_invalid_checkpoint_path_raises(self, tmp_path) -> None:
        with pytest.raises(CrossEncoderRerankerError):
            CrossEncoderReranker(str(tmp_path / "does_not_exist"))

    def test_score_pair_returns_finite_float(self, tiny_cross_encoder_dir: str) -> None:
        reranker = CrossEncoderReranker(tiny_cross_encoder_dir)
        score = reranker.score_pair("what is the leave policy", "casual leave is 12 days per year")
        assert isinstance(score, float)
        assert score == score  # not NaN

    def test_score_pair_empty_query_raises(self, tiny_cross_encoder_dir: str) -> None:
        reranker = CrossEncoderReranker(tiny_cross_encoder_dir)
        with pytest.raises(ValueError):
            reranker.score_pair("   ", "some text")

    def test_score_pair_empty_candidate_raises(self, tiny_cross_encoder_dir: str) -> None:
        reranker = CrossEncoderReranker(tiny_cross_encoder_dir)
        with pytest.raises(ValueError):
            reranker.score_pair("some query", "   ")

    def test_rerank_returns_ranked_candidates(self, tiny_cross_encoder_dir: str) -> None:
        reranker = CrossEncoderReranker(tiny_cross_encoder_dir)
        candidates = [
            _candidate("c1", "Casual leave is 12 days per year."),
            _candidate("c2", "The office coffee machine is broken."),
            _candidate("c3", "VPN access requires manager approval."),
        ]
        ranked = reranker.rerank("what is the leave policy", candidates, top_n=2)
        assert len(ranked) == 2
        assert [r.rank for r in ranked] == [1, 2]
        assert all(isinstance(r, RankedCandidate) for r in ranked)

    def test_rerank_sorted_descending_by_score(self, tiny_cross_encoder_dir: str) -> None:
        reranker = CrossEncoderReranker(tiny_cross_encoder_dir)
        candidates = [_candidate(f"c{i}", f"text number {i}") for i in range(5)]
        ranked = reranker.rerank("query text", candidates, top_n=5)
        scores = [r.rerank_score for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_preserves_original_candidate_provenance(self, tiny_cross_encoder_dir: str) -> None:
        reranker = CrossEncoderReranker(tiny_cross_encoder_dir)
        original = _candidate("c1", "Casual leave is 12 days per year.", fused_score=0.77)
        ranked = reranker.rerank("leave policy", [original], top_n=1)
        assert ranked[0].original_candidate.fused_score == pytest.approx(0.77)

    def test_rerank_empty_candidates_returns_empty_list(self, tiny_cross_encoder_dir: str) -> None:
        reranker = CrossEncoderReranker(tiny_cross_encoder_dir)
        assert reranker.rerank("query", [], top_n=5) == []

    def test_rerank_top_n_caps_results(self, tiny_cross_encoder_dir: str) -> None:
        reranker = CrossEncoderReranker(tiny_cross_encoder_dir)
        candidates = [_candidate(f"c{i}", f"text {i}") for i in range(10)]
        ranked = reranker.rerank("query", candidates, top_n=3)
        assert len(ranked) == 3

    def test_rerank_empty_query_raises(self, tiny_cross_encoder_dir: str) -> None:
        reranker = CrossEncoderReranker(tiny_cross_encoder_dir)
        with pytest.raises(ValueError):
            reranker.rerank("   ", [_candidate("c1", "text")], top_n=1)

    def test_rerank_non_positive_top_n_raises(self, tiny_cross_encoder_dir: str) -> None:
        reranker = CrossEncoderReranker(tiny_cross_encoder_dir)
        with pytest.raises(ValueError):
            reranker.rerank("query", [_candidate("c1", "text")], top_n=0)

    def test_module_level_score_pair(self, tiny_cross_encoder_dir: str) -> None:
        reranker = CrossEncoderReranker(tiny_cross_encoder_dir)
        score = score_pair(reranker, "query", "candidate text")
        assert isinstance(score, float)

    def test_module_level_rerank(self, tiny_cross_encoder_dir: str) -> None:
        reranker = CrossEncoderReranker(tiny_cross_encoder_dir)
        candidates = [_candidate("c1", "text one"), _candidate("c2", "text two")]
        ranked = rerank(reranker, "query", candidates, top_n=1)
        assert len(ranked) == 1


# --------------------------------------------------------------------------
# train.py
# --------------------------------------------------------------------------


class TestRelevanceTriple:
    def test_valid_construction(self) -> None:
        triple = RelevanceTriple("query", "relevant passage", "irrelevant passage")
        assert triple.query == "query"

    @pytest.mark.parametrize("field_index", [0, 1, 2])
    def test_empty_field_raises(self, field_index: int) -> None:
        args = ["query", "relevant", "irrelevant"]
        args[field_index] = "   "
        with pytest.raises(TrainSchemaValidationError):
            RelevanceTriple(*args)


_TRAINING_TRIPLES = [
    RelevanceTriple(
        "what is the leave policy",
        "casual leave is 12 days per year",
        "the coffee machine is broken",
    ),
    RelevanceTriple(
        "how do I get vpn access",
        "vpn access requires manager approval",
        "finance department budget report",
    ),
]


class TestFineTuneCrossEncoder:
    def test_trains_and_returns_usable_reranker(self, tiny_cross_encoder_dir: str, tmp_path) -> None:
        output_dir = str(tmp_path / "trained_ce")
        reranker = fine_tune_cross_encoder(
            _TRAINING_TRIPLES,
            base_model=tiny_cross_encoder_dir,
            output_dir=output_dir,
            num_epochs=1,
            batch_size=2,
        )
        assert isinstance(reranker, CrossEncoderReranker)
        score = reranker.score_pair("test query", "test passage")
        assert isinstance(score, float)

    def test_saves_checkpoint_to_output_dir(self, tiny_cross_encoder_dir: str, tmp_path) -> None:
        output_dir = tmp_path / "trained_ce_2"
        fine_tune_cross_encoder(
            _TRAINING_TRIPLES,
            base_model=tiny_cross_encoder_dir,
            output_dir=str(output_dir),
            num_epochs=1,
            batch_size=2,
        )
        assert (output_dir / "config.json").exists()

    def test_empty_triples_raises(self, tiny_cross_encoder_dir: str, tmp_path) -> None:
        with pytest.raises(RerankerTrainingError):
            fine_tune_cross_encoder([], base_model=tiny_cross_encoder_dir, output_dir=str(tmp_path / "out"))

    def test_invalid_num_epochs_raises(self, tiny_cross_encoder_dir: str, tmp_path) -> None:
        with pytest.raises(ValueError):
            fine_tune_cross_encoder(
                _TRAINING_TRIPLES,
                base_model=tiny_cross_encoder_dir,
                output_dir=str(tmp_path / "out"),
                num_epochs=0,
            )

    def test_invalid_batch_size_raises(self, tiny_cross_encoder_dir: str, tmp_path) -> None:
        with pytest.raises(ValueError):
            fine_tune_cross_encoder(
                _TRAINING_TRIPLES,
                base_model=tiny_cross_encoder_dir,
                output_dir=str(tmp_path / "out"),
                batch_size=0,
            )

    def test_invalid_base_model_raises(self, tmp_path) -> None:
        with pytest.raises(RerankerTrainingError):
            fine_tune_cross_encoder(
                _TRAINING_TRIPLES,
                base_model=str(tmp_path / "does_not_exist"),
                output_dir=str(tmp_path / "out"),
            )
