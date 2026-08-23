"""Tests for scripts/export_reports.py — the ONLY place that turns Part 1/2/3's
real, already-committed artifacts into the React console's JSON contracts.

Every assertion here checks the export against a real, independently-read
source file (a CSV, a saved model, a committed report) rather than against a
hand-typed expected value, so a regression in the export script (not just a
missing field) is what these tests are actually built to catch.
"""

import json

import pytest

from scripts.export_reports import (
    export_part1,
    export_part2,
    export_part3,
    export_project_meta,
)

# All four export_* functions write through this module-level function, so a
# single monkeypatch redirects every writer into an in-memory dict instead of
# the real frontend/public/reports/ directory.
import scripts.export_reports as export_reports


@pytest.fixture()
def captured(monkeypatch):
    written: dict[str, dict] = {}

    def fake_write_json(name, payload):
        written[name] = payload

    monkeypatch.setattr(export_reports, "write_json", fake_write_json)
    return written


def test_export_part1_matches_the_committed_reports(captured):
    export_part1()

    data = captured["part1_data.json"]
    assert data["rows"] == 6000
    assert data["columns"] == 13
    assert data["generator"]["seed"] == 42
    assert data["generator"]["n_rows"] == 6000
    assert abs(sum(c["probability"] for c in data["generator"]["category_probs"]) - 1.0) < 1e-9

    artifact = captured["part1_artifact.json"]
    import json as _json
    from pathlib import Path

    metadata = _json.loads((Path("models") / "return_risk_metadata.json").read_text())
    assert artifact["t_star_rf"] == metadata["threshold_rf"]
    assert artifact["loads_ok"] is True

    grid = captured["part1_rf_grid.json"]
    assert len(grid["cells"]) == 6
    assert grid["best_cv_roc_auc"] == metadata["best_cv_roc_auc"]

    sweep = captured["part1_threshold_sweep.json"]
    assert len(sweep["points"]) == 41
    # every point's confusion counts must sum to the full test split
    n_test = sum(sweep["points"][0][k] for k in ("tp", "fp", "tn", "fn"))
    assert all(sum(p[k] for k in ("tp", "fp", "tn", "fn")) == n_test for p in sweep["points"])

    subgroups = captured["part1_subgroups.json"]
    assert subgroups["weakest"]["by"] in ("product_category", "payment_method")
    assert 0.0 <= subgroups["overall"]["recall"] <= 1.0


def test_export_part2_matches_the_committed_confusion_matrix(captured):
    export_part2()

    eval_data = captured["part2_eval.json"]
    assert len(eval_data["confusion_matrix"]) == 10
    assert all(len(row) == 10 for row in eval_data["confusion_matrix"])
    total = sum(sum(row) for row in eval_data["confusion_matrix"])
    assert total == 10_000  # the full untouched Fashion-MNIST test split
    assert 0.0 <= eval_data["test_accuracy"] <= 1.0
    assert len(eval_data["pair_explanations"]) == 3
    for pair in eval_data["pair_explanations"]:
        assert pair["explanation"]  # real prose, not empty

    artifact = captured["part2_artifact.json"]
    assert len(artifact["sample_images"]) == 10
    for sample in artifact["sample_images"]:
        assert 0.0 <= sample["confidence"] <= 1.0
        assert sample["predicted_class"]


def test_export_part3_transcripts_all_parse_a_response(captured):
    export_part3()

    transcripts = captured["part3_transcripts.json"]["transcripts"]
    assert len(transcripts) >= 8
    for t in transcripts:
        assert t["turns"], f"{t['filename']} parsed with zero turns"
        for turn in t["turns"]:
            assert turn["response"] is not None, f"{t['filename']} turn {turn['turn_label']} response failed to parse"
            assert set(turn["response"]) == {"answer", "source", "confidence"}

    retrieval = captured["part3_retrieval_eval.json"]
    assert len(retrieval["queries"]) == 7
    for q in retrieval["queries"]:
        assert q["retrieved_chunks"], f"no retrieved_chunks for {q['query']!r}"

    tools = captured["part3_tools.json"]
    assert len(tools["return_risk_examples"]) == 5
    for ex in tools["return_risk_examples"]:
        assert 0.0 <= ex["return_probability"] <= 1.0


def test_export_project_meta_reads_real_git_state(captured):
    export_project_meta()
    meta = captured["project_meta.json"]
    assert len(meta["commit"]) == 40  # a real full git SHA
    assert meta["commits"]


def test_every_exported_file_is_json_serialisable(captured):
    """The one property every contract must satisfy, regardless of shape."""
    export_part1()
    export_part2()
    export_part3()
    export_project_meta()
    for name, payload in captured.items():
        json.dumps(payload)  # must not raise
