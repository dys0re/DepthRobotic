from copy import deepcopy
import pytest
from depthrobotic.config import RunConfig
from depthrobotic import results, validate


def row(task=0):
    return dict(
        suite="libero_spatial",
        task_id=task,
        episode=0,
        instruction="put the bowl on the plate",
        success=False,
        sim_steps=280,
        step_budget=280,
        error="StepBudgetExceeded: exhausted",
    )


def test_writer_rejects_duplicate_or_unplanned_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(results, "artifact_manifest", lambda config: {})
    config = RunConfig(
        "writer", cases="libero_spatial:0,libero_spatial:1", out_dir=str(tmp_path)
    )
    writer = results.ResultWriter(config)
    writer.append(row())
    with pytest.raises(validate.ValidationError):
        writer.append(row())
    with pytest.raises(validate.ValidationError):
        writer.append(row(2))
    payload = results.load(writer.path)
    assert payload["episodes"] == 1 and payload["complete"] is False
    assert not writer.path.with_suffix(".json.tmp").exists()
    writer.append(row(1))
    assert results.load(writer.path)["complete"] is True
    with pytest.raises(FileExistsError):
        results.ResultWriter(config)


def test_swapped_suite_counts_are_rejected():
    config = RunConfig("counts", cases="libero_spatial:0,libero_object:0")
    data = results.build_payload(
        [row(), dict(row(), suite="libero_object", success=True, error=None)],
        config,
        provenance={},
    )
    broken = deepcopy(data)
    broken["by_suite"]["libero_spatial"]["successes"] = 1
    broken["by_suite"]["libero_object"]["successes"] = 0
    with pytest.raises(validate.ValidationError):
        validate.check_payload(broken)


def test_infrastructure_failure_does_not_enter_denominator():
    with pytest.raises(validate.ValidationError):
        validate.check_episode(
            dict(row(), sim_steps=12, error="NameError: missing controller helper")
        )


def test_empty_complete_population_is_rejected():
    config = RunConfig("empty", cases="libero_spatial:0")
    payload = results.build_payload([], config, provenance={})
    assert payload["complete"] is False
    with pytest.raises(validate.ValidationError):
        validate.require_reportable(payload)
