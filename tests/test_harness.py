"""Protocol, accounting, and comparison tests."""

from __future__ import annotations
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from depthrobotic import compare, protocol, validate
from depthrobotic import config as config_module

REFERENCE = json.loads(
    (REPO_ROOT / "tests" / "reference" / "v30_libero40.json").read_text()
)


def episode(**overrides) -> dict:
    row = {
        "suite": "libero_spatial",
        "task_id": 0,
        "episode": 0,
        "instruction": "pick up the black bowl",
        "success": True,
        "error": None,
        "sim_steps": 145,
        "step_budget": 280,
    }
    row.update(overrides)
    return row


def payload(rows: list[dict], *, planned: int | None = None) -> dict:
    by_suite: dict[str, dict] = {}
    for row in rows:
        entry = by_suite.setdefault(row["suite"], {"successes": 0, "episodes": 0})
        entry["successes"] += bool(row["success"])
        entry["episodes"] += 1
    for entry in by_suite.values():
        entry["success_rate"] = entry["successes"] / entry["episodes"]
    successes = sum((1 for row in rows if row["success"]))
    planned = len(rows) if planned is None else planned
    return {
        "successes": successes,
        "episodes": len(rows),
        "planned_episodes": planned,
        "complete": len(rows) == planned,
        "by_suite": by_suite,
        "results": rows,
        "provenance": {"git": {"commit": "0" * 40, "dirty": False}},
    }


class ZeroStepEpisodes(unittest.TestCase):
    """Reject episodes that never advanced simulation."""

    def test_zero_step_episode_is_rejected(self):
        with self.assertRaises(validate.ValidationError) as caught:
            validate.check_episode(episode(success=False, sim_steps=0, error=None))
        self.assertIn("never ran", str(caught.exception))

    def test_a_real_failure_is_accepted(self):
        validate.check_episode(
            episode(
                success=False,
                sim_steps=280,
                error="StepBudgetExceeded: horizon reached",
            )
        )

    def test_success_with_an_error_is_contradictory(self):
        with self.assertRaises(validate.ValidationError):
            validate.check_episode(episode(success=True, error="RuntimeError: boom"))

    def test_steps_beyond_the_official_horizon_are_rejected(self):
        with self.assertRaises(validate.ValidationError):
            validate.check_episode(episode(sim_steps=281, step_budget=280))

    def test_missing_fields_are_rejected(self):
        broken = episode()
        del broken["sim_steps"]
        with self.assertRaises(validate.ValidationError):
            validate.check_episode(broken)


class SelfContradictoryPayloads(unittest.TestCase):
    """Reject inconsistent result totals."""

    def test_hand_edited_total_is_caught(self):
        rows = [episode(task_id=i, success=i < 6) for i in range(10)]
        data = payload(rows)
        data["successes"] = 8
        with self.assertRaises(validate.ValidationError) as caught:
            validate.check_payload(data)
        self.assertIn("successes=8", str(caught.exception))

    def test_stale_by_suite_is_caught(self):
        rows = [episode(task_id=i, success=i < 6) for i in range(10)]
        data = payload(rows)
        data["by_suite"]["libero_spatial"]["successes"] = 8
        with self.assertRaises(validate.ValidationError) as caught:
            validate.check_payload(data)
        self.assertIn("by_suite", str(caught.exception))

    def test_duplicate_rows_are_caught(self):
        rows = [episode(task_id=0), episode(task_id=0)]
        with self.assertRaises(validate.ValidationError):
            validate.check_payload(payload(rows))

    def test_consistent_payload_passes(self):
        validate.check_payload(
            payload([episode(task_id=i, success=i < 6) for i in range(10)])
        )


class TruncatedRuns(unittest.TestCase):
    """Require complete populations before reporting scores."""

    def test_incomplete_run_is_not_reportable(self):
        data = payload([episode(task_id=i) for i in range(27)], planned=90)
        with self.assertRaises(validate.ValidationError) as caught:
            validate.require_reportable(data)
        self.assertIn("27/90", str(caught.exception))

    def test_dirty_tree_is_not_reportable(self):
        data = payload([episode()])
        data["provenance"]["git"] = {
            "commit": "abc",
            "dirty": True,
            "dirty_paths": ["x.py"],
        }
        with self.assertRaises(validate.ValidationError):
            validate.require_reportable(data)

    def test_complete_clean_run_is_reportable(self):
        validate.require_reportable(payload([episode(task_id=i) for i in range(4)]))


class PairedComparison(unittest.TestCase):
    """16/40 vs 17/40 is p=1.00 on marginal rates; pairing is what can see a change."""

    def test_six_clean_wins_reach_significance(self):
        self.assertLess(compare.mcnemar_exact(6, 0), 0.05)

    def test_five_clean_wins_do_not(self):
        self.assertGreater(compare.mcnemar_exact(5, 0), 0.05)

    def test_small_unpaired_rate_difference_is_not_significant(self):
        self.assertGreater(compare.mcnemar_exact(2, 0), 0.05)

    def test_no_discordant_pairs_is_p_one(self):
        self.assertEqual(compare.mcnemar_exact(0, 0), 1.0)

    def test_gains_and_losses_are_reported_separately(self):
        base = payload([episode(task_id=i, success=i < 3) for i in range(6)])
        cand = payload([episode(task_id=i, success=i >= 3) for i in range(6)])
        result = compare.compare(base, cand)
        self.assertEqual(len(result.gains), 3)
        self.assertEqual(len(result.losses), 3)
        self.assertFalse(result.promotable)

    def test_a_win_with_a_regression_is_not_promotable(self):
        base = payload([episode(task_id=i, success=i < 6) for i in range(10)])
        rows = [episode(task_id=i, success=i < 7) for i in range(10)]
        rows[0]["success"] = False
        result = compare.compare(base, payload(rows))
        self.assertEqual(len(result.losses), 1)
        self.assertFalse(result.promotable)

    def test_unpaired_episodes_are_excluded_and_flagged(self):
        base = payload([episode(task_id=i) for i in range(6)])
        cand = payload([episode(task_id=i) for i in range(4)])
        result = compare.compare(base, cand)
        self.assertEqual(result.shared, 4)
        self.assertEqual(len(result.baseline_only), 2)
        self.assertIn("NOT PAIRED", result.report())


class ProtocolAndConfig(unittest.TestCase):

    def test_libero_40_is_forty_cases(self):
        self.assertEqual(len(protocol.LIBERO_40), 40)
        self.assertEqual(len(protocol.parse_cases(",".join(protocol.LIBERO_40))), 40)

    def test_official_horizons(self):
        self.assertEqual(protocol.budget_for("libero_spatial"), 280)
        self.assertEqual(protocol.budget_for("libero_goal"), 300)
        self.assertEqual(protocol.budget_for("libero_10"), 520)
        self.assertEqual(protocol.budget_for("libero_90"), 400)

    def test_unknown_suite_has_no_horizon(self):
        with self.assertRaises(ValueError):
            protocol.budget_for("libero_imaginary")

    def test_duplicate_cases_are_rejected(self):
        with self.assertRaises(ValueError):
            protocol.parse_cases("libero_spatial:0,libero_spatial:0")

    def test_baseline_pins_the_historical_image_convention(self):
        baseline = config_module.load(
            REPO_ROOT / "configs" / "baseline_v30.yaml", cases="libero_spatial:0"
        )
        self.assertEqual(
            baseline.orient_fix, 0, "every reported v30 number used ORIENT_FIX=0"
        )
        self.assertEqual(baseline.seed, 0)
        self.assertEqual(baseline.image_size, 832)
        self.assertEqual(baseline.support_z_offset, 0.24)

    def test_undeclared_drift_is_refused(self):
        baseline = config_module.load(
            REPO_ROOT / "configs" / "baseline_v30.yaml", cases="libero_spatial:0"
        )
        drifted = replace(baseline, orient_fix=1, workers=3)
        with self.assertRaises(ValueError) as caught:
            config_module.require_declared_diff(drifted, baseline, declared={"workers"})
        message = str(caught.exception)
        self.assertIn("orient_fix", message)
        self.assertNotIn("workers: baseline", message)

    def test_declared_change_is_allowed(self):
        baseline = config_module.load(
            REPO_ROOT / "configs" / "baseline_v30.yaml", cases="libero_spatial:0"
        )
        changed = replace(baseline, orient_fix=1)
        self.assertEqual(
            config_module.require_declared_diff(
                changed, baseline, declared={"orient_fix"}
            ),
            {"orient_fix": (0, 1)},
        )

    def test_run_name_is_not_a_behavioural_difference(self):
        baseline = config_module.load(
            REPO_ROOT / "configs" / "baseline_v30.yaml", cases="libero_spatial:0"
        )
        self.assertEqual(
            config_module.diff(replace(baseline, name="other"), baseline), {}
        )


class LiveObservation(unittest.TestCase):
    """The model was asked to correct spatial errors it had never been shown.

    Most tools returned text alone; the only routine image was the wrist view,
    taken from the arm, so the arm is not in it. The one agentview frame in the
    conversation predated the arm entering the scene. Three runs of the same
    configuration chose two different retry adjustments at the same decision
    point in ``libero_goal:2``, which is what an undetermined context looks like.
    """

    def _messages(self, count):
        return [
            {
                "role": "tool",
                "content": [
                    {"type": "image", "image": f"img{i}"},
                    {"type": "text", "text": f"t{i}"},
                ],
            }
            for i in range(count)
        ]

    def _images(self, messages):
        return [
            block["image"]
            for message in messages
            if isinstance(message.get("content"), list)
            for block in message["content"]
            if isinstance(block, dict) and block.get("type") == "image"
        ]

    def test_only_the_most_recent_images_are_kept(self):
        from depthrobotic import observation

        messages = self._messages(5)
        self.assertEqual(observation.keep_recent_images(messages, 2), 3)
        self.assertEqual(self._images(messages), ["img3", "img4"])

    def test_dropped_images_become_text_not_holes(self):
        """A removed block would leave a dangling placeholder in the rendered text."""
        from depthrobotic import observation

        messages = self._messages(3)
        blocks_before = [len(m["content"]) for m in messages]
        observation.keep_recent_images(messages, 1)
        self.assertEqual([len(m["content"]) for m in messages], blocks_before)
        self.assertEqual(
            messages[0]["content"][0], {"type": "text", "text": observation.OMITTED}
        )

    def test_nothing_is_dropped_below_the_limit(self):
        from depthrobotic import observation

        messages = self._messages(2)
        self.assertEqual(observation.keep_recent_images(messages, 4), 0)
        self.assertEqual(self._images(messages), ["img0", "img1"])

    def test_robot_state_reports_height_above_the_support_plane(self):
        from depthrobotic import observation

        text = observation.robot_state_text(
            [0.1, -0.2, 1.05], [0.039, -0.039], 0.9, 137
        )
        self.assertIn("z=1.050", text)
        self.assertIn("15.0 cm above the support plane", text)
        self.assertIn("0.078", text)
        self.assertIn("137 simulator steps remain", text)

    def test_appender_sends_both_real_cameras(self):
        """pi0.5-LIBERO reads exactly these two; its third slot is zero-filled."""
        from types import SimpleNamespace
        from depthrobotic import observation

        executor = SimpleNamespace(
            obs={
                "robot0_eef_pos": [0.0, 0.0, 1.0],
                "robot0_gripper_qpos": [0.04, -0.04],
            },
            support_z=0.9,
            env=SimpleNamespace(budget=280, steps=140),
        )
        messages = []
        observation.make_appender(lambda obs, camera: camera, keep=4)(
            messages, executor, "Grasp FAILED with strategy 'rim'."
        )
        self.assertEqual(self._images(messages), list(observation.CAMERAS))
        self.assertIn("Grasp FAILED", messages[0]["content"][-1]["text"])
        self.assertIn("140 simulator steps remain", messages[0]["content"][-1]["text"])

    def test_baseline_config_does_not_enable_it(self):
        baseline = config_module.load(
            REPO_ROOT / "configs" / "baseline_v30.yaml", cases="libero_spatial:0"
        )
        self.assertFalse(baseline.live_observation)

    def test_enabling_it_is_a_declared_behavioural_change(self):
        baseline = config_module.load(
            REPO_ROOT / "configs" / "baseline_v30.yaml", cases="libero_spatial:0"
        )
        live = replace(baseline, live_observation=True)
        self.assertEqual(
            config_module.diff(live, baseline), {"live_observation": (False, True)}
        )
        with self.assertRaises(ValueError):
            config_module.require_declared_diff(live, baseline, declared=set())


class RelaxedBudget(unittest.TestCase):
    """A relaxed horizon is a separate axis, not a better number.

    Measured on the baseline's six spatial successes, a first-attempt success
    costs 184 of its 280 steps and a second attempt about 140: inside the
    official horizon this agent gets one attempt. Testing whether that is the
    binding constraint means leaving the protocol, so the protocol number has to
    survive alongside the relaxed one and the two must not merge.
    """

    def _config(self, **overrides):
        base = config_module.load(
            REPO_ROOT / "configs" / "baseline_v30.yaml", cases="libero_spatial:0"
        )
        return replace(base, **overrides)

    def test_the_baseline_is_the_official_protocol(self):
        self.assertEqual(self._config().budget_scale, 1.0)
        self.assertEqual(self._config().budget_for("libero_spatial"), 280)
        self.assertEqual(self._config().grasp_attempts(), protocol.MAX_GRASP_ATTEMPTS)

    def test_scaling_moves_the_horizon_and_the_attempt_cap_together(self):
        """Raising the horizon alone only trades budget_refusal for attempt_limit."""
        doubled = self._config(budget_scale=2.0)
        self.assertEqual(doubled.budget_for("libero_spatial"), 560)
        self.assertEqual(doubled.budget_for("libero_10"), 1040)
        self.assertEqual(doubled.grasp_attempts(), 6)

    def test_protocol_horizons_are_never_mutated(self):
        self._config(budget_scale=3.0).budget_for("libero_spatial")
        self.assertEqual(protocol.budget_for("libero_spatial"), 280)

    def test_shortening_the_horizon_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            self._config(budget_scale=0.5).validate()
        self.assertIn("incomparable", str(caught.exception))

    def test_a_shortened_horizon_is_refused_at_the_episode_too(self):
        row = {
            "suite": "libero_spatial",
            "task_id": 0,
            "episode": 0,
            "instruction": "i",
            "success": False,
            "error": None,
            "sim_steps": 100,
            "step_budget": 200,
            "official_horizon": 280,
        }
        with self.assertRaises(validate.ValidationError) as caught:
            validate.check_episode(row)
        self.assertIn("below the official", str(caught.exception))

    def test_a_late_success_is_not_an_official_success(self):
        """The relaxed run must not be quotable as the protocol number."""
        row = {
            "suite": "libero_spatial",
            "task_id": 0,
            "episode": 0,
            "instruction": "i",
            "success": True,
            "error": None,
            "sim_steps": 340,
            "step_budget": 560,
            "official_horizon": 280,
            "within_official_horizon": False,
        }
        validate.check_episode(row)
        early = dict(row, sim_steps=145, within_official_horizon=True)
        validate.check_episode(early)

    def test_within_official_horizon_cannot_be_set_on_a_failure(self):
        row = {
            "suite": "libero_spatial",
            "task_id": 0,
            "episode": 0,
            "instruction": "i",
            "success": False,
            "error": None,
            "sim_steps": 145,
            "step_budget": 560,
            "official_horizon": 280,
            "within_official_horizon": True,
        }
        with self.assertRaises(validate.ValidationError):
            validate.check_episode(row)

    def test_relaxed_config_declares_exactly_one_change_from_its_control(self):
        control = config_module.load(
            REPO_ROOT / "configs" / "exp_live_observation.yaml",
            cases="libero_spatial:0",
        )
        relaxed = config_module.load(
            REPO_ROOT / "configs" / "exp_budget_relaxed.yaml", cases="libero_spatial:0"
        )
        self.assertEqual(
            config_module.diff(relaxed, control), {"budget_scale": (1.0, 2.0)}
        )


class FailureTaxonomy(unittest.TestCase):
    """One number per run could not steer nine versions of policy work.

    The taxonomy is ported from the recorded 130-episode audit so counts stay
    comparable with it, and it is deterministic string/structure matching on
    purpose: a model in the classifier would put nondeterminism back into the
    measuring instrument.
    """

    def _episode(
        self, *, calls, success=False, steps=200, budget=280, error=None, plan=None
    ):
        return {
            "suite": "libero_spatial",
            "task_id": 0,
            "episode": 0,
            "success": success,
            "error": error,
            "sim_steps": steps,
            "step_budget": budget,
            "plan": plan if plan is not None else [{"action": "pick_place"}],
            "perceptions": [{"tool_calls": calls}],
        }

    def test_stages_are_mutually_exclusive_and_ordered_by_progress(self):
        from depthrobotic import diagnose

        place = [{"tool": "place_object", "round": 3, "result": "ok"}]
        grasp = [{"tool": "grasp", "round": 1, "result": "ok"}]
        approach = [{"tool": "approach_source", "round": 0, "result": "ok"}]
        self.assertEqual(
            diagnose.stage_of(self._episode(calls=grasp + place)), "placement_attempted"
        )
        self.assertEqual(
            diagnose.stage_of(self._episode(calls=approach + grasp)),
            "grasp_attempted_no_placement",
        )
        self.assertEqual(
            diagnose.stage_of(self._episode(calls=approach)), "approach_only"
        )
        row = self._episode(calls=[])
        row["perceptions"] = [{"action": "articulate"}]
        self.assertEqual(diagnose.stage_of(row), "no_pick_place_executor")

    def test_symptoms_come_from_the_recorded_tool_text(self):
        from depthrobotic import diagnose

        row = self._episode(
            calls=[
                {
                    "tool": "grasp",
                    "round": 1,
                    "result": "Grasp FAILED with strategy 'rim'. Gripper gap: 0.0812 (empty grasp).",
                }
            ]
        )
        self.assertIn("empty_grasp", diagnose.symptoms_of(row))

    def test_budget_left_is_only_counted_against_failures(self):
        from depthrobotic import diagnose

        calls = [{"tool": "grasp", "round": 1, "result": "ok"}]
        failed = self._episode(calls=calls, success=False, steps=150, budget=280)
        won = self._episode(calls=calls, success=True, steps=150, budget=280)
        self.assertIn("failed_with_100_steps_left", diagnose.symptoms_of(failed))
        self.assertNotIn("failed_with_100_steps_left", diagnose.symptoms_of(won))

    def test_one_recovery_of_a_kind_is_not_a_repeat(self):
        from depthrobotic import diagnose

        once = self._episode(calls=[{"recovery": {"failure": "empty"}}])
        twice = self._episode(calls=[{"recovery": {"failure": "empty"}}] * 2)
        self.assertNotIn("repeated_failure_kind", diagnose.symptoms_of(once))
        self.assertIn("repeated_failure_kind", diagnose.symptoms_of(twice))

    def test_observe_then_done_needs_done_to_follow_the_fresh_view(self):
        from depthrobotic import diagnose

        offered = {
            "tool": "done",
            "round": 4,
            "result": diagnose.FRESH_VIEW_MARKER + " ...",
        }
        gave_up = self._episode(
            calls=[offered, {"tool": "done", "round": 5, "result": "finished"}]
        )
        recovered = self._episode(
            calls=[offered, {"tool": "retry_place", "round": 5, "result": "ok"}]
        )
        self.assertIn("observe_then_done", diagnose.symptoms_of(gave_up))
        self.assertIn("active_observation", diagnose.symptoms_of(recovered))
        self.assertNotIn("observe_then_done", diagnose.symptoms_of(recovered))

    def test_symptoms_are_observations_not_causes(self):
        """Successful trajectories carry them too; that is the point of the caveat."""
        from depthrobotic import diagnose

        rows = (
            json.loads(
                (
                    REPO_ROOT / "results" / "v30_libero40_gate_r1" / "results.json"
                ).read_text()
            )["results"]
            if (
                REPO_ROOT / "results" / "v30_libero40_gate_r1" / "results.json"
            ).exists()
            else None
        )
        if rows is None:
            self.skipTest("baseline run not present")
        wins = diagnose.tally(rows, failures_only=False)
        losses = diagnose.tally(rows)
        self.assertGreater(
            wins["symptoms"]["high_contact"], losses["symptoms"]["high_contact"]
        )

    def test_the_taxonomy_reproduces_the_recorded_baseline(self):
        """Locks the rules: editing one changes these counts and fails here."""
        from depthrobotic import diagnose

        path = REPO_ROOT / "results" / "v30_libero40_gate_r1" / "results.json"
        if not path.exists():
            self.skipTest("baseline run not present")
        tally = diagnose.tally(json.loads(path.read_text())["results"])
        self.assertEqual(tally["episodes"], 24)
        self.assertEqual(
            tally["stages"],
            {
                "grasp_attempted_no_placement": 13,
                "placement_attempted": 9,
                "approach_only": 1,
                "no_pick_place_executor": 1,
            },
        )
        for symptom, count in [
            ("high_contact", 12),
            ("repeated_failure_kind", 11),
            ("failed_with_100_steps_left", 10),
            ("empty_grasp", 6),
            ("observe_then_done", 5),
        ]:
            self.assertEqual(tally["symptoms"][symptom], count, symptom)

    def test_delta_reports_both_directions(self):
        from depthrobotic import diagnose

        worse = self._episode(
            calls=[{"tool": "grasp", "round": 1, "result": "empty grasp"}]
        )
        better = self._episode(
            calls=[{"tool": "grasp", "round": 1, "result": "fingers landed on top"}]
        )
        moved = diagnose.delta([worse], [better])
        self.assertEqual(moved["empty_grasp"], (1, 0))
        self.assertEqual(moved["high_contact"], (0, 1))


class Reference(unittest.TestCase):
    """The gate's reference data, independent of any previous script."""

    def test_reference_is_the_v30_baseline(self):
        self.assertEqual(REFERENCE["successes"], 16)
        self.assertEqual(REFERENCE["episodes"], 40)
        self.assertEqual(
            {k: v["successes"] for k, v in REFERENCE["by_suite"].items()},
            {"libero_spatial": 6, "libero_object": 5, "libero_goal": 5, "libero_10": 0},
        )

    def test_reference_is_internally_consistent(self):
        validate.check_payload(
            REFERENCE | {"provenance": {"git": {"commit": "x", "dirty": False}}}
        )

    def test_reference_covers_the_whole_libero_40_population(self):
        covered = {f"{r['suite']}:{r['task_id']}" for r in REFERENCE["results"]}
        self.assertEqual(covered, set(protocol.LIBERO_40))


if __name__ == "__main__":
    unittest.main(verbosity=2)
