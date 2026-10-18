"""Pick/place controller assembled from responsibility-specific components."""

from depthrobotic.perception.images import observation_image
from depthrobotic.perception.parsing import normalize_point, safe_ask_structured
from depthrobotic.skills.pick_place.locate import (
    DEST_RELATION_DIRS,
    parse_dest_relation,
)
from depthrobotic.skills.pick_place.state import (
    ObjectState,
    TaskState,
    region_descriptor,
)
import numpy as np
from .approach import SourceApproach
from .grasp import Grasping
from .placement import Placement
from .lifecycle import ExecutorLifecycle


class PickPlaceExecutor(ExecutorLifecycle, SourceApproach, Grasping, Placement):

    def __init__(
        self,
        env,
        obs,
        grounder,
        instruction,
        source,
        destination,
        size,
        support_z,
        events,
        perception_log,
        *,
        context,
    ):
        self.context = context
        self.env = env
        self.obs = obs
        self.grounder = grounder
        self.instruction = instruction
        self.source = source
        self.destination = destination
        self.size = size
        self.support_z = support_z
        self.events = events
        self.perception_log = perception_log
        self.travel_z = max(
            float(np.asarray(obs["robot0_eef_pos"])[2] - 0.04), support_z + 0.2
        )
        self.source_world = None
        self.source_surface_z = None
        self.wrist_center = None
        self.grasp_world = None
        self.grasp_eef_z = None
        self.held = False
        self.grasp_attempt = 0
        self.current_strategy = "center"
        self.held_offset = np.zeros(2)
        self.record = {
            "action": "pick_place_react_v1",
            "source": source,
            "destination": destination,
            "tool_calls": [],
        }
        self.perception_log.append(self.record)
        self.dest_pre_point = None
        try:
            img = observation_image(obs, "agentview_image", context=self.context)
            prompt = f'Robot instruction: {instruction}\nLocate the visual center of the usable receiving region of: {destination}. For a container choose the interior. Return JSON: {{{{"point":[x,y]}}}}.'
            _, v = safe_ask_structured(grounder, img, prompt, 96)
            self.dest_pre_point = normalize_point(v.get("point"))
        except Exception:
            self.dest_pre_point = None
        self.placed_once = False
        self._done_refusals = 0
        self.done = False
        self.last_dest_desc = destination
        self.last_is_container = False
        self.last_place_travel_z = self.travel_z
        self.place_retries = 0
        self.top_z = None
        self.rim_z = None
        self.box = None
        self.width_finger = None
        self.held_offset3 = None
        self.infeasible_count = 0
        self.servo_plane_done = None
        self.probe_done = False
        self.fingers_closed = False
        self.failure_history = []
        self.ocr_candidate_px = None
        self.dest_anchor = None
        self.release_attached_count = 0
        self.place_blocked = None
        self.record["v16"] = {
            "proposer": "GroundingDINO",
            "sam3": False,
            "selector": "wrist identity check (unchanged v12)",
            "last_resort_only_on_final_step": True,
        }
        self._prior = None
        self._pool = None
        self._used: list[np.ndarray] = []
        self.record["v17"] = {
            "proposer": "GroundingDINO",
            "relation_aware": True,
            "detector_only_after_v12_rejection": True,
            "geometry_source": "v12_camera_projection",
        }
        self._relation_pool = None
        self.record["v23"] = {
            "alternate_ranking": "DepthVLM numbered-mark multiple choice over the v17 pool",
            "trigger": "only after v12 identity rejection (v17 _next_source_candidate)",
            "fallback_on_abstain": "v17 rank-1",
            "measurement_height_grasp_place": "v12 unchanged",
        }
        self._candidate_selections = 0
        self.last_failure = None
        self.record["v28_revision"] = "rebuild-r1"
        self.record["execution_states"] = []
        self._capture_receiving_reference()
        self.record["action"] = "pick_place_v30_object_state"
        self.state = TaskState(
            ObjectState("source", self.source), ObjectState("target", self.destination)
        )
        self.record["v30_state"] = self.state.snapshot()
        self.source_reference = None
        self.source_candidate_used = False
        self.recovery_observations = 0
        self.held_descriptor = None
        self.source_descriptor = None
        self.release_descriptor = None
        self.release_association = "UNKNOWN"
        self.last_identity = None
        self.target_relation = parse_dest_relation(self.destination)
        self.target_label = (
            self.target_relation["anchor"]
            if self.target_relation
            and self.target_relation["relation"] in DEST_RELATION_DIRS
            else self.destination
        )
        if self.memory and self.target_label == self.destination:
            self._adopt_target(self.memory)
        else:
            ref = self._reference(self.target_label, "receiving region")
            if ref:
                self.memory = ref
                self._adopt_target(ref)
        self.source_reference = self._reference(
            self.source, "source object specified by the instruction"
        )
        if self.source_reference:
            ref = self.source_reference
            self.source_descriptor = region_descriptor(ref["features"], ref["box"])
            self.state.source.observe(
                self.env.steps,
                point=ref["point"],
                box=ref["box"],
                surface_z=ref["z"],
                identity="GROUNDED",
                association="REFERENCE",
            )
