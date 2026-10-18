"""Camera and proprioception feedback for tool conversations."""

from __future__ import annotations

OMITTED = "[earlier frame omitted]"
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
GUIDANCE = "Current views: front camera -- the arm is in frame, so read the gripper's position relative to the object from it -- and wrist camera."


def keep_recent_images(messages: list[dict], keep: int) -> int:
    """Retain only the ``keep`` most recent images. Returns how many were dropped.

    Every round re-encodes the whole conversation, so images cost quadratically
    in the number of rounds. Two arrive per round once live observation is on.
    """
    if keep < 1:
        raise ValueError("keep must be >= 1")
    positions = [
        (i, j)
        for i, message in enumerate(messages)
        if isinstance(message.get("content"), list)
        for j, block in enumerate(message["content"])
        if isinstance(block, dict) and block.get("type") == "image"
    ]
    stale = positions[: max(len(positions) - keep, 0)]
    for i, j in stale:
        messages[i]["content"][j] = {"type": "text", "text": OMITTED}
    return len(stale)


def robot_state_text(eef, gripper_qpos, support_z: float, steps_left: int) -> str:
    """The proprioception pi0.5 receives as ``observation.state``.

    The v30 loop sent none of it. The executor has had all of these numbers all
    along; they simply never reached the conversation.
    """
    x, y, z = (float(v) for v in eef[:3])
    gap = abs(float(gripper_qpos[0]) - float(gripper_qpos[1]))
    return f"Gripper at x={x:+.3f} y={y:+.3f} z={z:.3f} m, {(z - support_z) * 100:.1f} cm above the support plane. Finger gap {gap:.3f} m (0.080 = open, 0.000 = closed). {int(steps_left)} simulator steps remain."


def make_appender(image_of, keep: int):
    """Build the episode-local observation appender.

    ``image_of(obs, camera)`` renders one camera from an observation; it is
    passed in rather than imported so that this module stays free of the
    simulator and can be tested without it.
    """

    def append_live_observation(
        messages: list[dict], executor, result_text: str
    ) -> None:
        state = robot_state_text(
            executor.obs["robot0_eef_pos"],
            executor.obs["robot0_gripper_qpos"],
            executor.support_z,
            executor.env.budget - executor.env.steps,
        )
        messages.append(
            {
                "role": "tool",
                "content": [
                    *(
                        {"type": "image", "image": image_of(executor.obs, camera)}
                        for camera in CAMERAS
                    ),
                    {"type": "text", "text": f"{result_text}\n\n{GUIDANCE}\n{state}"},
                ],
            }
        )
        keep_recent_images(messages, keep)

    return append_live_observation
