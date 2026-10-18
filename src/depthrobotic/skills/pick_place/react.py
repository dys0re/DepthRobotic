"""skills / pick_place / react: v30 control behavior with explicit dependencies."""

from __future__ import annotations
from depthrobotic.perception.images import observation_image
from depthrobotic.skills.pick_place.executor import PickPlaceExecutor
import json
import re


def parse_tool_call(raw):
    """Extract the first <tool_call> from model output."""
    match = re.search("<tool_call>\\s*(\\{.*?\\})\\s*</tool_call>", raw, re.S)
    if not match:
        return None
    try:
        call = json.loads(match.group(1))
        return (call.get("name", ""), call.get("arguments", {}))
    except json.JSONDecodeError:
        return None


def strip_special_tail(raw):
    """Remove trailing <|im_end|> and similar."""
    return re.sub("<\\|[a-z_]+\\|>\\s*$", "", raw).strip()


PICK_PLACE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "approach_source",
            "description": "Locate the source object in the front camera image, move the robot arm above it, and refine position using the wrist camera. A close-up wrist camera image will be returned so you can observe the object shape.",
            "parameters": {
                "type": "object",
                "properties": {
                    "object_description": {
                        "type": "string",
                        "description": "The source object to find, including spatial qualifiers from the instruction",
                    }
                },
                "required": ["object_description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grasp",
            "description": "Attempt to grasp the object below the gripper. Choose a strategy based on the object's shape and properties that you observed in the wrist camera image. The robot first probes the object's height with closed fingers and measures its width; if the object is too wide for the gripper it tells you and you must choose 'rim'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": ["rim", "side", "top", "center"],
                        "description": "rim: grasp the edge/rim (bowls, cups, open containers). side: grasp the body from the side (bottles, tall objects). top: pinch from above (flat items: books, boxes, packaged goods). center: grasp the center (regular solid objects, mugs by body).",
                    }
                },
                "required": ["strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "retry_grasp",
            "description": "The previous grasp was empty. Adjust position or strategy and try again. Think about WHY it failed: wrong height? wrong lateral position? wrong strategy?",
            "parameters": {
                "type": "object",
                "properties": {
                    "adjustment": {
                        "type": "string",
                        "enum": [
                            "lower",
                            "higher",
                            "shift_left",
                            "shift_right",
                            "shift_forward",
                            "shift_back",
                            "change_strategy",
                        ],
                        "description": "How to adjust before retrying: lower/higher move the grasp height by 2 cm; shift_* move the grasp point 1.5 cm; change_strategy re-plans with the new strategy.",
                    },
                    "strategy": {
                        "type": "string",
                        "enum": ["rim", "side", "top", "center"],
                        "description": "Grasp strategy for this retry attempt",
                    },
                },
                "required": ["adjustment", "strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place_object",
            "description": "Transport the held object to the destination and place it. Set is_container=true if the destination has an interior to place INTO (basket, drawer, bowl, compartment, caddy, microwave). Set is_container=false for placing ON a surface (plate, stove, cabinet top, table).",
            "parameters": {
                "type": "object",
                "properties": {
                    "destination_description": {
                        "type": "string",
                        "description": "The destination to place the object at",
                    },
                    "is_container": {
                        "type": "boolean",
                        "description": "true if placing INSIDE a container, false if placing ON a surface",
                    },
                },
                "required": ["destination_description", "is_container"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "done",
            "description": "Signal that this pick-place step is complete.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def execute_pick_place(
    env,
    obs,
    grounder,
    instruction,
    step,
    size,
    support_z,
    events,
    perception_log,
    *,
    context,
):
    source, destination = (step["source"], step["destination"])
    executor = PickPlaceExecutor(
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
        context=context,
    )
    scene_image = observation_image(obs, "agentview_image", context=context)
    system_msg = {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "You are a robot manipulation agent controlling a Franka Panda arm with a parallel-jaw gripper. You see through a front camera and a downward-facing wrist camera. Complete the given pick-and-place task by calling tools step by step.\n\nWorkflow:\n1. Call approach_source to find and approach the source object\n2. Observe the wrist camera image to understand the object's shape\n3. Call grasp with an appropriate strategy based on the object shape\n4. If grasp fails, call retry_grasp with an adjustment\n5. Once grasped, call place_object with the destination info\n6. Call done to finish\n\nImportant rules:\n- If approach_source fails twice for the same object, call done (the target may be unreachable)\n- retry_grasp only works after a failed grasp, not after a failed approach\n- After calling done, the system moves to the next step in the plan\n\nChoose grasp strategy based on object shape:\n- Bowl, cup, or open container → rim\n- Bottle or tall upright object → side\n- Book, flat box, or packaged item → top\n- Regular solid object → center",
            }
        ],
    }
    user_msg = {
        "role": "user",
        "content": [
            {"type": "image", "image": scene_image},
            {
                "type": "text",
                "text": f"Task: {instruction}\nCurrent step: pick up '{source}' and place it at '{destination}'.\nStart by calling approach_source.",
            },
        ],
    }
    messages = [system_msg, user_msg]
    for round_idx in range(context.settings.max_tool_rounds):
        raw = grounder.generate_with_tools(
            messages, PICK_PLACE_TOOLS, max_new_tokens=300
        )
        raw_clean = strip_special_tail(raw)
        parsed = parse_tool_call(raw)
        if parsed is None:
            executor.record["tool_calls"].append(
                {"round": round_idx, "raw": raw_clean, "no_tool_call": True}
            )
            messages.append({"role": "assistant", "content": raw_clean})
            messages.append(
                {
                    "role": "user",
                    "content": "Your reply contained no tool call. Respond with exactly one <tool_call> now.",
                }
            )
            raw = grounder.generate_with_tools(
                messages, PICK_PLACE_TOOLS, max_new_tokens=200
            )
            raw_clean = strip_special_tail(raw)
            parsed = parse_tool_call(raw)
            if parsed is None:
                m = re.search("\\b(rim|side|top|center)\\b", raw_clean.lower())
                if m and (not executor.held) and (executor.wrist_center is not None):
                    parsed = ("grasp", {"strategy": m.group(1)})
                    executor.record["tool_calls"].append(
                        {"round": round_idx, "fallback_from_prose": True}
                    )
                else:
                    executor.record["tool_calls"].append(
                        {"round": round_idx, "raw": raw_clean, "no_tool_call": True}
                    )
                    break
        tool_name, tool_args = parsed
        executor.record["tool_calls"].append(
            {"round": round_idx, "tool": tool_name, "args": tool_args}
        )
        result_text, result_image = executor.dispatch(tool_name, tool_args)
        executor.record["tool_calls"][-1]["result"] = result_text
        print(
            f"  [react r{round_idx}] {tool_name}({json.dumps(tool_args, ensure_ascii=False)}) -> {result_text[:120]}",
            flush=True,
        )
        messages.append({"role": "assistant", "content": raw_clean})
        if context.settings.live_observation and context.append_observation is not None:
            context.append_observation(messages, executor, result_text)
        elif result_image is not None:
            messages.append(
                {
                    "role": "tool",
                    "content": [
                        {"type": "image", "image": result_image},
                        {"type": "text", "text": result_text},
                    ],
                }
            )
        else:
            messages.append({"role": "tool", "content": result_text})
        if executor.done:
            break
    return executor.obs
