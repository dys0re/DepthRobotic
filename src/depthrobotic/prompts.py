"""prompts: v30 control behavior with explicit dependencies."""

from __future__ import annotations

DEPTH_PROMPT = "Given this image, estimate the metric depth (in meters, rounded to two decimal places) for every pixel of the image."


def push_prompt(source, destination, instruction):
    return f'Instruction: {instruction}\nPush {source} to {destination}. Locate a contact point on the side of the source opposite the desired motion, and locate the final center of the destination region. Both points use normalized image coordinates from 0..1000. Return JSON only: {{"contact":[x,y],"destination":[x,y]}}.'


def articulation_prompt(target, mode, instruction):
    return f'Instruction: {instruction}\nThe required operation is to {mode} {target}. Locate the exact handle or movable panel contact point. Also give a 2D unit direction [dx,dy] in the image for the direction that contact point must move to complete the operation. For a drawer, outward means the direction the drawer pulls toward the robot; inward means the reverse. Coordinates are normalized to 0..1000 and direction components are from -1 to 1. Return JSON only: {{"contact":[x,y],"motion":[dx,dy]}}.'


def toggle_prompt(target, mode, instruction):
    return f'Instruction: {instruction}\nOperate {target} to turn it {mode}. Locate the center of the visible movable switch or knob that the robot should grasp or press. Use normalized image coordinates from 0..1000. Return JSON only: {{"point":[x,y]}}.'


def mark_prompt(instruction, desc, n):
    return f"""Robot instruction: {instruction}\nThis is the robot's front camera. Numbered markers 1 to {n} are drawn on candidate objects. Which marker is on the {desc}? Use the spatial relations in the instruction to reject same-category distractors. If no marker is on it, answer 0. JSON only, reason at most 10 words: {{"mark": <number>, "reason": "<short>"}}"""


def crop_center_prompt(instruction):
    return f'Robot instruction: {instruction}\nThis is a crop around the coarse source prediction. Locate the visual center of the exact source object. Return JSON only with one point normalized 0..1000 within this crop: {{"point":[x,y]}}.'


def plan_prompt(instruction):
    return f'Robot instruction: {instruction}\nDecompose it into ordered physical steps using only these actions: pick_place, articulate, push_object, toggle. Use pick_place to move a loose object and include source and destination. Use articulate to open or close a drawer, door, or lid and include target and mode=open or mode=close. Use push_object to slide a loose object to a destination and include source and destination. Use toggle for a switch or knob and include target and mode=on or mode=off. Split instructions involving multiple actions or objects into ordered steps. Preserve spatial words such as left, right, front, back, top, middle, and bottom in the object phrases. Include only actions explicitly requested by the instruction; never add an action just because its object is visible. If the instruction only asks to pick up and place an object, return exactly one pick_place step. Return JSON only. Example schema: {{"steps":[{{"action":"articulate","target":"top drawer of cabinet","mode":"open"}},{{"action":"pick_place","source":"black bowl","destination":"inside top drawer"}}]}}'


def _door_prompt(furniture, instruction):
    return f"""Instruction: {instruction}\nThis is the robot's front camera. Return the bounding box of the open door of the {furniture}: the whole swinging panel only, from its hinge edge to its free edge, not the body of the appliance. JSON only, normalized 0..1000: {{"bbox_2d":[x1,y1,x2,y2]}}"""


def _point_prompt(target, instruction, camera):
    return f"""Instruction: {instruction}\nThis is the robot's {camera} camera. Point to the center of the handle of the {target} -- the bar the fingers can grip, not the door or drawer face. JSON only, normalized 0..1000: {{"point":[x,y]}}"""


def _furniture_prompt(furniture, instruction):
    return f'Instruction: {instruction}\nReturn the bounding box of the {furniture}: the free-standing piece with drawers or a door that stands on the table, not the built-in kitchen counters in the background. JSON only, normalized 0..1000: {{"bbox_2d":[x1,y1,x2,y2]}}'


def _handles_prompt(furniture):
    return f'This crop of the front camera shows the {furniture}. List the bounding boxes of ALL of its handle bars (one per drawer or door; the bars the fingers can grip). JSON only, normalized 0..1000 within this crop: [{{"bbox_2d":[x1,y1,x2,y2]}}, ...]'


def wrist_center_prompt(phrase, instruction, role):
    if role == "source":
        description = f"visual center of the exact source object {phrase}"
    else:
        description = f"center of the usable receiving region of {phrase}"
    return f'Robot instruction: {instruction}\nThis is the wrist camera looking down near the {role}. Locate the {description}; ignore the gripper fingers and held object. Return JSON only with a point normalized 0..1000 in this wrist image: {{"point":[x,y]}}.'
