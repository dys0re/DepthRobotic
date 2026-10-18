"""Explicit plan-action registry. Importing it never loads model weights."""

from .pick_place.react import execute_pick_place
from .articulate.handle import execute_articulate
from .simple import execute_push_object, execute_toggle

SKILLS = {
    "pick_place": execute_pick_place,
    "articulate": execute_articulate,
    "push_object": execute_push_object,
    "toggle": execute_toggle,
}
