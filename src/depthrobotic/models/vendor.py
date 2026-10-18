"""Load the DepthVLM extension under a private package name."""

import hashlib
import importlib.util
from pathlib import Path
import sys


def load_depthvlm(source_path):
    package = Path(source_path).resolve() / "model"
    entry = package / "__init__.py"
    if not entry.is_file():
        raise FileNotFoundError(f"DepthVLM model package is missing: {entry}")
    name = (
        "_depthrobotic_depthvlm_"
        + hashlib.sha256(str(package).encode()).hexdigest()[:12]
    )
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, entry, submodule_search_locations=[str(package)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    del sys.modules[key]
            raise
    return sys.modules[name].Qwen3VLForConditionalGeneration
