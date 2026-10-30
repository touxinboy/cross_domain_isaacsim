"""Load pinned upstream function definitions without Habitat/client initialization.

The original source files are kept unchanged. Only explicitly named top-level
function definitions are compiled; imports, assignments and main loops are not
executed. File hashes must match the source manifest before loading.
"""
import ast
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def reviewed_tree(relative_path):
    manifest = json.loads((PROJECT / "configs/navigation_sources.json").read_text(encoding="utf-8"))
    expected = manifest["files"].get(relative_path)
    path = PROJECT / relative_path
    if not path.is_file():
        raise RuntimeError("Missing pinned upstream source: " + relative_path)
    raw = path.read_bytes()
    if not expected or hashlib.sha256(raw).hexdigest() != expected:
        raise RuntimeError("Upstream source hash differs from the reviewed version: " + relative_path)
    return ast.parse(raw.decode("utf-8"), filename=str(path)), path


def load_functions(relative_path, names, namespace):
    tree, path = reviewed_tree(relative_path)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise RuntimeError("Required upstream function definitions are missing")
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def load_string_assignment(relative_path, function_name, variable_name):
    tree, _ = reviewed_tree(relative_path)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    for node in ast.walk(function):
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == variable_name for target in node.targets):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError("Required upstream prompt constant is missing")
