from __future__ import annotations

import inspect
from typing import Any

JSON_TO_PY: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}
PY_TO_JSON: dict[type, str] = {py: js for js, py in JSON_TO_PY.items()}


def schema_from_signature(func: Any) -> dict[str, Any]:
    """Best-effort JSON schema for a function's keyword arguments, from its type hints."""
    try:
        hints = inspect.get_annotations(func, eval_str=True)
    except Exception:  # noqa: BLE001 - fall back to no hints rather than fail tool registration
        hints = {}
    signature = inspect.signature(func)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, param in signature.parameters.items():
        if param_name == "self":
            continue
        annotation = hints.get(param_name, str)
        properties[param_name] = {"type": PY_TO_JSON.get(annotation, "string")}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def signature_from_json_schema(parameters: dict[str, Any]) -> inspect.Signature:
    """Inverse of schema_from_signature: builds a real python Signature from a JSON
    schema, so it can be attached (via __signature__) to a **kwargs wrapper function
    that a schema-introspecting framework (e.g. MCP's FastMCP) can read normally."""
    properties: dict[str, Any] = parameters.get("properties", {})
    required = set(parameters.get("required", []))
    ordered_names = sorted(properties, key=lambda n: n not in required)  # required first, stable within group
    sig_params = []
    for prop_name in ordered_names:
        py_type = JSON_TO_PY.get(properties[prop_name].get("type", "string"), str)
        if prop_name in required:
            sig_params.append(inspect.Parameter(prop_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=py_type))
        else:
            sig_params.append(
                inspect.Parameter(
                    prop_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=py_type, default=None
                )
            )
    return inspect.Signature(sig_params)
