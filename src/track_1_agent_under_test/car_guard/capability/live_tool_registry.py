"""Strict per-session validation for only the currently supplied live tools."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Iterable, Mapping
from typing import Any

from ..domain import OfficialToolCall, ProposedToolCall


class ToolDefinitionError(ValueError):
    """A live tool definition is malformed or ambiguous."""


class ToolArgumentValidationError(ValueError):
    """Arguments do not conform exactly to the current live JSON Schema."""

    def __init__(self, code: str, path: str, message: str) -> None:
        self.code = code
        self.path = path
        super().__init__(f"{path}: {message}")


class LiveToolRegistry:
    """A defensive copy of one session's live tool definitions.

    The registry intentionally has no static catalog, task classification, or
    inventory-difference API. Unknown parameters are rejected when a schema
    omits ``additionalProperties`` so outbound calls stay within declared live
    properties even for non-strict provider schemas.
    """

    def __init__(self, live_tools: Iterable[Mapping[str, Any]]) -> None:
        tools: dict[str, dict[str, Any]] = {}
        for index, raw_tool in enumerate(live_tools):
            if not isinstance(raw_tool, Mapping):
                raise ToolDefinitionError(f"live tool {index} must be an object")
            function = raw_tool.get("function", raw_tool)
            if not isinstance(function, Mapping):
                raise ToolDefinitionError(
                    f"live tool {index} function must be an object"
                )
            if raw_tool.get("type", "function") != "function":
                raise ToolDefinitionError(f"live tool {index} is not a function tool")

            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ToolDefinitionError(
                    f"live tool {index} requires a non-empty name"
                )
            name = name.strip()
            if name in tools:
                raise ToolDefinitionError(f"duplicate live tool name: {name}")

            parameters = function.get(
                "parameters",
                {"type": "object", "properties": {}, "additionalProperties": False},
            )
            if not isinstance(parameters, Mapping):
                raise ToolDefinitionError(
                    f"parameters for {name} must be an object schema"
                )
            schema = copy.deepcopy(dict(parameters))
            _prune_optional_false_property_schemas(
                schema,
                path=f"{name}.parameters",
            )
            _validate_schema_definition(schema, path=f"{name}.parameters", root=True)
            normalized = copy.deepcopy(dict(raw_tool))
            normalized_function = copy.deepcopy(dict(function))
            normalized_function["name"] = name
            normalized_function["parameters"] = schema
            if "function" in raw_tool:
                normalized["function"] = normalized_function
            else:
                normalized = normalized_function
            tools[name] = normalized
        self._tools = tools

    def __len__(self) -> int:
        return len(self._tools)

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def validate_arguments(self, name: str, args: Mapping[str, Any]) -> dict[str, Any]:
        """Return a defensive copy when ``args`` exactly satisfy the live schema."""

        schema = self._parameter_schema(name)
        if not isinstance(args, Mapping):
            raise ToolArgumentValidationError(
                "type", "$", "tool arguments must be an object"
            )
        copied = copy.deepcopy(dict(args))
        _validate_value(copied, schema, path="$")
        return copied

    def required_parameters(self, name: str) -> frozenset[str]:
        required = self._parameter_schema(name).get("required", ())
        return frozenset(required)

    def parameter_domain(self, name: str, parameter: str) -> dict[str, Any] | None:
        properties = self._parameter_schema(name).get("properties", {})
        if parameter not in properties:
            return None
        return copy.deepcopy(dict(properties[parameter]))

    def as_definitions(self) -> list[dict[str, Any]]:
        """Return defensive copies of this session's current definitions."""

        return copy.deepcopy(list(self._tools.values()))

    def definition(self, name: str) -> dict[str, Any]:
        try:
            return copy.deepcopy(self._tools[name])
        except KeyError as exc:
            raise ToolArgumentValidationError(
                "tool", "$.tool_name", f"tool is not live: {name}"
            ) from exc

    def description(self, name: str) -> str:
        definition = self.definition(name)
        function = definition.get("function", definition)
        value = function.get("description", "")
        return value if isinstance(value, str) else ""

    def serialize_official(
        self, call: OfficialToolCall | ProposedToolCall | Mapping[str, Any]
    ) -> dict[str, Any]:
        """Validate a call and emit only the official benchmark payload fields."""

        if isinstance(call, (OfficialToolCall, ProposedToolCall)):
            name = call.tool_name
            arguments = call.arguments
        elif isinstance(call, Mapping):
            name = call.get("tool_name", call.get("name"))
            arguments = call.get("arguments", {})
        else:
            raise TypeError("call must be a domain tool call or mapping")
        if not isinstance(name, str) or not name:
            raise ToolArgumentValidationError(
                "tool", "$.tool_name", "invalid tool name"
            )
        normalized = self.validate_arguments(name, arguments)
        return OfficialToolCall(tool_name=name, arguments=normalized).to_a2a_payload()

    def _parameter_schema(self, name: str) -> dict[str, Any]:
        try:
            tool = self._tools[name]
        except KeyError as exc:
            raise ToolArgumentValidationError(
                "tool", "$.tool_name", f"tool is not live: {name}"
            ) from exc
        function = tool.get("function", tool)
        return function["parameters"]


def _prune_optional_false_property_schemas(
    schema: dict[str, Any], *, path: str
) -> None:
    """Prune unusable properties only from closed object schemas.

    Removing one from an open object would turn a forbidden named property
    into an allowed additional property.
    """

    properties = schema.get("properties")
    if isinstance(properties, Mapping):
        parent_is_closed = schema.get("additionalProperties", False) is False
        required = schema.get("required", [])
        required_names = (
            {item for item in required if isinstance(item, str)}
            if isinstance(required, list)
            else set()
        )
        normalized_properties = dict(properties)
        for key, child in properties.items():
            child_path = f"{path}.properties.{key}"
            if child is False:
                if key in required_names:
                    raise ToolDefinitionError(
                        f"{child_path} is required but has an unsatisfiable schema"
                    )
                if parent_is_closed:
                    normalized_properties.pop(key, None)
                continue
            if isinstance(child, Mapping):
                normalized_child = copy.deepcopy(dict(child))
                _prune_optional_false_property_schemas(
                    normalized_child,
                    path=child_path,
                )
                normalized_properties[key] = normalized_child
        schema["properties"] = normalized_properties

    additional = schema.get("additionalProperties")
    if isinstance(additional, Mapping):
        normalized_additional = copy.deepcopy(dict(additional))
        _prune_optional_false_property_schemas(
            normalized_additional,
            path=f"{path}.additionalProperties",
        )
        schema["additionalProperties"] = normalized_additional

    items = schema.get("items")
    if isinstance(items, Mapping):
        normalized_items = copy.deepcopy(dict(items))
        _prune_optional_false_property_schemas(
            normalized_items,
            path=f"{path}.items",
        )
        schema["items"] = normalized_items

    for keyword in ("allOf", "anyOf", "oneOf"):
        choices = schema.get(keyword)
        if not isinstance(choices, list):
            continue
        normalized_choices = []
        for index, child in enumerate(choices):
            if not isinstance(child, Mapping):
                normalized_choices.append(child)
                continue
            normalized_child = copy.deepcopy(dict(child))
            _prune_optional_false_property_schemas(
                normalized_child,
                path=f"{path}.{keyword}[{index}]",
            )
            normalized_choices.append(normalized_child)
        schema[keyword] = normalized_choices


def _validate_schema_definition(
    schema: Mapping[str, Any], *, path: str, root: bool = False
) -> None:
    declared_type = schema.get("type")
    if declared_type is not None:
        types = [declared_type] if isinstance(declared_type, str) else declared_type
        if not isinstance(types, list) or not types:
            raise ToolDefinitionError(f"{path}.type must be a string or non-empty list")
        supported = {
            "array",
            "boolean",
            "integer",
            "null",
            "number",
            "object",
            "string",
        }
        if any(item not in supported for item in types):
            raise ToolDefinitionError(f"{path}.type contains an unsupported JSON type")
    if root and declared_type not in (None, "object"):
        if not (isinstance(declared_type, list) and "object" in declared_type):
            raise ToolDefinitionError(f"{path} must describe an object")

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ToolDefinitionError(f"{path}.properties must be an object")
    for key, child in properties.items():
        if not isinstance(key, str) or not isinstance(child, Mapping):
            raise ToolDefinitionError(f"{path}.properties contains an invalid entry")
        _validate_schema_definition(child, path=f"{path}.properties.{key}")

    required = schema.get("required", [])
    if not isinstance(required, list) or any(
        not isinstance(item, str) for item in required
    ):
        raise ToolDefinitionError(f"{path}.required must be a list of strings")
    if len(required) != len(set(required)):
        raise ToolDefinitionError(f"{path}.required contains duplicates")
    additional = schema.get("additionalProperties", False)
    if not isinstance(additional, (bool, Mapping)):
        raise ToolDefinitionError(
            f"{path}.additionalProperties must be boolean or schema"
        )
    if isinstance(additional, Mapping):
        _validate_schema_definition(additional, path=f"{path}.additionalProperties")

    items = schema.get("items")
    if items is not None:
        if not isinstance(items, Mapping):
            raise ToolDefinitionError(f"{path}.items must be a schema")
        _validate_schema_definition(items, path=f"{path}.items")
    for keyword in ("allOf", "anyOf", "oneOf"):
        choices = schema.get(keyword)
        if choices is not None:
            if not isinstance(choices, list) or not choices:
                raise ToolDefinitionError(f"{path}.{keyword} must be a non-empty list")
            for index, child in enumerate(choices):
                if not isinstance(child, Mapping):
                    raise ToolDefinitionError(
                        f"{path}.{keyword}[{index}] must be a schema"
                    )
                _validate_schema_definition(child, path=f"{path}.{keyword}[{index}]")

    enum = schema.get("enum")
    if enum is not None and (not isinstance(enum, list) or not enum):
        raise ToolDefinitionError(f"{path}.enum must be a non-empty list")
    multiple_of = schema.get("multipleOf")
    if multiple_of is not None and (not _is_number(multiple_of) or multiple_of <= 0):
        raise ToolDefinitionError(f"{path}.multipleOf must be positive")


def _validate_value(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    if "allOf" in schema:
        for child in schema["allOf"]:
            _validate_value(value, child, path=path)
    if "anyOf" in schema and not any(
        _matches(value, child, path=path) for child in schema["anyOf"]
    ):
        _fail("anyOf", path, "value does not match any allowed schema")
    if "oneOf" in schema:
        matches = sum(_matches(value, child, path=path) for child in schema["oneOf"])
        if matches != 1:
            _fail("oneOf", path, "value must match exactly one allowed schema")

    if "const" in schema and not _json_equal(value, schema["const"]):
        _fail("const", path, "value does not match the required constant")
    if "enum" in schema and not any(
        _json_equal(value, candidate) for candidate in schema["enum"]
    ):
        _fail("enum", path, f"value is outside the allowed enum {schema['enum']!r}")

    declared = schema.get("type")
    if declared is not None:
        allowed = [declared] if isinstance(declared, str) else declared
        if not any(_has_json_type(value, item) for item in allowed):
            _fail("type", path, f"expected JSON type {' or '.join(allowed)}")

    if isinstance(value, Mapping):
        _validate_object(value, schema, path=path)
    elif isinstance(value, list):
        _validate_array(value, schema, path=path)
    elif isinstance(value, str):
        _validate_string(value, schema, path=path)
    elif _is_number(value):
        _validate_number(value, schema, path=path)


def _validate_object(
    value: Mapping[str, Any], schema: Mapping[str, Any], *, path: str
) -> None:
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    absent = [name for name in required if name not in value]
    if absent:
        _fail("required", path, f"missing required properties: {absent!r}")
    additional = schema.get("additionalProperties", False)
    for key, item in value.items():
        if not isinstance(key, str):
            _fail("type", path, "object property names must be strings")
        child_path = f"{path}.{key}"
        if key in properties:
            _validate_value(item, properties[key], path=child_path)
        elif additional is False:
            _fail("additionalProperties", child_path, "undeclared property")
        elif isinstance(additional, Mapping):
            _validate_value(item, additional, path=child_path)


def _validate_array(value: list[Any], schema: Mapping[str, Any], *, path: str) -> None:
    if "minItems" in schema and len(value) < schema["minItems"]:
        _fail("minItems", path, "array is too short")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        _fail("maxItems", path, "array is too long")
    if schema.get("uniqueItems") and any(
        _json_equal(left, right)
        for index, left in enumerate(value)
        for right in value[index + 1 :]
    ):
        _fail("uniqueItems", path, "array items must be unique")
    items = schema.get("items")
    if items is not None:
        for index, item in enumerate(value):
            _validate_value(item, items, path=f"{path}[{index}]")


def _validate_string(value: str, schema: Mapping[str, Any], *, path: str) -> None:
    if "minLength" in schema and len(value) < schema["minLength"]:
        _fail("minLength", path, "string is too short")
    if "maxLength" in schema and len(value) > schema["maxLength"]:
        _fail("maxLength", path, "string is too long")
    if "pattern" in schema:
        try:
            matches = re.search(schema["pattern"], value)
        except (re.error, TypeError) as exc:
            raise ToolDefinitionError(f"invalid JSON Schema pattern at {path}") from exc
        if matches is None:
            _fail("pattern", path, "string does not match the required pattern")


def _validate_number(
    value: int | float, schema: Mapping[str, Any], *, path: str
) -> None:
    if not math.isfinite(float(value)):
        _fail("type", path, "JSON numbers must be finite")
    limits = (
        ("minimum", lambda actual, limit: actual >= limit),
        ("maximum", lambda actual, limit: actual <= limit),
        ("exclusiveMinimum", lambda actual, limit: actual > limit),
        ("exclusiveMaximum", lambda actual, limit: actual < limit),
    )
    for keyword, predicate in limits:
        if keyword in schema and not predicate(value, schema[keyword]):
            _fail(keyword, path, f"number violates {keyword}={schema[keyword]!r}")
    if "multipleOf" in schema:
        quotient = float(value) / float(schema["multipleOf"])
        if not math.isclose(quotient, round(quotient), abs_tol=1e-9):
            _fail(
                "multipleOf",
                path,
                f"number is not a multiple of {schema['multipleOf']!r}",
            )


def _matches(value: Any, schema: Mapping[str, Any], *, path: str) -> bool:
    try:
        _validate_value(value, schema, path=path)
    except ToolArgumentValidationError:
        return False
    return True


def _has_json_type(value: Any, expected: str) -> bool:
    return {
        "null": value is None,
        "boolean": type(value) is bool,
        "integer": type(value) is int,
        "number": _is_number(value),
        "string": isinstance(value, str),
        "array": isinstance(value, list),
        "object": isinstance(value, Mapping),
    }[expected]


def _is_number(value: Any) -> bool:
    return type(value) in {int, float}


def _json_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    return left == right


def _fail(code: str, path: str, message: str) -> None:
    raise ToolArgumentValidationError(code, path, message)


__all__ = [
    "LiveToolRegistry",
    "ToolArgumentValidationError",
    "ToolDefinitionError",
]
