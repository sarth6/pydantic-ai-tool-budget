# pyright: reportUnnecessaryIsInstance=false, reportUnknownArgumentType=false
"""Custom VCR serializer that parses JSON bodies for readable cassettes.

Simplified from pydantic-ai's test infrastructure.
"""

# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import json
import unicodedata
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from yaml import Dumper, SafeLoader
else:
    try:
        from yaml import CDumper as Dumper
        from yaml import CSafeLoader as SafeLoader
    except ImportError:
        from yaml import Dumper, SafeLoader

FILTERED_HEADER_PREFIXES = ["x-"]
FILTERED_HEADERS = {
    "authorization",
    "date",
    "request-id",
    "server",
    "user-agent",
    "via",
    "set-cookie",
    "api-key",
}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _normalize_body(obj: Any) -> Any:
    if isinstance(obj, str):
        return _normalize(obj)
    elif isinstance(obj, dict):
        return {k: _normalize_body(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_body(item) for item in obj]
    return obj


class LiteralDumper(Dumper):
    pass


def _str_presenter(dumper: Dumper, data: str) -> Any:
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


LiteralDumper.add_representer(str, _str_presenter)


def deserialize(cassette_string: str) -> Any:
    cassette_dict = yaml.load(cassette_string, Loader=SafeLoader)
    for interaction in cassette_dict["interactions"]:
        for kind, data in interaction.items():
            parsed_body = data.pop("parsed_body", None)
            if parsed_body is not None:
                dumped_body = json.dumps(parsed_body)
                data["body"] = {"string": dumped_body} if kind == "response" else dumped_body
    return cassette_dict


def serialize(cassette_dict: Any) -> str:
    for interaction in cassette_dict["interactions"]:
        for _kind, data in interaction.items():
            headers: dict[str, list[str]] = data.get("headers", {})
            headers = {k.lower(): v for k, v in headers.items()}
            headers = {k: v for k, v in headers.items() if k not in FILTERED_HEADERS}
            headers = {k: v for k, v in headers.items() if not any(k.startswith(p) for p in FILTERED_HEADER_PREFIXES)}
            data["headers"] = headers

            content_type = headers.get("content-type", [])
            if any(isinstance(h, str) and h.startswith("application/json") for h in content_type):
                body = data.get("body", None)
                if body is not None:
                    if isinstance(body, dict):
                        body = body.get("string")
                    if body:
                        if isinstance(body, bytes):
                            body = body.decode("utf-8")
                        parsed = json.loads(body)
                        data["parsed_body"] = _normalize_body(parsed)
                        del data["body"]

    return yaml.dump(cassette_dict, Dumper=LiteralDumper, allow_unicode=True, width=120)  # type: ignore[no-any-return]
