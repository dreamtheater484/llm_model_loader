from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable

from .config import default_opencode_db_path
from .scripts import parse_script
from .storage import new_id, now, store


TOKEN_FIELDS = (
    "input_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "output_tokens",
    "reasoning_tokens",
)
RATE_FIELDS = {
    "input_tokens": "price_input_per_million",
    "cache_read_tokens": "price_cache_read_per_million",
    "cache_write_tokens": "price_cache_write_per_million",
    "output_tokens": "price_output_per_million",
    "reasoning_tokens": "price_reasoning_per_million",
}
DISPLAY_FIELDS = {
    "input_tokens": "input",
    "cache_read_tokens": "cache_read",
    "cache_write_tokens": "cache_write",
    "output_tokens": "output",
    "reasoning_tokens": "reasoning",
}


class UsageUnavailable(RuntimeError):
    pass


def _empty_stats() -> dict[str, Any]:
    return {
        **{field: 0 for field in TOKEN_FIELDS},
        "total_tokens": 0,
        "requests": 0,
        "cost": Decimal("0"),
        "cost_by_field": {field: Decimal("0") for field in TOKEN_FIELDS},
        "missing_rates": set(),
        "reasoning_not_reported": False,
        "reasoning_reported": False,
    }


def _to_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _to_seconds(value: Any) -> float:
    number = float(value or 0)
    return number / 1000 if number > 100_000_000_000 else number


def _iso_timestamp(value: float | None) -> str | None:
    if not value:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _normalise_range(value: str) -> str:
    value = (value or "all").lower()
    if value not in {"7d", "30d", "all"}:
        raise ValueError("range must be 7d, 30d, or all")
    return value


def _json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value:
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _opencode_path(configured: str | Path | None = None) -> Path:
    if configured:
        return Path(configured).expanduser()
    return default_opencode_db_path()


def _open_opencode(path: Path) -> sqlite3.Connection:
    path = path.expanduser()
    if not path.exists() or not path.is_file():
        raise UsageUnavailable(f"OpenCode database not found at {path}")
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=0.5)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma query_only = on")
        return connection
    except (OSError, sqlite3.Error) as exc:
        raise UsageUnavailable(f"OpenCode usage unavailable: {exc}") from exc


def _reasoning_message_ids(connection: sqlite3.Connection) -> set[str]:
    try:
        rows = connection.execute("select message_id, data from part").fetchall()
    except sqlite3.Error:
        return set()
    result: set[str] = set()
    for row in rows:
        data = _json(row["data"])
        if data.get("type") == "reasoning":
            result.add(str(row["message_id"]))
    return result


def _read_source(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    connection = _open_opencode(path)
    try:
        try:
            session_rows = connection.execute(
                "select id, parent_id, title, agent, model, time_created, time_updated from session"
            ).fetchall()
            message_rows = connection.execute(
                "select id, session_id, time_created, time_updated, data from message"
            ).fetchall()
        except sqlite3.Error as exc:
            raise UsageUnavailable(f"OpenCode usage unavailable: {exc}") from exc
        reasoning_ids = _reasoning_message_ids(connection)
    finally:
        connection.close()

    sessions: dict[str, dict[str, Any]] = {}
    for row in session_rows:
        session = dict(row)
        model = _json(session.get("model"))
        session["provider_id"] = model.get("providerID") or model.get("provider_id")
        session["external_model_id"] = model.get("id") or model.get("modelID") or model.get("model_id")
        session["time_created"] = _to_seconds(session.get("time_created"))
        session["time_updated"] = _to_seconds(session.get("time_updated"))
        sessions[str(session["id"])] = session

    events: list[dict[str, Any]] = []
    for row in message_rows:
        data = _json(row["data"])
        if data.get("role") != "assistant":
            continue
        tokens = data.get("tokens") or {}
        cache = tokens.get("cache") or {}
        session = sessions.get(str(row["session_id"]), {})
        event = {
            "id": str(row["id"]),
            "session_id": str(row["session_id"]),
            "timestamp": _to_seconds(row["time_updated"] or row["time_created"]),
            "provider_id": data.get("providerID") or data.get("provider_id") or session.get("provider_id") or "unknown",
            "external_model_id": data.get("modelID") or data.get("model_id") or session.get("external_model_id") or "unknown",
            "agent": data.get("agent") or session.get("agent") or "",
            "reasoning_part": str(row["id"]) in reasoning_ids,
        }
        event.update(
            input_tokens=_to_int(tokens.get("input")),
            cache_read_tokens=_to_int(cache.get("read")),
            cache_write_tokens=_to_int(cache.get("write")),
            output_tokens=_to_int(tokens.get("output")),
            reasoning_tokens=_to_int(tokens.get("reasoning")),
        )
        event["reasoning_not_reported"] = event["reasoning_part"] and event["reasoning_tokens"] == 0
        events.append(event)
    return events, sessions


def _loader_context(loader_store: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = loader_store.rows("select * from models order by display_order asc, created_at desc")
    scripts = loader_store.rows("select model_id, raw_script from scripts")
    bindings = loader_store.rows(
        "select id, model_id, provider_id, external_model_id, created_at from model_usage_bindings"
    )
    for model in models:
        model["bindings"] = [binding for binding in bindings if binding["model_id"] == model["id"]]
    return models, scripts


def _identity_key(provider_id: Any, external_model_id: Any) -> tuple[str, str]:
    return (str(provider_id or "").strip().casefold(), str(external_model_id or "").strip().casefold())


def _model_mapping(models: list[dict[str, Any]], scripts: list[dict[str, Any]], loader_store: Any) -> dict[tuple[str, str], str | None]:
    explicit = loader_store.rows(
        "select provider_id, external_model_id, model_id from model_usage_bindings"
    )
    mapping: dict[tuple[str, str], str | None] = {
        _identity_key(row["provider_id"], row["external_model_id"]): row["model_id"] for row in explicit
    }
    aliases: dict[str, set[str]] = defaultdict(set)

    def add_alias(alias: Any, model_id: str) -> None:
        if alias is not None and str(alias).strip():
            aliases[str(alias).strip().casefold()].add(model_id)

    for model in models:
        add_alias(model.get("name"), model["id"])
    for script in scripts:
        try:
            info = parse_script(script.get("raw_script") or "")
            add_alias(info.alias, script["model_id"])
            add_alias(info.model_ref, script["model_id"])
        except Exception:
            continue
    for alias, model_ids in aliases.items():
        model_id = next(iter(model_ids)) if len(model_ids) == 1 else None
        # Automatic matches are provider-independent, but explicit bindings
        # always win for identities that were saved by the user.
        mapping.setdefault(("", alias), model_id)
    return mapping


def _resolve_model(event: dict[str, Any], mapping: dict[tuple[str, str], str | None]) -> str | None:
    provider, external = _identity_key(event["provider_id"], event["external_model_id"])
    explicit = mapping.get((provider, external))
    if (provider, external) in mapping:
        return explicit
    return mapping.get(("", external))


def _parse_rate(value: Any) -> Decimal | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        rate = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return None
    return rate if rate >= 0 else None


def _event_cost(event: dict[str, Any], model: dict[str, Any] | None) -> tuple[Decimal, set[str], dict[str, Decimal]]:
    components = {field: Decimal("0") for field in TOKEN_FIELDS}
    if not model:
        return Decimal("0"), {DISPLAY_FIELDS[field] for field in TOKEN_FIELDS if event[field]}, components
    rates = {field: _parse_rate(model.get(rate_field)) for field, rate_field in RATE_FIELDS.items()}
    if rates["reasoning_tokens"] is None:
        rates["reasoning_tokens"] = rates["output_tokens"]
    total = Decimal("0")
    missing: set[str] = set()
    for field in TOKEN_FIELDS:
        tokens = event[field]
        if not tokens:
            continue
        rate = rates[field]
        if rate is None:
            missing.add(DISPLAY_FIELDS[field])
        else:
            components[field] = Decimal(tokens) * rate / Decimal(1_000_000)
            total += components[field]
    return total, missing, components


def _add_event(
    stats: dict[str, Any],
    event: dict[str, Any],
    cost: Decimal,
    missing: Iterable[str],
    components: dict[str, Decimal] | None = None,
) -> None:
    for field in TOKEN_FIELDS:
        stats[field] += event[field]
    stats["total_tokens"] = sum(stats[field] for field in TOKEN_FIELDS)
    stats["requests"] += 1
    stats["cost"] += cost
    for field in TOKEN_FIELDS:
        stats["cost_by_field"][field] += (components or {}).get(field, Decimal("0"))
    stats["missing_rates"].update(missing)
    stats["reasoning_not_reported"] |= bool(event["reasoning_not_reported"])
    stats["reasoning_reported"] |= bool(event["reasoning_tokens"])


def _public_stats(stats: dict[str, Any]) -> dict[str, Any]:
    if not stats["requests"]:
        cost_status = "no_usage"
    elif stats["missing_rates"]:
        cost_status = "partially_priced" if stats["cost"] else "unpriced"
    else:
        cost_status = "priced"
    if stats["reasoning_not_reported"]:
        reasoning_status = "not_reported"
    elif stats["reasoning_reported"]:
        reasoning_status = "reported"
    else:
        reasoning_status = "none"
    def money_text(value: Decimal) -> str:
        return format(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP), "f")

    cost_text = money_text(stats["cost"])
    component_text = {
        DISPLAY_FIELDS[field]: money_text(stats["cost_by_field"][field])
        for field in TOKEN_FIELDS
    }
    component_text["cache"] = money_text(
        stats["cost_by_field"]["cache_read_tokens"] + stats["cost_by_field"]["cache_write_tokens"]
    )
    component_text["total"] = cost_text
    return {
        **{field: int(stats[field]) for field in TOKEN_FIELDS},
        "total_tokens": int(stats["total_tokens"]),
        "requests": int(stats["requests"]),
        "cost": cost_text,
        "costs": component_text,
        "cost_status": cost_status,
        "missing_rates": sorted(stats["missing_rates"]),
        "reasoning_status": reasoning_status,
    }


def _new_group() -> dict[str, Any]:
    stats = _empty_stats()
    stats["model_ids"] = set()
    stats["model_names"] = set()
    stats["last_timestamp"] = 0.0
    return stats


def _group_public(stats: dict[str, Any]) -> dict[str, Any]:
    result = _public_stats(stats)
    result["model_ids"] = sorted(stats.get("model_ids", set()))
    result["model_names"] = sorted(stats.get("model_names", set()))
    result["updated_at"] = _iso_timestamp(stats.get("last_timestamp"))
    return result


def _root_id(session_id: str, sessions: dict[str, dict[str, Any]]) -> str:
    seen: set[str] = set()
    current = session_id
    while current in sessions and sessions[current].get("parent_id") and current not in seen:
        seen.add(current)
        current = str(sessions[current]["parent_id"])
    return current


def usage_snapshot(
    range_value: str = "all",
    model_id: str | None = None,
    page: int = 0,
    page_size: int = 8,
    opencode_path: str | Path | None = None,
    loader_store: Any | None = None,
) -> dict[str, Any]:
    range_value = _normalise_range(range_value)
    page = max(0, int(page or 0))
    loader_store = loader_store or store
    path = _opencode_path(opencode_path)
    events, sessions = _read_source(path)
    models, scripts = _loader_context(loader_store)
    mapping = _model_mapping(models, scripts, loader_store)
    model_by_id = {model["id"]: model for model in models}
    since = 0.0
    if range_value != "all":
        since = now() - (7 if range_value == "7d" else 30) * 86400

    filtered: list[dict[str, Any]] = []
    for event in events:
        if event["timestamp"] < since:
            continue
        event["model_id"] = _resolve_model(event, mapping)
        if model_id and event["model_id"] != model_id:
            continue
        event["model_name"] = model_by_id.get(event["model_id"], {}).get("name")
        event["cost_value"], event["missing_rates"], event["cost_components"] = _event_cost(event, model_by_id.get(event["model_id"]))
        filtered.append(event)

    summary = _new_group()
    model_groups: dict[str, dict[str, Any]] = {model["id"]: _new_group() for model in models}
    unmapped_groups: dict[tuple[str, str], dict[str, Any]] = {}
    task_groups: dict[str, dict[str, Any]] = {}
    session_groups: dict[tuple[str, str], dict[str, Any]] = {}
    for event in filtered:
        _add_event(summary, event, event["cost_value"], event["missing_rates"], event["cost_components"])
        group = model_groups.get(event["model_id"]) if event["model_id"] else None
        if group is not None:
            _add_event(group, event, event["cost_value"], event["missing_rates"], event["cost_components"])
            group["model_ids"].add(event["model_id"])
            group["model_names"].add(event["model_name"] or "Unmapped")
        else:
            key = _identity_key(event["provider_id"], event["external_model_id"])
            group = unmapped_groups.setdefault(key, _new_group())
            _add_event(group, event, event["cost_value"], event["missing_rates"], event["cost_components"])
        root_id = _root_id(event["session_id"], sessions)
        task = task_groups.setdefault(root_id, _new_group())
        _add_event(task, event, event["cost_value"], event["missing_rates"], event["cost_components"])
        task["model_ids"].add(event["model_id"] or "")
        task["model_names"].add(event["model_name"] or f"{event['provider_id']} / {event['external_model_id']}")
        session_key = (root_id, event["session_id"])
        child = session_groups.setdefault(session_key, _new_group())
        _add_event(child, event, event["cost_value"], event["missing_rates"], event["cost_components"])
        child["model_ids"].add(event["model_id"] or "")
        child["model_names"].add(event["model_name"] or f"{event['provider_id']} / {event['external_model_id']}")
        for target in (summary, group, task, child):
            target["last_timestamp"] = max(target["last_timestamp"], event["timestamp"])

    model_output = []
    for model in models:
        output = {
            "model_id": model["id"],
            "model_name": model["name"],
            "bindings": model.get("bindings", []),
            "price_currency": model.get("price_currency") or "USD",
            "price_input_per_million": model.get("price_input_per_million"),
            "price_cache_read_per_million": model.get("price_cache_read_per_million"),
            "price_cache_write_per_million": model.get("price_cache_write_per_million"),
            "price_output_per_million": model.get("price_output_per_million"),
            "price_reasoning_per_million": model.get("price_reasoning_per_million"),
            "usage": _group_public(model_groups[model["id"]]),
        }
        model_output.append(output)

    task_items = []
    for root_id, task in task_groups.items():
        root = sessions.get(root_id, {})
        main = session_groups.get((root_id, root_id), _new_group())
        subagents = _new_group()
        children = []
        for (task_root, session_id), child in session_groups.items():
            if task_root != root_id:
                continue
            if session_id != root_id:
                _merge_group(subagents, child)
                session = sessions.get(session_id, {})
                children.append(
                    {
                        "session_id": session_id,
                        "title": session.get("title") or session.get("agent") or "Subagent",
                        "agent": session.get("agent") or "",
                        "usage": _group_public(child),
                    }
                )
        children.sort(key=lambda item: item["usage"].get("updated_at") or "", reverse=True)
        task_items.append(
            {
                "session_id": root_id,
                "title": root.get("title") or "Untitled OpenCode task",
                "updated_at": _iso_timestamp(task["last_timestamp"]),
                "model_names": sorted(name for name in task["model_names"] if name),
                "main": _group_public(main),
                "subagents": _group_public(subagents),
                "usage": _group_public(task),
                "children": children,
            }
        )
    task_items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
    recent = task_items[0] if task_items else None
    start = page * page_size
    history = task_items[start : start + page_size]

    unmapped = []
    for (provider_id, external_model_id), group in sorted(unmapped_groups.items()):
        item = _group_public(group)
        item.update({"provider_id": provider_id, "external_model_id": external_model_id})
        unmapped.append(item)

    return {
        "available": True,
        "path": str(path),
        "range": range_value,
        "summary": _group_public(summary),
        "models": model_output,
        "recent_task": recent,
        "history": {
            "items": history,
            "total": len(task_items),
            "page": page,
            "page_size": page_size,
        },
        "unmapped": unmapped,
    }


def _merge_group(target: dict[str, Any], source: dict[str, Any]) -> None:
    target["cost"] += source["cost"]
    for field in TOKEN_FIELDS:
        target[field] += source[field]
        target["cost_by_field"][field] += source["cost_by_field"][field]
    target["total_tokens"] = sum(target[field] for field in TOKEN_FIELDS)
    target["requests"] += source["requests"]
    target["missing_rates"].update(source["missing_rates"])
    target["reasoning_not_reported"] |= source["reasoning_not_reported"]
    target["reasoning_reported"] |= source["reasoning_reported"]
    target["model_ids"].update(source.get("model_ids", set()))
    target["model_names"].update(source.get("model_names", set()))
    target["last_timestamp"] = max(target["last_timestamp"], source.get("last_timestamp", 0.0))


def _validated_price(value: Any, label: str) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{label} must be a non-negative decimal number") from exc
    if number < 0:
        raise ValueError(f"{label} must be a non-negative decimal number")
    return format(number, "f")


def save_model_usage_settings(
    model_id: str,
    payload: dict[str, Any],
    loader_store: Any | None = None,
) -> dict[str, Any]:
    loader_store = loader_store or store
    model = loader_store.row("select * from models where id=?", (model_id,))
    if not model:
        raise ValueError("Model not found.")
    currency = str(payload.get("currency") or payload.get("price_currency") or model.get("price_currency") or "USD").strip().upper()
    if not currency or len(currency) > 8:
        raise ValueError("Currency must be a short code such as USD.")
    def incoming(name: str, column: str) -> Any:
        if name in payload:
            return payload[name]
        price_name = column
        return payload[price_name] if price_name in payload else model.get(column)

    prices = {
        "price_input_per_million": _validated_price(incoming("input_per_million", "price_input_per_million"), "Input price"),
        "price_cache_read_per_million": _validated_price(incoming("cache_read_per_million", "price_cache_read_per_million"), "Cache-read price"),
        "price_cache_write_per_million": _validated_price(incoming("cache_write_per_million", "price_cache_write_per_million"), "Cache-write price"),
        "price_output_per_million": _validated_price(incoming("output_per_million", "price_output_per_million"), "Output price"),
        "price_reasoning_per_million": _validated_price(incoming("reasoning_per_million", "price_reasoning_per_million"), "Reasoning price"),
    }
    bindings = payload.get("bindings")
    if bindings is None:
        bindings = loader_store.rows(
            "select provider_id, external_model_id from model_usage_bindings where model_id=?",
            (model_id,),
        )
    clean_bindings: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for binding in bindings or []:
        provider_id = str(binding.get("provider_id") or "").strip()
        external_model_id = str(binding.get("external_model_id") or "").strip()
        if not provider_id or not external_model_id:
            raise ValueError("Each binding needs a provider and external model ID.")
        key = _identity_key(provider_id, external_model_id)
        if key in seen:
            continue
        seen.add(key)
        clean_bindings.append((provider_id, external_model_id))
    with loader_store.connect() as connection:
        conflict = connection.execute(
            "select provider_id, external_model_id, model_id from model_usage_bindings"
        ).fetchall()
        for row in conflict:
            key = _identity_key(row["provider_id"], row["external_model_id"])
            if key in seen and row["model_id"] != model_id:
                raise ValueError(f"That OpenCode identity is already bound to another model: {row['provider_id']} / {row['external_model_id']}")
        connection.execute(
            "update models set price_currency=?, price_input_per_million=?, price_cache_read_per_million=?, price_cache_write_per_million=?, price_output_per_million=?, price_reasoning_per_million=? where id=?",
            (currency, *prices.values(), model_id),
        )
        connection.execute("delete from model_usage_bindings where model_id=?", (model_id,))
        for provider_id, external_model_id in clean_bindings:
            connection.execute(
                "insert into model_usage_bindings(id, model_id, provider_id, external_model_id, created_at) values(?, ?, ?, ?, ?)",
                (new_id("binding"), model_id, provider_id, external_model_id, now()),
            )
    updated = loader_store.row("select * from models where id=?", (model_id,)) or {"id": model_id}
    updated["bindings"] = loader_store.rows(
        "select id, model_id, provider_id, external_model_id, created_at from model_usage_bindings where model_id=?",
        (model_id,),
    )
    return updated
