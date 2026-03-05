"""Scope routing must use bool.should (OR), not device/equip AND filters."""

from __future__ import annotations

from collections.abc import Mapping


def _normalize_values(values: list[str] | None) -> list[str]:
    if not values:
        return []
    normalized = [str(value).strip() for value in values if str(value).strip()]
    return sorted(set(normalized))


def build_scope_filter_by_doc_ids(
    allowed_devices: list[str] | None,
    allowed_equip_ids: list[str] | None,
    *,
    shared_doc_ids: list[str] | None,
    device_doc_types: list[str] | None,
    equip_doc_types: list[str] | None,
) -> dict[str, object] | None:
    devices = _normalize_values(allowed_devices)
    equip_ids = _normalize_values(allowed_equip_ids)
    shared_ids = _normalize_values(shared_doc_ids)
    device_types = _normalize_values(device_doc_types)
    equip_types = _normalize_values(equip_doc_types)

    should_clauses: list[object] = []

    if shared_ids:
        should_clauses.append({"terms": {"doc_id": shared_ids}})

    if devices and device_types:
        device_must = [
            {"terms": {"doc_type": device_types}},
            {"terms": {"device_name": devices}},
        ]
        if shared_ids:
            device_clause = {
                "bool": {
                    "must": device_must,
                    "must_not": [{"terms": {"doc_id": shared_ids}}],
                }
            }
        else:
            device_clause = {"bool": {"must": device_must}}
        should_clauses.append(device_clause)

    equip_scope_filter = None
    if equip_ids:
        equip_scope_filter = {"terms": {"equip_id": equip_ids}}
    elif devices:
        equip_scope_filter = {"terms": {"device_name": devices}}

    if equip_scope_filter and equip_types:
        should_clauses.append(
            {
                "bool": {
                    "must": [
                        {"terms": {"doc_type": equip_types}},
                        equip_scope_filter,
                    ]
                }
            }
        )

    if not should_clauses:
        return None
    return {"bool": {"should": should_clauses, "minimum_should_match": 1}}


def build_scope_filter_by_fields(
    allowed_devices: list[str] | None,
    allowed_equip_ids: list[str] | None,
) -> dict[str, object]:
    devices = _normalize_values(allowed_devices)
    equip_ids = _normalize_values(allowed_equip_ids)

    should_clauses: list[object] = [
        {
            "bool": {
                "must": [
                    {"term": {"scope_level": "shared"}},
                    {"term": {"is_shared": True}},
                ]
            }
        }
    ]

    if devices:
        should_clauses.append(
            {
                "bool": {
                    "must": [
                        {"term": {"scope_level": "device"}},
                        {"terms": {"device_name": devices}},
                    ]
                }
            }
        )

    equip_scope_filter = None
    if equip_ids:
        equip_scope_filter = {"terms": {"equip_id": equip_ids}}
    elif devices:
        equip_scope_filter = {"terms": {"device_name": devices}}

    if equip_scope_filter:
        should_clauses.append(
            {
                "bool": {
                    "must": [
                        {"term": {"scope_level": "equip"}},
                        equip_scope_filter,
                    ]
                }
            }
        )

    return {"bool": {"should": should_clauses, "minimum_should_match": 1}}


def apply_scope_filter(
    base_filter: Mapping[str, object] | None,
    scope_filter: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if base_filter is None:
        return dict(scope_filter) if scope_filter is not None else None
    if scope_filter is None:
        return dict(base_filter)
    return {"bool": {"filter": [base_filter, scope_filter]}}


__all__ = [
    "apply_scope_filter",
    "build_scope_filter_by_doc_ids",
    "build_scope_filter_by_fields",
]
