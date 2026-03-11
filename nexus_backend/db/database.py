"""
Database loader — loads all JSON mock databases at startup into module-level dicts.
Provides typed query functions used by all agents.
"""
from __future__ import annotations

import json
from pathlib import Path

import structlog

log = structlog.get_logger("database")

_DB_DIR = Path(__file__).parent

# ── Module-level storage ──────────────────────────────────────────────────────
_clients: dict = {}
_products: dict = {}
_errors: dict = {}
_knowledge_base: list[dict] = []


def load_all() -> dict:
    """
    Load all JSON databases. Called once at startup.
    Returns the combined dict for injection into agents.
    """
    global _clients, _products, _errors, _knowledge_base

    _clients = _load_json("client_db.json")
    _products = _load_json("product_db.json")
    _errors = _load_json("error_db.json")
    _knowledge_base = _load_json("knowledge_base.json")

    log.info(
        "database.loaded",
        clients=len(_clients),
        products=len(_products),
        error_categories=len(_errors),
        kb_docs=len(_knowledge_base),
    )

    return {
        "clients": _clients,
        "products": _products,
        "errors": _errors,
        "knowledge_base": _knowledge_base,
    }


def _load_json(filename: str) -> dict | list:
    path = _DB_DIR / filename
    with open(path) as f:
        return json.load(f)


# ── Query functions ────────────────────────────────────────────────────────────

def get_client(client_id: str) -> dict | None:
    return _clients.get(client_id)


def get_product(product_name: str) -> dict | None:
    return _products.get(product_name)


def get_active_incident(product_name: str) -> dict | None:
    """Returns active incident dict if the product has one, else None."""
    product = get_product(product_name)
    if product:
        return product.get("active_incident")
    return None


def get_error_entry(product_name: str, error_code: str) -> dict | None:
    product_errors = _errors.get(product_name, {})
    return product_errors.get(error_code)


def get_all_error_codes(product_name: str) -> list[str]:
    return list(_errors.get(product_name, {}).keys())


def get_recent_tickets(client_id: str) -> list[dict]:
    client = get_client(client_id)
    if client:
        return client.get("recent_tickets", [])
    return []


def count_recurring_issue(client_id: str, error_code: str) -> int:
    """Count how many recent tickets match the given error code."""
    tickets = get_recent_tickets(client_id)
    return sum(1 for t in tickets if t.get("error") == error_code)


def get_knowledge_base_docs() -> list[dict]:
    return _knowledge_base


def all_data() -> dict:
    """Return all loaded data as a single dict (for agent injection)."""
    return {
        "clients": _clients,
        "products": _products,
        "errors": _errors,
        "knowledge_base": _knowledge_base,
    }
