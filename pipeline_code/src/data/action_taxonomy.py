"""
Tool classification for tau-bench retail (and airline).

>>> P2 RESPONSIBILITY <<<
Verify tool names against the actual tau-bench source:
  external/tau-bench/tau_bench/envs/retail/tools/
After installing tau-bench run:
  python -c "from tau_bench.envs import get_env; e = get_env('retail', ...); print(e.tools)"
and update the sets below if names differ.

>>> DO NOT CHANGE THE FUNCTION SIGNATURES <<<
GatedEnv and all agents call is_mutating(), is_reversible(), get_policy_text() by name.
"""

from __future__ import annotations

# ── Retail tools ────────────────────────────────────────────────────────────

RETAIL_READONLY: set[str] = {
    "get_user_details",
    "find_user_id_by_email",
    "find_user_id_by_name_zip",
    "get_order_details",
    "find_order_id_by_invoice",
    "get_product_details",
    "list_all_product_types",
    "calculate",
    "think",
}

# Mutating tools that CAN be undone by a follow-up action within the same session.
RETAIL_REVERSIBLE: set[str] = {
    "cancel_pending_order",
    "modify_pending_order_items",
    "modify_pending_order_payment",
    "modify_pending_order_address",
    "modify_user_address",
    "return_delivered_order_items",   # just initiates; reversible until processed
}

# Mutating tools whose effects cannot easily be undone.
RETAIL_IRREVERSIBLE: set[str] = {
    "exchange_delivered_order_items",
    "transfer_to_human_agents",
}

RETAIL_MUTATING: set[str] = RETAIL_REVERSIBLE | RETAIL_IRREVERSIBLE


# ── Airline tools (stretch) — P2 to fill in ─────────────────────────────────

AIRLINE_READONLY: set[str] = {
    "get_reservation_details",
    "get_flight_info",
    "get_passenger_details",
    "calculate",
    "think",
}

AIRLINE_REVERSIBLE: set[str] = {
    "update_reservation_baggages",
    "update_reservation_flights",
    "update_reservation_passengers",
}

AIRLINE_IRREVERSIBLE: set[str] = {
    "cancel_reservation",
    "transfer_to_human_agents",
}

AIRLINE_MUTATING: set[str] = AIRLINE_REVERSIBLE | AIRLINE_IRREVERSIBLE


# ── Public API ───────────────────────────────────────────────────────────────

def is_mutating(tool_name: str, env_name: str = "retail") -> bool:
    if env_name == "retail":
        return tool_name in RETAIL_MUTATING
    if env_name == "airline":
        return tool_name in AIRLINE_MUTATING
    raise ValueError(f"Unknown env: {env_name}")


def is_reversible(tool_name: str, env_name: str = "retail") -> bool:
    if env_name == "retail":
        return tool_name in RETAIL_REVERSIBLE
    if env_name == "airline":
        return tool_name in AIRLINE_REVERSIBLE
    raise ValueError(f"Unknown env: {env_name}")


def get_policy_text(env_name: str = "retail") -> str:
    if env_name == "retail":
        return RETAIL_POLICY
    if env_name == "airline":
        return AIRLINE_POLICY
    raise ValueError(f"Unknown env: {env_name}")


# ── Policy texts (fed verbatim to critic prompt) ─────────────────────────────
# P2: Expand/correct these based on the actual tau-bench retail policy doc.

RETAIL_POLICY = """\
RETAIL BUSINESS RULES:
- cancel_pending_order: Only allowed when order status is PENDING.
  Requires a valid reason string. Cannot cancel SHIPPED or DELIVERED orders.
- modify_pending_order_items: Only allowed when status is PENDING.
  New items must exist and be in stock.
- modify_pending_order_payment: Only allowed when status is PENDING.
  Payment method must belong to this user.
- modify_pending_order_address: Only allowed when status is PENDING.
- modify_user_address: Updates the user's profile address, not a specific order.
- exchange_delivered_order_items: Only for DELIVERED orders within the return window.
  Requires the user to have purchased the items being exchanged.
- return_delivered_order_items: Only for DELIVERED orders within the return window.
- transfer_to_human_agents: Use ONLY when unable to resolve, or customer explicitly requests.
  This ends the session; cannot be undone.
GENERAL:
- NEVER perform an action without first confirming the relevant order/user details.
- NEVER assume order status; always call get_order_details first.
- NEVER modify or cancel an order that belongs to a different user.""".strip()


AIRLINE_POLICY = """\
AIRLINE BUSINESS RULES (stretch — P2 to expand):
- cancel_reservation: Only within the cancellation window.
- update_reservation_flights: Subject to fare rules and availability.
- transfer_to_human_agents: Ends the session; irreversible.
GENERAL:
- Always verify reservation and passenger details before any mutation.""".strip()
