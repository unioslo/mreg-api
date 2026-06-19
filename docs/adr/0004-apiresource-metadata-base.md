# `endpoint()` metadata lives on a concrete `APIResource` base

The "this type is an API-addressable resource" contract is a small **concrete base
class**, `APIResource`, in `models/abstracts.py`. It carries the endpoint metadata
that survives the model-behavior strip (task 02): abstract `endpoint() -> Endpoint`
plus the pure helpers built on it (`id_for_endpoint()`, `endpoint_with_id`).

## Context

`endpoint()` is pure metadata (no I/O) and must remain reachable on every model so
managers can find the URL/id-field. Three layers need the contract:

- managers — the `T = TypeVar(bound=APIResource)` bound and `_model`;
- `events.py` — `ObjectRef.new(obj)` calls `obj.endpoint().external_id_field()`;
- models — must expose `endpoint()`.

The spike defined `APIResource` as a `Protocol` inside `managers.py`. That's the wrong
owner: `events.py` (a lower layer) would have to import from `managers.py` (the top
layer), and a Protocol leaves the contract implicit and not runtime-checkable —
despite `ObjectRef.new` calling `endpoint()` at runtime.

## Decision

- A concrete `APIResource` base in `models/abstracts.py`; models inherit it. The
  `Protocol` in `managers.py` is removed; managers import the base.
- `events.py` retargets its single `APIMixin` reference (TYPE_CHECKING import +
  `ObjectRef.new` annotation) to `APIResource` — a ~2-line change, no rewrite.
- The base holds `endpoint()` (abstract) + `id_for_endpoint()` + `endpoint_with_id`
  — the exact members managers (`_endpoint_with_id`) and `events.py` use. Everything
  else from the old `APIMixin` (the API calls) is deleted, not moved here.

## Consequences

- `APIResource` is the honest, `isinstance`-able marker for "API resource model".
- Layer direction is correct: models → events/managers import the base from models.
- Task 02 owns the implementation; task 04 drops the `# noqa: PLC0415` cycle imports
  once models no longer import the client.
