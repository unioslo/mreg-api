# Zone and delegation managers: one public manager over private per-type fetchers

## Context

The model layer splits zones into `Zone` (abstract) → `ForwardZone` / `ReverseZone`,
and delegations into `Delegation` → `ForwardZoneDelegation` / `ReverseZoneDelegation`.
Each concrete subclass exists for exactly one reason: its `endpoint()` classmethod
returns a different URL (`/zones/forward/` vs `/zones/reverse/`, and the matching
`.../delegations/` paths). The forward/reverse pairs are otherwise identical in
fields and behavior.

The manager base (`ResourceManager`) is built around **one endpoint per manager**:
every fetch path calls the argument-less `self._endpoint()`. The CLI confirms the
distinction barely surfaces: `Zone.get_zone_or_raise(name)` is used type-agnostically
everywhere (delete, SOA, TTL, nameservers, delegations); the only places that need an
explicit type are `zone list --forward/--reverse` and `ForwardZone.get_from_host`
(forward-only, 4 call sites).

## Decision

**One public `ZoneManager` (`client.zones`)**, not three. It is a thin facade over two
**private** single-endpoint sub-managers (`_ForwardZoneManager`, `_ReverseZoneManager`),
each a clean `NamedResourceManager` honoring the one-endpoint invariant. Public methods
dispatch by name shape (`is_reverse_zone_name`): `get_by_name`/`create` return the
`ForwardZone | ReverseZone` union; `list_forward()` / `list_reverse()` give the narrow
typed lists; `get_from_host` delegates to the forward sub-manager.

**Delegations get their own `DelegationManager`** (zone-scoped methods: `create(zone, …)`,
`list_by_zone(zone)`, `delete`, `set_comment`), even though delegations have no standalone
endpoint and their type is derived from the parent zone. This keeps it composition-ready
for a future `client.zones.delegations`.

## Considered options

- **Flat `ZoneManager` branching internally** — rejected: the base threads the endpoint
  implicitly through `_fetch_by_field`, so branching forces either re-implementing the
  fetch logic in `ZoneManager` or threading an explicit endpoint through the whole base.
- **Three public managers mirroring the model** — rejected: forward/reverse is an artifact
  of the model's per-type `endpoint()` binding, not a domain distinction; the CLI treats
  zones type-agnostically almost everywhere.
- **Delegation methods on `ZoneManager`** — rejected in favor of a separate manager to keep
  the future composed-manager refactor (`client.zones.delegations`, `client.networks.policies`)
  cheap.

## Consequences

- The forward/reverse manager classes exist but are private; the public surface is just
  `client.zones`.
- Callers that know they want one type narrow via `list_forward()`/`list_reverse()`;
  `get_by_name` returns a union they rarely need to narrow.
