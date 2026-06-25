# `get_first` in the base manager; `get_count` as an opt-in capability

## Context

`get_first` and `get_count` currently live on the models' `APIMixin`
(`abstracts.py`), so every model gets both. Task 03 strips API behavior off models,
so both need a home in the manager layer. They are not equivalent in reach:

- **`get_first`** is universally implementable — fetch the list, take the first item
  (or `None`). Works on every endpoint, including non-paginated ones like
  `dhcphosts` (it just over-fetches when the server won't honor a limit).
- **`get_count`** is *not* universal. It needs server-side count (DRF pagination
  total). Most list endpoints support it; the meta `dhcphosts` endpoints do not
  (mreg-tools stubs it with a `mock_count()` returning 0). A `get_count` that
  silently full-fetches or returns a sentinel is a lie.

## Decision

- **`get_first` / `get_first_or_raise` go on the base `ResourceManager`** — every
  manager gets them for free.
- **`get_count` goes on a new opt-in capability mixin, `CountableResourceManager`**,
  mirroring `HistoryManager`. A manager gets `get_count` only by explicitly inheriting
  it. Managers combine capabilities by multiple inheritance, as `HostManager` already
  does (`NamedResourceManager` + `HistoryManager`).
- **`CountableResourceManager` inherits from `WriteResourceManager`**, not the bare
  `ResourceManager`. Countability rides on write capability because read-only ∩
  countable = ∅: the only read-only managers are meta endpoints, and those are exactly
  the ones that cannot count. This also smooths a planned future fold of
  `WriteResourceManager` into `ResourceManager` (with meta endpoints simply throwing on
  write attempts).

## Considered options

- **`get_count` in the base, opt *out* for the few non-countable endpoints** —
  rejected. Fewer lines, but it invites rot: a newly added non-paginated endpoint that
  forgets to opt out ships a lying `get_count`. Opt-in makes "has count" a deliberate,
  audited per-manager fact.

## Consequences

- DhcpHost managers (`client.dhcp_host_ipv4` / `ipv6` / `ipv6byipv4`) are bare,
  read-only `ResourceManager[T]`: they get `get_first` but **not** `get_count`.
- mreg-tools' `get-dhcphosts` command, which passes `count_func=...get_count`, must
  adapt (e.g. `len(get_list())` or caching). Migration-guide item.
