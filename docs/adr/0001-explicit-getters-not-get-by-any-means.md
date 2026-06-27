# Resolution ("get by any means") lives in the CLI, not the library

The library exposes only **explicit, per-kind getters** on managers — `get` (by
external id-field), `get_by_id`, `get_by_name`, `get_by_ip`, `get_by_mac`,
`get_by_network` — each taking one unambiguous identifier kind (addresses passed as
strings). It does **not** expose a single `get(identifier)` that guesses the kind.

The old `Host.get_by_any_means` / `Network.get_by_any_means` took one free-text arg
and tried kinds in a fixed order (Host: id→IP→MAC→name→CNAME; Network:
IP→network CIDR→numeric id). `get_by_any_means_or_raise` was the raising variant;
both are dropped. That order-of-resolution heuristic exists because a human types one
ambiguous string into the CLI — it is a CLI/UX concern. Baking it into the library
makes `get("42")` ambiguous (host named "42" vs host id 42; hostname that looks like
a MAC), which is unacceptable for a typed library API.

**Decision:** the resolution heuristic moves into the CLI, which calls the library's
explicit getters in the documented order. The library stays unambiguous; collisions
become impossible at the library layer.

**Consequences:**
- The CLI needs a documented mapping from the old `get_by_any_means` to the new
  explicit-getter chain. See `.dev/work/singleton-refactor/migration-guide.md`.
- **`inform_as_cname` / `inform_as_ptr` are dropped from the library entirely.** The
  two side-facts are different in kind:
  - **CNAME** is no longer a library event at all. Under explicit getters the CLI
    *composes* the fallback (`get_by_name` → `cnames.get` → `get_by_id(cname.host)`),
    so it already knows it followed a CNAME and decides itself whether to print. "I
    know it's a CNAME" = CLI calls `cnames.get` directly; "ambiguous identifier that
    resolved to a CNAME" = CLI's `resolve_host` chain prints. No flag, no event.
  - **PTR** is genuinely library-internal: `get_by_ip` silently falls back from the
    direct IP to a PTR override, so the caller can't see which matched. `get_by_ip`
    therefore **always** records the `RESOLUTION` event (no `inform_as_ptr` flag).
- Opt-in/out of *showing* events is a **CLI display-time filter**, not library state:
  the CLI drains `client.events` after an operation (scoped by `correlation_id` /
  `EventKind` / `EventLevel`) and decides what to surface in its command context.
