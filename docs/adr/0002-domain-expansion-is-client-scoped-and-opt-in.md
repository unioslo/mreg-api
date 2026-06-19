# Bare-name → FQDN expansion is client-scoped and opt-in; models don't expand

Bare hostnames (`web` → `web.example.com`) are expanded only by the client that
performs the operation, only when a `domain` was passed to the client constructor,
and only on **input**. Models never expand.

## Context

The old `HostName` Pydantic validator appended a domain read from a module-global
`ContextVar` (`fields.py:34`, default `"uio.no"`), written by every `MregClient` on
construct. With the singleton gone, that global remained: two clients with different
domains clobber each other, so `a.hosts.get("web")` could be validated against
client *b*'s domain. That is the exact shared-mutable-state bug the refactor exists
to remove — the headline "two clients, different domains, no shared state" goal in
the plan's verification section would fail.

`HostName` appears only as the **output** fields `Host.name` and `CNAME.name`
(server already returns FQDNs), so expansion is never needed on deserialization — it
is purely an input-normalization concern at the manager boundary.

## Decision

- Remove job 4 (domain append) and the `"uio.no"` default from `HostName`. No global
  domain state anywhere.
- Add `MregClient.fqdn(name) -> HostName`, the single shared expander: strip trailing
  dot → return; contains a dot → return as-is (assume qualified); else append
  `client.domain` iff set; else return bare. Managers route **every** hostname input
  through it (anything that is a `HostName` today); the CLI uses it for input
  convenience too.
- `HostName` becomes a runtime-erased `NewType('HostName', str)` — a *marker* meaning
  "produced through our normalization", not a Pydantic-validated type. It signifies a
  validated, normalized hostname but is **not** necessarily fully qualified (a bare
  name with no client domain is still a valid `HostName`).
- Normalization (lowercase, strip trailing dot, shape-validate, **no** expansion)
  lives in a module helper `parse_hostname(s) -> HostName`. `client.fqdn` is the
  primary constructor: it decides expansion, then delegates to `parse_hostname`.
  `parse_hostname` is also the construction path for tests and other non-client code.
  `HostName` is not constructed by hand otherwise.
- **API output is trusted, not transformed.** `name` fields deserialize the server's
  string as-is (no validator). Re-normalizing data from the source would signal
  distrust in it; mreg normalizes hostnames server-side.

## Consequences

- No default domain: a client built without `domain=` does no expansion;
  `client.fqdn("web") == "web"`.
- Expansion uses the fetching client's domain — correct under multiple clients.
- `HostName` carries no runtime validation; correctness depends on always
  constructing via `client.fqdn` / `parse_hostname`. A stray `HostName(x)` is an
  unchecked lie — code review must keep construction funnelled through the helpers.
- Read path no longer normalizes; relies on the server returning clean hostnames.
