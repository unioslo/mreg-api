# mreg-api

Python client library for the mreg DNS/host registration API. Wraps the HTTP API
in typed Pydantic models and an explicit, instance-based client.

## Language

**Client**:
The `MregClient` instance. Owns HTTP/auth/cache/event state and the resource
managers. All API access goes through a client instance — there is no singleton.
_Avoid_: singleton, global client

**Manager**:
The client-bound object that owns all operations for one resource type
(`client.hosts`, `client.networks`). Holds the model type and a back-reference to
the client; fetches data and builds models from it. Models themselves are dumb data.
_Avoid_: repository, service, DAO

**Resource**:
An API-backed entity type a manager operates on (Host, Network, Zone, …). Conforms
to the `APIResource` Protocol by exposing an `endpoint()` classmethod; otherwise a
plain Pydantic model that carries data only, no I/O. URL construction lives on the
manager, not the model.
_Avoid_: entity, record (when meaning the type)

**External ID field**:
The field the server uses as the URL identifier for a resource, which is not always
`id`: Host→`name`, Network→`network`, HInfo/Location→`host`, most others→`id`.
Declared on `Endpoint` via `external_id_field()` and mirrored as the manager's
`_url_identifier` ClassVar, which the manager uses to construct endpoint URLs.
_Avoid_: lookup key, natural key

**Resolution** (a.k.a. "get by any means"):
Taking one free-text human identifier and figuring out _what_ it is before fetching.
Two flavors, both a **CLI concern, not a library one**: discriminating _identifier
kind_ (id vs IP vs MAC vs hostname vs CNAME for a Host) and discriminating _resource
type_ (is this name a Role or an Atom?). The library exposes explicit per-kind /
per-type getters (`client.roles.get_by_name`, `client.atoms.get_by_name`, …); the
CLI composes them in a fixed order.
_Avoid_: get_by_any_means (as a library method), smart get, magic get,
get_role_or_atom (as a library method)

**HostName**:
A marker (`NewType` over `str`) meaning "produced through our normalization":
lowercased, trailing dot stripped, shape-checked. **Not** necessarily fully
qualified — a bare label with no client domain is still a valid HostName. Constructed
only via `client.fqdn` or `parse_hostname`, never by hand; carries no runtime
validation of its own.
_Avoid_: fqdn (as a type), domain name

**Domain expansion** (`client.fqdn`):
Turning a bare label into an FQDN by appending the client's domain (`web` →
`web.example.com`). Client-scoped and opt-in: happens only when the client was given
a `domain`, only on input, only via `client.fqdn`. Models never expand.
_Avoid_: hostname_domain, default domain

**Event**:
A structured record (`client.events.record(...)`) the library emits for noteworthy
side-facts during an operation — e.g. "this name was a CNAME pointing elsewhere".
The library's neutral substitute for printing to the user; consumers (the CLI)
decide how to surface them.
_Avoid_: log line, output message
