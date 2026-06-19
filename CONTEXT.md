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
An API-backed entity type a manager operates on (Host, Network, Zone, …). Inherits
the concrete `APIResource` base, which carries the endpoint metadata (`endpoint()`,
`id_for_endpoint()`, `endpoint_with_id`) as pure data; carries no I/O.
_Avoid_: entity, record (when meaning the type)

**External ID field**:
The field the server uses as the URL identifier for a resource, which is not always
`id`: Host→`name`, Network→`network`, Hinfo/Loc→`host`, most others→`id`. Derived
from the endpoint (`Endpoint.external_id_field()`).
_Avoid_: lookup key, natural key

**Resolution** (a.k.a. "get by any means"):
Taking one free-text human identifier and figuring out *what kind* it is (id vs IP
vs MAC vs hostname vs CNAME) before fetching. A **CLI concern**, not a library one:
the library exposes explicit per-kind getters (`get_by_id`, `get_by_ip`, …) and the
CLI composes them in a fixed order.
_Avoid_: get_by_any_means (as a library method), smart get, magic get

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
