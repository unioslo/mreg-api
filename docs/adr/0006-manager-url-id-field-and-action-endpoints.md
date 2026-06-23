# URL identifier field lives on the manager; sub-resource actions call the client directly

The mreg API uses non-numeric fields as URL path identifiers for some resources
(`/api/v1/hosts/{name}`, `/api/v1/networks/{network}`). Previously, `Endpoint`
carried a dispatch table (`external_id_field()`, `requires_search_for_id()`) that
mapped each endpoint to its identifier field. This coupled URL-template metadata to
resource-identity metadata on the same object, and required updating a central
function in `endpoints.py` every time a new resource type with a non-numeric
identifier was added.

**Decision:** The URL identifier field is declared as `_url_identifier: ClassVar[str]`
on `ResourceManager` (default `"id"`). Concrete managers override it where the
resource uses a non-numeric path identifier (`"name"`, `"network"`, `"host"`).
`ResourceManager._fetch_by_field` and `_endpoint_with_id` read `self._url_identifier`
directly instead of delegating to the endpoint. `Endpoint.external_id_field()` and
`Endpoint.requires_search_for_id()` are no longer called by managers; they remain
temporarily while `APIMixin` (the old model-layer API mixin) is still being migrated
and will be removed once that migration is complete.

For sub-resource actions that use dedicated endpoints (e.g.
`Endpoint.HostGroupsAddHostGroups.with_params(group.name)`), manager methods call
`self._client.post` / `self._client.delete` directly with the action endpoint, then
call `self._refetch(obj)` to return a fresh model. The inherited `_create` / `delete`
helpers are for top-level resource CRUD only and do not apply to these operations.

**Consequences:**

- Every manager for a name/network/host-keyed resource must declare `_url_identifier`.
  Omitting it silently inherits `"id"` and breaks `get()` and `ensure_absent()`.
- Once `APIMixin` is fully removed, `Endpoint.external_id_field()` and
  `Endpoint.requires_search_for_id()` can be deleted, leaving `Endpoint` as pure URL
  templates with no resource-identity knowledge.
- Sub-resource action methods are not inherited from a base class; they are
  manager-specific and must be implemented per concrete manager.
