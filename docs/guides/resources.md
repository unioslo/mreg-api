---
icon: lucide/layers
---

# Working with resources

Every MREG resource type is an attribute on the client, and they all expose the same
manager interface. Learn it once and it applies everywhere:

```python
client.host      # manage hosts
client.network   # manage networks
client.cname     # manage CNAME records
# ...
```

## The manager interface

| Method | Purpose |
|---|---|
| `get(x)` | Fetch by name, ID, or model object |
| `get_by_name(name)` | Fetch by name (named resources only) |
| `list(**filters)` | List, optionally filtered |
| `first(**filters)` | Fetch the first match |
| `count()` | Count resources |
| `create(**fields)` | Create a resource |
| `update(target, **fields)` | Update a resource |
| `delete(target)` | Delete a resource |
| `history(name)` | Change history (history-enabled resources) |

Not every resource supports every method — read-only resources have no `create`/`update`/
`delete`, and only some support `get_by_name` or `history`. The per-resource
[Managers reference](../reference/managers.md) shows exactly what each manager provides.

### Fetching

`get` interprets the argument type: a string (name), an int (ID), or an existing model
object (to refresh it). If a `domain` is set on the client, bare names are expanded to
FQDNs automatically.

```python
host = client.host.get("myhost.example.com")
host = client.host.get("myhost")  # domain appended automatically if set
host = client.host.get(123)       # by ID
host = client.host.get(host)      # refresh a model object
```

By default a missing resource raises. Pass `required=False` to get `None` instead:

```python
host = client.host.get("myhost", required=False)
if host is not None:
    ...
```

### Listing and filtering

`list` returns resources of that type. Keyword arguments map to model fields for
filtering and are forwarded to the API. `list` returns up to `limit` results (default
500); pass `limit=None` to remove the cap.

```python
all_hosts = client.host.list()
hosts = client.host.list(name__startswith="test-")
first = client.host.first(name__startswith="test-")
n = client.host.count(name__startswith="test-")
```

### Creating

`create` takes keyword arguments matching the model's fields. Endpoints aren't guaranteed
to return the created resource, so the return type is `T | None`. Pass
`fetch_after_create=True` to fetch and return the newly created resource.

```python
new_host = client.host.create(
    name="newhost.example.com",
    comment="Created via the API",
    fetch_after_create=True,
)
```

### Updating

`update` accepts a model object or an identifier (name, ID), plus the fields to change.

```python
client.host.update(
    new_host or "newhost.example.com",
    comment="Updated comment",
    ttl=3600,
)
```

### Deleting

`delete` accepts a model object or an identifier.

```python
client.host.delete(new_host)               # by object
client.host.delete("newhost.example.com")  # by name
client.host.delete(123)                     # by ID
```

## All managers

Every attribute below is a manager on the client. Follow the link for its full method
reference.

| Attribute | Resource | Reference |
|---|---|---|
| `client.host` | Hosts | [`HostManager`](../reference/managers.md#mreg_api.managers.HostManager) |
| `client.hostgroup` | Host groups | [`HostGroupManager`](../reference/managers.md#mreg_api.managers.HostGroupManager) |
| `client.ipaddress` | IP addresses | [`IPAddressManager`](../reference/managers.md#mreg_api.managers.IPAddressManager) |
| `client.network` | Networks | [`NetworkManager`](../reference/managers.md#mreg_api.managers.NetworkManager) |
| `client.networkpolicy` | Network policies | [`NetworkPolicyManager`](../reference/managers.md#mreg_api.managers.NetworkPolicyManager) |
| `client.networkpolicyattribute` | Network policy attributes | [`NetworkPolicyAttributeManager`](../reference/managers.md#mreg_api.managers.NetworkPolicyAttributeManager) |
| `client.cname` | CNAME records | [`CNAMEManager`](../reference/managers.md#mreg_api.managers.CNAMEManager) |
| `client.mx` | MX records | [`MXManager`](../reference/managers.md#mreg_api.managers.MXManager) |
| `client.txt` | TXT records | [`TXTManager`](../reference/managers.md#mreg_api.managers.TXTManager) |
| `client.naptr` | NAPTR records | [`NAPTRManager`](../reference/managers.md#mreg_api.managers.NAPTRManager) |
| `client.srv` | SRV records | [`SrvManager`](../reference/managers.md#mreg_api.managers.SrvManager) |
| `client.sshfp` | SSHFP records | [`SSHFPManager`](../reference/managers.md#mreg_api.managers.SSHFPManager) |
| `client.ptroverride` | PTR overrides | [`PTROverrideManager`](../reference/managers.md#mreg_api.managers.PTROverrideManager) |
| `client.hinfo` | HINFO records | [`HInfoManager`](../reference/managers.md#mreg_api.managers.HInfoManager) |
| `client.location` | Locations | [`LocationManager`](../reference/managers.md#mreg_api.managers.LocationManager) |
| `client.bacnetid` | BACnet IDs | [`BacnetIDManager`](../reference/managers.md#mreg_api.managers.BacnetIDManager) |
| `client.role` | Host policy roles | [`RoleManager`](../reference/managers.md#mreg_api.managers.RoleManager) |
| `client.atom` | Host policy atoms | [`AtomManager`](../reference/managers.md#mreg_api.managers.AtomManager) |
| `client.label` | Labels | [`LabelManager`](../reference/managers.md#mreg_api.managers.LabelManager) |
| `client.zone` | Zones | [`ZoneManager`](../reference/managers.md#mreg_api.managers.ZoneManager) |
| `client.delegation` | Zone delegations | [`DelegationManager`](../reference/managers.md#mreg_api.managers.DelegationManager) |
| `client.nameserver` | Nameservers | [`NameServerManager`](../reference/managers.md#mreg_api.managers.NameServerManager) |
| `client.dhcphostipv4` | IPv4 DHCP hosts | [`DhcpHostIPv4Manager`](../reference/managers.md#mreg_api.managers.DhcpHostIPv4Manager) |
| `client.dhcphostipv6` | IPv6 DHCP hosts | [`DhcpHostIPv6Manager`](../reference/managers.md#mreg_api.managers.DhcpHostIPv6Manager) |
| `client.dhcphostipv6byipv4` | IPv6-via-IPv4 DHCP hosts | [`DhcpHostIPv6ByIPv4Manager`](../reference/managers.md#mreg_api.managers.DhcpHostIPv6ByIPv4Manager) |
| `client.permission` | Permissions | [`PermissionManager`](../reference/managers.md#mreg_api.managers.PermissionManager) |
| `client.meta` | Server metadata & health | [`MetaManagerNamespace`](../reference/managers.md#mreg_api.managers.MetaManagerNamespace) |

### Nested sub-managers

A few managers are reached through another manager rather than directly on the client:

- `client.network.policy` and `client.network.community`
- `client.networkpolicy.attribute`
- `client.meta.version`, `client.meta.libraries`, `client.meta.userinfo`,
  `client.meta.ldap`, `client.meta.heartbeat`, `client.meta.health`
