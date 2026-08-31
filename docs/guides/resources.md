---
icon: lucide/layers
---

# Resources

Every MREG resource type is an attribute on the client, and they all expose more or less the same interface. Learn it once and it applies everywhere:

``` python
client = MregClient(..., domain="example.com")

client.host      # manage hosts
client.network   # manage networks
client.cname     # manage CNAME records


host = client.host.get("myhost")
network = client.network.get("10.0.0.0/24")

networks = client.network.list()

cname = client.cname.create(host="myhost.example.com", name="alias.example.com")

# Identifier or model object can be used for updates
client.host.update(host, description="New description")
client.network.update("10.0.0.0/24", description="New description")

# Identifier or model object can be used for deletion
client.host.delete("myhost")
client.network.delete(network)
```

## Methods

| Method                                                                    | Purpose                                    |
| ------------------------------------------------------------------------- | ------------------------------------------ |
| [`get(ident)`][mreg_api.managers.ResourceManager.get]                     | Fetch by name, ID, or model object         |
| [`get_by_name(name)`][mreg_api.managers.NamedResourceManager.get_by_name] | Fetch by name (named resources only)       |
| [`list(**query)`][mreg_api.managers.ResourceManager.list]                 | List, optionally filtered                  |
| [`first(**query)`][mreg_api.managers.ResourceManager.first]               | Fetch the first match                      |
| [`count()`][mreg_api.managers.ResourceManager.count]                      | Count resources                            |
| [`create(**fields)`][mreg_api.managers.HostManager.create]                | Create a resource                          |
| [`update(target, **fields)`][mreg_api.managers.HostManager.update]        | Update a resource                          |
| [`delete(target)`][mreg_api.managers.WriteResourceManager.delete]         | Delete a resource                          |
| [`history(name)`][mreg_api.managers.HistoryManager.history]               | Change history (history-enabled resources) |

Not every resource supports every method — read-only resources have no `create`/`update`/
`delete`, and only some support `get_by_name` or `history`. The per-resource
[Managers reference](../reference/managers.md) shows exactly what each manager provides.

### Fetching

`get` interprets the argument type: a string (name), an int (ID), or an existing model
object (to refresh it). If a `domain` is set on the client, bare names are expanded to
FQDNs automatically.

``` python
host = client.host.get("myhost.example.com")
host = client.host.get("myhost")  # domain appended automatically if set
host = client.host.get(123)       # by ID
host = client.host.get(host)      # refresh a model object
```

By default a missing resource (404) raises a [`GetError`][mreg_api.exceptions.GetError]. Pass `required=False` to return `None` on 404 instead:

``` python
host = client.host.get("myhost", required=False)
if host is not None:
    ...
```

### Listing and filtering

`list` returns resources of that type. Keyword arguments map to model fields for
filtering and are forwarded to the API. `list` returns up to `limit` results (default
no limit). Fetching all hosts is _very_ slow, and should be avoided at all costs.

``` python
all_hosts = client.host.list()
hosts = client.host.list(name__startswith="test-", limit=500)
first = client.host.first(name__startswith="test-")
n = client.host.count(name__startswith="test-")
```

### Creating

`create` takes keyword arguments matching the model's fields. Every create method returns the created object on success. A [`PostError`][mreg_api.exceptions.PostError] is raised if the creation fails.

``` python
host = client.host.create(
    name="newhost.example.com",
    comment="Created via the API",
    fetch_after_create=True,
)
```

### Updating

`update` accepts a model object or an identifier (name, ID), plus the fields to change. The available fields vary depending on the resource. A [`PatchError`][mreg_api.exceptions.PatchError] is raised if the update fails.

``` python
client.host.update(
    host, # or name or ID
    comment="Updated comment",
    ttl=3600,
)
```

#### Refreshing after update

Updating a resource does not modify the client-side copy of the object. You can call `refresh()` to fetch the updated state of the object:

``` python
client.host.update(host, comment="Updated comment")
refreshed = client.host.refresh(host)
```

### Deleting

`delete` accepts a model object or an identifier. A [`DeleteError`][mreg_api.exceptions.DeleteError] is raised if the deletion fails.

``` python
client.host.delete(host)               # by object
client.host.delete("newhost.example.com")  # by name
client.host.delete(123)                    # by ID
```

## All managers

Every attribute below is a manager on the client. Follow the link for its full method
reference.

| Attribute                                                                      | Resource                  | Reference                                                                                                   |
| ------------------------------------------------------------------------------ | ------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `MregClient.host`                                                              | Hosts                     | [`HostManager`](../reference/managers.md#mreg_api.managers.HostManager)                                     |
| `MregClient.hostgroup`                                                         | Host groups               | [`HostGroupManager`](../reference/managers.md#mreg_api.managers.HostGroupManager)                           |
| `MregClient.ipaddress`                                                         | IP addresses              | [`IPAddressManager`](../reference/managers.md#mreg_api.managers.IPAddressManager)                           |
| `MregClient.network`                                                           | Networks                  | [`NetworkManager`](../reference/managers.md#mreg_api.managers.NetworkManager)                               |
| `MregClient.networkpolicy`<br />`MregClient.network.policy`                    | Network policies          | [`NetworkPolicyManager`](../reference/managers.md#mreg_api.managers.NetworkPolicyManager)                   |
| `MregClient.networkpolicyattribute`<br />`MregClient.network.policy.attribute` | Network policy attributes | [`NetworkPolicyAttributeManager`](../reference/managers.md#mreg_api.managers.NetworkPolicyAttributeManager) |
| `MregClient.community`<br />`MregClient.network.community`                     | Network communities       | [`NetworkCommunityManager`](../reference/managers.md#mreg_api.managers.CommunityManager)                    |
| `MregClient.cname`                                                             | CNAME records             | [`CNAMEManager`](../reference/managers.md#mreg_api.managers.CNAMEManager)                                   |
| `MregClient.mx`                                                                | MX records                | [`MXManager`](../reference/managers.md#mreg_api.managers.MXManager)                                         |
| `MregClient.txt`                                                               | TXT records               | [`TXTManager`](../reference/managers.md#mreg_api.managers.TXTManager)                                       |
| `MregClient.naptr`                                                             | NAPTR records             | [`NAPTRManager`](../reference/managers.md#mreg_api.managers.NAPTRManager)                                   |
| `MregClient.srv`                                                               | SRV records               | [`SrvManager`](../reference/managers.md#mreg_api.managers.SrvManager)                                       |
| `MregClient.sshfp`                                                             | SSHFP records             | [`SSHFPManager`](../reference/managers.md#mreg_api.managers.SSHFPManager)                                   |
| `MregClient.ptroverride`                                                       | PTR overrides             | [`PTROverrideManager`](../reference/managers.md#mreg_api.managers.PTROverrideManager)                       |
| `MregClient.hinfo`                                                             | HINFO records             | [`HInfoManager`](../reference/managers.md#mreg_api.managers.HInfoManager)                                   |
| `MregClient.location`                                                          | Locations                 | [`LocationManager`](../reference/managers.md#mreg_api.managers.LocationManager)                             |
| `MregClient.bacnetid`                                                          | BACnet IDs                | [`BacnetIDManager`](../reference/managers.md#mreg_api.managers.BacnetIDManager)                             |
| `MregClient.role`<br />`MregClient.policy.role`                                | Host policy roles         | [`RoleManager`](../reference/managers.md#mreg_api.managers.RoleManager)                                     |
| `MregClient.atom`<br />`MregClient.policy.atom`                                | Host policy atoms         | [`AtomManager`](../reference/managers.md#mreg_api.managers.AtomManager)                                     |
| `MregClient.label`<br />`MregClient.policy.label`                              | Labels                    | [`LabelManager`](../reference/managers.md#mreg_api.managers.LabelManager)                                   |
| `MregClient.zone`                                                              | Zones                     | [`ZoneManager`](../reference/managers.md#mreg_api.managers.ZoneManager)                                     |
| `MregClient.delegation`                                                        | Zone delegations          | [`DelegationManager`](../reference/managers.md#mreg_api.managers.DelegationManager)                         |
| `MregClient.nameserver`                                                        | Nameservers               | [`NameServerManager`](../reference/managers.md#mreg_api.managers.NameServerManager)                         |
| `MregClient.dhcphostipv4`                                                      | IPv4 DHCP hosts           | [`DhcpHostIPv4Manager`](../reference/managers.md#mreg_api.managers.DhcpHostIPv4Manager)                     |
| `MregClient.dhcphostipv6`                                                      | IPv6 DHCP hosts           | [`DhcpHostIPv6Manager`](../reference/managers.md#mreg_api.managers.DhcpHostIPv6Manager)                     |
| `MregClient.dhcphostipv6byipv4`                                                | IPv6-via-IPv4 DHCP hosts  | [`DhcpHostIPv6ByIPv4Manager`](../reference/managers.md#mreg_api.managers.DhcpHostIPv6ByIPv4Manager)         |
| `MregClient.permission`                                                        | Permissions               | [`PermissionManager`](../reference/managers.md#mreg_api.managers.PermissionManager)                         |
| `MregClient.meta`                                                              | Server metadata & health  | [`MetaManagerNamespace`](../reference/managers.md#mreg_api.managers.MetaManagerNamespace)                   |

### Nested sub-managers

A few managers are also composed logically on top of each other, so as to categorize related operations.

- `client.network` ⇾ `client.network.policy`, `client.network.community`
- `client.network.policy` ⇾ `client.network.policy.attribute`
- `client.meta` ⇾ `client.meta.version`, `client.meta.libraries`, `client.meta.userinfo`,
  `client.meta.ldap`, `client.meta.heartbeat`, `client.meta.health`
- `client.policy` ⇾ `client.policy.role`, `client.policy.atom`, `client.policy.label`
