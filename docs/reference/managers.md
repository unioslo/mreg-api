---
icon: lucide/boxes
---

# Managers

Every resource type is reachable as a manager attribute on an
[`MregClient`][mreg_api.client.MregClient] instance — for example `client.host` or
`client.network`. These managers are exposed on the client as attributes, so they never need to be instantiated directly.

All managers share the same core interface (`get`, `list`, `create`, `update`,
`delete`, …). That interface, and how to use it, is described in
[Working with resources](../guides/resources.md). This page is the per-manager API
reference.

!!! note "Inherited methods"
    Each manager inherits its base methods from internal base classes. Those base
    classes are not part of the public API, so the inherited methods are shown inline on
    each manager below rather than on separate pages.

## Hosts

::: mreg_api.managers.HostManager

::: mreg_api.managers.HostGroupManager

## IP addresses

::: mreg_api.managers.IPAddressManager

## Networks

::: mreg_api.managers.NetworkManager

::: mreg_api.managers.NetworkPolicyManager

::: mreg_api.managers.NetworkPolicyAttributeManager

::: mreg_api.managers.CommunityManager

## DNS records

::: mreg_api.managers.CNAMEManager

::: mreg_api.managers.MXManager

::: mreg_api.managers.TXTManager

::: mreg_api.managers.NAPTRManager

::: mreg_api.managers.SrvManager

::: mreg_api.managers.SSHFPManager

::: mreg_api.managers.PTROverrideManager

::: mreg_api.managers.HInfoManager

::: mreg_api.managers.LocationManager

::: mreg_api.managers.BacnetIDManager

## Host policy

::: mreg_api.managers.RoleManager

::: mreg_api.managers.AtomManager

::: mreg_api.managers.LabelManager

## Zones

::: mreg_api.managers.ZoneManager

::: mreg_api.managers.DelegationManager

::: mreg_api.managers.NameServerManager

## DHCP

::: mreg_api.managers.DhcpHostIPv4Manager

::: mreg_api.managers.DhcpHostIPv6Manager

::: mreg_api.managers.DhcpHostIPv6ByIPv4Manager

## Access control

::: mreg_api.managers.PermissionManager

## Server metadata and health

Reached through `client.meta` (e.g. `client.meta.version`, `client.meta.health`).

::: mreg_api.managers.MetaManagerNamespace

::: mreg_api.managers.ServerVersionManager

::: mreg_api.managers.ServerLibrariesManager

::: mreg_api.managers.UserInfoManager

::: mreg_api.managers.LDAPHealthManager

::: mreg_api.managers.HeartbeatHealthManager

::: mreg_api.managers.HealthManager
