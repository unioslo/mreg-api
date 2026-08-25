---
icon: lucide/boxes
---

# Managers

This page describes the per-resource managers exposed on [`MregClient`][mreg_api.client.MregClient] instances.
A manager is a class that encapsulates the logic for interacting with a specific MREG resource type.
Depending on the resource, a manager may implement all CRUD operations (create, read, update, delete)
or just a subset, in addition to other methods for more granular resource-specific operations.
The CRUD interface is documented in [Resources](../guides/resources.md).This page is the per-manager API reference.

Some managers have additional methods, such as [`HostManager.get_by_ip`][mreg_api.managers.HostManager.get_by_ip] or [`PermissionManager.get_by_triplet`][mreg_api.managers.PermissionManager.get_by_triplet].

Other managers, such as the read-only meta-endpoint managers exposed via [`MregClient.meta`][mreg_api.client.MregClient.meta], have a more limited interface, exposing only a `get` method.

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

## Base Manager Classes

::: mreg_api.managers.ResourceManager
      options:
        inherited_members: false

::: mreg_api.managers.WriteResourceManager
      options:
        inherited_members: false

::: mreg_api.managers.NamedResourceManager
      options:
        inherited_members: false

::: mreg_api.managers.HistoryManager
      options:
        inherited_members: false

::: mreg_api.managers.GetManager
      options:
        inherited_members: false
