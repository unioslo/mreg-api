# `update()` distinguishes unchanged / null / value via an internal sentinel

Manager `update()` methods need three distinct intents per field, two of which would
otherwise both be `None`:

| intent | wire result |
|---|---|
| leave unchanged | field omitted from PATCH body |
| set to null | `{"field": null}` |
| set to value | `{"field": value}` |

## Decision

For **genuinely-nullable** fields, the parameter defaults to an internal sentinel:

```python
class _UnsetType:
    def __repr__(self) -> str: return "UNSET"
UNSET = _UnsetType()   # module-private; NOT exported from the package

def update(self, ref, *, policy: int | None | _UnsetType = UNSET, ...) -> Network:
    data: dict[str, Any] = {}
    if policy is not UNSET:        # narrows to int | None
        data["policy"] = policy    # None here means "send null" (unset)
    ...
```

- `UNSET` (default) → omit → unchanged.
- `None` → send `null` → actively unset (replaces the old `unset_policy` /
  `unset_max_communities` model methods).
- a value → set it.

- The sentinel is **not exported**. Callers (the CLI) never reference it: they omit
  the arg to leave unchanged, pass `None` to unset, or pass a value to set. The CLI
  includes/omits kwargs based on the command being run.
- Applied **only to fields the server can actually null**. Non-nullable fields stay a
  plain `T | None` parameter where `None` simply means "unchanged" (ADR convention as
  in `HostManager.update`). The sentinel must be visually obvious in signatures so the
  two conventions don't get confused on the same model.
- `is not UNSET` narrows cleanly under basedpyright — no overloads needed.

## Why ADR

Cross-cutting (every `update` with a nullable field), surprising (`None` means
"unset", not "unchanged", on *some* params but not others), and a real trade-off
(sentinel machinery vs separate `unset_*` verbs). Pinning it once prevents 29 managers
inventing three different spellings.
