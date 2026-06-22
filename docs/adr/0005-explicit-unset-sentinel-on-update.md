# `update()` distinguishes unchanged / null / value via an internal sentinel

Manager `update()` methods need three distinct intents per field, two of which would
otherwise both be `None`:

| intent | wire result |
|---|---|
| leave unchanged | field omitted from PATCH body |
| set to null | `{"field": null}` |
| set to value | `{"field": value}` |

## Decision

Every optional `update()` parameter defaults to an internal sentinel, `UNSET`:

```python
class _UnsetType:
    def __repr__(self) -> str: return "UNSET"
UNSET = _UnsetType()   # module-private; NOT exported from the package

def update(self, ref, *,
    name: str | _UnsetType = UNSET,        # non-nullable: no None in the union
    comment: str | _UnsetType = UNSET,     # non-nullable
    ttl: int | None | _UnsetType = UNSET,  # nullable: None allowed, means "send null"
) -> Host:
    data: dict[str, Any] = {}
    if name is not UNSET:        # narrows to str
        data["name"] = str(name)
    if comment is not UNSET:     # narrows to str
        data["comment"] = comment
    if ttl is not UNSET:         # narrows to int | None
        data["ttl"] = ttl        # None here → JSON null → server unsets ttl
    ...
```

One convention, applied uniformly:

- `UNSET` (default) → omit → **unchanged**. Always.
- `None` → send `null` → **actively unset** (replaces the old `unset_policy` /
  `unset_max_communities` model methods). Always.
- a value → set it.

The key rule: **`None` appears in a parameter's type only for genuinely-nullable fields** — fields the server can actually null. For those, `None` means "set null".
Non-nullable fields are `T | _UnsetType = UNSET` with no `None` in the union, so
passing `None` to them is a static type error rather than a silent no-op. There is no
second meaning of `None` anywhere; `UNSET` is the only "unchanged" signal in `update` methods.

- The sentinel is **not exported**. Callers (the CLI) never reference it: they omit the
  arg to leave unchanged, pass `None` to unset (where allowed), or pass a value to set.
  The CLI includes/omits kwargs based on the command being run.
- `is not UNSET` narrows cleanly under basedpyright — no overloads needed.
- Future: PEP 661 adds a builtin `sentinel()` in Python 3.15 that supersedes the
  hand-rolled `_UnsetType` (typed, picklable, no import). Revisit once the project is on
  3.15; the call sites (`is not UNSET`) and `None`-only-when-nullable rule are unchanged.

## Why ADR

Cross-cutting (every `update` with an optional field), surprising (`None` means "unset",
not "unchanged") and a real trade-off: a uniform sentinel default puts `UNSET` on
non-nullable params too — slightly noisier signatures — in exchange for `None` having a
single, unambiguous meaning across the whole API. The rejected alternative (sentinel only
on nullable fields, `None` = unchanged elsewhere) made `None` mean two different things on
the same model. Pinning one spelling prevents 29 managers inventing three.
