# Manager verb conventions (get / ensure_absent / history / delete)

The shared verbs on `ResourceManager` follow one consistent contract, set once here
because ~29 managers and the whole CLI migration depend on it.

## get — collapsed, with `required`

Replaces the old `get_x` / `get_x_or_raise` trio with one method:

```python
@overload
def get(self, ident, *, required: Literal[True]) -> T: ...
@overload
def get(self, ident, *, required: Literal[False] = False) -> T | None: ...
```

- `required=False` (default) → `T | None`.
- `required=True` → `T`, raises `EntityNotFound` if missing.
- Kwarg is `required`, not `should_exist` (reads cleaner; `should_exist` was awkward).
- Same shape for `get_by_name` on `NamedResourceManager`, and for the explicit Host
  getters (`get_by_id`/`get_by_ip`/…), see ADR-0001.

## ensure_absent — the "must not exist" guard

The old `should_exist=False` (raise if present, return `None`) is **not** an overload
of `get` — a return-typed-`None` value is dead and tempts misuse. It is its own verb:

```python
def ensure_absent(self, ident) -> None:   # raises EntityAlreadyExists if present
```

Keeps the CLI-useful "fail if it already exists" check (e.g. host add) explicit.

## history — returns `[]` on no history

`HistoryCapableManager.history(name)` returns `list[HistoryItem]`, **empty when there
is no history**. No-history is a valid state, not a not-found error. (The old
`HistoryItem.get` raised `EntityNotFound("No history found")`.)

> CLI impact: callers that relied on the raise must check for `[]` themselves.

## delete — returns `None`

`delete()` returns `None`. The client's `delete()` raises `DeleteError` on failure,
so success is implied by returning normally. The old `bool` was a lie — it was really
`NoReturn | Literal[True]`, yet ~14 CLI sites do `if x.delete():` / `if not
x.delete():` assuming a real boolean. Those branches are silently inverted by this
change and **must** be rewritten (see migration guide).

## Why ADR

Surprising (`delete` no longer truthy; `history` no longer raises), hard to reverse
(29 managers + every CLI call site bind to these signatures), and each is a real
trade-off (honest return vs source-compat with the old API). Setting them once stops
the pattern being copied inconsistently across managers.
