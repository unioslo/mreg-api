# ~~`endpoint()` metadata lives on a concrete `APIResource` base~~ — SUPERSEDED

**Superseded by the model strip (sessions 12–13).** `APIResource` and `APIMixin` were
removed entirely. Models are now plain `MregModel` (frozen Pydantic base, no HTTP
methods, no `endpoint()` requirement). `endpoint()` lives on managers only, except
`HostList` which keeps it as pure metadata for a type-contract test.

The `T = TypeVar(bound=APIResource)` manager bound was retargeted to
`T = TypeVar(bound=MregModel)`. `events.py` `ObjectRef.new` was updated accordingly.
The original decision (concrete base in `models/abstracts.py`) was valid while the
strip was in progress but became moot once all models were pure data.
