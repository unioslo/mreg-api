---
icon: lucide/key-round
---

# Authentication

The client does not authenticate on construction and never reads credentials from the
environment or a token file on its own — authentication is always explicit. There are two
ways to authenticate.

## Username and password

[`login()`][mreg_api.client.MregClient.login] exchanges credentials for a token, stores
it on the client, and returns it. Reading credentials from the environment is a common
pattern:

```python
import os
from mreg_api import MregClient

client = MregClient(url="https://mreg.example.com", domain="example.com")
client.login(
    username=os.environ["MREG_USERNAME"],
    password=os.environ["MREG_PASSWORD"],
)
```

`login()` raises [`LoginFailedError`][mreg_api.exceptions.LoginFailedError] on a
connection failure or bad credentials.

## Existing token

!!! note

    Token auth is primarily relevant to persistent sessions for applications such as [mreg-cli](https://github.com/unioslo/mreg-cli), not for scripts or ephemeral sessions.

If you already have a token, set it directly with
[`set_token()`][mreg_api.client.MregClient.set_token] — no `login()` call needed:

```python
client = MregClient(url="https://mreg.example.com")
client.set_token(os.environ["MREG_TOKEN"])
```

## Managing the token

```python
client.get_token()    # current token, or None
client.test_auth()    # raises InvalidAuthTokenError if the token is invalid
client.unset_token()  # clear the token locally
client.logout()       # invalidate the token on the server
```

## Correlation IDs

To make requests easier to trace in server logs, attach a correlation ID that is sent
with every subsequent request:

```python
client.set_correlation_id("nightly-sync")
client.get_correlation_id()
```
