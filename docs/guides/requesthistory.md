---
icon: lucide/timeline
---

# Request/Response History

The client records a request/response history of each interaction with the server. The history is a bounded deque, sized by the `history_size` argument to [`MregClient`][mreg_api.client.MregClient]. You can access the history via the `client.history` property, which returns a list of [`RequestRecord`][mreg_api.client.RequestRecord] instances, which contain the request and response objects for each interaction, as well as shortcuts for the request method, URL, data and json payloads, as well as response status code.

```python
client = MregClient(...)
for record in client.get_client_history():
    print(record.request.method, record.request.url, record.status)
    assert record.status == record.response.status_code
    print("Request payload:", record.data or record.json)
```
