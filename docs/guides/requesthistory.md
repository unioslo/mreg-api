---
icon: lucide/timeline
---

# Request/Response History

The client records a request/response history of each interaction with the server. The history is a bounded deque, sized by the `history_size` argument to [`MregClient`][mreg_api.client.MregClient].
You can access the history via the `client.history` property, which returns a list of [`RequestRecord`][mreg_api.client.RequestRecord] instances, which contain the request and response objects for each interaction, as well as shortcuts for the request method, URL, data and json payloads, as well as response status code.

``` python
client = MregClient(...)
for record in client.history:
    print(record.request.method, record.request.url, record.status)
    assert record.status == record.response.status_code
    print("Request payload:", record.data or record.json)
```

## Get the last record

To get the record for the last response, you can use the `last` method:

``` python
record = client.history.last()
```

## Get all records

To get the record for the last response, you can use the `get` method:

``` python

records = client.history.get()
```

## Filtering

Each request record has a method, URL, status code, and optionally data and JSON payloads. You can filter the request history using the `get` method with the `status`, `method`, and `url` arguments:

``` python
all_200 = client.history.get(status=200)
all_posts = client.history.get(method="POST")
all_example = client.history.get(url="https://example.com/api")
```

### Combining filters

Filters can be combined to further narrow down the results:

``` python
all_200_posts = client.history.get(status=200, method="POST")
```

### Advanced filtering

You can also perform more advanced filtering by providing a callable to the `where` argument.
The callable is a predicate function that receives a `RequestRecord` instance and returns `True` if the record should be included in the results.

``` python
def is_post_to_example_and_not_201(record: RequestRecord) -> bool:
    return (
        record.method == "POST"
        and record.url.startswith("https://example.com/api")
        and record.status != 201
    )

filtered_records = client.history.get(where=is_post_to_example_and_not_201)
```

This can also be achieved with a lambda function, which is a more concise way to define one-time
filtering logic that doesn't need to be reused elsewhere:

``` python
filtered_records = client.history.get(
    where=lambda r: r.method == "POST" and r.url.startswith("https://example.com/api") and r.status != 201
)
```
