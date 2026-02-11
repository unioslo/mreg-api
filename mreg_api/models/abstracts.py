"""Abstract models for the API."""

from __future__ import annotations

import functools
from abc import ABC
from abc import abstractmethod
from collections.abc import Mapping
from datetime import datetime
from typing import Callable
from typing import Self
from typing import TypeVar
from typing import cast
from weakref import ReferenceType
from weakref import ref

from pydantic import AliasChoices
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import PrivateAttr
from pydantic.fields import FieldInfo
from typing_extensions import override

from mreg_api.endpoints import Endpoint
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.exceptions import GetError
from mreg_api.exceptions import InternalError
from mreg_api.exceptions import PatchError
from mreg_api.exceptions import PostError
from mreg_api.types import ClientProtocol
from mreg_api.types import Json
from mreg_api.types import JsonMapping
from mreg_api.types import QueryParams


def get_field_aliases(field_info: FieldInfo) -> set[str]:
    """Get all aliases for a Pydantic field."""
    aliases: set[str] = set()

    if field_info.alias:
        aliases.add(field_info.alias)

    if field_info.validation_alias:
        if isinstance(field_info.validation_alias, str):
            aliases.add(field_info.validation_alias)
        elif isinstance(field_info.validation_alias, AliasChoices):
            for choice in field_info.validation_alias.choices:
                if isinstance(choice, str):
                    aliases.add(choice)
    return aliases


def get_model_aliases(model: BaseModel) -> dict[str, str]:
    """Get a mapping of aliases to field names for a Pydantic model.

    Includes field names, alias, and validation alias(es).
    """
    fields: dict[str, str] = {}
    for field_name, field_info in model.__class__.model_fields.items():
        aliases = get_field_aliases(field_info)
        if model.model_config.get("populate_by_name"):
            aliases.add(field_name)
        # Assign aliases to field name in mapping
        for alias in aliases:
            fields[alias] = field_name
    return fields


R = TypeVar("R")


def manager_only(func: Callable[..., R]) -> Callable[..., R]:
    """Allow classmethod calls only from a client manager."""

    @functools.wraps(func)
    def wrapper(*args: object, **kwargs: object) -> R:
        mutable_kwargs = cast(dict[str, object], kwargs)
        manager_call = bool(mutable_kwargs.pop("_manager", False))
        client_obj = args[1] if len(args) > 1 else mutable_kwargs.get("client")
        client = cast(ClientProtocol | None, client_obj) if client_obj is not None else None
        if client is None:
            if not manager_call:
                message = "".join(
                    [
                        "Direct classmethod usage is not supported. Use a client manager, ",
                        "e.g. client.host().get_by_id(...).",
                    ]
                )
                raise RuntimeError(message)
            return func(*args, **mutable_kwargs)

        depth = client._manager_call_depth
        if not manager_call and depth == 0:
            message = "".join(
                [
                    "Direct classmethod usage is not supported. Use a client manager, ",
                    "e.g. client.host().get_by_id(...).",
                ]
            )
            raise RuntimeError(message)

        client._manager_call_depth = depth + 1
        try:
            return func(*args, **mutable_kwargs)
        finally:
            client._manager_call_depth = depth

    return cast(Callable[..., R], wrapper)


def validate_patched_model(model: BaseModel, fields: dict[str, object]) -> None:
    """Validate that model fields were patched correctly."""
    aliases = get_model_aliases(model)

    validators = cast(
        dict[type, Callable[[object, object], bool]],
        {
            list: _validate_lists,
            dict: _validate_dicts,
        },
    )
    for key, value in fields.items():
        field_name = key
        if key in aliases:
            field_name = aliases[key]

        try:
            nval = cast(object, getattr(model, field_name))
        except AttributeError as e:
            raise PatchError(f"Could not get value for {field_name} in patched object.") from e

        # Ensure patched value is the one we tried to set
        validator = validators.get(type(nval), _validate_default)
        if not validator(nval, value):
            raise PatchError(
                f"Patch failure! Tried to set {key} to {value!r}, but server returned {nval!r}."
            )


def normalize_patch_fields(
    fields: Mapping[str, object] | None,
    field_kwargs: Mapping[str, object],
) -> dict[str, object]:
    """Normalize patch input from either a mapping or keyword arguments."""
    if fields is not None and field_kwargs:
        raise PatchError("Provide either `fields` or keyword patch arguments, not both.")

    normalized: dict[str, object] = dict(fields) if fields is not None else dict(field_kwargs)
    if not normalized:
        raise PatchError("No fields provided for patch operation.")

    return normalized


def _validate_lists(new: list[object], old: list[object]) -> bool:
    """Validate that two lists are equal."""
    if len(new) != len(old):
        return False
    return all(x in old for x in new)


def _validate_dicts(new: dict[str, object], old: dict[str, object]) -> bool:
    """Validate that two dictionaries are equal."""
    if len(new) != len(old):
        return False
    return all(old.get(k) == v for k, v in new.items())


def _validate_default(new: object, old: object) -> bool:
    """Validate that two values are equal."""
    return str(new) == str(old)


class MregBaseModel(BaseModel):
    """Base Pydantic model for this library."""


class FrozenModel(MregBaseModel):
    """Model for an immutable object."""

    _notes: list[str] = PrivateAttr(default_factory=list)
    """Internal notes for the object."""

    def add_note(self, note: str) -> None:
        """Add a note regarding the object."""
        self._notes.append(note)

    def get_notes(self) -> list[str]:
        """Get all notes regarding the object."""
        return self._notes

    @override
    def __setattr__(self, name: str, value: object):
        """Raise an exception when trying to set an attribute."""
        raise AttributeError("Cannot set attribute on a frozen object")

    @override
    def __delattr__(self, name: str):
        """Raise an exception when trying to delete an attribute."""
        raise AttributeError("Cannot delete attribute on a frozen object")

    model_config = ConfigDict(
        # Freeze model to make it immutable and thus hashable.
        frozen=True,
    )


class FrozenModelWithTimestamps(FrozenModel):
    """Model with created_at and updated_at fields."""

    created_at: datetime
    updated_at: datetime


class APIMixin(ABC):
    """A mixin for API-related methods."""

    _client: "ReferenceType[ClientProtocol] | None" = PrivateAttr(default=None)

    def _set_client(self, client: "ClientProtocol") -> None:
        object.__setattr__(self, "_client", ref(client))

    def _get_client(self) -> "ClientProtocol | None":
        if self._client is None:
            return None
        return self._client()

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Ensure that the subclass inherits from BaseModel."""
        super().__init_subclass__(**kwargs)
        if BaseModel not in cls.__mro__:
            raise TypeError(f"{cls.__name__} must be applied on classes inheriting from BaseModel.")

    def bind(self, client: "ClientProtocol") -> Self:
        """Bind this object to a client instance."""
        self._set_client(client)
        return self

    def _require_client(self) -> "ClientProtocol":
        client = self._get_client()
        if client is None:
            message = "{} instance is not bound to a client. {}".format(
                self.__class__.__name__,
                "Fetch it via a client manager or bind it explicitly.",
            )
            raise RuntimeError(message)
        return client

    def id_for_endpoint(self) -> int | str:
        """Return the appropriate id for the object for its endpoint.

        :returns: The correct identifier for the endpoint.
        """
        field = self.endpoint().external_id_field()
        identifier = cast(object, getattr(self, field))
        if isinstance(identifier, (int, str)):
            return identifier
        raise InternalError(
            "Identifier for endpoint field {field!r} on {name} must be int or str, got {type_name}."
            .format(
                field=field,
                name=self.__class__.__name__,
                type_name=type(identifier).__name__,
            )
        )

    @classmethod
    @abstractmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the method."""
        raise NotImplementedError("You must define an endpoint.")

    @classmethod
    @manager_only
    def get(cls, client: "ClientProtocol", _id: int) -> Self | None:
        """Get an object.

        This function is at its base a wrapper around the get_by_id function,
        but it can be overridden to provide more specific functionality.

        :param _id: The ID of the object.
        :returns: The object if found, None otherwise.
        """
        return cls.get_by_id(client, _id)

    @classmethod
    @manager_only
    def get_list_by_id(cls, client: "ClientProtocol", _id: int) -> list[Self]:
        """Get a list of objects by their ID.

        :param _id: The ID of the object.
        :returns: A list of objects if found, an empty list otherwise.
        """
        endpoint = cls.endpoint()
        if endpoint.requires_search_for_id():
            return cls.get_list_by_field(client, "id", _id)

        data = client.get(endpoint.with_id(_id), ok404=True)
        if not data:
            return []

        payload = cast(object, data.json())
        if not isinstance(payload, list):
            raise GetError(f"Unexpected response for {cls.__name__}: expected list, got {type(payload)}.")

        payload_items = cast(list[object], payload)
        ret: list[Self] = []
        for item in payload_items:
            if not isinstance(item, Mapping):
                raise GetError(
                    f"Unexpected list item for {cls.__name__}: expected mapping, got {type(item)}."
                )
            ret.append(cls(**cast(Mapping[str, object], item)))
        return client._bind_client(ret)

    @classmethod
    @manager_only
    def get_by_id(cls, client: "ClientProtocol", _id: int) -> Self | None:
        """Get an object by its ID.

        Note that for Hosts, the ID is the name of the host.

        :param _id: The ID of the object.
        :returns: The object if found, None otherwise.
        """
        endpoint = cls.endpoint()

        # Some endpoints do not use the ID field as the endpoint identifier,
        # and in these cases we need to search for the ID... Lovely.
        if endpoint.requires_search_for_id():
            data = client.get_item_by_key_value(cls.endpoint(), "id", str(_id))
        else:
            data = client.get(cls.endpoint().with_id(_id), ok404=True)
            if not data:
                return None
            payload = cast(object, data.json())
            if not isinstance(payload, Mapping):
                raise GetError(
                    f"Unexpected response for {cls.__name__}: expected mapping, got {type(payload)}."
                )
            data = cast(Mapping[str, object], payload)

        if not data:
            return None

        obj = cls(**data)
        return client._bind_client(obj)

    @classmethod
    @manager_only
    def get_by_field(cls, client: "ClientProtocol", field: str, value: str | int) -> Self | None:
        """Get an object by a field.

        Note that some endpoints do not use the ID field for lookups. We do some
        magic mapping via endpoint introspection to perform the following mapping for
        classes and their endpoint "id" fields:

          - Hosts -> name
          - Networks -> network

        This implies that doing a get_by_field("name", value) on Hosts will *not*
        result in a search, but a direct lookup at ../endpoint/name which is what
        the mreg server expects for Hosts (and similar for Network).

        :param field: The field to search by.
        :param value: The value to search for.

        :returns: The object if found, None otherwise.
        """
        endpoint = cls.endpoint()

        if endpoint.requires_search_for_id() and field == endpoint.external_id_field():
            data = client.get(endpoint.with_id(value), ok404=True)
            if not data:
                return None
            payload = cast(object, data.json())
            if not isinstance(payload, Mapping):
                raise GetError(
                    f"Unexpected response for {cls.__name__}: expected mapping, got {type(payload)}."
                )
            data = cast(Mapping[str, object], payload)
        else:
            data = client.get_item_by_key_value(cls.endpoint(), field, value, ok404=True)

        if not data:
            return None

        obj = cls(**data)
        return client._bind_client(obj)

    @classmethod
    @manager_only
    def get_by_field_or_raise(
        cls,
        client: "ClientProtocol",
        field: str,
        value: str,
        exc_type: type[Exception] = EntityNotFound,
        exc_message: str | None = None,
    ) -> Self:
        """Get an object by a field and raise if not found.

        Used for cases where the object must exist for the operation to continue.

        :param field: The field to search by.
        :param value: The value to search for.
        :param exc_type: The exception type to raise.
        :param exc_message: The exception message. Overrides the default message.

        :returns: The object if found.
        """
        obj = cls.get_by_field(client, field, value)
        if not obj:
            if not exc_message:
                exc_message = f"{cls.__name__} with {field} {value!r} not found."
            raise exc_type(exc_message)
        return obj

    @classmethod
    @manager_only
    def get_by_field_and_raise(
        cls,
        client: "ClientProtocol",
        field: str,
        value: str,
        exc_type: type[Exception] = EntityAlreadyExists,
        exc_message: str | None = None,
    ) -> None:
        """Get an object by a field and raise if found.

        Used for cases where the object must NOT exist for the operation to continue.

        :param field: The field to search by.
        :param value: The value to search for.
        :param exc_type: The exception type to raise.
        :param exc_message: The exception message. Overrides the default message.

        :raises Exception: If the object is found.
        """
        obj = cls.get_by_field(client, field, value)
        if obj:
            if not exc_message:
                exc_message = f"{cls.__name__} with {field} {value!r} already exists."
            raise exc_type(exc_message)
        return None

    @classmethod
    @manager_only
    def get_list(
        cls, client: "ClientProtocol", params: QueryParams | None = None, limit: int | None = None
    ) -> list[Self]:
        """Get a list of all objects.

        Optionally filtered by query parameters and limited by limit.

        :param params: The query parameters to filter by.
        :param limit: The maximum number of hits to allow (default 500)

        :returns: A list of objects if found, an empty list otherwise.
        """
        return client.get_typed(cls.endpoint(), list[cls], params=params, limit=limit)

    @classmethod
    @manager_only
    def get_by_query(
        cls,
        client: "ClientProtocol",
        query: QueryParams,
        ordering: str | None = None,
        limit: int | None = 500,
    ) -> list[Self]:
        """Get a list of objects by a query.

        :param query: The query to search by.
        :param ordering: The ordering to use when fetching the list.
        :param limit: The maximum number of hits to allow (default 500)

        :returns: A list of objects if found, an empty list otherwise.
        """
        if ordering:
            query["ordering"] = ordering
        return cls.get_list(client, params=query, limit=limit)

    @classmethod
    @manager_only
    def get_list_by_field(
        cls,
        client: "ClientProtocol",
        field: str,
        value: str | int,
        ordering: str | None = None,
        limit: int = 500,
    ) -> list[Self]:
        """Get a list of objects by a field.

        :param field: The field to search by.
        :param value: The value to search for.
        :param ordering: The ordering to use when fetching the list.
        :param limit: The maximum number of hits to allow (default 500)

        :returns: A list of objects if found, an empty list otherwise.
        """
        query: QueryParams = {field: value}
        return cls.get_by_query(client, query=query, ordering=ordering, limit=limit)

    @classmethod
    @manager_only
    def get_by_query_unique_or_raise(
        cls,
        client: "ClientProtocol",
        query: QueryParams,
        exc_type: type[Exception] = EntityNotFound,
        exc_message: str | None = None,
    ) -> Self:
        """Get an object by a query and raise if not found.

        Used for cases where the object must exist for the operation to continue.

        :param query: The query to search by.
        :param exc_type: The exception type to raise.
        :param exc_message: The exception message. Overrides the default message.

        :returns: The object if found.
        """
        obj = cls.get_by_query_unique(client, query)
        if not obj:
            if not exc_message:
                exc_message = f"{cls.__name__} with query {query} not found."
            raise exc_type(exc_message)
        return obj

    @classmethod
    @manager_only
    def get_by_query_unique_and_raise(
        cls,
        client: "ClientProtocol",
        query: QueryParams,
        exc_type: type[Exception] = EntityAlreadyExists,
        exc_message: str | None = None,
    ) -> None:
        """Get an object by a query and raise if found.

        Used for cases where the object must NOT exist for the operation to continue.

        :param query: The query to search by.
        :param exc_type: The exception type to raise.
        :param exc_message: The exception message. Overrides the default message.

        :raises Exception: If the object is found.
        """
        obj = cls.get_by_query_unique(client, query)
        if obj:
            if not exc_message:
                exc_message = f"{cls.__name__} with query {query} already exists."
            raise exc_type(exc_message)
        return None

    @classmethod
    @manager_only
    def get_by_query_unique(cls, client: "ClientProtocol", data: QueryParams) -> Self | None:
        """Get an object with the given data.

        :param data: The data to search for.
        :returns: The object if found, None otherwise.
        """
        obj_dict = client.get_list_unique(cls.endpoint(), params=data)
        if not obj_dict:
            return None
        obj = cls(**obj_dict)
        return client._bind_client(obj)

    @classmethod
    @manager_only
    def get_first(cls, client: "ClientProtocol") -> Self | None:
        """Get the first object from the list.

        :raises EntityNotFound: If no items are found.
        :returns: The first item from the list.
        """
        try:
            return cls.get_first_or_raise(client)
        except EntityNotFound:
            return None

    @classmethod
    @manager_only
    def get_first_or_raise(cls, client: "ClientProtocol") -> Self:
        """Get the first object from the list.

        :raises EntityNotFound: If no items are found.
        :returns: The first item from the list.
        """
        obj = client.get_first(cls.endpoint())
        if not obj:
            raise EntityNotFound("No items found.")
        result = cls(**obj)
        return client._bind_client(result)

    @classmethod
    @manager_only
    def get_count(cls, client: "ClientProtocol") -> int:
        """Get the count of items from the list.

        :returns: The count of items.
        """
        return client.get_count(cls.endpoint())

    def refetch(self) -> Self:
        """Fetch an updated version of the object.

        Note that the caller (self) of this method will remain unchanged and can contain
        outdated information. The returned object will be the updated version.

        :returns: The fetched object.
        """
        id_field = self.endpoint().external_id_field()
        identifier_value = cast(object, getattr(self, id_field, None))
        if identifier_value is None:
            raise InternalError(
                f"Could not get identifier for {self.__class__.__name__} via {id_field}."
            )

        lookup: int | str
        # If we have and ID field, a refetch based on that is cleaner as a rename
        # will change the name or whatever other insane field that are used for lookups...
        # Let this be a lesson to you all, don't use mutable fields as identifiers. :)
        model_cls = cast(type[BaseModel], self.__class__)
        if "id" in model_cls.model_fields:
            lookup_obj = cast(object, getattr(self, "id", None))
            if lookup_obj is None:
                raise InternalError(f"Could not get ID for {self.__class__.__name__} via 'id'.")
            if not isinstance(lookup_obj, int):
                raise InternalError(
                    "Could not get ID for {name} via 'id': expected int, got {type_name}.".format(
                        name=self.__class__.__name__,
                        type_name=type(lookup_obj).__name__,
                    )
                )
            lookup = lookup_obj
        else:
            if not isinstance(identifier_value, (int, str)):
                raise InternalError(
                    "Could not use identifier for {name}: expected int or str, got {type_name}."
                    .format(
                        name=self.__class__.__name__,
                        type_name=type(identifier_value).__name__,
                    )
                )
            lookup = identifier_value

        client = self._require_client()
        obj = self.__class__.get_by_id(client, lookup, _manager=True)
        if not obj:
            raise GetError(f"Could not refresh {self.__class__.__name__} with ID {identifier_value}.")

        return obj

    def patch(
        self,
        fields: Mapping[str, object] | None = None,
        validate: bool = True,
        **field_kwargs: object,
    ) -> Self:
        """Patch the object with the given values.

        Notes:
        -----
          1. Depending on the endpoint, the server may not return the patched object.
          2. Patching with None may not clear the field if it isn't nullable (which few fields
             are). Odds are you want to pass an empty string instead.

        :param fields: The values to patch.
        :param validate: Whether to validate the patched object.
        :param field_kwargs: Keyword patch arguments.
        :returns: The object refetched from the server.

        """
        patch_fields = normalize_patch_fields(fields, field_kwargs)
        patch_payload = cast(dict[str, Json], patch_fields)
        client = self._require_client()
        client.patch(self.endpoint().with_id(self.id_for_endpoint()), params=None, **patch_payload)
        new_object = self.refetch()

        if validate:
            # __init_subclass__ guarantees we inherit from BaseModel
            # but we can't signal this to the type checker, so we cast here.
            validate_patched_model(cast(BaseModel, new_object), patch_fields)  # pyright: ignore[reportInvalidCast] # we know what we are doing here (...?!)

        return new_object

    def patch_raw(self, fields: dict[str, object], validate: bool = True) -> Self:
        """Patch the object using an explicit dictionary payload."""
        return self.patch(fields=fields, validate=validate)

    def delete(self) -> bool:
        """Delete the object.

        :returns: True if the object was deleted, False otherwise.
        """
        client = self._require_client()
        response = client.delete(self.endpoint().with_id(self.id_for_endpoint()))

        if response and response.is_success:
            return True

        return False

    @classmethod
    @manager_only
    def create(
        cls, client: "ClientProtocol", params: JsonMapping, fetch_after_create: bool = True
    ) -> Self | None:
        """Create the object.

        Note that several endpoints do not support location headers for created objects,
        so we can't fetch the object after creation. In these cases, we return None even
        if the object was created successfully...

        :param params: The parameters to create the object with.
        :raises CreateError: If the object could not be created.
        :raises GetError: If the object could not be fetched after creation.
        :returns: The object if created and its fetchable, None otherwise.
        """
        response = client.post(cls.endpoint(), params=None, ok404=False, **params)

        if response and response.is_success:
            # NOTE: Headers.__getitem__ returns a str, while
            # Headers.get returns Any. Hence, check -> access.
            if "Location" in response.headers:
                location = response.headers["Location"]
                if fetch_after_create:
                    return client.get_typed(location, cls)
            # else:
            # Lots of endpoints don't give locations on creation,
            # so we can't fetch the object, but it's not an error...
            # Per se.
            # raise APIError("No location header in response.")

        else:
            raise PostError(f"Failed to create {cls} with {params} @ {cls.endpoint()}.")

        return None
