"""Abstract models for the API."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from datetime import datetime
from typing import Any
from typing import Self

from pydantic import BaseModel
from pydantic import ConfigDict

from mreg_api.endpoints import Endpoint
from mreg_api.exceptions import EntityAlreadyExists
from mreg_api.exceptions import EntityNotFound
from mreg_api.exceptions import GetError
from mreg_api.exceptions import InternalError
from mreg_api.exceptions import PostError
from mreg_api.types import JsonMapping
from mreg_api.types import QueryParams


class FrozenModel(BaseModel):
    """Model for an immutable object."""

    def __setattr__(self, name: str, value: Any):
        """Raise an exception when trying to set an attribute."""
        raise AttributeError("Cannot set attribute on a frozen object")

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

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Ensure that the subclass inherits from BaseModel."""
        super().__init_subclass__(**kwargs)
        if BaseModel not in cls.__mro__:
            raise TypeError(f"{cls.__name__} must be applied on classes inheriting from BaseModel.")

    def id_for_endpoint(self) -> int | str:
        """Return the appropriate id for the object for its endpoint.

        Returns:
            The correct identifier for the endpoint.
        """
        field = self.endpoint().external_id_field()
        return getattr(self, field)

    @classmethod
    @abstractmethod
    def endpoint(cls) -> Endpoint:
        """Return the endpoint for the method."""
        raise NotImplementedError("You must define an endpoint.")

    @classmethod
    def get(cls, _id: int) -> Self | None:
        """Get an object.

        This function is at its base a wrapper around the get_by_id function,
        but it can be overridden to provide more specific functionality.

        Args:
            _id: The ID of the object.

        Returns:
            The object if found, None otherwise.
        """
        return cls.get_by_id(_id)

    @classmethod
    def get_list_by_id(cls, _id: int) -> list[Self]:
        """Get a list of objects by their ID.

        Args:
            _id: The ID of the object.

        Returns:
            A list of objects if found, an empty list otherwise.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        endpoint = cls.endpoint()
        if endpoint.requires_search_for_id():
            return cls.get_list_by_field("id", _id)

        data = MregClient().get(endpoint.with_id(_id), ok404=True)
        if not data:
            return []

        return [cls(**item) for item in data.json()]

    @classmethod
    def get_by_id(cls, _id: int) -> Self | None:
        """Get an object by its ID.

        Note that for Hosts, the ID is the name of the host.

        Args:
            _id: The ID of the object.

        Returns:
            The object if found, None otherwise.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        client = MregClient()

        endpoint = cls.endpoint()

        # Some endpoints do not use the ID field as the endpoint identifier,
        # and in these cases we need to search for the ID... Lovely.
        if endpoint.requires_search_for_id():
            data = client.get_item_by_key_value(cls.endpoint(), "id", str(_id))
        else:
            data = client.get(cls.endpoint().with_id(_id), ok404=True)
            if not data:
                return None
            data = data.json()

        if not data:
            return None

        return cls(**data)

    @classmethod
    def get_by_field(cls, field: str, value: str | int) -> Self | None:
        """Get an object by a field.

        Note that some endpoints do not use the ID field for lookups. We do some
        magic mapping via endpoint introspection to perform the following mapping for
        classes and their endpoint "id" fields:

          - Hosts -> name
          - Networks -> network

        This implies that doing a get_by_field("name", value) on Hosts will *not*
        result in a search, but a direct lookup at ../endpoint/name which is what
        the mreg server expects for Hosts (and similar for Network).

        Args:
            field: The field to search by.
            value: The value to search for.

        Returns:
            The object if found, None otherwise.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        client = MregClient()
        endpoint = cls.endpoint()

        if endpoint.requires_search_for_id() and field == endpoint.external_id_field():
            data = client.get(endpoint.with_id(value), ok404=True)
            if not data:
                return None
            data = data.json()
        else:
            data = client.get_item_by_key_value(cls.endpoint(), field, value, ok404=True)

        if not data:
            return None

        return cls(**data)

    @classmethod
    def get_by_field_or_raise(
        cls,
        field: str,
        value: str,
        exc_type: type[Exception] = EntityNotFound,
        exc_message: str | None = None,
    ) -> Self:
        """Get an object by a field and raise if not found.

        Used for cases where the object must exist for the operation to continue.

        Args:
            field: The field to search by.
            value: The value to search for.
            exc_type: The exception type to raise.
            exc_message: The exception message. Overrides the default message.

        Returns:
            The object if found.
        """
        obj = cls.get_by_field(field, value)
        if not obj:
            if not exc_message:
                exc_message = f"{cls.__name__} with {field} {value!r} not found."
            raise exc_type(exc_message)
        return obj

    @classmethod
    def get_by_field_and_raise(
        cls,
        field: str,
        value: str,
        exc_type: type[Exception] = EntityAlreadyExists,
        exc_message: str | None = None,
    ) -> None:
        """Get an object by a field and raise if found.

        Used for cases where the object must NOT exist for the operation to continue.

        Args:
            field: The field to search by.
            value: The value to search for.
            exc_type: The exception type to raise.
            exc_message: The exception message. Overrides the default message.

        Raises:
            Exception: If the object is found.
        """
        obj = cls.get_by_field(field, value)
        if obj:
            if not exc_message:
                exc_message = f"{cls.__name__} with {field} {value!r} already exists."
            raise exc_type(exc_message)
        return None

    @classmethod
    def get_list(
        cls,
        params: QueryParams | None = None,
        limit: int | None = None,
        endpoint: Endpoint | str | None = None,
    ) -> list[Self]:
        """Get a list of all objects.

        Optionally filtered by query parameters and limited by limit.

        Args:
            params: The query parameters to filter by.
            limit: The maximum number of hits to allow.
            endpoint: Override default model endpoint.

        Returns:
            A list of objects if found, an empty list otherwise.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        # Use default endpoint if omitted
        if endpoint is None:
            endpoint = cls.endpoint()

        return MregClient().get_typed(endpoint, list[cls], params=params, limit=limit)

    @classmethod
    def get_by_query(
        cls, query: QueryParams, ordering: str | None = None, limit: int | None = 500
    ) -> list[Self]:
        """Get a list of objects by a query.

        Args:
            query: The query to search by.
            ordering: The ordering to use when fetching the list.
            limit: The maximum number of hits to allow (default 500).

        Returns:
            A list of objects if found, an empty list otherwise.
        """
        if ordering:
            query["ordering"] = ordering
        return cls.get_list(params=query, limit=limit)

    @classmethod
    def get_list_by_field(
        cls, field: str, value: str | int, ordering: str | None = None, limit: int = 500
    ) -> list[Self]:
        """Get a list of objects by a field.

        Args:
            field: The field to search by.
            value: The value to search for.
            ordering: The ordering to use when fetching the list.
            limit: The maximum number of hits to allow (default 500).

        Returns:
            A list of objects if found, an empty list otherwise.
        """
        query: QueryParams = {field: value}
        return cls.get_by_query(query=query, ordering=ordering, limit=limit)

    @classmethod
    def get_by_query_unique_or_raise(
        cls,
        query: QueryParams,
        exc_type: type[Exception] = EntityNotFound,
        exc_message: str | None = None,
    ) -> Self:
        """Get an object by a query and raise if not found.

        Used for cases where the object must exist for the operation to continue.

        Args:
            query: The query to search by.
            exc_type: The exception type to raise.
            exc_message: The exception message. Overrides the default message.

        Returns:
            The object if found.
        """
        obj = cls.get_by_query_unique(query)
        if not obj:
            if not exc_message:
                exc_message = f"{cls.__name__} with query {query} not found."
            raise exc_type(exc_message)
        return obj

    @classmethod
    def get_by_query_unique_and_raise(
        cls,
        query: QueryParams,
        exc_type: type[Exception] = EntityAlreadyExists,
        exc_message: str | None = None,
    ) -> None:
        """Get an object by a query and raise if found.

        Used for cases where the object must NOT exist for the operation to continue.

        Args:
            query: The query to search by.
            exc_type: The exception type to raise.
            exc_message: The exception message. Overrides the default message.

        Raises:
            Exception: If the object is found.
        """
        obj = cls.get_by_query_unique(query)
        if obj:
            if not exc_message:
                exc_message = f"{cls.__name__} with query {query} already exists."
            raise exc_type(exc_message)
        return None

    @classmethod
    def get_by_query_unique(cls, data: QueryParams) -> Self | None:
        """Get an object with the given data.

        Args:
            data: The data to search for.

        Returns:
            The object if found, None otherwise.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        obj_dict = MregClient().get_list_unique(cls.endpoint(), params=data)
        if not obj_dict:
            return None
        return cls(**obj_dict)

    @classmethod
    def get_first(cls) -> Self | None:
        """Get the first object from the list.

        Returns:
            The first item from the list, or None if no items are found.
        """
        try:
            return cls.get_first_or_raise()
        except EntityNotFound:
            return None

    @classmethod
    def get_first_or_raise(cls) -> Self:
        """Get the first object from the list.

        Raises:
            EntityNotFound: If no items are found.

        Returns:
            The first item from the list.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        obj = MregClient().get_first(cls.endpoint())
        if not obj:
            raise EntityNotFound("No items found.")
        return cls(**obj)

    @classmethod
    def get_count(cls) -> int:
        """Get the count of items from the list.

        Returns:
            The count of items.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        return MregClient().get_count(cls.endpoint())

    def refetch(self) -> Self:
        """Fetch an updated version of the object.

        Note that the caller (self) of this method will remain unchanged and can contain
        outdated information. The returned object will be the updated version.

        Returns:
            The fetched object.
        """
        id_field = self.endpoint().external_id_field()
        identifier = getattr(self, id_field, None)
        if not identifier:
            raise InternalError(
                f"Could not get identifier for {self.__class__.__name__} via {id_field}."
            )

        lookup = None
        # If we have and ID field, a refetch based on that is cleaner as a rename
        # will change the name or whatever other insane field that are used for lookups...
        # Let this be a lesson to you all, don't use mutable fields as identifiers. :)
        if hasattr(self, "id"):
            lookup = getattr(self, "id", None)
            if not lookup:
                raise InternalError(f"Could not get ID for {self.__class__.__name__} via 'id'.")
        else:
            lookup = getattr(self, identifier)

        obj = self.__class__.get_by_id(lookup)
        if not obj:
            raise GetError(f"Could not refresh {self.__class__.__name__} with ID {identifier}.")
        return obj

    def patch(self, data: JsonMapping, *, params: QueryParams | None = None) -> Self:
        """Patch the object with the given values.

        Note:
            1. Depending on the endpoint, the server may not return the patched object.
            2. Patching with None may not clear the field if it isn't nullable (which few fields
               are). Odds are you want to pass an empty string instead.

        Args:
            data: The values to patch.
            params: Optional query parameters.

        Returns:
            The object refetched from the server.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        MregClient().patch(self.endpoint().with_id(self.id_for_endpoint()), json=data, params=params)
        new_object = self.refetch()

        return new_object

    def delete(self) -> bool:
        """Delete the object.

        Returns:
            True if the object was deleted, False otherwise.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        response = MregClient().delete(self.endpoint().with_id(self.id_for_endpoint()))

        if response and response.is_success:
            return True

        return False

    @classmethod
    def create(cls, data: JsonMapping, *, fetch_after_create: bool = True) -> Self | None:
        """Create the object.

        Note that several endpoints do not support location headers for created objects,
        so we can't fetch the object after creation. In these cases, we return None even
        if the object was created successfully...

        Args:
            data: The data to create the object with.
            fetch_after_create: Whether to fetch the object after creation.

        Raises:
            PostError: If the object could not be created.
            GetError: If the object could not be fetched after creation.

        Returns:
            The object if created and fetchable, None otherwise.
        """
        from mreg_api.client import MregClient  # noqa: PLC0415

        client = MregClient()

        response = client.post(cls.endpoint(), json=data)

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
            raise PostError(f"Failed to create {cls} with {data} @ {cls.endpoint()}.")

        return None
