# ruff: noqa: INP001  (loaded by file path, not imported as a package)
"""Zensical macros. Loaded by the macros extension (see zensical.toml)."""

from __future__ import annotations

import inspect
from typing import Any

from mreg_api import exceptions
from mreg_api import managers


def define_env(env: Any) -> None:
    """Register documentation macros with the Zensical macros environment."""

    # NOTE: Claude recreated this from a .pyc file...
    #       but it begs the question, which branch did I define this in?
    #       Why did it never get merged...?
    @env.macro
    def history_managers() -> list[type[managers.HistoryManager[Any]]]:
        """Concrete managers that expose history() (subclass HistoryManager)."""
        return sorted(
            (
                cls
                for cls in vars(managers).values()
                if inspect.isclass(cls)
                and issubclass(cls, managers.HistoryManager)
                and cls is not managers.HistoryManager
                and not inspect.isabstract(cls)
            ),
            key=lambda c: c.__name__,
        )

    @env.macro
    def exception_tree() -> str:
        """Render the `mreg_api.exceptions` class hierarchy as a markdown list.

        Introspects every exception defined in the module (in definition order,
        skipping deprecated ones) and prints it as an indented markdown list
        rooted at `MregApiBaseError`.
        """
        root = exceptions.MregApiBaseError
        classes = [
            cls
            for cls in vars(exceptions).values()
            if inspect.isclass(cls)
            and cls.__module__ == exceptions.__name__
            and issubclass(cls, root)
            and not getattr(cls, "__deprecated__", None)
        ]

        children: dict[type, list[type]] = {cls: [] for cls in classes}
        for cls in classes:
            if cls is root:
                continue
            # Attach to the nearest ancestor that is part of the hierarchy.
            for parent in cls.__mro__[1:]:
                if parent in children:
                    children[parent].append(cls)
                    break

        lines: list[str] = []

        def render(node: type, depth: int) -> None:
            lines.append(f"{'    ' * depth}- [`{node.__name__}`][{node.__module__}.{node.__name__}]")
            for kid in children[node]:
                render(kid, depth + 1)

        render(root, 0)
        return "\n".join(lines)
