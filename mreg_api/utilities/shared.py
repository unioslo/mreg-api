"""Shared utilities for the mreg_cli package."""

from __future__ import annotations


def convert_wildcard_to_regex(param: str, arg: str, autoWildcards: bool = False) -> tuple[str, str]:
    """Convert wildcard filter "foo*bar*" to something DRF will understand.

    E.g. "foo*bar*" -> "?name__regex=$foo.*bar.*"

    :param param: The parameter to filter on
    :param arg: The argument to filter on
    :param autoWildcards: If True, add wildcards to the beginning and end of the argument if
                          they are not already present.
    """
    if "*" not in arg:
        if autoWildcards:
            arg = f"*{arg}*"
        else:
            return (param, arg)

    args = arg.split("*")
    args_len = len(args) - 1
    regex = ""
    for i, piece in enumerate(args):
        if i == 0 and piece:
            regex += f"^{piece}"
        elif i == args_len and piece:
            regex += f"{piece}$"
        elif piece:
            regex += f".*{piece}.*"
    #        if i == 0 and piece:
    #            parts.append(f'{param}__startswith={piece}')
    #        elif i == args_len and piece:
    #            parts.append(f'{param}__endswith={piece}')
    #        elif piece:
    #            parts.append(f'{param}__contains={piece}')

    if arg == "*":
        regex = "."

    return (f"{param}__regex", regex)
