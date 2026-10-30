"""Project exceptions."""

from typing import NoReturn


class DataLoadError(Exception):
    """Raised when a required Transfermarkt file is missing or invalid."""


class DataValidationError(Exception):
    """Raised when a table or frame fails a schema or quality check."""


class LeakageError(Exception):
    """Raised when a feature uses information after the season cutoff."""


class InvalidParameterError(Exception):
    """Raised when a caller passes an illegal argument."""


class NotBuiltError(NotImplementedError):
    """Raised by a signature stub until the owning member implements it."""

    def __init__(self, owner: str, name: str) -> None:
        self.owner = owner
        self.name = name
        super().__init__(
            f"{name} is a signature stub (owner: {owner}). "
            "Implement the body; do not change the signature without a team PR."
        )


def not_built(owner: str, name: str) -> NoReturn:
    """Fail fast from an unimplemented contract."""
    raise NotBuiltError(owner, name)
