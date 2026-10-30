"""Project exceptions."""


class DataLoadError(Exception):
    """Raised when a required Transfermarkt file is missing or invalid."""


class LeakageError(Exception):
    """Raised when a feature uses information after the season cutoff."""
