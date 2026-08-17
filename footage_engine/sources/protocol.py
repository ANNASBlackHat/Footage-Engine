"""Abstract URL Provider protocol for external databases or feeds."""

from typing import Any, Iterator, Protocol


class URLProvider(Protocol):
    """Protocol for external databases, spreadsheets, or third-party APIs supplying URLs."""

    def fetch_urls(self) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yields (url, metadata) tuples."""
        ...


class IterableURLProvider:
    """Convenience in-memory or generator-based URLProvider implementation."""

    def __init__(self, items: list[tuple[str, dict[str, Any]]]):
        self.items = items

    def fetch_urls(self) -> Iterator[tuple[str, dict[str, Any]]]:
        for url, meta in self.items:
            yield url, meta
