"""Structured API errors."""

from __future__ import annotations


class ApiError(Exception):
    def __init__(self, message: str, *, status_code: int = 400, source: str = "api") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.source = source

