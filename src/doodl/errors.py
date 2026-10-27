"""Exception types. Each carries the HTTP status the API should answer with."""

from __future__ import annotations


class DooDLError(Exception):
    """Base class. The message is written for the person using the app."""

    status_code = 400
    code = "error"


class VideoError(DooDLError):
    status_code = 422
    code = "invalid_video"


class LimitError(DooDLError):
    status_code = 413
    code = "limit_exceeded"


class PromptError(DooDLError):
    status_code = 422
    code = "invalid_prompt"


class NotFoundError(DooDLError):
    status_code = 404
    code = "not_found"


class ConflictError(DooDLError):
    status_code = 409
    code = "conflict"


class BackendError(DooDLError):
    status_code = 503
    code = "backend_unavailable"


class AuthError(DooDLError):
    status_code = 401
    code = "unauthorized"
