# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenAI's error envelope.

Every OpenAI client parses failures out of ``{"error": {...}}``. FastAPI's
default ``HTTPException`` renders ``{"detail": ...}`` instead, which the
``openai`` SDK surfaces as a generic status error with no usable message --
so a 400 that says exactly which parameter is wrong arrives at the user as
"Error code: 400". Raising :class:`OpenAIError` instead keeps the message.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse

# OpenAI's ``error.type`` vocabulary. Clients branch on these strings (the SDK
# maps 429 + rate_limit_error onto its own RateLimitError, etc.), so they are
# part of the wire contract rather than free text.
INVALID_REQUEST_ERROR = "invalid_request_error"
AUTHENTICATION_ERROR = "authentication_error"
NOT_FOUND_ERROR = "not_found_error"
RATE_LIMIT_ERROR = "rate_limit_error"
SERVER_ERROR = "server_error"


class OpenAIError(Exception):
    """An error that serializes to OpenAI's envelope."""

    def __init__(
        self,
        status_code: int,
        message: str,
        *,
        err_type: str = INVALID_REQUEST_ERROR,
        param: str | None = None,
        code: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.err_type = err_type
        self.param = param
        self.code = code
        self.headers = headers or {}

    def body(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.err_type,
                "param": self.param,
                "code": self.code,
            }
        }


def unsupported_param(param: str, detail: str) -> OpenAIError:
    """400 for a parameter we deliberately refuse rather than silently drop.

    Accepting a parameter that changes what the model is supposed to produce
    and then ignoring it returns a plausible-looking 200 that is simply wrong
    -- the failure mode that costs the most to diagnose. Refusing is louder
    and cheaper.
    """
    return OpenAIError(400, f"{param!r} is not supported by this endpoint. {detail}", param=param)


async def openai_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, OpenAIError)
    return JSONResponse(status_code=exc.status_code, content=exc.body(), headers=exc.headers)


async def unhandled_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    """Anything uncaught still leaves as a well-formed OpenAI error.

    A bare traceback through FastAPI's default handler produces a 500 with an
    HTML/plain body that no OpenAI client can parse, which turns a backend
    hiccup into "connection error" at the caller.
    """
    err = OpenAIError(500, f"{type(exc).__name__}: {exc}", err_type=SERVER_ERROR)
    return JSONResponse(status_code=500, content=err.body())
