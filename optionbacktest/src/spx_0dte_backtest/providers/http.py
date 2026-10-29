"""Injectable JSON HTTP transport used by network providers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ProviderConnectionError(RuntimeError):
    """A provider request failed or returned an explicit provider error."""


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    payload: Mapping[str, Any]
    headers: Mapping[str, str]


class JsonTransport(Protocol):
    def get_json(
        self,
        url: str,
        params: dict[str, object] | None,
        *,
        timeout: float,
    ) -> ProviderResponse: ...


class UrllibJsonTransport:
    """Small standard-library transport; provider parsing remains separately tested."""

    def get_json(
        self,
        url: str,
        params: dict[str, object] | None,
        *,
        timeout: float,
    ) -> ProviderResponse:
        target = url if not params else f"{url}?{urlencode(params)}"
        request = Request(target, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                payload = json.loads(response.read().decode("utf-8"))
                headers = dict(response.headers.items())
        except HTTPError as exc:
            raise ProviderConnectionError(f"provider returned HTTP {exc.code} for {url}") from exc
        except URLError as exc:
            raise ProviderConnectionError(
                f"cannot connect to provider at {url}: {exc.reason}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ProviderConnectionError(f"provider returned invalid JSON for {url}") from exc
        if not isinstance(payload, dict):
            raise ProviderConnectionError("provider JSON root must be an object")
        return ProviderResponse(payload=payload, headers=headers)
