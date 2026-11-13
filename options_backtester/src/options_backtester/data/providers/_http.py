"""Shared HTTP plumbing for REST adapters: injectable transport, rate limiting, retries."""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ProviderError(RuntimeError):
    """A provider request failed after retries or returned an explicit error."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True, slots=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProviderError("provider returned invalid JSON") from exc

    def header(self, name: str) -> str | None:
        for key, value in self.headers.items():
            if key.lower() == name.lower():
                return str(value)
        return None


class Transport(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        data: bytes | None = None,
        timeout: float = 30.0,
    ) -> HttpResponse: ...


class UrllibTransport:
    """Standard-library transport. Returns non-2xx responses instead of raising."""

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        data: bytes | None = None,
        timeout: float = 30.0,
    ) -> HttpResponse:
        request = Request(url, data=data, headers=dict(headers or {}), method=method)
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310
                return HttpResponse(
                    response.status, dict(response.headers.items()), response.read()
                )
        except HTTPError as exc:
            return HttpResponse(
                exc.code, dict(exc.headers.items()) if exc.headers else {}, exc.read()
            )
        except URLError as exc:
            raise ConnectionError(f"cannot reach {url}: {exc.reason}") from exc


class RateLimiter:
    """Sliding-window limiter: at most ``max_calls`` per ``period`` seconds."""

    def __init__(
        self,
        max_calls: int,
        period: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_calls < 1 or period <= 0:
            raise ValueError("max_calls must be >= 1 and period positive")
        self.max_calls, self.period = max_calls, period
        self._clock, self._sleep = clock, sleep
        self._calls: deque[float] = deque()

    def acquire(self) -> None:
        now = self._clock()
        while self._calls and now - self._calls[0] >= self.period:
            self._calls.popleft()
        if len(self._calls) >= self.max_calls:
            self._sleep(self.period - (now - self._calls[0]))
            now = self._clock()
            while self._calls and now - self._calls[0] >= self.period:
                self._calls.popleft()
        self._calls.append(now)


RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass
class HttpClient:
    """GET/POST with rate limiting and exponential-backoff retries on transient failures."""

    transport: Transport = field(default_factory=UrllibTransport)
    limiter: RateLimiter | None = None
    max_attempts: int = 4
    backoff_seconds: float = 1.0
    timeout: float = 30.0
    sleep: Callable[[float], None] = time.sleep

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        data: bytes | None = None,
    ) -> HttpResponse:
        target = f"{url}?{urlencode(params)}" if params else url
        failure = "no attempt made"
        for attempt in range(self.max_attempts):
            if self.limiter is not None:
                self.limiter.acquire()
            delay = self.backoff_seconds * 2**attempt
            try:
                response = self.transport.request(
                    method, target, headers=headers, data=data, timeout=self.timeout
                )
            except (ConnectionError, TimeoutError) as exc:
                failure = str(exc)
            else:
                if response.status not in RETRY_STATUSES:
                    return response
                failure = f"HTTP {response.status}"
                retry_after = response.header("Retry-After")
                if retry_after and retry_after.replace(".", "", 1).isdigit():
                    delay = max(delay, float(retry_after))
            if attempt < self.max_attempts - 1:
                self.sleep(delay)
        raise ProviderError(f"{method} {url} failed after {self.max_attempts} attempts: {failure}")

    def get_json(
        self,
        url: str,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> tuple[Any, HttpResponse]:
        response = self.request("GET", url, params=params, headers=headers)
        if response.status >= 400:
            raise ProviderError(f"GET {url} returned HTTP {response.status}", response.status)
        return response.json(), response
