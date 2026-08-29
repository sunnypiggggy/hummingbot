"""Resilient HTTP reads whose retry policy can never replay writes."""

from __future__ import annotations

import time
from typing import Any, Mapping

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


def _error_text(error: BaseException | str) -> str:
    value = str(error).replace("\r", " ").replace("\n", " ").strip()
    return f"{type(error).__name__}: {value}" if isinstance(error, BaseException) else value


def read_retry_session(*, retry_total: int = 2) -> requests.Session:
    """Return a Session whose urllib3 retries are restricted to GET."""
    session = requests.Session()
    retry = Retry(
        total=max(0, int(retry_total)),
        connect=max(0, int(retry_total)),
        read=max(0, int(retry_total)),
        status=max(0, int(retry_total)),
        backoff_factor=0.1,
        allowed_methods=frozenset({"GET"}),
        status_forcelist=tuple(sorted(RETRYABLE_STATUS)),
        respect_retry_after_header=True,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


class GetOnlyReadClient:
    """Persistent read client with at most two retries after the first GET.

    The first retry stays in urllib3 so ``Retry-After`` and ordinary backoff
    are honored.  If it still fails, the final GET uses a brand-new no-retry
    Session.  Non-GET methods are issued once and are never replayed.
    """

    def __init__(
        self, base: str, *, auth: tuple[str, str] | None = None,
        pooled_retries: int = 1,
    ) -> None:
        self.base = str(base).rstrip("/")
        self.auth = auth
        self.pooled_retries = max(0, int(pooled_retries))
        self.session = self._new_session(retry_total=self.pooled_retries)
        self._read_retry_events: list[dict[str, Any]] = []

    def _new_session(self, *, retry_total: int) -> requests.Session:
        session = read_retry_session(retry_total=retry_total)
        if self.auth is not None:
            session.auth = self.auth
        return session

    @staticmethod
    def _retry_history(response: requests.Response) -> list[Any]:
        retries = getattr(getattr(response, "raw", None), "retries", None)
        return list(getattr(retries, "history", ()) or ())

    @staticmethod
    def _retryable_transport(error: BaseException) -> bool:
        return isinstance(error, (requests.exceptions.ConnectionError,
                                  requests.exceptions.Timeout))

    def _replace_session(self) -> None:
        previous = self.session
        self.session = self._new_session(retry_total=self.pooled_retries)
        previous.close()

    def consume_read_retry_events(self) -> list[dict[str, Any]]:
        events, self._read_retry_events = self._read_retry_events, []
        return events

    def _fresh_get(
        self, *, url: str, path: str, payload: Mapping[str, Any] | None,
        params: Mapping[str, Any] | None, timeout: float,
        first_error: BaseException | str,
    ) -> requests.Response:
        self.session.close()
        one_shot = self._new_session(retry_total=0)
        started = time.monotonic()
        try:
            response = one_shot.request(
                "GET", url, json=payload, params=params, timeout=timeout,
            )
        finally:
            one_shot.close()
            self.session = self._new_session(retry_total=self.pooled_retries)
        self._read_retry_events.append({
            "path": path, "attempts": 2, "pool_replaced": True,
            "duration_seconds": max(0.0, time.monotonic() - started),
            "reason": _error_text(first_error),
        })
        return response

    def request(
        self, method: str, path: str,
        payload: Mapping[str, Any] | None = None, *,
        params: Mapping[str, Any] | None = None, timeout: float = 30,
    ) -> Any:
        method = method.upper()
        url = path if str(path).startswith(("http://", "https://")) else f"{self.base}{path}"
        response: requests.Response
        try:
            response = self.session.request(
                method, url, json=payload, params=params, timeout=timeout,
            )
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as exc:
            if method != "GET" or not self._retryable_transport(exc):
                raise
            response = self._fresh_get(
                url=url, path=path, payload=payload, params=params,
                timeout=timeout, first_error=exc,
            )
        else:
            history = self._retry_history(response) if method == "GET" else []
            if method == "GET" and int(response.status_code) in RETRYABLE_STATUS:
                status_code = int(response.status_code)
                response.close()
                response = self._fresh_get(
                    url=url, path=path, payload=payload, params=params,
                    timeout=timeout,
                    first_error=f"HTTP {status_code} after pooled retry",
                )
            elif history:
                reasons = [_error_text(item.error) for item in history if item.error]
                self._read_retry_events.append({
                    "path": path, "attempts": len(history),
                    "pool_replaced": True, "duration_seconds": 0.0,
                    "reason": reasons[-1] if reasons else "transient GET retry",
                })
                self._replace_session()
        response.raise_for_status()
        return response.json() if response.content else {}
