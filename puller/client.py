"""Thin client for the Massive REST API (https://massive.com/docs).

Massive serves the endpoints this pilot needs under three shapes:

    /v2/aggs/ticker/{ticker}/range/...     OHLC bars, built from *trades*
    /v3/reference/options/contracts        the contract index, with as_of
    /v3/quotes/{optionsTicker}             tick-level NBBO, nanosecond stamps

There is no quote-bar endpoint. See quote_bars() in pilot.py for what we do
about that; this module only speaks HTTP.

The key lives in MASSIVE_API_KEY, read from the environment or from a .env
file next to the repo root. Get one at https://massive.com/dashboard/keys.
"""

import os
import random
import threading
import time
from datetime import date, datetime, time as dtime
from zoneinfo import ZoneInfo

import requests

BASE = "https://api.massive.com"
MARKET_TZ = ZoneInfo("America/New_York")

# Regular session. SPY options keep quoting until 16:15, which is why the
# quote window and the bar grid are allowed to end at different times.
SESSION_OPEN = dtime(9, 30)
SESSION_CLOSE = dtime(16, 0)
OPTION_QUOTE_CLOSE = dtime(16, 15)

SIGNUP = "https://massive.com/dashboard/signup"
KEYPAGE = "https://massive.com/dashboard/keys"


class MassiveError(RuntimeError):
    pass


class AuthError(MassiveError):
    pass


def load_dotenv(path=None):
    """Minimal .env reader. Does not overwrite variables already exported."""
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            os.environ.setdefault(k, v)


class RateLimiter:
    """Token-bucket-ish spacer. rpm<=0 disables it."""

    def __init__(self, rpm):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        if not self.interval:
            return
        with self._lock:
            now = time.monotonic()
            sleep = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if sleep:
            time.sleep(sleep)


class Massive:
    def __init__(self, api_key=None, rpm=None, max_retries=5, timeout=30,
                 session=None, pool=64):
        load_dotenv()
        self.api_key = api_key or os.environ.get("MASSIVE_API_KEY", "").strip()
        if not self.api_key:
            raise AuthError(
                "MASSIVE_API_KEY is not set.\n"
                f"  Sign up:  {SIGNUP}\n"
                f"  Get key:  {KEYPAGE}\n"
                "  Then put it in .env as  MASSIVE_API_KEY=your_key_here"
            )
        if rpm is None:
            rpm = float(os.environ.get("MASSIVE_RPM", "0") or 0)
        self.limiter = RateLimiter(rpm)
        self.max_retries = max_retries
        self.timeout = timeout
        self.s = session or requests.Session()
        # requests defaults to a 10-connection pool. Above that, workers queue on
        # each other and connections get torn down and rebuilt, which shows up as
        # multi-second stalls that look like server latency but are self-inflicted.
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=pool, pool_maxsize=pool, max_retries=0)
        self.s.mount("https://", adapter)
        self.s.mount("http://", adapter)
        self.s.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "User-Agent": "paper1-pilot/0.1",
        })
        self.calls = 0

    # ---- transport ------------------------------------------------------

    def get(self, url, params=None):
        """GET with retry on 429/5xx. `url` may be a path or a full next_url."""
        if not url.startswith("http"):
            url = BASE + url
        last = None
        for attempt in range(self.max_retries + 1):
            self.limiter.wait()
            try:
                r = self.s.get(url, params=params, timeout=self.timeout)
                self.calls += 1
            except requests.RequestException as exc:
                last = exc
                if attempt == self.max_retries:
                    raise MassiveError(f"{url}: {exc}") from exc
                time.sleep(min(30.0, 2 ** attempt) + random.random())
                continue

            if r.status_code == 401 or r.status_code == 403:
                raise AuthError(
                    f"{r.status_code} from {url}. The key was rejected or the plan does "
                    f"not cover this endpoint.\n  Keys: {KEYPAGE}\n  Body: {r.text[:300]}"
                )
            if r.status_code == 429:
                delay = float(r.headers.get("Retry-After", 0) or 0) or min(60.0, 2 ** attempt)
                time.sleep(delay + random.random())
                last = MassiveError("429 rate limited")
                continue
            if r.status_code >= 500:
                last = MassiveError(f"{r.status_code} {r.text[:200]}")
                if attempt == self.max_retries:
                    raise last
                time.sleep(min(30.0, 2 ** attempt) + random.random())
                continue
            if r.status_code != 200:
                raise MassiveError(f"{r.status_code} from {url}: {r.text[:300]}")

            try:
                return r.json()
            except ValueError as exc:
                raise MassiveError(f"non-JSON from {url}: {r.text[:200]}") from exc
        raise MassiveError(f"exhausted retries on {url}: {last}")

    def paginate(self, path, params=None, max_pages=1000):
        """Yield `results` rows across next_url pages. Returns pages consumed."""
        url, page = path, 0
        while url and page < max_pages:
            payload = self.get(url, params=params if page == 0 else None)
            for row in payload.get("results") or []:
                yield row
            url = payload.get("next_url")
            page += 1
        if url:
            raise MassiveError(f"{path}: hit max_pages={max_pages} with more data pending")

    # ---- endpoints ------------------------------------------------------

    def stock_bars(self, ticker, day, multiplier=5, timespan="minute", limit=50000):
        """/v2/aggs — OHLCV built from qualifying trades. Bar `t` is the window START."""
        path = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{day}/{day}"
        payload = self.get(path, {"adjusted": "true", "sort": "asc", "limit": limit})
        return payload.get("results") or []

    def option_contracts(self, underlying, expiration_date, as_of=None, expired=None,
                         contract_type=None, limit=1000):
        """/v3/reference/options/contracts.

        A same-day-expiry contract queried after the fact is *expired*, so the
        default (expired=false) returns nothing. We try expired=true first and
        fall back, rather than assuming which side of the flag the date is on.
        """
        params = {
            "underlying_ticker": underlying,
            "expiration_date": expiration_date,
            "limit": limit,
            "sort": "strike_price",
            "order": "asc",
        }
        if as_of:
            params["as_of"] = as_of
        if contract_type:
            params["contract_type"] = contract_type

        tries = [expired] if expired is not None else [True, False]
        for flag in tries:
            p = dict(params, expired="true" if flag else "false")
            rows = list(self.paginate("/v3/reference/options/contracts", p))
            if rows:
                return rows, flag
        return [], tries[0]

    def option_quotes(self, options_ticker, start_ns, end_ns, limit=50000, max_pages=400):
        """/v3/quotes — every NBBO tick in [start_ns, end_ns]. Ascending."""
        params = {
            "timestamp.gte": int(start_ns),
            "timestamp.lte": int(end_ns),
            "order": "asc",
            "sort": "timestamp",
            "limit": limit,
        }
        return self.paginate(f"/v3/quotes/{options_ticker}", params, max_pages=max_pages)

    def last_quote_at_or_before(self, options_ticker, ts_ns, floor_ns=None):
        """The single most recent NBBO at or before ts_ns.

        One call instead of a full tick pull. floor_ns bounds how far back we
        will reach, so a contract that has not quoted all session returns None
        rather than a stale quote from the open.
        """
        params = {
            "timestamp.lte": int(ts_ns),
            "order": "desc",
            "sort": "timestamp",
            "limit": 1,
        }
        if floor_ns is not None:
            params["timestamp.gte"] = int(floor_ns)
        payload = self.get(f"/v3/quotes/{options_ticker}", params)
        rows = payload.get("results") or []
        return rows[0] if rows else None


# ---- small helpers the pilot shares --------------------------------------

def occ_ticker(underlying, expiry, right, strike):
    """O:SPY251015C00580000 — strike is 8 digits of strike*1000."""
    if isinstance(expiry, str):
        expiry = date.fromisoformat(expiry)
    return (f"O:{underlying}{expiry:%y%m%d}{right.upper()[0]}"
            f"{int(round(strike * 1000)):08d}")


def et(ts_ms=None, ts_ns=None):
    """Unix ms or ns -> tz-aware New York datetime."""
    secs = (ts_ms / 1e3) if ts_ms is not None else (ts_ns / 1e9)
    return datetime.fromtimestamp(secs, tz=ZoneInfo("UTC")).astimezone(MARKET_TZ)


def ns(day, clock):
    """A New York wall-clock time on `day` -> Unix nanoseconds."""
    if isinstance(day, str):
        day = date.fromisoformat(day)
    dt = datetime.combine(day, clock, tzinfo=MARKET_TZ)
    return int(dt.timestamp() * 1_000_000_000)
