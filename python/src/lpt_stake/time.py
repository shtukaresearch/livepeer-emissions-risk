"""Round ↔ datetime conversion utilities.

Provides two kinds of conversion:

**Estimated** (linear approximation) — for reindexing data, setting simulation
horizons, and labelling plots.  These assume rounds are exactly
``SECONDS_PER_ROUND`` apart, anchored to the reference point in
``constants.py``.  Function names include *estimate* to signal that they are
approximations.

**Real** (API-backed) — ``fetch_block_at_timestamp`` queries the Etherscan API
for the actual Arbitrum block number at a given wall-clock time.  This is used
by the data-fetching pipeline and is authoritative, not an estimate.
"""

from datetime import datetime, timedelta, timezone

import requests

from lpt_stake.constants import (
    REFERENCE_DATETIME,
    REFERENCE_ROUND,
    SECONDS_PER_ROUND,
)

# Etherscan v2 API endpoint (chain ID 42161 = Arbitrum One)
_ETHERSCAN_API_URL = "https://api.etherscan.io/v2/api?chainid=42161"


# ---------------------------------------------------------------------------
# Estimated conversions (linear approximation)
# ---------------------------------------------------------------------------


def estimate_round_number(dt: datetime) -> int:
    """Estimate the Livepeer round number at a given datetime.

    Uses a linear approximation anchored to ``REFERENCE_ROUND`` /
    ``REFERENCE_DATETIME``.  The result is floored to an integer, matching
    the on-chain contract's integer division.

    Parameters
    ----------
    dt
        A timezone-aware datetime.

    Returns
    -------
    int
        Estimated round number.

    Raises
    ------
    ValueError
        If *dt* is naive (has no timezone).
    """
    if dt.tzinfo is None:
        raise ValueError("dt must be timezone-aware")
    delta_seconds = (dt - REFERENCE_DATETIME).total_seconds()
    return int(REFERENCE_ROUND + delta_seconds // SECONDS_PER_ROUND)


def estimate_datetime(round_num: int) -> datetime:
    """Estimate the UTC datetime at which a given round starts.

    Uses a linear approximation anchored to ``REFERENCE_ROUND`` /
    ``REFERENCE_DATETIME``.

    Parameters
    ----------
    round_num
        Round number.

    Returns
    -------
    datetime
        Timezone-aware UTC datetime.
    """
    delta_seconds = (round_num - REFERENCE_ROUND) * SECONDS_PER_ROUND
    return REFERENCE_DATETIME + timedelta(seconds=delta_seconds)


def estimate_round_count(delta: timedelta) -> int:
    """Convert a timedelta to an estimated number of complete rounds.

    Pure arithmetic using ``SECONDS_PER_ROUND``; no reference point needed.
    The result is floored to count only complete rounds.

    Parameters
    ----------
    delta
        A timedelta to convert.

    Returns
    -------
    int
        Estimated number of complete rounds.
    """
    return int(delta.total_seconds() // SECONDS_PER_ROUND)


def estimate_timedelta(rounds: int) -> timedelta:
    """Convert a number of rounds to an estimated timedelta.

    Pure arithmetic using ``SECONDS_PER_ROUND``; no reference point needed.

    Parameters
    ----------
    rounds
        Number of rounds.

    Returns
    -------
    timedelta
        Estimated wall-clock duration.
    """
    return timedelta(seconds=rounds * SECONDS_PER_ROUND)


# ---------------------------------------------------------------------------
# Real timestamp → block number conversion (Etherscan API)
# ---------------------------------------------------------------------------


def fetch_block_at_timestamp(
    timestamp: datetime,
    api_key: str,
    *,
    closest: str = "before",
    timeout: float = 10,
) -> int:
    """Query Etherscan for the Arbitrum block number at a given timestamp.

    This is an authoritative lookup, not an estimate.  It requires network
    access and a valid Etherscan API key.

    Parameters
    ----------
    timestamp
        A timezone-aware datetime.
    api_key
        Etherscan API key.
    closest
        ``"before"`` (default) or ``"after"`` — whether to return the last
        block before or the first block after *timestamp*.
    timeout
        HTTP request timeout in seconds.

    Returns
    -------
    int
        Arbitrum block number.

    Raises
    ------
    ValueError
        If *timestamp* is naive or the API returns an error.
    """
    if timestamp.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")

    params = {
        "module": "block",
        "action": "getblocknobytime",
        "timestamp": int(timestamp.timestamp()),
        "closest": closest,
        "apikey": api_key,
    }
    response = requests.get(_ETHERSCAN_API_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()

    if payload.get("status") == "1":
        return int(payload["result"])
    raise ValueError(
        f"Etherscan API error: {payload.get('message', 'unknown error')}"
    )
