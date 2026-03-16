"""Protocol-level numeric constants for the Livepeer network.

All constants here are properties of the protocol itself, not of this library's
modelling choices.  They are used throughout the library for unit conversions
and time estimation.

Historical note: ``ETH_SLOTS_PER_ROUND`` was changed from 5760 to 6377 by
LIP-83 on 2022-10-06, to preserve the historical round duration after the
Ethereum Merge (2022-09-15) fixed block time at 12 s.  The pre-Merge average
block time was ~13.29 s, giving rounds of ~76 525 s.  At 12 s/slot the
equivalent is 76 525 / 12 ≈ 6377 slots.

The on-chain round number is ``lastRoundLengthUpdateRound + (blockNum -
lastRoundLengthUpdateStartBlock) / roundLength``, where ``blockNum`` is
Solidity's ``block.number``.  On Arbitrum, ``block.number`` currently tracks
the L1 block number (synced ~every minute).  A future ArbOS upgrade could
change this semantic, which would require another ``setRoundLength`` call and
an update to the constants here.
"""

from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Round geometry
# ---------------------------------------------------------------------------

SECONDS_PER_ETH_SLOT: int = 12
"""Duration of one Ethereum L1 consensus slot, in seconds."""

ETH_SLOTS_PER_ROUND: int = 6377
"""Number of L1 slots that make up one Livepeer round (post-LIP-83)."""

SECONDS_PER_ROUND: int = SECONDS_PER_ETH_SLOT * ETH_SLOTS_PER_ROUND
"""Nominal duration of one Livepeer round, in seconds (76 524 s ≈ 21.3 h)."""

ROUNDS_PER_YEAR: float = 365.25 * 86_400 / SECONDS_PER_ROUND
"""Approximate number of rounds per Julian year (≈ 412.5)."""

# ---------------------------------------------------------------------------
# Reference point for estimated round ↔ datetime conversion
# ---------------------------------------------------------------------------

REFERENCE_ROUND: int = 4048
"""A known Livepeer round number at ``REFERENCE_DATETIME``.

Verified on-chain via ``RoundsManager.currentRound()`` on 2026-03-15.
"""

REFERENCE_DATETIME: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)
"""UTC wall-clock time at which ``REFERENCE_ROUND`` was current."""
