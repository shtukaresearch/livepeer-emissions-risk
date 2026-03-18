"""Tests for lpt_stake.constants.

The self-consistency tests run offline.  The on-chain verification tests
require ``ARB_RPC_URL`` and ``ETHERSCAN_API_KEY`` environment variables and
are skipped when those are not set.
"""

import os
from datetime import timezone

import pytest

try:
    import requests
except ModuleNotFoundError:
    requests = None

from lpt_stake.constants import (
    ETH_SLOTS_PER_ROUND,
    REFERENCE_DATETIME,
    REFERENCE_ROUND,
    ROUNDS_PER_YEAR,
    SECONDS_PER_ETH_SLOT,
    SECONDS_PER_ROUND,
)

# ---------------------------------------------------------------------------
# Markers for live tests
# ---------------------------------------------------------------------------

_ARB_RPC_URL = os.environ.get("ARB_RPC_URL")
_ETHERSCAN_API_KEY = os.environ.get("ETHERSCAN_API_KEY")

requires_rpc = pytest.mark.skipif(
    not _ARB_RPC_URL or requests is None,
    reason="ARB_RPC_URL not set or requests not installed",
)
requires_etherscan = pytest.mark.skipif(
    not _ETHERSCAN_API_KEY or requests is None,
    reason="ETHERSCAN_API_KEY not set or requests not installed",
)

_ROUNDS_MANAGER = "0xdd6f56DcC28D3F5f27084381fE8Df634985cc39f"


def _eth_call(to: str, data: str, block: str = "latest") -> str:
    """Make an eth_call against the Arbitrum RPC."""
    resp = requests.post(
        _ARB_RPC_URL,
        json={
            "jsonrpc": "2.0",
            "method": "eth_call",
            "params": [{"to": to, "data": data}, block],
            "id": 1,
        },
        timeout=10,
    )
    return resp.json()["result"]


# ---------------------------------------------------------------------------
# Self-consistency (offline)
# ---------------------------------------------------------------------------


class TestRoundGeometry:
    """Verify that derived constants are consistent with their inputs."""

    def test_seconds_per_round_is_product(self):
        assert SECONDS_PER_ROUND == SECONDS_PER_ETH_SLOT * ETH_SLOTS_PER_ROUND

    def test_rounds_per_year_consistent_with_seconds(self):
        julian_year_seconds = 365.25 * 86_400
        expected = julian_year_seconds / SECONDS_PER_ROUND
        assert ROUNDS_PER_YEAR == expected

    def test_rounds_per_year_approximate(self):
        """Sanity: roughly 412–413 rounds per year."""
        assert 412 < ROUNDS_PER_YEAR < 413

    def test_round_duration_is_roughly_21_hours(self):
        """Sanity: a round is roughly 21 hours."""
        hours = SECONDS_PER_ROUND / 3600
        assert 21.0 < hours < 21.5


# ---------------------------------------------------------------------------
# On-chain verification (requires RPC / Etherscan)
# ---------------------------------------------------------------------------


class TestRoundLengthOnChain:
    """Verify ETH_SLOTS_PER_ROUND matches the live contract."""

    @requires_rpc
    def test_round_length_matches_contract(self):
        """roundLength() on the RoundsManager should equal ETH_SLOTS_PER_ROUND."""
        # roundLength() selector: 0x8b649b94
        result = _eth_call(_ROUNDS_MANAGER, "0x8b649b94")
        on_chain_round_length = int(result, 16)
        assert on_chain_round_length == ETH_SLOTS_PER_ROUND


class TestReferencePointOnChain:
    """Verify the reference point against live chain state."""

    @requires_rpc
    @requires_etherscan
    def test_reference_round_matches_contract(self):
        """currentRound() at a block shortly after the reference datetime
        should return REFERENCE_ROUND."""
        # Query a few minutes after the round start to allow for the
        # ~1 minute lag in Arbitrum's L1 block.number tracking.
        ts = int(REFERENCE_DATETIME.timestamp()) + 300
        resp = requests.get(
            "https://api.etherscan.io/v2/api",
            params={
                "chainid": 42161,
                "module": "block",
                "action": "getblocknobytime",
                "timestamp": ts,
                "closest": "before",
                "apikey": _ETHERSCAN_API_KEY,
            },
            timeout=10,
        )
        arb_block = int(resp.json()["result"])
        arb_block_hex = hex(arb_block)

        # currentRound() selector: 0x8a19c8bc
        result = _eth_call(_ROUNDS_MANAGER, "0x8a19c8bc", arb_block_hex)
        on_chain_round = int(result, 16)
        assert on_chain_round == REFERENCE_ROUND

    @requires_rpc
    @requires_etherscan
    def test_reference_datetime_is_round_boundary(self):
        """The reference datetime should correspond to the actual start block
        of REFERENCE_ROUND, verified via currentRoundStartBlock() and the
        L1 block timestamp."""
        # Get an Arbitrum block during REFERENCE_ROUND
        ts = int(REFERENCE_DATETIME.timestamp()) + 60  # just after round start
        resp = requests.get(
            "https://api.etherscan.io/v2/api",
            params={
                "chainid": 42161,
                "module": "block",
                "action": "getblocknobytime",
                "timestamp": ts,
                "closest": "after",
                "apikey": _ETHERSCAN_API_KEY,
            },
            timeout=10,
        )
        arb_block_hex = hex(int(resp.json()["result"]))

        # currentRoundStartBlock() selector: 0x8fa148f2
        result = _eth_call(_ROUNDS_MANAGER, "0x8fa148f2", arb_block_hex)
        l1_start_block = int(result, 16)

        # Look up L1 block timestamp via Etherscan mainnet
        resp = requests.get(
            "https://api.etherscan.io/v2/api",
            params={
                "chainid": 1,
                "module": "block",
                "action": "getblockreward",
                "blockno": l1_start_block,
                "apikey": _ETHERSCAN_API_KEY,
            },
            timeout=10,
        )
        l1_timestamp = int(resp.json()["result"]["timeStamp"])

        assert REFERENCE_DATETIME.tzinfo is timezone.utc
        assert int(REFERENCE_DATETIME.timestamp()) == l1_timestamp
