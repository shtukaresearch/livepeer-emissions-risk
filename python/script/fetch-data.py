"""Fetch historic Livepeer chain state for the internal dashboard."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import arrow
import requests
import urllib3
from pytz import UTC
from requests.exceptions import ReadTimeout
from requests.adapters import HTTPAdapter
from web3 import Web3

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
RAW_CHAIN_DIR = PYTHON_DIR / "data" / "raw" / "chain"
TICKS_FILE = RAW_CHAIN_DIR / "arbitrum-daily-blocks.json"
DATA_FILE = RAW_CHAIN_DIR / "lpt-daily-data.json"

DEPLOYMENTS_DIR = Path(
    os.getenv("LPT_DEPLOYMENTS_DIR", REPO_ROOT / "protocol" / "deployments" / "arbitrumMainnet")
)
MINTER_DEPLOYMENT_JSON = DEPLOYMENTS_DIR / "Minter.json"
BONDING_MANAGER_DEPLOYMENT_JSON = DEPLOYMENTS_DIR / "BondingManager.json"
BONDING_MANAGER_IMPLEMENTATION_JSON = DEPLOYMENTS_DIR / "BondingManagerTarget.json"

API_URL = "https://api.etherscan.io/v2/api?chainid=42161"

retries = urllib3.util.Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)


def _session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_deployments() -> None:
    required = [
        MINTER_DEPLOYMENT_JSON,
        BONDING_MANAGER_DEPLOYMENT_JSON,
        BONDING_MANAGER_IMPLEMENTATION_JSON,
    ]
    missing = [path for path in required if not path.exists()]
    if not missing:
        return

    raise FileNotFoundError(
        "Missing Livepeer deployment artifacts. Initialize the `protocol` submodule "
        "or set LPT_DEPLOYMENTS_DIR to a directory containing the arbitrum deployment JSON files."
    )


def get_block_number_by_time(
    session: requests.Session,
    apikey: str,
    timestamp: datetime,
) -> int:
    params = {
        "module": "block",
        "action": "getblocknobytime",
        "timestamp": int(timestamp.timestamp()),
        "closest": "before",
        "apikey": apikey,
    }
    response = session.get(API_URL, params=params, timeout=20)
    response.raise_for_status()
    data = response.json()
    if data["status"] == "1":
        return int(data["result"])
    raise ValueError(f"Error fetching block number: {data['message']}")


def save_json(path: Path, payload: dict) -> None:
    _ensure_parent(path)
    with path.open("w") as handle:
        json.dump(payload, handle)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def load_existing_ticks(start: datetime, path: Path = TICKS_FILE) -> dict[str, list] | None:
    if not path.exists():
        return None

    payload = load_json(path)

    if not payload.get("date"):
        return None

    first_date = datetime.strptime(payload["date"][0], "%Y-%m-%d").replace(tzinfo=UTC)
    if first_date != start:
        return None

    return payload


def fetch_arbitrum_daily_blocks(
    apikey: str,
    start: datetime,
    num_days: int,
    *,
    existing_blocks: dict[str, list] | None = None,
    checkpoint_path: Path | None = TICKS_FILE,
) -> dict[str, list]:
    """Fetch Arbitrum block numbers at daily intervals."""

    session = _session()
    blocks = existing_blocks or load_existing_ticks(start, checkpoint_path or TICKS_FILE) or {
        "date": [],
        "block": [],
    }
    current_time = start + timedelta(days=len(blocks["date"]))

    for _ in range(len(blocks["date"]), num_days):
        print(f"tick {len(blocks['date']) + 1}/{num_days}: {current_time.date()}")
        retries_left = 3
        while retries_left > 0:
            try:
                block_number = get_block_number_by_time(session, apikey, current_time)
                blocks["block"].append(block_number)
                blocks["date"].append(current_time.strftime("%Y-%m-%d"))
                if checkpoint_path is not None:
                    save_json(checkpoint_path, blocks)
                break
            except (ValueError, ReadTimeout) as exc:
                print(f"Error fetching block number: {exc}. Retrying...")
                retries_left -= 1
                time.sleep(2 ** (3 - retries_left))
        else:
            raise RuntimeError(
                f"Failed to fetch block number for {current_time} after multiple retries."
            )

        current_time += timedelta(days=1)
        time.sleep(0.2)
    return blocks


def arbitrum_w3() -> Web3:
    """Create a Web3 object from the configured Arbitrum RPC URL."""

    arb_rpc_url = os.getenv("ARB_RPC_URL")
    if not arb_rpc_url:
        raise ValueError(
            "ARB_RPC_URL is not set. Provide an Arbitrum archive node RPC URL."
        )
    return Web3(Web3.HTTPProvider(arb_rpc_url))


def load_contract_from_json(w3: Web3, path: Path):
    """Construct a Contract object from a deployment JSON file."""

    with path.open("r") as file:
        contract_json = json.load(file)
    contract_abi = contract_json["abi"]
    contract_address = contract_json["address"]
    return w3.eth.contract(address=contract_address, abi=contract_abi)


def bonding_manager(w3: Web3):
    with BONDING_MANAGER_IMPLEMENTATION_JSON.open() as handle:
        contract_abi = json.load(handle)["abi"]
    with BONDING_MANAGER_DEPLOYMENT_JSON.open() as handle:
        contract_address = json.load(handle)["address"]
    return w3.eth.contract(address=contract_address, abi=contract_abi)


def fetch_historic(callable_obj, blocks: list[int]) -> list[int]:
    results = []
    for block in blocks:
        results.append(callable_obj.call(block_identifier=block))
        time.sleep(0.05)
    return results


def help_lines() -> list[str]:
    return [
        "Set environment variables:",
        "ETHERSCAN_API_KEY=<Etherscan API key>\t\t(for fetching Arbitrum block numbers)",
        "ARB_RPC_URL=<Arbitrum archive node RPC URL>\t(for fetching historic state)",
        "Optional: LPT_DEPLOYMENTS_DIR=<path to arbitrum deployment JSONs>",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Livepeer staking data from Arbitrum.")
    parser.add_argument(
        "--extend",
        type=Path,
        help="Path to an existing ticks or state file to extend.",
    )
    parser.add_argument("--ticks", action="store_true", help="Fetch Arbitrum block numbers.")
    parser.add_argument(
        "--state",
        action="store_true",
        help="Fetch historic state at ticks in data/raw/chain. If --ticks is present, fetch ticks first.",
    )
    parser.add_argument(
        "--start-date",
        type=lambda s: arrow.get(s).datetime,
        required=True,
        help="Start date (inclusive) in YYYY-MM-DD format.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--end-date",
        type=lambda s: arrow.get(s).datetime,
        help="End date (exclusive) in YYYY-MM-DD format.",
    )
    group.add_argument("--num-days", type=int, help="Number of days from the start date.")
    return parser.parse_args()


def fetch_ticks(
    start_date: datetime,
    num_days: int,
    *,
    output_path: Path | None = TICKS_FILE,
    existing_blocks: dict[str, list] | None = None,
) -> dict[str, list]:
    if num_days <= 0:
        return {"date": [], "block": []}

    arbiscan_api_key = os.getenv("ETHERSCAN_API_KEY")
    if not arbiscan_api_key:
        print(*help_lines(), sep="\n")
        sys.exit(1)

    print("Fetching Arbitrum block numbers...")
    block_nums = fetch_arbitrum_daily_blocks(
        apikey=arbiscan_api_key,
        start=start_date,
        num_days=num_days,
        existing_blocks=existing_blocks,
        checkpoint_path=output_path,
    )

    if output_path is not None:
        save_json(output_path, block_nums)

    return block_nums


def fetch_state(block_nums: dict[str, list], *, output_path: Path | None = DATA_FILE) -> dict:
    if not block_nums.get("block"):
        return {
            "inflation": [],
            "total-supply": [],
            "bonded": [],
            "date": block_nums.get("date", []),
            "block": block_nums.get("block", []),
        }

    _ensure_deployments()
    w3 = arbitrum_w3()

    minter = load_contract_from_json(w3, MINTER_DEPLOYMENT_JSON)
    bonding = bonding_manager(w3)

    callables = {
        "inflation": minter.functions.inflation(),
        "total-supply": minter.functions.getGlobalTotalSupply(),
        "bonded": bonding.functions.getTotalBonded(),
    }

    print("Fetching historic data...")
    results = {key: fetch_historic(value, block_nums["block"]) for key, value in callables.items()}
    results = results | block_nums

    if output_path is not None:
        save_json(output_path, results)

    return results


def extend_state(old: dict, extension: list[dict]) -> dict:
    before, after = extension
    return {key: before[key] + old[key] + after[key] for key in old}


def extend_ticks(old: dict[str, list], extension: list[dict[str, list]]) -> dict[str, list]:
    before, after = extension
    return {
        "date": before.get("date", []) + old.get("date", []) + after.get("date", []),
        "block": before.get("block", []) + old.get("block", []) + after.get("block", []),
    }


if __name__ == "__main__":
    try:
        args = parse_args()

        if args.end_date:
            num_days = (args.end_date - args.start_date).days
            end_date = args.end_date
        else:
            num_days = args.num_days
            end_date = args.start_date + timedelta(days=num_days)
            if num_days <= 0:
                raise ValueError("Number of days must be greater than 0.")

        if args.extend:
            print(f"Extending existing data files {args.extend}...")
            old_payload = load_json(args.extend)

            old_start_date = datetime.strptime(old_payload["date"][0], "%Y-%m-%d").replace(
                tzinfo=UTC
            )
            old_end_date = (
                datetime.strptime(old_payload["date"][-1], "%Y-%m-%d") + timedelta(days=1)
            ).replace(tzinfo=UTC)
            if old_start_date < args.start_date or old_end_date > args.end_date:
                raise ValueError("New date range is not an extension of range in file.")

            tick_extension = []
            state_extension = []
            for start, days in [
                (args.start_date, (old_start_date - args.start_date).days),
                (old_end_date, (end_date - old_end_date).days),
            ]:
                ticks = fetch_ticks(start, days, output_path=None)
                tick_extension.append(ticks)
                if args.state:
                    state_extension.append(fetch_state(ticks, output_path=None))

            if args.ticks:
                old_ticks = {
                    "date": old_payload.get("date", []),
                    "block": old_payload.get("block", []),
                }
                save_json(TICKS_FILE, extend_ticks(old_ticks, tick_extension))

            if args.state:
                save_json(args.extend, extend_state(old_payload, state_extension))

        else:
            if args.ticks:
                block_nums = fetch_ticks(args.start_date, num_days)
            else:
                block_nums = None

            if args.state:
                if block_nums is None:
                    with TICKS_FILE.open() as handle:
                        block_nums = json.load(handle)
                results = fetch_state(block_nums)
                print(json.dumps(results))
    except (FileNotFoundError, RuntimeError, ValueError, requests.RequestException) as exc:
        raise SystemExit(str(exc)) from exc
