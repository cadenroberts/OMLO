#!/usr/bin/env python3
"""
Emit an MDC-style state transition for the current oram workspace.

This is a minimal, local execution of the MDC-V2-INFINITE-COORDINATION
contract against the oram project. It:

- Locates the latest snapshot hash for the single shard "oram/project"
- If none exists, emits a GENESIS transition and records the snapshot
- Stores state locally under results/mdc_state/
- Uses canonical YAML (sorted keys) for content-addressed hashing

This script is intentionally self-contained and does not attempt to
implement networked routers; it models state://, merge://, and
validate:// URIs as metadata, while actually persisting everything
to the local filesystem.
"""

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


SYSTEM_ID = "oram"
DOMAIN = "privacy-preserving-ml"
SHARD_ID = "oram/project"


@dataclass
class StateTransition:
    kind: str
    spec: str
    system_id: str
    domain: str
    shard_id: str
    operator: str
    from_snapshot_hash: Optional[str]
    to_snapshot_hash: str
    delta: Dict[str, Any]
    checks: Dict[str, Any]
    validation_model: Dict[str, Any]
    merge_model: Dict[str, Any]
    status: str
    reason: str
    timestamp: str


def project_root() -> Path:
    """Return the project root (parent of experiments/)."""
    return Path(__file__).resolve().parent.parent


def compute_sha256(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return f"sha256:{h.hexdigest()}"


def canonical_yaml(obj: Any) -> bytes:
    """
    Canonical-ish YAML serialization: sorted keys, stable formatting.
    This is sufficient for a local, self-consistent content hash.
    """
    text = yaml.safe_dump(
        obj,
        sort_keys=True,
        default_flow_style=False,
    )
    return text.encode("utf-8")


def state_dir() -> Path:
    root = project_root()
    d = root / "results" / "mdc_state"
    d.mkdir(parents=True, exist_ok=True)
    return d


def state_file() -> Path:
    return state_dir() / "state.json"


def load_latest_snapshot_hash() -> Optional[str]:
    sf = state_file()
    if not sf.exists():
        return None
    try:
        with sf.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("latest_snapshot_hash")
    except Exception:
        # If state is corrupted, treat as no state and start fresh.
        return None


def save_latest_snapshot_hash(snapshot_hash: str) -> None:
    sf = state_file()
    data = {
        "system_id": SYSTEM_ID,
        "domain": DOMAIN,
        "shard_id": SHARD_ID,
        "latest_snapshot_hash": snapshot_hash,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with sf.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def describe_workspace() -> Dict[str, Any]:
    """
    Extract a small, checkable description of the oram workspace.
    This keeps the delta minimal while still meaningfully advancing state.
    """
    root = project_root()
    src_exists = (root / "src").exists()
    experiments_exists = (root / "experiments").exists()
    data_exists = (root / "data").exists()

    files = []
    for rel in [
        "README.md",
        "requirements.txt",
        os.path.join("src", "oram_storage.py"),
        os.path.join("src", "oram_dataloader.py"),
        os.path.join("src", "oram_trainer.py"),
        os.path.join("experiments", "run_oram.py"),
        os.path.join("experiments", "run_baseline.py"),
    ]:
        p = root / rel
        files.append(
            {
                "path": rel,
                "exists": p.exists(),
                "size": p.stat().st_size if p.exists() else None,
            }
        )

    return {
        "project_root": str(root),
        "has_src": src_exists,
        "has_experiments": experiments_exists,
        "has_data": data_exists,
        "files": files,
    }


def build_genesis_transition() -> StateTransition:
    """
    Build a GENESIS transition when no prior snapshot exists.
    """
    workspace_facts = describe_workspace()
    snapshot_payload = {
        "system_id": SYSTEM_ID,
        "domain": DOMAIN,
        "shard_id": SHARD_ID,
        "facts": workspace_facts,
    }
    snapshot_hash = compute_sha256(canonical_yaml(snapshot_payload))

    delta = {
        "facts": workspace_facts,
        "artifacts": [],
    }

    checks = {
        "preconditions": [],
        "hard_constraints": [],
        "postconditions": [
            "snapshot_hash is content-addressed from workspace_facts",
        ],
    }

    validation_model = {
        "uri": "validate://router/v2",
        "purity": True,
        "acceptance": "ALL",
    }

    merge_model = {
        "uri": "merge://router/v2",
        "serialization": "per_shard",
    }

    return StateTransition(
        kind="state_transition",
        spec="MDC-V2-INFINITE-COORDINATION",
        system_id=SYSTEM_ID,
        domain=DOMAIN,
        shard_id=SHARD_ID,
        operator="GENESIS",
        from_snapshot_hash=None,
        to_snapshot_hash=snapshot_hash,
        delta=delta,
        checks=checks,
        validation_model=validation_model,
        merge_model=merge_model,
        status="GENESIS",
        reason="Initial MDC snapshot for the oram project workspace.",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def write_transition(transition: StateTransition) -> Path:
    """
    Write the transition as a frontmatter-only YAML artifact.
    """
    dir_ = state_dir()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    op = transition.operator.lower()
    path = dir_ / f"transition_{ts}_{op}.yaml"

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            asdict(transition),
            f,
            sort_keys=False,
            default_flow_style=False,
        )

    return path


def main() -> None:
    latest = load_latest_snapshot_hash()

    if latest is None:
        # Per locate_latest rule: if no snapshot is available, emit GENESIS and stop.
        transition = build_genesis_transition()
        write_path = write_transition(transition)
        save_latest_snapshot_hash(transition.to_snapshot_hash)
        print(f"Emitted GENESIS transition to: {write_path}")
        print(f"New snapshot hash: {transition.to_snapshot_hash}")
        return

    # For now, we respect U006 (smallest delta) by doing nothing if a snapshot
    # already exists; in a more complete integration this branch would emit
    # a non-GENESIS transition tied to a specific oram operation.
    print("Existing snapshot detected; no new transition emitted.")
    print(f"Latest snapshot hash: {latest}")


if __name__ == "__main__":
    main()

