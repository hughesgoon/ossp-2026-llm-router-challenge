from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from router.allocate import allocate
from router.artifact import load
from router.constants import MODEL_IDS, TIER_BUDGET, TIERS
from router.predict import predict_batch

_PACKAGE = Path(__file__).resolve().parent


def _default_artifact() -> Path:
    local = _PACKAGE / "router-v2.json"
    if local.exists():
        return local
    return _PACKAGE.parent / "artifacts" / "router-v2.json"


DEFAULT_ARTIFACT = _default_artifact()
POLICY_ID = "ossp-2026-prompt-router-v1"


def episode_text(episode: dict) -> str:
    prompt = episode.get("prompt")
    if prompt is not None:
        return prompt
    return "\n".join(m.get("content", "") for m in episode.get("messages", []))


def write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, separators=(",", ":"))
        os.chmod(temp, 0o644)
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise


def run(input_path: Path, tier: str, output_path: Path, artifact_path: Path) -> int:
    batch = json.loads(input_path.read_text("utf-8"))
    episodes = batch["episodes"]
    if not episodes:
        raise ValueError("input contains no episodes")

    artifact = load(artifact_path)
    scores, costs, _ = predict_batch(artifact, [episode_text(e) for e in episodes])
    selection = allocate(scores, costs, tier, TIER_BUDGET[tier])

    write_atomic(
        output_path,
        {
            "schema_version": 1,
            "challenge_id": batch["challenge_id"],
            "policy_id": POLICY_ID,
            "split": batch["split"],
            "tier": tier,
            "decisions": [
                {"episode_id": e["episode_id"], "model_id": MODEL_IDS[selection[i]]}
                for i, e in enumerate(episodes)
            ],
        },
    )
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="router-run")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tier", choices=TIERS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args(argv)
    try:
        return run(args.input, args.tier, args.output, args.artifact)
    except Exception as error:
        print(f"router-run: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
