#!/usr/bin/env python
"""Every way to open a sampler, so you can see which ones the session page lists.

    uv run python recipes/04_samplers.py

Every sampler is created through the platform service, which opens it under
the inference-side session it keeps for this Service Session. A sampler draws from one
of two kinds of weights:

* a **public base model** is already deployed, so the sampler is ready at once.
* **trained weights** (a saved ``model_path``) must be deployed first. The
  platform service creates that temporary deployment and releases it when a
  newer one for the same trainer is ready or the session ends.

This recipe opens one of the first kind and three of the second (one named
save, two per-step snapshots), samples from each, then prints what the page
should show. It also tries the two creates the platform rejects -- trained
weights without a base model, and ``kind="inflight"`` -- and prints the 400.

Change the ``CONFIG`` block below to probe a different model or relay.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import mind_recursive as mr
@dataclass
class Config:
    """Edit this, then run the file."""

    #: Base model. ``mock-model`` consumes no training quota.
    model: str = "mock-model"
    #: Relay origin. ``None`` reads ``MIND_RECURSIVE_BASE_URL`` from the environment.
    base_url: str | None = None
    lora_rank: int = 8
    #: Training steps between the per-step snapshots.
    steps: int = 2


CONFIG = Config()

PROMPT = [1, 2, 3]
PARAMS = mr.SamplingParams(max_tokens=8)


def sample_once(label: str, sampler: object) -> None:
    response = sampler.sample(PROMPT, num_samples=2, sampling_params=PARAMS).result()
    print(f"  {label:<24} {sampler.sampling_session_id}  {len(response.sequences)} sequence(s)")


def main() -> int:
    args = CONFIG
    with mr.ServiceClient(base_url=args.base_url, client_id="samplers-recipe") as sc:
        print(f"session {sc.session_id}")
        sc.log("samplers: start")

        # 1. Public base model: also creates a fresh deployment, then a sampler.
        print("\n1. public base model (deployment, then sampler)")
        base = sc.create_sampler(base_model=args.model)
        sample_once("base_model", base)
        sc.log(f"samplers: base-model sampler {base.sampling_session_id}")

        trainer = sc.create_lora_trainer(base_model=args.model, rank=args.lora_rank)
        batch = [mr.Datum.from_tokens([1, 2, 3, 4, 5])]

        def train(n: int) -> None:
            for _ in range(n):
                trainer.forward_backward(batch).result()
                trainer.optim_step().result()

        # 2. A named save: deployed by the platform service, listed.
        print("\n2. named save (platform service deploys it, listed)")
        train(args.steps)
        path = trainer.save_weights_only("samplers-named").result()
        named = trainer.create_sampler(model_path=path)
        sample_once(path.rsplit("/", 1)[-1], named)
        sc.log(f"samplers: named sampler on {path}")

        # 3. Per-step snapshots, the RL pattern: each step is a new
        #    deployment; on the platform the newer one releases the older.
        print("\n3. per-step snapshots (one deployment per step, listed)")
        for step in range(2):
            train(args.steps)
            snap = trainer.save_weights_and_get_sampling_client()
            sample_once(f"snapshot step {step + 1}", snap)
            sc.log(f"samplers: snapshot {step + 1} {snap.sampling_session_id}")

        # 4. What the platform refuses.
        print("\n4. rejected creates")
        try:
            sc.create_sampler(model_path=path)
        except mr.BadRequestError as exc:
            print(f"  model_path without base_model -> 400: {exc.message}")
        try:
            trainer.create_sampler(kind="inflight")
        except mr.BadRequestError as exc:
            print(f"  kind='inflight'               -> 400: {exc.message}")

        sc.summary({"samplers": 4})

    print(
        "\nOpen the Web console -> 我的训练 -> this session -> sampler tab: four rows\n"
        "(the base-model deployment, the named save and two snapshots), all 已释放 now\n"
        "that the session has ended."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
