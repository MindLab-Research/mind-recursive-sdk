#!/usr/bin/env python
"""Environment self-check: connect, train one step, save weights, sample.

Run this first. It answers "is my key, network and model name right?" in about
20 seconds, before you spend an hour debugging a training script.

    uv run python recipes/00_smoke.py

Change the ``CONFIG`` block below to probe a different model or relay.

``mock-model`` is the intended target: it exercises the full request path --
session, protobuf forward_backward, future polling, weight save -- without
consuming real training quota. Alpha offers it for the same purpose.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import mind_recursive as mr
@dataclass
class Config:
    """Edit this, then run the file."""

    #: Base model to probe. ``mock-model`` consumes no training quota.
    model: str = "mock-model"
    #: Relay origin. ``None`` reads ``MIND_RECURSIVE_BASE_URL`` from the environment.
    base_url: str | None = None
    lora_rank: int = 8


CONFIG = Config()


def main() -> int:
    args = CONFIG

    print(f"model={args.model}")

    # 1. Ask what the backend serves *before* opening a session, so a wrong
    #    key or an unreachable relay is reported without leaving a session
    #    record behind. This is why the surface is two clients: UserClient
    #    creates nothing.
    try:
        user = mr.UserClient(base_url=args.base_url)
    except mr.ConfigurationError as exc:
        print(f"\nconfiguration problem: {exc}")
        return 2

    with user:
        print(f"relay   : {user.config.origin}")
        print(f"key     : {user.config.redacted()['api_key']}")

        caps = user.get_server_capabilities()
        names = caps.names()
        print(f"\nserver offers {len(names)} model(s)")
        for name in names[:8]:
            print(f"  - {name}")

        if args.model not in names:
            print(
                f"\n{args.model!r} is not offered here. "
                "Set CONFIG.model to one of the above."
            )
            return 1

    # 2. Only now commit to a session. Constructing a ServiceClient creates
    #    one, so this line is the first billable step.
    with mr.ServiceClient(base_url=args.base_url) as sc:
        # 3. create_model is where a bad model name or exhausted quota shows up.
        print(f"\nopening a training client on {args.model} ...")
        trainer = sc.create_lora_trainer(
            base_model=args.model, rank=args.lora_rank
        )
        print(f"  session  {sc.session_id}")
        print(f"  model_id {trainer.model_id}")
        # 一条运行日志，确认上报这条路也通：控制台「运行日志」tab 只渲染
        # type="log"，也就是 sc.log 写的内容。
        sc.log(f"smoke: connected to {sc.config.origin}, model {args.model}")
        sc.log(f"smoke: session {sc.session_id}, trainer {trainer.model_id} created")

        # 4. One real step: protobuf out, future polled, optimiser applied.
        print("\nrunning one forward_backward + optim_step ...")
        batch = [mr.Datum.from_tokens([1, 2, 3, 4, 5])]
        out = trainer.forward_backward(batch).result()
        trainer.optim_step().result()
        print(f"  loss       {out.loss}")
        print(f"  num_tokens {out.num_tokens}")
        sc.log(f"smoke: one step done, loss {out.loss}")

        # 5. The artefact a deployment would consume.
        print("\nsaving sampler weights ...")
        path = trainer.save_weights_only("smoke-step-1").result()
        print(f"  {path}")
        sc.log(f"smoke: sampler weights saved to {path}")

        # 6. The other plane: sample from the weights just saved. Training and
        #    sampling are separate surfaces, so "saved" is not the same as
        #    "can be served".
        print("\nsampling from the saved weights ...")
        sampler = trainer.create_sampler(model_path=path)
        response = sampler.sample(
            [1, 2, 3],
            num_samples=1,
            sampling_params=mr.SamplingParams(max_tokens=8)
        ).result()
        print(f"  {len(response.sequences)} sequence(s)")
        sc.log(f"smoke: sampled {len(response.sequences)} sequence(s) from {path}")

    print("\nOK — the full training path works. Try recipes/01_sft.py next.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
