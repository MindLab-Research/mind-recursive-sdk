#!/usr/bin/env python
"""SFT with the things a real run needs.

Adds to ``01_sft.py``: a validation split, LR warmup + cosine decay, periodic
checkpoints, early stopping, and resume from a checkpoint.

Run it with no arguments; change the ``CONFIG`` block below to change the run::

    uv run python recipes/02_sft_full.py

Every addition here is orchestration you write in the recipe, not a feature
the SDK hides. That is the division: the SDK owns the wire, the recipe owns
the training policy.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import mind_recursive as mr
# Recipes are standalone scripts, runnable from any directory, so the shared
# helper module sits next to them rather than on the import path.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DEMO_SFT, Timer, batched, fmt, load_jsonl

# Rendering is recipe-local: it is not the SDK's job. See _renderer.py.
from _renderer import get_renderer


@dataclass
class Config:
    """Everything this run needs. Edit ``CONFIG`` below, then run the file."""

    #: Training JSONL, or ``None`` to use the built-in tiny set.
    data: str | None = None
    #: Validation JSONL. ``None`` holds out ``val_split`` of the training set.
    val_data: str | None = None
    #: Fraction held out when ``val_data`` is absent. 0 disables validation.
    val_split: float = 0.2
    model: str = "Qwen/Qwen3.8-27B"
    #: Relay origin. ``None`` reads ``MIND_RECURSIVE_BASE_URL`` from the environment.
    base_url: str | None = None
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 1e-4
    lora_rank: int = 32
    max_length: int = 4096
    #: Steps between validation passes. 0 disables.
    eval_every: int = 20
    #: Steps between checkpoints. 0 disables.
    save_every: int = 0
    log_every: int = 1
    #: Validation passes without improvement before stopping. 0 disables.
    patience: int = 3
    #: ``mint://`` train-state path to resume from, or ``None`` to start fresh.
    resume_from: str | None = None


CONFIG = Config()


def lr_at(step: int, total: int, peak: float, warmup_frac: float = 0.03) -> float:
    """Linear warmup then cosine decay to 10% of peak.

    Warmup matters more than usual with LoRA: the adapters start at zero, so
    the first steps see large gradients that a cold high LR turns into a loss
    spike.
    """
    warmup = max(1, int(total * warmup_frac))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def evaluate(trainer: mr.LoraTrainer, data: list, batch_size: int) -> float:
    """Mean loss over the validation set.

    ``forward_only=True`` runs the forward pass without accumulating
    gradients, so evaluating does not disturb the optimiser state. Losses are
    token-weighted, because a plain mean over batches would over-count short
    ones.
    """
    total_loss = 0.0
    total_tokens = 0
    for batch in batched(data, batch_size):
        out = trainer.forward_backward(batch, forward_only=True).result()
        tokens = out.num_tokens or len(batch)
        if out.loss is not None:
            total_loss += out.loss * tokens
            total_tokens += tokens
    return total_loss / total_tokens if total_tokens else float("nan")


def train(args: Config, train_conv: list, val_conv: list) -> str:
    renderer = get_renderer(args.model, max_length=args.max_length)
    train_data = renderer.render_many(train_conv)
    val_data = renderer.render_many(val_conv) if val_conv else []
    if not train_data:
        raise SystemExit("no usable training data after rendering")
    print(f"train {len(train_data)}  val {len(val_data)}")

    steps_per_epoch = (len(train_data) + args.batch_size - 1) // args.batch_size
    total_steps = args.epochs * steps_per_epoch

    sc = mr.ServiceClient(base_url=args.base_url)
    with sc:
        trainer = sc.create_lora_trainer(
            base_model=args.model, rank=args.lora_rank, train_unembed=False
        )
        sc.log(f"session {sc.session_id}, trainer {trainer.model_id} created")

        if args.resume_from:
            print(f"resuming from {args.resume_from}")
            trainer.load_weights(args.resume_from).result()
            sc.log(f"resumed from {args.resume_from}")

        # 运行日志是给人读的时间线，控制台「运行日志」tab 只渲染 type="log"。
        # 训练指标走 sc.metric，两者不要混：一个画曲线，一个讲发生了什么。
        sc.log(
            f"SFT (full) started: {len(train_data)} train / {len(val_data)} val, "
            f"{total_steps} steps, model {args.model}",
            step=0
        )
        best_val = float("inf")
        best_path = ""
        since_improvement = 0
        step = 0
        last_loss = None

        try:
            for epoch in range(args.epochs):
                for batch in batched(train_data, args.batch_size):
                    lr = lr_at(step, total_steps, args.learning_rate)
                    with Timer() as forward:
                        out = trainer.forward_backward(batch).result()
                    with Timer() as optim:
                        trainer.optim_step(mr.AdamParams(learning_rate=lr)).result()

                    step += 1
                    last_loss = out.loss
                    sc.metric(
                        {
                            "loss/train": out.loss,
                            "optimizer/lr": lr,
                            "progress/epoch": epoch,
                            "tokens/count": out.num_tokens,
                        },
                        step=step,
                        stage="training",
                        total=total_steps,
                        timings={
                            "forward_backward": forward.seconds,
                            "optim_step": optim.seconds,
                        },
                    )
                    if step % args.log_every == 0 or step == total_steps:
                        print(
                            f"epoch {epoch + 1}/{args.epochs}  step {step}/{total_steps}  "
                            f"loss {fmt(out.loss)}  lr {lr:.2e}"
                        )
                        sc.log(
                            f"epoch {epoch + 1}/{args.epochs} "
                            f"step {step}/{total_steps} loss {fmt(out.loss)} "
                            f"lr {lr:.2e}",
                            step=step,
                        )

                    # -- validation ------------------------------------------
                    if val_data and args.eval_every and step % args.eval_every == 0:
                        val_loss = evaluate(trainer, val_data, args.batch_size)
                        sc.metric({"loss/eval": val_loss}, step=step, stage="evaling")
                        improved = val_loss < best_val
                        print(
                            f"  val_loss {fmt(val_loss)}"
                            f"{'  (best)' if improved else ''}"
                        )
                        sc.log(
                            f"eval step {step}: val_loss {fmt(val_loss)}"
                            f"{' (best)' if improved else ''}",
                            step=step,
                        )
                        if improved:
                            best_val = val_loss
                            since_improvement = 0
                            best_path = trainer.save_weights_only(
                                f"best-step-{step}",
                                user_metadata={
                                    "val_loss": f"{val_loss:.6f}",
                                    "step": str(step),
                                },
                            ).result()
                            sc.log(f"best weights saved: {best_path}", step=step)
                        else:
                            since_improvement += 1
                            if args.patience and since_improvement >= args.patience:
                                print(
                                    f"  no improvement in {since_improvement} evals, stopping"
                                )
                                sc.log(
                                    f"early stop at step {step}: no improvement in "
                                    f"{since_improvement} evals, best val {best_val:.4f}",
                                    level="warning",
                                    step=step,
                                )
                                raise _EarlyStop

                    # -- periodic checkpoint ---------------------------------
                    if args.save_every and step % args.save_every == 0:
                        state = trainer.save_state(f"step-{step}").result()
                        print(f"  checkpoint {state}")
                        sc.log(f"checkpoint saved: {state}", step=step)

                sc.log(
                    f"epoch {epoch + 1}/{args.epochs} done, "
                    f"last loss {fmt(last_loss)}",
                    step=step,
                )

        except _EarlyStop:
            pass
        except Exception as exc:
            # 日志给人看；event 给平台看，带稳定的 type/code 供它处理失败。
            sc.log(f"run failed: {type(exc).__name__}: {exc}", level="error")
            sc.event(
                "run.failed",
                level="error",
                code=type(exc).__name__,
                message=str(exc),
            )
            raise

        # -- final artefact -------------------------------------------------
        final_path = best_path or trainer.save_weights_only(
            f"final-step-{step}"
        ).result()
        sc.log(f"training done, weights {final_path}", step=step)

        # 存下来的权重直接拿去采样，确认训练产物可部署、可调用。
        # 采样面和训练面是两条路径，这一步走的是 create_sampler + sample。
        prompt = [
            message for message in train_conv[0] if message.get("role") != "assistant"
        ] or train_conv[0]
        sampler = trainer.create_sampler(model_path=final_path)
        sc.log(
            f"sampler {sampler.sampling_session_id} opened on {final_path}",
            step=step
        )
        response = sampler.sample(
            renderer.prompt_tokens(prompt),
            num_samples=2,
            sampling_params=mr.SamplingParams(max_tokens=64, temperature=0.7)
        ).result()

        first_sample = ""
        for index, sequence in enumerate(response.sequences):
            text = renderer.decode(sequence.tokens).strip()
            first_sample = first_sample or text
            print(f"sample {index}      : {text[:200]}")
            sc.log(f"post-train sample {index}: {text[:500]}", step=step)

        sc.summary(
            final_loss=last_loss,
            best_val_loss=None if best_val == float("inf") else best_val,
            steps=step,
            sample=first_sample[:200]
        )

        print(f"\nsteps run     : {step}")
        if best_val != float("inf"):
            print(f"best val loss : {best_val:.4f}")
        print(f"weights       : {final_path}")
        return final_path


class _EarlyStop(Exception):
    """Internal signal; not an error, so it does not mark the run failed."""


def main() -> int:
    config = CONFIG
    if config.data:
        conversations = load_jsonl(config.data)
    else:
        conversations = DEMO_SFT

    if config.val_data:
        val = load_jsonl(config.val_data)
        train_conv = conversations
    elif config.val_split > 0 and len(conversations) >= 5:
        cut = max(1, int(len(conversations) * config.val_split))
        val, train_conv = conversations[:cut], conversations[cut:]
    else:
        val, train_conv = [], conversations

    try:
        train(config, train_conv, val)
    except mr.ConfigurationError as exc:
        print(f"configuration problem: {exc}", file=sys.stderr)
        return 2
    except mr.RequestFailedError as exc:
        print(f"the server rejected the work: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
