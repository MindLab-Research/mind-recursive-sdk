#!/usr/bin/env python
"""SFT：监督微调，最短的可用版本。

不带参数运行；改下面的 ``CONFIG`` 块来改变这次运行：

    uv run python recipes/01_sft.py

数据格式，一行一个对话：

    {"messages": [{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}]}

这个文件想展示的边界：

* **渲染在 SDK 外面。** 训练面只接收 token 和每个 token 的 loss 权重，
  从不接收 chat 格式的消息。用哪个模板、监督哪些 turn、样本最长多少 ——
  这些是数据集策略，所以放在 recipes 旁边的 ``_renderer.py``，而不是
  ``mind_recursive`` 里面。
* **SDK 只看到 datums。** ``load_datums()`` 把 JSONL 变成这个列表；
  ``main()`` 随后原地跑训练循环 —— 没有隐藏的 helper，SDK 的每个调用都在
  同一个块里看得见。

下一步读 ``02_sft_full.py``：验证集、checkpointing 和 LR schedule。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import mind_recursive as mr
# recipe 是独立脚本，可以从任意目录运行，所以共享 helper 放在它旁边，
# 而不是挂到 import path 上。
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DEMO_SFT, Timer, batched, fmt, load_jsonl

# 渲染属于 recipe 本地逻辑，不是 SDK 的职责。见 _renderer.py。
from _renderer import get_renderer

#: demo 数据放在 recipes/data/ 下，新 clone 的仓库不用先生成数据就能跑这个 recipe。
#: 要用自己的数据，把 ``CONFIG.data`` 指向对应文件。
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "recipes" / "data" / "sft_demo.jsonl"


@dataclass
class Config:
    """这次运行需要的全部配置。改 ``CONFIG``，然后直接运行文件。"""

    #: 一行一个对话；设为 ``None`` 则使用内置数据集。
    data: str | Path | None = DEFAULT_DATA
    model: str = "Qwen/Qwen3.8-27B"
    #: 8 条 demo 数据、batch_size=4 时 1 个 epoch 只有 2 步，很快就跑完。
    #: 想观察更多过程就调大这个值；连真实后端时记得改回 1，否则会重复训练多遍。
    epochs: int = 30
    batch_size: int = 4
    learning_rate: float = 1e-4
    lora_rank: int = 32
    max_length: int = 4096
    #: 每多少步写一条运行日志（控制台「运行日志」tab）。0 关闭。
    #: 默认每步一条：日志是给人读的时间线，metric 画曲线，两者都要能看见。
    #: 真机跑几千步时调大，避免把上报通道刷满。
    log_every: int = 1
    #: 训练前打印一次 loss mask。换数据集或换模型时保持开启：
    #: 掩码错了不会报错，只会安静地训练错东西。
    inspect_mask: bool = True


CONFIG = Config()


def load_datums(
    *,
    data: str | Path | None,
    model: str,
    max_length: int | None,
    inspect_mask: bool,
) -> tuple[list, list]:
    """JSONL -> 对话 -> ``Datum`` 列表，同时把原始对话返回给调用方。

    训练只需要 datums；原始对话留着，是因为训练结束后要拿第一条 prompt
    去 sampler 做生成抽检。

    这是 recipe 侧的那一半契约。``render_many`` 返回 token id 和每个 token
    对应的权重，正好就是 ``forward_backward`` 接收的东西；SDK 不需要知道
    它们由哪个 chat template 产生。
    """
    # 数据准备、renderer 不是 sdk 的任务，SDK TITO 原则，传入 Datums 对象
    conversations = load_jsonl(data) if data else DEMO_SFT
    renderer = get_renderer(model, max_length=max_length)

    # 只有 assistant 的 token 参与监督；掩码错了不会报错，所以每个数据集看一次。
    if inspect_mask:
        sample = renderer.render(conversations[0])
        print("\n--- loss mask check ([bracketed] text is trained on) ---")
        print(sample.debug_spans(renderer.tokenizer))
        print(f"--- {sample.supervised_tokens}/{len(sample)} tokens supervised ---\n")

    datums = renderer.render_many(conversations)
    if not datums:
        raise SystemExit("every conversation failed to render; check the data format")
    print(f"{len(datums)} of {len(conversations)} conversations usable")
    return datums, conversations


def main() -> int:
    """整个 recipe 都在这里：先渲染，再训练。"""

    config = CONFIG

    try:
        # ── recipe 侧：JSONL -> token + 每个 token 的 loss 权重 ─────────────
        # 从这里往下，没有任何代码知道 chat template 或数据集。
        datums, conversations = load_datums(
            data=config.data,
            model=config.model,
            max_length=config.max_length,
            inspect_mask=config.inspect_mask
        )
        # get_renderer 有缓存，这里拿到的是 load_datums 里那一份；
        # 训练结束后要用它把 sample 出来的 token 解码回文本。
        renderer = get_renderer(config.model, max_length=config.max_length)
        print(f"data ready: {len(datums)}/{len(conversations)} conversations")

        # ── SDK 侧：下面每个调用都会发 HTTP 请求 ───────────────────────────
        # 不传 base_url：SDK 自己读环境变量 MIND_RECURSIVE_BASE_URL。
        # 构造 ServiceClient 会立刻执行
        #   POST client/config -> POST create_session
        # 从这一刻起 session 就存在了，控制台里也马上能看到。只查模型列表 /
        # 健康检查请用 mr.UserClient()，它不创建 session。
        sc = mr.ServiceClient()
        with sc:
            # 离开这个 with 块会停止心跳并调用 POST /sessions/{id}/finish，
            # 即使中途抛了异常也一样。

            # session 已经由构造函数创建；这里只是在它下面 POST create_model。
            # base_model / loss_fn 属于 model / step，不属于 session。
            trainer = sc.create_lora_trainer(
                base_model=config.model,
                rank=config.lora_rank,
                train_unembed=False,  # SDK 默认值；训练 LM head 会导致权重无法部署
            )
            print(f"session {sc.session_id}  model {trainer.model_id}")

            # 上报通过 client 自己的 transport 走 HTTP，失败只 warning，
            # 不会打断训练。
            total_steps = config.epochs * (
                (len(datums) + config.batch_size - 1) // config.batch_size
            )
            # sc.log 写的是给人按顺序读的运行日志，控制台「运行日志」tab
            # 只渲染这个 type。metric 画曲线，log 解释这次 run 在干什么。
            sc.log(
                f"sessions {sc.session_id}, trainer {trainer.model_id} created",
                step=0,
            )
            sc.log(
                f"SFT started: {len(datums)} examples, {total_steps} steps, "
                f"model {config.model}, rank {config.lora_rank}",
                step=0,
            )

            try:
                adam = mr.AdamParams(learning_rate=config.learning_rate)
                step = 0

                for epoch in range(config.epochs):
                    for batch in batched(datums, config.batch_size):
                        # forward_backward 返回的是 future，不是结果：服务端先回
                        # request_id，loss 要再通过 retrieve_future 取。``.result()``
                        # 做的就是这件事。relay 也使用同样的两段式协议，所以忘了
                        # .result() 在本地一样会出问题。
                        with Timer() as forward:
                            out = trainer.forward_backward(batch).result()

                        # optim_step 应用 forward/backward 累积的梯度，并记录本步
                        # 使用的 learning rate。
                        with Timer() as optim:
                            trainer.optim_step(adam).result()

                        step += 1
                        sc.metric(
                            {
                                "loss/train": out.loss,
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
                        print(
                            f"epoch {epoch + 1}/{config.epochs}  "
                            f"step {step}/{total_steps}  "
                            f"loss {fmt(out.loss)}  "
                            f"{forward.seconds + optim.seconds:.1f}s"
                        )
                        if config.log_every and (
                            step % config.log_every == 0 or step == total_steps
                        ):
                            sc.log(
                                f"epoch {epoch + 1}/{config.epochs} "
                                f"step {step}/{total_steps} loss {fmt(out.loss)}",
                                step=step,
                            )

                # sampler 权重是可部署的产物：只有 adapter，不含 optimizer state。
                # ``mint://`` 是之后 sampler 和平台使用的句柄。
                path = trainer.save_weights_only(f"step-{step}").result()
                sc.log(f"training done, sampler weights saved to {path}", step=step)
                print(f"\nsampler weights: {path}")

                # 再用刚存下的权重做一次生成抽检：训练面和采样面是两条路径，
                # 这一步确认产物不是只能存、不能调用。
                prompt = [
                    message
                    for message in conversations[0]
                    if message.get("role") != "assistant"
                ] or conversations[0]
                sampler = trainer.create_sampler(model_path=path)
                sc.log(
                    f"sampler {sampler.sampling_session_id} opened on {path}",
                    step=step,
                )
                response = sampler.sample(
                    renderer.prompt_tokens(prompt),
                    num_samples=2,
                    sampling_params=mr.SamplingParams(
                        max_tokens=64, temperature=0.7
                    ),
                ).result()

                first_sample = ""
                for index, sequence in enumerate(response.sequences):
                    text = renderer.decode(sequence.tokens).strip()
                    first_sample = first_sample or text
                    print(f"sample {index}: {text[:200]}")
                    sc.log(f"post-train sample {index}: {text[:500]}", step=step)

                sc.summary(
                    final_loss=out.loss,
                    steps=step,
                    sample=first_sample[:200],
                )

            except Exception as exc:
                # 失败先写日志（人看），再发结构化 event（平台看）。event
                # 是跨侧交互用的：这里给它一个稳定的 type/code，平台才能据此
                # 标记这次 run，而不是靠解析人写的日志。
                sc.log(f"run failed: {type(exc).__name__}: {exc}", level="error")
                sc.event(
                    "run.failed",
                    level="error",
                    code=type(exc).__name__,
                    message=str(exc),
                )
                raise

    except mr.ConfigurationError as exc:
        print(f"configuration problem: {exc}", file=sys.stderr)
        return 2
    except mr.RequestFailedError as exc:
        print(f"the server rejected the work: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
