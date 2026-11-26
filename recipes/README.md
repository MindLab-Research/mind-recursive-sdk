# Recipes

Recipe = 训练流程编排。SDK 给原语（`forward_backward` / `optim_step` /
`sample`），recipe 决定**按什么顺序、用什么数据、什么时候存**。

这个目录里的东西是**可以直接跑的**，也是拿来改的起点。

`_renderer.py` 也在这里，而不在 SDK 里：把对话消息渲染成 token + loss mask 依赖
模型的 chat template 和数据集自己的监督策略，那是 recipe / cookbook 的事。SDK 只认
`Datum`（token + 张量）。

## 先跑这个

```bash
pip install "mind-recursive[recipes]"
export MIND_RECURSIVE_API_KEY=sk-...

# 不耗配额，验证环境通不通（打 mock-model）
python recipes/00_smoke.py
```

## 怎么改参数：改 `CONFIG`，不传命令行

每个 recipe 顶部有一个 `CONFIG` 块，跑之前改它：

```python
@dataclass
class Config:
    data: str | None = None          # None = 用内置的小数据集
    model: str = "Qwen/Qwen3.8-27B"
    base_url: str | None = None      # None = 读 MIND_RECURSIVE_BASE_URL
    epochs: int = 1
    batch_size: int = 4
    learning_rate: float = 1e-4
    ...

CONFIG = Config()
```

然后 `python recipes/01_sft.py`，不带任何参数。recipe 是**拿来改的起点**，不是要配置
的工具 —— 改一个 dataclass 字段比记一串 flag 直接，而且改完那份文件就是你这次跑的
完整记录。

## 目录

| 文件 | 做什么 | 行数 |
|---|---|---|
| [`00_smoke.py`](00_smoke.py) | 环境自检：能连上、能建 client、能跑一步、能存权重 | 119 |
| [`01_sft.py`](01_sft.py) | **SFT**。最常用的那条路，从 JSONL 到可部署权重 | 270 |
| [`02_sft_full.py`](02_sft_full.py) | SFT 加上该有的东西：验证集、断点续训、早停、LR schedule | 307 |
| [`03_grpo.py`](03_grpo.py) | **GRPO**。采样 → 打分 → 组内归一化 → 策略梯度 | 369 |
| [`_common.py`](_common.py) | 几个 recipe 共用的小工具（读数据、分批、计时） | 134 |
| [`_renderer.py`](_renderer.py) | 消息 → token + loss mask。**不是 SDK 的一部分**，归 cookbook 的那类代码 | 376 |

读的顺序建议 `00` → `01` → `02` → `03`。`01` 是骨架，`02` 和 `03` 都是在它上面加东西。

## 一个 SFT 长什么样

把 `01_sft.py` 剥到只剩骨头，就是这 15 行：

```python
import mind_recursive as mr
from _renderer import get_renderer          # recipe 本地的，不是 SDK 的

sc = mr.ServiceClient()
renderer = get_renderer(MODEL)
trainer = sc.create_lora_trainer(base_model=MODEL, rank=32)

data = [renderer.render_to_datum(c) for c in conversations]

for step, batch in enumerate(batches(data, size=8)):
    out = trainer.forward_backward(batch).result()        # 前向 + 反向
    trainer.optim_step(mr.AdamParams(learning_rate=1e-4)).result()  # 更新
    print(step, out.loss)

path = trainer.save_weights_only("step-100").result()  # 可部署的权重
```

剩下的约 250 行都是：`CONFIG` 块、`sc.log/metric/...` 上报、详细的逐步注释和错误处理。
**编排本身就是这么短**，这是 SDK 的目标。

## 数据格式

SFT 的输入是 JSONL，一行一个对话：

```json
{"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]}
```

GRPO 的输入是 JSONL，一行一个 prompt（加上打分需要的东西）：

```json
{"messages": [{"role": "user", "content": "2+2?"}], "answer": "4"}
```

`01_sft.py` 默认读 `recipes/data/sft_demo.jsonl`（8 条 demo 对话），不需要先准备数据；
其他 recipe 的 `CONFIG.data` 默认是 `None`，这时用内置的小数据集。要用自己的数据就把它改成
JSONL 的路径。

## loss mask 是 recipe 最容易写错的地方

`_renderer.py` 的 `render_to_datum(messages)` 做的事是：消息 → token + **每个 token
的 loss 权重**。只有 assistant 回答的那些 token 权重是 1，用户输入和模板脚手架是 0。

写错了**不会报错**，只是训出来的模型行为漂。所以 recipe 里都留了一行：

```python
print(rendered.debug_spans(renderer.tokenizer))
# 输出：<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n[答案<|im_end|>\n]
#                                                             ^^^^ 方括号内是算 loss 的
```

**第一次换数据集或换模型时，一定看一眼这个。**

## 自己写 recipe

抄 `01_sft.py` 改。三件事是框架性的、不建议动：

1. **`sc.close()` 要执行到**（用 `with` 或 `try/finally`）。不然 session 泄漏，
   GPU 槽位要等服务端 idle 回收。
2. **上报直接挂在 client 上**：`sc.log` / `sc.event` / `sc.metric` / `sc.artifact` /
   `sc.summary`，没有单独的 reporter 对象，也没有 `start` / `finish`。
   `log(text)` 只是 `event("log", message=text)` 的语法糖。
   run 的生命周期跟着 `ServiceClient` 的 session 走，`sc.close()` 会结束 session。
3. **`.result()` 不要忘**。`forward_backward` 返回的是 future，不 `.result()`
   就只是提交了没等结果，下一步的 `optim_step` 会基于不完整的梯度。

其他随便改。

## 这些 recipe 验证到哪一步

| 验证 | 状态 |
|---|---|
| 逻辑跑通（fake relay，真 socket） | ✅ `tests/test_recipes.py`，39 个测试 |
| loss mask 正确（真 tokenizer） | ✅ `tests/test_recipe_renderer.py` |
| 打真机 alpha `mock-model` | ⬜ 待跑 `00_smoke.py` |
| 真实 SFT 收敛、权重可部署 | ⬜ |

诚实说：**逻辑是通的，但还没在真机上训出过模型。**
`00_smoke.py` 就是为了让你用一条命令确认前半段。
