# 采样：Sampler

`Sampler` 做 token 级采样：token id 进，token id + logprob 出。

## 创建

### 从 ServiceClient

采基座模型或一份已保存的权重，二选一：

```python
# 采基座
sampler = sc.create_sampler(base_model="Qwen/Qwen3.8-27B")

# 采保存下来的权重：要带上训练时的 base_model
sampler = sc.create_sampler(
    model_path="mint://.../adapters/step-100", base_model="Qwen/Qwen3.8-27B"
)
```

两条路径在服务端都会先在推理平台上新建一个部署：基座采样不再复用已有的公开部署，
训练权重则要加载到指定基座上。因此 `base_model` 传的是**基座名字**（如
`Qwen/Qwen3.8-27B`），不是推理平台上的部署 ID；基座采样也要等部署就绪，通常几十秒。

采训练权重时，`model_path` 必须和 `base_model` 一起传；只传 `model_path` 会返回 400。
从 trainer 创建（下一节）会自动带上，推荐用那种写法。

### 从 LoraTrainer

```python
# 保存一份再采：权重固定，跨 step 可比
sampler = trainer.save_weights_and_get_sampler("step-100")

# 等价于分两步
path = trainer.save_weights_only("step-100").result()
sampler = trainer.create_sampler(model_path=path)

# 不保存，跟随当前训练权重
sampler = trainer.create_sampler(kind="inflight")
```

`kind` 决定采哪份权重：

| kind | 采什么 | 典型用途 |
|---|---|---|
| `"snapshot"`（默认） | `base_model` 或 `model_path` 指向的固定权重 | 评测、benchmark |
| `"inflight"` | 训练模型的当前权重，随 `optim_step` 变化 | on-policy RL，不必每步存 checkpoint |

`"inflight"` 只能从 trainer 创建，因为它需要 `model_id`；它不接受 `base_model` 或
`model_path`。

**本版本服务端还不支持 `"inflight"`**，创建时返回 400
（`inflight sampling is not supported yet`）：推理平台目前只能采已部署的 checkpoint。
on-policy RL 先用 `save_weights_and_get_sampler()`，每步存一份再采。

## sample

```python
resp = sampler.sample(
    prompt,
    num_samples=1,
    sampling_params=None,
    include_prompt_logprobs=None,
    topk_prompt_logprobs=0,
    topk_sample_logprobs=0,
    target_prompt_logprobs=None,
    prompt_alt_tokens_k=0,
    prompt_logprobs_last_n=None,
    *,
    seq_id=None,
    prompt_logprobs=False,
    record_stability_info=False,
).result()
```

`prompt` 可以是 `ModelInput`，也可以是 token id 列表。

| 参数 | 说明 |
|---|---|
| `num_samples` | 采样几条 continuation，至少 1 |
| `sampling_params` | 解码参数；不传用默认值 |
| `include_prompt_logprobs` | Tinker 拼写，等价于 `prompt_logprobs`；两者冲突时本地报错 |
| `seq_id` | 本轮在 sampling session 内的序号；多轮采样时每轮给不同值 |
| `prompt_logprobs` | 额外返回 prompt 逐 token 的 logprob |
| `prompt_logprobs_last_n` | 只给最后 N 个 prompt token 打分；自动开启 `prompt_logprobs` 的前置校验 |
| `topk_prompt_logprobs` | 每个 prompt 位置返回 top-k |
| `topk_sample_logprobs` | 每个采样位置返回 top-k |
| `target_prompt_logprobs` | 给指定 token、指定 prompt 位置打分 |
| `prompt_alt_tokens_k` | 每个 prompt 位置额外抽 k 个候选 token |
| `record_stability_info` | 让服务端记录逐序列稳定性信息，供 `stable_reinforce` 使用 |

### 本地校验

以下组合在发请求前就会报 `ValueError`：

- `num_samples < 1`；
- `prompt_logprobs_last_n` 没配 `prompt_logprobs=True`；
- `prompt_logprobs_last_n < 1`，或超过 `len(prompt) - 1`；
- `topk_prompt_logprobs` / `topk_sample_logprobs` / `prompt_alt_tokens_k` 为负；
- `seq_id < 0`；
- `include_prompt_logprobs=False` 与 `prompt_logprobs=True` 冲突。

### compute_logprobs

Tinker 风格的 prompt 打分入口，内部就是一次 `max_tokens=1`、开启 prompt
logprobs 的 `sample`，不发额外请求：

```python
logprobs = sampler.compute_logprobs(prompt).result()
logprobs = sampler.compute_logprobs(prompt, prompt_logprobs_last_n=8).result()
```

## 返回

```python
resp = sampler.sample(tokens, num_samples=4).result()

resp.sequences                   # list[SampledSequence]
resp.prompt_cache_hit_tokens
```

每条序列：

```python
seq = resp.sequences[0]
seq.tokens          # list[int]
seq.logprobs        # list[float] | None，按 token 对齐
seq.topk_logprobs   # TopkLogprobs | None
seq.stop_reason     # "stop" | "length"
seq.sequence_id
len(seq)
```

`SampleResponse` 的其它字段只有在请求里显式要求时才有值：

| 字段 | 要求 |
|---|---|
| `prompt_logprobs` | `prompt_logprobs=True` |
| `topk_prompt_logprobs` | `topk_prompt_logprobs=k` |
| `target_prompt_logprobs` | `target_prompt_logprobs=...` |
| `prompt_alt_tokens` | `prompt_alt_tokens_k=k` |

## logprobs

采样出来的 token 的 logprob **无条件返回**，在 `seq.logprobs` 里。prompt 侧的
logprob 需要额外 prefill，所以都要显式开启。

### prompt 逐 token 打分

```python
resp = sampler.sample(tokens, prompt_logprobs=True).result()
resp.prompt_logprobs
```

只给最后 N 个 token 打分：

```python
resp = sampler.sample(
    tokens,
    prompt_logprobs=True,
    prompt_logprobs_last_n=16,
).result()
```

没打分的位置是 `None`。前面的前缀可以走服务端 prefix cache，所以给长 prompt 的短
后缀打分，开销接近后缀本身。

### top-k

```python
resp = sampler.sample(tokens, topk_sample_logprobs=5).result()
topk = resp.sequences[0].topk_logprobs

topk.row(0)        # [(token_id, logprob), ...]，best first
topk.to_rows()     # 每个位置一行
topk.k, topk.length
```

内部是行主序：位置 `i`、第 `j` 名在 `i * k + j`。空格子的 token id 是 0，logprob
是 `-99999.0`。

### 指定 token 打分

`target_prompt_logprobs` 接收一个 int64 张量，形状 `[len(prompt) - 1, K]`：

- 第 `i` 行对应 prompt 位置 `i + 1`；
- `-1` 表示该格不打分；
- 返回同形状张量，请求里是 `-1` 的格子返回 `0.0`。

```python
import numpy as np

ids = np.full((len(prompt_tokens) - 1, 1), -1, dtype=np.int64)
ids[position - 1, 0] = candidate_token_id

resp = sampler.sample(
    prompt_tokens,
    sampling_params=mr.SamplingParams(max_tokens=1),
    target_prompt_logprobs=mr.TensorData(ids, "int64"),
).result()
resp.target_prompt_logprobs
```

配 `max_tokens=1` 就是一次纯 prefill 打分。稀疏 CSR 也支持：给
`TensorData` 传 `sparse_crow_indices` / `sparse_col_indices`，只发送和返回点名的格子。

## 解码参数

```python
sampling_params=mr.SamplingParams(
    max_tokens=512,
    temperature=1.0,
    top_p=1.0,
    top_k=None,
    seed=None,
    stop=None,
    stop_tokens=None,
)
```

不传 `sampling_params` 时使用上面这组默认值。

`stop` 是文本停止序列，`stop_tokens` 是 token id 停止序列，两者在 wire 上合并成同一个
`stop` 字段。同时给两个会在本地报 `ValueError`。

## prompt_cache_hit_tokens

`resp.prompt_cache_hit_tokens` 是**单份** prompt 的缓存命中数。`num_samples=N` 时，
其余 `N-1` 份 prompt 按缓存命中计费；估算成本时不要把这个数乘以 N。

## 相关文档

- [训练](trainer.md)
- [Checkpoint 与续训](checkpoints.md)
- [Future 与错误](futures-and-errors.md)
