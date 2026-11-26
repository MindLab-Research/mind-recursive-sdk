# 数据与 batch

训练和采样接收的都是 **token id**，不是文本。SDK 定义了三层数据对象：

```text
Datum
  model_input: ModelInput      # 喂给模型的 token / chunk 序列
  loss_fn_inputs: dict[str, TensorData]   # 算 loss 的张量
```

## ModelInput

最常见的形式是一段 token：

```python
model_input = mr.ModelInput.from_ints([151644, 872, 198, ...])
```

也可以由多个 chunk 组成：

| chunk | 内容 |
|---|---|
| `EncodedTextChunk(tokens=[...])` | token id 序列 |
| `ImageChunk(data=b"...", format="png", expected_tokens=None)` | 内联图片字节 |
| `DmelChunk(dmel=b"...")` | 序列化音频张量 |

```python
model_input = mr.ModelInput(chunks=[
    mr.EncodedTextChunk(tokens=prompt_tokens),
    mr.ImageChunk(data=image_bytes, format="png"),
])
```

`ModelInput.to_ints()` 只在全部是文本 chunk 时可用；有图片或音频会抛 `ValueError`。
`len(model_input)` 返回各 chunk 长度之和；`model_input.length` 等价。

采样请求里的 `prompt` 可以直接传 `ModelInput`，也可以传 token id 列表。

## Datum

一个 `Datum` 是一条训练样本：

```python
datum = mr.Datum(
    model_input=mr.ModelInput.from_ints(tokens),
    loss_fn_inputs={
        "weights": mask,
        "target_tokens": targets,
    },
)
```

`loss_fn_inputs` 的值可以是：

- `TensorData`；
- numpy 数组；
- Python list；
- 任何能转成 numpy 的对象（包括 CPU torch tensor）。

构造时会自动包成 `TensorData`。

### loss_fn_inputs 的键

键由 `loss_fn` 决定：

| loss_fn | 常用键 |
|---|---|
| `cross_entropy` | `target_tokens`、`weights` |
| `importance_sampling` | `target_tokens`、`weights`、`advantages`、`logprobs` |
| `ppo` / `cispo` / `dro` / `gspo` / `stable_reinforce` | 目标、权重，以及各算法需要的 advantage / logprob 张量 |

SFT 的 `weights` 是逐 token 的 loss mask：

```text
0 = 这个位置不计入 loss
1 = 计入 loss
```

屏蔽 prompt、只在 assistant 回复上训练，靠的就是它。

### dtype

已知键的 dtype 由键名决定：

| 键 | dtype |
|---|---|
| `target_tokens` | `int64` |
| `weights` | `float32` |
| `advantages` | `float32` |
| `logprobs` | `float32` |
| `clip_low_threshold` / `clip_high_threshold` | `float32` |
| `sequence_ratio_mask` | `float32` |

所以 `"weights": [1, 0, 1]` 在线上仍然是 float32，不会因为写成整数就变成 int64。
未知键按 numpy 数组的 kind 推断：浮点是 float32，其它是 int64。

公开 wire 只暴露两种 dtype：`float32` 和 `int64`。

### 快捷构造

```python
datum = mr.Datum.from_tokens(tokens)
```

等价于：

- `target_tokens` = `tokens[1:] + [0]`（标准的下一个 token 预测）；
- `weights` 全为 `1.0`。

全 1 mask 会连 prompt 一起训练，只适合 smoke test。真实训练请显式传 `weights` 和
`target_tokens`：

```python
datum = mr.Datum.from_tokens(
    tokens,
    weights=mask,
    target_tokens=targets,
)
```

### frozen

`Datum` 是 frozen dataclass。攒进 batch 之后再改会创建新对象，不会悄悄改变已经
提交的那一步训练内容。

## batch

`forward_backward` 接收 `Sequence[Datum]`：

```python
batch = [datum_1, datum_2, ...]
out = trainer.forward_backward(batch, loss_fn="cross_entropy").result()
```

SDK 会按服务端协商的 chunk 预算自动切分，见 [训练](trainer.md#自动分块)。
单个 datum 超过预算时整体透传，服务端返回 413。

## 消息渲染不属于 SDK

把对话消息渲染成 token + loss mask 依赖模型的 chat template 和数据集自己的监督策略，
不在 SDK 内。SDK 只接收 `Datum`。

仓库的 `recipes/_renderer.py` 有一份可用实现，基于 Hugging Face chat template 和
offset mapping，供 recipe 直接使用。它属于 recipe / cookbook，不是库的功能。

loss mask 错一位不会报错，只会训错位置。换数据集或换模型时，建议把 mask 打印出来
检查：把每个 datum 的 `weights` 非零比例打印出来。

## 相关文档

- [训练](trainer.md)
- [采样](sampler.md)
- [recipes/README.md](../../recipes/README.md)
