# 训练：LoraTrainer

`LoraTrainer` 是一个训练模型的句柄，由 `ServiceClient` 创建：

```python
trainer = sc.create_lora_trainer(
    base_model="Qwen/Qwen3.8-27B",
    rank=32,
    train_unembed=False,
    train_mlp=True,
    train_attn=True,
    seed=None,
    user_metadata=None,
)
```

`base_model` 可以位置传，其余参数是关键字：

```python
trainer = sc.create_lora_trainer("Qwen/Qwen3.8-27B", rank=16)
```

创建会提交 `create_model` 并等到 `model_id` 返回。trainer 不持有连接，只是 session
下的一个服务端模型。

## 方法

| 方法 | 返回 | 说明 |
|---|---|---|
| `forward_backward(data, loss_fn="cross_entropy", loss_fn_config=None, *, forward_only=False)` | `APIFuture[ForwardBackwardOutput]` | 前向 + 反向，累积梯度 |
| `optim_step(optim_params=None, *, adam_params=None)` | `APIFuture[dict]` | 应用累积的梯度；只接受 `AdamParams` |
| `get_info()` | `ModelInfo` | 服务端对这个模型的记录；同步调用 |
| `create_sampler(*, model_path=None, kind="snapshot", record_stability_info=False)` | `Sampler` | 从这个模型创建采样客户端 |
| `save_weights_and_get_sampler(*, record_stability_info=False, kind="snapshot")` | `Sampler` | 临时导出权重并创建 sampler；阻塞到导出完成。不接 `name`／`ttl_seconds` |
| `save_weights_only(name, ...)` | `APIFuture[str]` | 只保存权重，见 [Checkpoint](checkpoints.md) |
| `save_state(name, ...)` | `APIFuture[str]` | 保存权重 + 优化器状态，见 [Checkpoint](checkpoints.md) |
| `load_weights(path, *, optimizer=True)` | `APIFuture[dict]` | 向当前模型加载权重 |
| `load_state(path)` / `load_state_with_optimizer(path)` | `APIFuture[dict]` | 上游拼写，分别固定只加载权重 / 连优化器一起加载 |

属性：`model_id`、`base_model`、`lora_config`、`session_id`。

## 一个训练 step

```python
out = trainer.forward_backward(batch, loss_fn="cross_entropy").result()
trainer.optim_step(mr.AdamParams(learning_rate=1e-4)).result()
print(out.loss, out.num_tokens)
```

`batch` 是 `list[Datum]`。`Datum` 的构造和 loss mask 见 [数据与 batch](data.md)。

验证集只做前向：

```python
val = trainer.forward_backward(val_batch, forward_only=True).result()
```

`forward_only=True` 不累积梯度，不改变优化器状态。

`loss_fn` 可取值：

```text
cross_entropy  importance_sampling  ppo  cispo  dro  gspo  stable_reinforce  custom
```

`loss_fn_config` 是传给服务端的可选参数 dict。

## Adam 参数

```python
trainer.optim_step(mr.AdamParams(learning_rate=1e-4)).result()
```

MinT 的 FSDP2 后端只接受这一组：

```text
beta1=0.9, beta2=0.999, eps=1e-8, lora_plus_ratio=1
```

`AdamParams` 的默认值就是这组，所以只设置 `learning_rate` 不会触发约束。显式改
`beta1` / `beta2` / `eps` 时 SDK 原样发送并打 warning，服务端会拒绝那一步并让
future 以 `RequestFailedError` 结束。

| 字段 | 默认值 |
|---|---|
| `learning_rate` | `1e-4` |
| `beta1` | `0.9` |
| `beta2` | `0.999` |
| `eps` | `1e-8` |
| `weight_decay` | `0.0` |
| `grad_clip_norm` | `0.0` |

## 自动分块

`client/config` 返回 `fwdbwd_max_chunk_len` 和 `fwdbwd_max_chunk_bytes_count`。
超过任一预算的 batch 会被 SDK 自动切成多块分别提交，再把各块的 metrics 合并：

- `:sum` 直接相加；
- `:mean` 按各块 token 数加权；
- `:max` / `:min` 取极值；
- 其它键取简单平均。

单个 datum 超过字节预算时整体透传，由服务端返回 413。调用方不需要手动控制 batch
大小。

## 训练输出

`ForwardBackwardOutput`：

| 字段 | 内容 |
|---|---|
| `metrics` | 服务端返回的标量，如 `loss:mean`、`num_tokens:sum` |
| `loss` | `metrics["loss:mean"]`，没有则为 `None` |
| `num_tokens` | `metrics["num_tokens:sum"]`，没有则为 `None` |
| `loss_fn_outputs` | 服务端返回的逐 datum 张量 |
| `loss_fn_output_type` | 输出类型标识 |

## get_info

```python
info = trainer.get_info()
info.model_id
info.model_name
info.is_lora
info.lora_rank        # wire 上叫 lora_rank，不是 rank
info.optimizer_config
info.model_data
info.extra            # 未建模的服务端字段
```

`get_info()` 是同步调用。所有字段都可选；缺失表示服务端没有返回，不是错误。

从 checkpoint 创建 trainer 时，如果没有同时显式给出 `rank`、`train_mlp`、
`train_attn`，`trainer.lora_config` 是 `None`，而 `get_info()` 也查不到 effective
rank：这两个字段只反映建模请求里显式带上的值。续训不传 `rank` 时 `is_lora` 是
`False`、`lora_rank` 是 `None`，与 checkpoint 实际存的结构无关。见
[Checkpoint 与续训](checkpoints.md)。

## 从这里采样

```python
# 临时导出：自动命名、24 小时后过期，适合 RL 每步采样
sampler = trainer.save_weights_and_get_sampler()

# 要留下这份权重：自己存，再开 sampler
path = trainer.save_weights_only("step-100").result()
sampler = trainer.create_sampler(model_path=path)

# 跟随当前训练权重，不保存（本版本服务端还不支持，返回 400）
sampler = trainer.create_sampler(kind="inflight")
```

采样参数和返回结构见 [采样](sampler.md)。

## Tinker 拼写

| Tinker 拼写 | 原生方法 |
|---|---|
| `forward` | `forward_backward(..., forward_only=True)` 的薄壳 |
| `optim_step(optim_params=...)` | `optim_step`（只接受 `AdamParams`） |
| `save_weights_for_sampler` | `save_weights_only`（别名） |
| `save_state` | `save_state`（参数顺序已对齐） |
| `create_sampling_client` | `create_sampler` 的薄壳 |
| `save_weights_and_get_sampling_client` | `save_weights_and_get_sampler` 的真别名（同一个函数对象） |
| `load_state` | 固定 `load_weights(path, optimizer=False)` |
| `load_state_with_optimizer` | 固定 `load_weights(path, optimizer=True)` |

```python
trainer.forward(data, "cross_entropy").result()
trainer.optim_step(tinker.types.AdamParams(learning_rate=1e-5)).result()
trainer.save_state("step-1", 3600, False).result()
trainer.save_weights_for_sampler("step-1", 3600).result()
sampler = trainer.save_weights_and_get_sampling_client()   # 返回 Sampler，不是 future
```

`save_state` 在两套命名里相同。`load_state` 不是 `load_weights` 的别名：它只加载
权重，不动优化器状态。

`save_weights_and_get_sampling_client` 是 `save_weights_and_get_sampler` 的真别名
（同一个函数对象），两跳（训练和采样在两个不同的后端）：先在 `/train` 存一次权重
（SDK 自动命名 `ephemeral-<seq>`，每个 trainer 内递增，24 小时 TTL），拿到 `mint://` path；再经
`/service/sample` 由平台把这份权重部署到推理服务上并开 sampler。推理服务只能采样
已部署的权重，所以第二跳要等一次部署，SDK 自动等待；每步一个新部署，新部署就绪后
释放同一训练模型的上一个。

这条路径上的导出一律是用完就扔：不接 `name`、不接 `ttl_seconds`、不登记 artifact。
**要留下来的权重用 `save_weights_only(name)`**，再用 `create_sampler(model_path=...)`。
见 [Tinker 迁移](tinker.md)。

## 协议细节

`forward_backward` 的请求体是 protobuf，没有 JSON 路径，所以 `numpy` 是硬依赖。
服务端通过 `client/config` 要求 zstd 压缩时，安装 `mind_recursive[zstd]` 即可启用；没装会
warning 一次，然后发送未压缩请求体，服务端两种都接受。

## 相关文档

- [数据与 batch](data.md)
- [Checkpoint 与续训](checkpoints.md)
- [未来与错误](futures-and-errors.md)
- [采样](sampler.md)
