# Checkpoint 与续训

这一页讲四件事：保存什么、怎么加载、怎么从 checkpoint 起一个新的训练、以及续训时
选哪个入口。

## 两种保存

`LoraTrainer` 有两个公开的保存方法：

| 方法 | 保存内容 | 路径 | 用途 |
|---|---|---|---|
| `save_weights_only(name)` | 只有 LoRA 权重 | `mint://.../adapters/...` | 采样、部署 |
| `save_state(name)` | 权重 + 优化器状态 | `mint://.../checkpoints/...` | 续训 |

```python
weights_path = trainer.save_weights_only("step-100").result()
state_path = trainer.save_state("step-100").result()
```

两者的返回值都是 `APIFuture[str]`，`.result()` 得到路径。

命名规则：`name` 必须非空，只能包含 ASCII 字母、数字、`-`、`_`、`.`，且不能是
`.` 或 `..`。SDK 在本地校验，不合法会抛 `ConfigurationError`，不会发请求。

两者都接受 `overwrite=False`：同名保存默认报冲突（`ConflictError`），要覆盖必须
显式传 `overwrite=True`。

`save_weights_only` 不能用 `ephemeral` 前缀作名字。`save_weights_and_get_sampling_client()`
每步把临时导出存成 `ephemeral-<seq>`，所以这些名字是保留的，SDK 在本地直接拒绝，
抛 `ConfigurationError`。

两种保存的区别：

- **持久保存**：传 `name`（`save_weights_only` / `save_weights_and_get_sampler`），
  想留下的权重走这条路。
- **临时保存**：`save_weights_and_get_sampling_client()`，不传 `name`。每步存一个
  `ephemeral-<seq>`、24 小时后过期；推理服务要先部署这份权重才能采样，所以每步会建
  一个临时部署，新部署就绪后释放上一个。适合 RL 每步采样。

保存成功后，路径会自动登记为 artifact：`save_weights_only` 登记
`sampler_weights`，`save_state` 登记 `checkpoint`。不需要再调用 `sc.artifact()`。
保存失败不会登记。

## 加载到当前模型

`load_weights(path, *, optimizer=True)` 把 checkpoint 加载进一个已经存在的模型：

```python
trainer.load_weights(state_path).result()                 # 权重 + 优化器
trainer.load_weights(weights_path, optimizer=False).result()  # 只加载权重
```

`optimizer=True` 是默认值，用于续训；`optimizer=False` 用于只换权重、保留当前优化器
状态（例如回滚到某个权重快照）。

上游拼写的两个方法固定了 `optimizer`：

```python
trainer.load_state(path).result()                   # 只加载权重
trainer.load_state_with_optimizer(path).result()    # 权重 + 优化器
```

`load_state` 不是 `load_weights` 的别名，它只加载权重。

## 从 checkpoint 起训

`create_lora_trainer_from_state` 在一次提交里创建模型并加载 checkpoint：

```python
trainer = sc.create_lora_trainer_from_state(
    "mint://.../checkpoints/sft-step-500",
    base_model="Qwen/Qwen3.8-27B",
)
```

不会再先随机初始化一份 adapter 再覆盖。

### 参数

| 参数 | 必填 | 说明 |
|---|---|---|
| `path` | 是 | `save_state` 返回的 `checkpoints/...` 路径 |
| `base_model` | 是 | 这次训练使用的 API 层模型名 |
| `rank` | 否 | 覆盖 checkpoint 的 rank |
| `train_mlp` | 否 | 覆盖 MLP 是否参与训练 |
| `train_attn` | 否 | 覆盖 attention 是否参与训练 |
| `load_optimizer` | 否 | `None` 跟随 checkpoint；`False` 只加载权重；`True` 要求 checkpoint 含优化器 |
| `user_metadata` | 否 | 记录在资源上的标签 |

不传的结构参数不会出现在请求里，由服务端从 checkpoint 解析：

```python
# 只覆盖 rank，其余沿用 checkpoint
trainer = sc.create_lora_trainer_from_state(
    state_path,
    base_model="Qwen/Qwen3.8-27B",
    rank=16,
)
```

`trainer.lora_config` 只有在 `rank`、`train_mlp`、`train_attn` **三个都显式给出**
时才有值，否则是 `None`。effective rank 用：

```python
trainer.get_info().lora_rank
```

### 路径类型

`path` 必须是训练状态（`checkpoints/...`）。`save_weights_only` 产出的
`adapters/...` 不是训练状态，服务端会拒绝。SDK 不检查 `mint://` 前缀或路径形状，
只校验非空字符串；路径能否解析由服务端决定。

### base model 身份

`base_model` 必须是这个 checkpoint 对应的模型。当前服务端不会从 artifact 推断
`base_model`，SDK 也不做本地身份校验；传错时可能加载成功，所以续训时要自己保证
`base_model` 与保存 checkpoint 时一致。

## 两种续训方式怎么选

| 场景 | 用 |
|---|---|
| 新建一个模型，从 checkpoint 继续训练 | `sc.create_lora_trainer_from_state(path, base_model=...)` |
| 往一个已经存在的模型加载权重 / 回滚 / warm start | `trainer.load_weights(path, optimizer=...)` |

完整例子：

```python
with mr.ServiceClient() as sc:
    # 第一次训练
    first = sc.create_lora_trainer(base_model="Qwen/Qwen3.8-27B", rank=32)
    ...
    state = first.save_state("step-500").result()

with mr.ServiceClient() as sc:
    # 新进程续训：一次提交完成建模型 + 加载
    resumed = sc.create_lora_trainer_from_state(
        state,
        base_model="Qwen/Qwen3.8-27B",
    )
    out = resumed.forward_backward(next_batch).result()
    resumed.optim_step(mr.AdamParams(learning_rate=1e-4)).result()
```

如果只想在同一个模型上继续，不需要新建：

```python
trainer.load_state_with_optimizer(state).result()
```

## 相关文档

- [训练](trainer.md)
- [数据与 batch](data.md)
- [上报](reporter.md)：保存成功后的 artifact 自动登记
