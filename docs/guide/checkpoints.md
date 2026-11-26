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

## checkpoint 会过期

`save_state` 默认在请求里带 `ttl_seconds=259200`（72 小时），所以 state path 默认
只有三天可用。要留更久就自己传：

```python
# 默认 72 小时
state = trainer.save_state("step-500").result()

# 自己定
state = trainer.save_state("step-500", ttl_seconds=86400 * 30).result()

# 不发这个字段
state = trainer.save_state("step-500", ttl_seconds=None).result()
```

`save_weights_only` 默认不带 `ttl_seconds`：这是留给长期保留的权重，需要按 TTL
回收时自己传。

命名规则：`name` 必须非空，只能包含 ASCII 字母、数字、`-`、`_`、`.`，且不能是
`.` 或 `..`。SDK 在本地校验，不合法会抛 `ConfigurationError`，不会发请求。

两者都接受 `overwrite=False`：同名保存默认报冲突（`ConflictError`），要覆盖必须
显式传 `overwrite=True`。

`ephemeral` 前缀留给 SDK 的临时导出：`save_weights_and_get_sampler()` 每次自动命名
`ephemeral-<seq>`（每个 trainer 内递增），并带 24 小时 TTL。因此
`save_weights_only` 不允许用 `ephemeral` 或 `ephemeral-` 开头作名字，SDK 在本地直接
拒绝，抛 `ConfigurationError`。

持久保存和临时导出的区别：

- **持久保存**：`save_weights_only(name)` / `save_state(name)`。必须传 `name`，会登记
  artifact。`save_state` 默认带 72 小时 TTL，`save_weights_only` 默认不带 TTL。
  **想留下的权重只走这条路。**
- **临时导出**：`save_weights_and_get_sampler()`（Tinker 拼法
  `save_weights_and_get_sampling_client()` 是它的真别名）。不接 `name`、不接
  `ttl_seconds`：SDK 自动命名、固定带 24 小时 TTL、不登记 artifact。推理服务要先
  部署这份权重才能采样，所以每步会建一个临时部署，新部署就绪后释放上一个。适合 RL
  每步采样。

要留下权重、同时又要采样，就分两步：

```python
path = trainer.save_weights_only("step-100").result()
sampler = trainer.create_sampler(model_path=path)
```

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
时才有值，否则是 `None`。

`get_info()` 也查不到 effective rank：它只反映这次建模请求里显式带上的 LoRA 字段。
续训时不传 `rank`，无论 checkpoint 存的是什么，都会得到：

```python
info = trainer.get_info()
info.is_lora    # False
info.lora_rank  # None
```

注意 `is_lora` 这里是 `False` 而不是 `None`——它是个错值，不是缺值。如果你的代码靠
它判断模型是不是 LoRA，会走错分支。

这只影响 `get_info()` 的回读，不影响续训本身。要让这两个字段有意义，续训时显式传
`rank`：

```python
trainer = sc.create_lora_trainer_from_state(
    state_path,
    base_model="Qwen/Qwen3.8-27B",
    rank=8,  # 必须与 checkpoint 一致，否则是覆盖而不是回读
)
trainer.get_info().lora_rank  # 8
```

显式传 `rank` 是覆盖参数，不是查询参数：传错值会改变这次训练的结构，所以只在你已经
知道 checkpoint 的 rank 时这样做。

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
