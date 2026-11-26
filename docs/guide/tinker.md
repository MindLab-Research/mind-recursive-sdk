# 从 Tinker 迁移

`mind_recursive.tinker` 提供 Tinker 的类名和方法名，背后是 `mind_recursive` 自己的类。它是一层
**迁移辅助**，不是完整的兼容实现：

- 名字和调用形状对得上的方法，直接别名或加一层薄壳；
- 能力本身不存在的部分（管理面、自定义 loss、外部权重导出、tokenizer 等）不假装
  支持，调用会明确失败。

所以迁移时不要假设“改个 import 就全都能跑”。先用下面的清单判断你的代码落在哪一边。

```python
# before
import tinker

# after
from mind_recursive import tinker
```

## 名字对应

| Tinker | mind_recursive.tinker |
|---|---|
| `ServiceClient` | `ServiceClient` |
| `TrainingClient` | `LoraTrainer` 的别名 |
| `SamplingClient` | `Sampler` 的别名 |
| `TinkerError` | `MindRecursiveError` 的别名 |
| `types` | `mind_recursive.types` |
| `Datum` / `ModelInput` / `TensorData` / `AdamParams` / `SamplingParams` 等 | 同名类型 |

`ServiceClient` 上：

| Tinker | mind_recursive |
|---|---|
| `create_lora_training_client` | `create_lora_trainer`（别名） |
| `create_sampling_client` | `create_sampler`（薄壳） |
| `create_training_client_from_state` | `create_lora_trainer_from_state`（薄壳） |
| `create_training_client_from_state_with_optimizer` | 同上的 `load_optimizer=True` |
| `get_server_capabilities()` / `_async()` | 兼容命名空间上的同名方法；原生 `ServiceClient` 只有 async 版 |
| `close(status, detail)` | `close()`（参数记入 `session.closing` 事件） |

`mind_recursive.tinker.ServiceClient` 是原生 `ServiceClient` 的子类，只额外补了
`get_server_capabilities` 这两个 Tinker 专属查询；其余方法都直接继承，所以两个命名空间
不会各维护一套实现。

`TrainingClient` 上：

| Tinker | mind_recursive |
|---|---|
| `forward` | `forward_backward(..., forward_only=True)` 的薄壳 |
| `forward_backward` | `forward_backward` |
| `optim_step(optim_params=...)` | `optim_step`（只支持 Adam） |
| `save_state` | `save_state` |
| `save_weights_for_sampler` | `save_weights_only`（别名） |
| `load_state` | `load_state`（只加载权重） |
| `load_state_with_optimizer` | `load_state_with_optimizer` |
| `create_sampling_client` | `create_sampler` 的薄壳 |
| `save_weights_and_get_sampling_client` | `save_weights_and_get_sampler` 的真别名；临时导出两跳：`/train` 存 `ephemeral-<seq>` 拿 path，再经 `/service/sample` 部署并开 sampler |

`SamplingClient` 上：

| Tinker | mind_recursive |
|---|---|
| `sample(prompt, num_samples, sampling_params, ...)` | `sample(...)`，参数顺序已对齐 |
| `compute_logprobs` | `compute_logprobs`，内部走一次 `sample` |
| `get_base_model` | `get_base_model`（不知道时返回 `None`） |

## 已经能对上的调用形状

下面这些写法可以直接迁移：

```python
sc = tinker.ServiceClient()

trainer = sc.create_lora_training_client(
    "Qwen/Qwen3.8-27B", 32, None, True, True, False
).result()

trainer.forward_backward(data, "cross_entropy", config).result()
trainer.forward(data, "cross_entropy").result()
trainer.optim_step(tinker.types.AdamParams(learning_rate=1e-5)).result()

trainer.save_state("step-1", 3600, False).result()
trainer.save_weights_for_sampler("step-1", 3600).result()

sampler = trainer.create_sampling_client("mint://run/adapters/step-1").result()
sampler = trainer.save_weights_and_get_sampling_client().result()
sampler = sc.create_sampling_client(model_path="mint://run/adapters/step-1").result()

result = sampler.sample(prompt, 4, tinker.types.SamplingParams(max_tokens=64), True).result()
logprobs = sampler.compute_logprobs(prompt).result()

sc.close("success", detail="training complete").result()
```

几个要点：

- 创建类调用在本 SDK 里是同步的，返回对象本身；`.result()` 是兼容壳，返回同一个
  对象。它不会像上游那样先返回 future、稍后再解析。
- `save_state` / `save_weights_for_sampler` 的 future 解析成路径字符串；这个字符串
  同时有 `.path` 属性，所以上游 `result().path` 的写法也能用。
- `sample` 的 `num_samples`、`sampling_params` 是位置参数，和上游一致；
  `include_prompt_logprobs` 是 `prompt_logprobs` 的上游拼写，二选一即可。
- `save_weights_and_get_sampling_client()` 是 `save_weights_and_get_sampler` 的真
  别名（同一个函数对象），临时导出走两跳（训练和采样是两个平台）：先在 `/train` 存
  一次权重拿 `mint://` path（SDK 自动命名 `ephemeral-<seq>`，每个 trainer 内递增，
  24h TTL），再经
  `/service/sample` 由平台把这份权重部署到推理服务上并开 sampler。推理服务只能采样
  已部署的权重，所以第二跳要等一次部署，SDK 自动等待；每步一个新部署，新部署就绪后
  释放上一个。临时存盘不上报 artifact。持久保存不能用 `ephemeral` 前缀。
- 这个方法**不接 `name`、`ttl_seconds` 或 `retry_config`**，上游传这些参数的调用会
  抛 `TypeError`。上游用 `name` 想留下的权重，在这里走 `save_weights_only(name)`
  再 `create_sampler(model_path=...)`。`kind="inflight"` 跳过导出直接绑活模型，
  同样经 `/service/sample`。

## 没有覆盖的部分

这些不是“以后再说”的占位，而是当前**没有对应能力**，调用会失败：

| 上游能力 | 状态 |
|---|---|
| `RestClient` / `AdminClient` | 不提供；用户级查询用 `UserClient`（只有 `health_check` / `get_server_capabilities`） |
| `get_console_url` | 不提供；没有控制台 URL 概念 |
| `copy_weights` | 不提供；平台没有 project 概念 |
| `create_custom_loss` / `forward_backward_custom` | 不提供；自定义 loss 图契约未定 |
| `save_weights_external` / `external_weights_upload_*` | 不提供；v1 用 `mint://` checkpoint |
| `get_tokenizer` | 不提供；tokenisation 留在 recipe 层，SDK 只认 token |
| `get_sampler_handle` / `from_sampler_handle` | 不提供；sampler 生命周期由 session 管理 |
| `on_queue_state_change` | 不提供；队列状态由 future 轮询内部处理 |
| telemetry / audit / billing | 不提供 |
| `*_async` / `await` | 提供；原生 async 传输，同名方法，后端接口不变 |
| `DimuonParams` 等非 Adam 优化器 | 不接受；`optim_params` 只接受 `AdamParams` |
| `retry_config` | 不接受；重试策略在 `ServiceClient(max_retries=...)` |
| `weights_access_token` | 不接受；平台不支持跨账号读取 |
| `create_training_client_from_state(path)` 不传 `base_model` | 不支持；本版本 `base_model` 必填 |
| `get_info().model_data.model_name` | 字段形状不同；本 SDK 用 `info.model_name` |

## 行为差异

### 类型身份

两边的类不同：

```python
import tinker as real_tinker
from mind_recursive import tinker

tinker.TrainingClient is mr.LoraTrainer        # True
isinstance(datum, real_tinker.types.Datum)          # False
```

捕获异常要改成：

```python
except mr.MindRecursiveError:
    ...
```

### `load_state` 与 `load_weights`

`load_state(path)` 只加载权重，`load_weights(path, optimizer=True)` 默认连优化器一起
加载。两者不是别名。迁移时如果原代码依赖 Tinker 的 `load_state`，继续用
`load_state` 即可保持语义。

### 默认值

| 参数 | mind_recursive | Tinker 0.39 |
|---|---|---|
| `AdamParams.beta2` | `0.999`（MinT/FSDP2 强制） | `0.95` |
| `AdamParams.eps` | `1e-8`（MinT/FSDP2 强制） | `1e-12` |
| `LoraConfig.train_unembed` | `False`（训 LM head 的权重无法部署） | `True` |

这些差异不会报错，但会改变训练结果，迁移时逐项确认。

### `close` 的 status/detail

`close("success", detail="...")` 的签名保留；`status` / `detail` 不会发给 finish
端点，而是记入 `session.closing` 生命周期事件。session 仍会被正常结束。

## 迁移检查清单

1. `import tinker` 改成 `from mind_recursive import tinker`；
2. `from tinker.types import ...` 改成 `from mind_recursive import types` 或 `tinker.types`；
3. 检查是否使用了 `RestClient` / `AdminClient`，这些需要改为 `UserClient` 或平台接口；
4. 检查 `get_tokenizer()`，SDK 不提供，需要在 recipe 里用 transformers 加载；
5. 检查 `except tinker.*Error`，改成 `except mr.MindRecursiveError`；
6. 检查 `isinstance` 对上游类型的判断；
7. 检查默认值：Adam 的 `beta2` / `eps`、`train_unembed`；
8. 检查 `create_training_client_from_state` 是否传了 `base_model`；
9. 检查 `retry_config`、`weights_access_token` 这些本 SDK 不接受的参数；
10. 检查训练调用是否都调用了 `.result()`。

## 相关文档

- [ServiceClient](service-client.md)
- [训练](trainer.md)
- [Checkpoint 与续训](checkpoints.md)
