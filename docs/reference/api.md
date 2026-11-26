# API 参考

公开符号都从 `mind_recursive` 导出。这一页按对象列出签名和返回类型；使用示例见
[guide/](../guide/README.md)。

## 顶层导出

```python
import mind_recursive as mr
mr.ServiceClient
mr.UserClient
mr.LoraTrainer
mr.Sampler
mr.APIFuture

mr.AdamParams
mr.Datum
mr.LoraConfig
mr.ModelInput
mr.ModelInfo
mr.SamplingParams
mr.TensorData
mr.ServerCapabilities
mr.SupportedModel
mr.SampleResponse
mr.SampledSequence
mr.ForwardBackwardOutput
mr.TopkLogprobs
mr.PromptAltTokens
mr.EncodedTextChunk
mr.ImageChunk
mr.DmelChunk
mr.TensorDtype
mr.LossFnType
mr.SamplerKind
mr.ClientConfig
mr.Config
```

`from mind_recursive import tinker` 是 Tinker 迁移命名空间，见
[Tinker 迁移](../guide/tinker.md)。

异常也都从这里导出，见 [错误](#错误)。

## UserClient

```python
UserClient(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
    connect_timeout: float = 30.0,
    max_retries: int = 3,
    trust_env: bool = False,
    config: Config | None = None,
)
```

构造不发请求。

| 方法 | 返回 | 说明 |
|---|---|---|
| `health_check()` | `bool` | training 面健康检查；不抛异常 |
| `get_server_capabilities()` | `ServerCapabilities` | 当前提供的模型 |
| `whoami()` | `Account` | 当前 key 的用户与训练准入 |
| `get_platform_capabilities()` | `PlatformCapabilities` | 平台开了哪些功能 |
| `close()` | `None` | 关闭连接 |

属性：`config`、`transport`。支持 `with`。

命名空间（走平台入口 `/api/mint/cli/v1`，详见
[数据集、托管训练、部署与推理](../guide/platform.md)）：

| 方法 | 返回 |
|---|---|
| `datasets.upload(file, *, type="sft", dataset_version=None, note=None, idempotency_key=None, wait=True, timeout=600.0, poll_interval=1.0)` | `Dataset` |
| `datasets.wait(dataset_version, *, timeout=600.0, poll_interval=1.0)` | `Dataset` |
| `datasets.list(*, type=None, page=None, page_size=None)` | `Page[Dataset]` |
| `datasets.get(dataset_version, *, owner_uid=None)` | `Dataset` |
| `datasets.rows(dataset_version, *, offset=None, limit=None, owner_uid=None)` | `list` |
| `datasets.validation_report(dataset_version, *, offset=None, limit=None, owner_uid=None)` | `ValidationReport` |
| `datasets.download(dataset_version, destination, *, timeout=600.0)` | `str` |
| `datasets.update_note(dataset_version, note)` | `Dataset` |
| `datasets.delete(dataset_version)` | `None` |
| `training.submit(*, dataset_version=None, base_model=None, config=None, kind="sft", display_name=None, package_version=None, base_version_id=None, dataset_owner_uid=None, base_version_owner_uid=None, idempotency_key=None)` | `Execution` |
| `training.retry(execution_id, *, continue_training=False, config=None, display_name=None, idempotency_key=None)` | `Execution` |
| `training.list(*, state=None, updated_since=None, page=None, page_size=None)` | `Page[Execution]` |
| `training.get(execution_id)` | `Execution` |
| `training.wait(execution_id, *, timeout=None, poll_interval=10.0)` | `Execution` |
| `training.cancel(execution_id)` | `Execution` |
| `training.rename(execution_id, display_name)` | `Execution` |
| `training.delete(execution_id)` | `None` |
| `training.progress(execution_id, *, tail=None)` | `list[str]` |
| `training.metrics(execution_id, *, cursor=None, limit=None, keys=None, include_snapshots=False)` | `MetricsPage` |
| `training.iter_metrics(execution_id, *, keys=None, limit=None, include_snapshots=False)` | `Iterator[MetricPoint]` |
| `training.metric_keys(execution_id)` | `list[MetricKey]` |
| `training.result(execution_id)` | `ExecutionResult` |
| `training.events(execution_id, *, page=None, page_size=None)` | `Page[ExecutionEvent]` |
| `models.list(*, page=None, page_size=None)` | `Page[ModelVersion]` |
| `models.get(version_id, *, owner_uid=None)` | `ModelVersion` |
| `models.lineage(version_id, *, owner_uid=None)` | `list[ModelVersion]` |
| `models.delete(version_id)` | `None` |
| `deployments.create(*, version_id, display_name=None, version_owner_uid=None, source=None)` | `Deployment` |
| `deployments.wait(deployment_id, *, until="ready", timeout=1800.0, poll_interval=10.0)` | `Deployment` |
| `deployments.list(*, state=None, page=None, page_size=None)` | `Page[Deployment]` |
| `deployments.get(deployment_id)` / `stop` / `start` / `delete` | `Deployment` |
| `chat.completions.create(*, model, messages, stream=False, timeout=600.0, **params)` | `ResponseObject`，或 `stream=True` 时 `Iterator[ResponseObject]` |

`chat.completions` 走公网推理入口 `{origin}/v1/*`，其余走平台入口。
`ResponseObject` 是 `dict` 子类，支持点号访问。这些命名空间暂时只有同步版本。

## ServiceClient

```python
ServiceClient(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 600.0,
    connect_timeout: float = 30.0,
    max_retries: int = 3,
    heartbeat_seconds: float = 10.0,
    name: str | None = None,
    client_id: str | None = None,
    trust_env: bool = False,
    config: Config | None = None,
    tags: list[str] | None = None,
    user_metadata: dict[str, Any] | None = None,
)
```

构造即创建 session。

| 方法 | 返回 | 说明 |
|---|---|---|
| `create_lora_trainer(base_model, rank=32, seed=None, train_mlp=True, train_attn=True, train_unembed=False, user_metadata=None, optimizer=None)` | `LoraTrainer` | 创建训练模型；`optimizer` 只接受 AdamW |
| `create_lora_trainer_from_state(path, *, base_model, rank=None, train_mlp=None, train_attn=None, load_optimizer=None, user_metadata=None)` | `LoraTrainer` | 创建模型并加载 checkpoint |
| `create_sampler(model_path=None, *, base_model=None, model_id=None, kind="snapshot", record_stability_info=False, deploy_base_model=None, training_model_id=None)` | `Sampler` | 创建采样客户端 |
| `log(message, *, level="info", step=None)` | `None` | 运行日志 |
| `event(type, *, level="info", code=None, message=None, data=None, step=None)` | `None` | 结构化事件 |
| `metric(metrics, *, step, stage=None, total=None, timings=None, counters=None)` | `None` | 逐 step 指标 |
| `artifact(uri, *, type, name=None, version=None, description=None, metadata=None)` | `None` | 登记产物引用 |
| `summary(mapping=None, /, **kwargs)` | `None` | 收尾标量 |
| `require_session()` | `str` | 当前 session id；已关闭时抛 `SessionClosedError` |
| `close(status=None, detail=None)` | 完成态 future-like | 停心跳、结束 session、关闭连接；`status` / `detail` 记入 `session.closing` 事件 |

属性：`session_id`、`name`、`client_id`、`config`、`client_config`、`transport`。

Tinker 拼写：

| 拼写 | 目标 |
|---|---|
| `create_lora_training_client` | `create_lora_trainer`（别名） |
| `create_sampling_client` | `create_sampler`（薄壳，接受上游位置参数） |
| `create_training_client_from_state` | `create_lora_trainer_from_state`（薄壳） |
| `create_training_client_from_state_with_optimizer` | 同上，`load_optimizer=True` |
| `get_server_capabilities_async` | 原生 `ServiceClient` 直接提供 |

兼容命名空间 `mind_recursive.tinker.ServiceClient` 额外提供同步 `get_server_capabilities()`，
原生 `ServiceClient` 只提供 async 版。

## LoraTrainer

由 `ServiceClient` 创建，不直接构造。

| 方法 | 返回 | 说明 |
|---|---|---|
| `forward_backward(data, loss_fn="cross_entropy", loss_fn_config=None, *, forward_only=False)` | `APIFuture[ForwardBackwardOutput]` | 前向 + 反向 |
| `forward(data, loss_fn="cross_entropy", loss_fn_config=None)` | `APIFuture[ForwardBackwardOutput]` | Tinker 拼写；`forward_only=True` |
| `optim_step(optim_params=None, *, adam_params=None)` | `APIFuture[dict[str, Any]]` | 应用梯度；只接受 `AdamParams` |
| `save_weights_only(name, ttl_seconds=None, user_metadata=None, *, overwrite=False)` | `APIFuture[str]` | 持久保存权重到 `adapters/`；默认不带 TTL；`ephemeral` 前缀保留给临时导出 |
| `save_state(name, ttl_seconds=None, overwrite=False, user_metadata=None)` | `APIFuture[str]` | 保存权重 + 优化器到 `checkpoints/`；默认带 72h TTL |
| `load_weights(path, *, optimizer=True)` | `APIFuture[dict[str, Any]]` | 向当前模型加载权重 |
| `load_state(path)` | `APIFuture[dict[str, Any]]` | 只加载权重 |
| `load_state_with_optimizer(path)` | `APIFuture[dict[str, Any]]` | 加载权重 + 优化器 |
| `get_info()` | `ModelInfo` | 服务端模型记录；同步 |
| `create_sampler(*, model_path=None, kind="snapshot", record_stability_info=False)` | `Sampler` | 从当前模型创建 sampler |
| `save_weights_and_get_sampler(*, record_stability_info=False, kind="snapshot")` | `Sampler` | 临时导出权重并创建 sampler；自动命名 `ephemeral-<seq>`（每个 trainer 内递增）、固定 24h TTL、不登记 artifact。不接 `name`／`ttl_seconds` |
| `create_sampling_client(model_path=None, retry_config=None, record_stability_info=False, *, kind="snapshot")` | `Sampler` | Tinker 薄壳；`retry_config` 会被拒绝 |
| `save_weights_and_get_sampling_client` | `Sampler` | Tinker 拼法，`save_weights_and_get_sampler` 的真别名（同一个函数对象） |
| `result()` | `LoraTrainer` | 兼容壳，返回自身 |

属性：`model_id`、`base_model`、`lora_config`、`session_id`。

Tinker 别名：

| 别名 | 目标 |
|---|---|
| `save_weights_for_sampler` | `save_weights_only` |

## Sampler

由 `ServiceClient` 或 `LoraTrainer` 创建。

```python
sample(
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
) -> APIFuture[SampleResponse]
```

`include_prompt_logprobs` 是 Tinker 拼写，和 `prompt_logprobs` 二选一；两者冲突时本地报错。

| 方法 | 返回 | 说明 |
|---|---|---|
| `compute_logprobs(prompt, prompt_logprobs_last_n=None)` | `APIFuture[list[float] \| None]` | 一次 `sample` 薄壳，返回 prompt logprobs |
| `get_base_model()` | `str \| None` | 已知时返回 base model；checkpoint-only sampler 为 `None` |
| `result()` | `Sampler` | 兼容壳，返回自身 |

属性：`sampling_session_id`、`base_model`、`model_path`、`kind`。

## APIFuture

```python
APIFuture[T]
```

| 成员 | 说明 |
|---|---|
| `request_id: str` | 服务端请求 id |
| `done() -> bool` | 是否已解析 |
| `result(timeout: float | None = None) -> T` | 阻塞取结果；结果缓存 |

超时抛 `FutureTimeoutError`，服务端请求仍在继续。

## 异步方法

同步方法 `foo(...)` 都有对应的 `await client.foo_async(...)`；语义、请求体和错误完全相同，
区别只在等待方式。完整清单见 [异步调用](../guide/async.md)。`APIFuture` 仍是同步方法的返回值；
`*_async` 直接返回最终结果。

## 数据类型

### Datum

```python
Datum(
    model_input: ModelInput,
    loss_fn_inputs: Mapping[str, TensorData] = {},
)
```

frozen。`loss_fn_inputs` 的值可以是 `TensorData`、numpy 数组或 list，构造时自动转换。

```python
Datum.from_tokens(
    tokens,
    *,
    weights=None,
    target_tokens=None,
) -> Datum
```

`weights` 默认全 1，`target_tokens` 默认左移一位。

### ModelInput

```python
ModelInput(chunks: list[ModelInputChunk] = [])
ModelInput.from_ints(tokens) -> ModelInput
ModelInput.empty() -> ModelInput
```

方法：`append_int(token)`、`to_ints()`、`length`、`len()`。

### TensorData

```python
TensorData(
    data: np.ndarray,
    dtype: "float32" | "int64" = "float32",
    shape: list[int] | None = None,
    sparse_crow_indices: list[int] | None = None,
    sparse_col_indices: list[int] | None = None,
)
```

方法：`from_numpy(arr)`、`to_numpy()`、`tolist()`、`to_wire()`、`is_sparse`。

### AdamParams

```python
AdamParams(
    learning_rate=1e-4,
    beta1=0.9,
    beta2=0.999,
    eps=1e-8,
    weight_decay=0.0,
    grad_clip_norm=0.0,
)
```

### LoraConfig

```python
LoraConfig(
    rank=32,
    seed=None,
    train_unembed=False,
    train_mlp=True,
    train_attn=True,
)
```

### SamplingParams

```python
SamplingParams(
    max_tokens=512,
    temperature=1.0,
    top_p=1.0,
    top_k=None,
    seed=None,
    stop=None,
    stop_tokens=None,
)
```

`stop` 和 `stop_tokens` 只能给一个。

### 返回类型

| 类型 | 主要字段 |
|---|---|
| `ForwardBackwardOutput` | `metrics`、`loss`、`num_tokens`、`loss_fn_outputs`、`loss_fn_output_type` |
| `SampleResponse` | `sequences`、`prompt_logprobs`、`prompt_cache_hit_tokens`、`topk_prompt_logprobs`、`target_prompt_logprobs`、`prompt_alt_tokens` |
| `SampledSequence` | `tokens`、`logprobs`、`topk_logprobs`、`stop_reason`、`sequence_id` |
| `TopkLogprobs` | `token_ids`、`logprobs`、`k`、`length`；方法 `row(i)`、`to_rows()` |
| `PromptAltTokens` | `tokens`、`logprobs` |
| `ModelInfo` | `model_id`、`model_name`、`is_lora`、`lora_rank`、`optimizer_config`、`model_data`、`extra` |
| `ServerCapabilities` | `supported_models`；方法 `names()`、`find(name)` |
| `SupportedModel` | `model_name`、`max_context_length`、`trainable`、`sampleable` |

### 其它类型

| 类型 | 说明 |
|---|---|
| `EncodedTextChunk(tokens)` | 文本 token chunk |
| `ImageChunk(data, format, expected_tokens=None)` | 内联图片 |
| `DmelChunk(dmel)` | 序列化音频 |
| `ClientConfig` | `client/config` 协商结果 |
| `LossFnType` | loss 名称的 `Literal` |
| `SamplerKind` | `Literal["snapshot", "inflight"]` |
| `TensorDtype` | `Literal["float32", "int64"]` |

## 错误

```text
MindRecursiveError
├── ConfigurationError
├── APIError
│   ├── APIConnectionError
│   │   └── APITimeoutError
│   ├── AuthenticationError
│   ├── PermissionDeniedError
│   ├── NotFoundError
│   ├── BadRequestError
│   ├── ConflictError
│   ├── GoneError
│   ├── PayloadTooLargeError
│   ├── RateLimitError
│   └── ServerError
├── RequestFailedError
├── FutureTimeoutError
└── SessionClosedError
```

`APIError` 属性：`status`、`category`、`path`、`body`。
`RequestFailedError` 属性：`error`、`category`、`request_id`。
`FutureTimeoutError` 属性：`request_id`、`timeout`、`queue_state`。

## 相关文档

- [Future 与错误](../guide/futures-and-errors.md)
- [ServiceClient](../guide/service-client.md)
- [训练](../guide/trainer.md)
- [采样](../guide/sampler.md)
