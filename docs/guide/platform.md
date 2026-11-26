# 数据集、托管训练、部署与推理

`UserClient` 上挂了平台资源的几个命名空间。它们都不需要 session，请求发到平台入口
`{origin}/api/mint/cli/v1`。

| 命名空间 | 做什么 |
|---|---|
| `uc.whoami()` | 当前 key 的用户、角色和训练准入 |
| `uc.get_platform_capabilities()` | 平台开了哪些功能 |
| `uc.datasets` | 上传、列表、详情、行、校验报告、下载、改备注、删除 |
| `uc.training` | 提交、重试托管训练；训练记录、状态、等待、取消、进度、指标、结果、事件 |
| `uc.models` | 托管训练产出的模型和续训链 |
| `uc.deployments` | 部署模型版本、等待就绪、停止、重启、删除 |
| `uc.chat.completions` | 调用已就绪的部署（OpenAI 格式） |

评测和组织共享不在 SDK 里，继续用 CLI 或 Web。

## 完整流程

```python
import mind_recursive as mr
with mr.UserClient() as uc:
    ds = uc.datasets.upload("train.jsonl", type="sft", note="first try")
    print(ds.dataset_version, ds.rows)

    run = uc.training.submit(
        dataset_version=ds.dataset_version,
        base_model="Qwen/Qwen3.8-27B",
        config={"learning_rate": 1e-4, "num_epochs": 1},
        display_name="sft-v1",
    )
    run = uc.training.wait(run.execution_id, poll_interval=30)
    print(run.state, run.result_reason, run.version_id)

    if run.state == "succeeded":
        dep = uc.deployments.create(version_id=run.version_id, display_name="sft-v1")
        dep = uc.deployments.wait(dep.deployment_id)          # 等到 ready
        resp = uc.chat.completions.create(
            model=dep.public_model_id,
            messages=[{"role": "user", "content": "你好"}],
            max_tokens=256,
        )
        print(resp.choices[0].message.content)
        uc.deployments.stop(dep.deployment_id)                # 释放资源
```

## 数据集

```python
uc.datasets.upload(path, type="sft", dataset_version=None, note=None,
                   idempotency_key=None, wait=True, timeout=600)   # Dataset
uc.datasets.wait(version, timeout=600)                              # Dataset
uc.datasets.list(type=None, page=None, page_size=None)              # Page[Dataset]
uc.datasets.get(version, owner_uid=None)                            # Dataset
uc.datasets.rows(version, offset=None, limit=None)                  # list[dict]
uc.datasets.validation_report(version, offset=None, limit=None)     # ValidationReport
uc.datasets.download(version, "out.jsonl")                          # str（本地路径）
uc.datasets.update_note(version, "note")                            # Dataset
uc.datasets.delete(version)
```

`upload` 和 CLI 一样分三步：先登记上传意图，再用返回的预签名 URL 把文件直接 PUT 到对象
存储，最后通知服务端文件已就位。服务端在后台校验。`wait=True`（默认）时一直轮询到
`available`；校验结果是 `failed` 时抛 `DatasetValidationError`，`err.dataset.failure_reason`
是原因，逐行问题用 `validation_report` 看。

重试同一次上传时传同一个 `idempotency_key`，服务端会返回同一个版本；不传时每次调用生成
一个新 key。

`download` 不覆盖已有文件：先写同目录的临时文件，传完再原子地链接到目标路径，失败时不
留下文件。

预签名 URL 的上传和下载都用一个干净的连接：不带 `X-API-Key`，不跟随重定向，报错信息里
也不出现 URL。

## 托管训练

提交、重试，以及之后查看训练记录，都在 `uc.training` 下面。

```python
# 提交与重试
uc.training.submit(dataset_version=..., base_model=..., config={...},
                   kind="sft", display_name=None, package_version=None,
                   base_version_id=None, idempotency_key=None)       # Execution
uc.training.retry(execution_id, continue_training=False, config=None)  # Execution

# 训练记录
uc.training.list(state=None, updated_since=None, page=None, page_size=None)  # Page[Execution]
uc.training.get(execution_id)                       # Execution
uc.training.wait(execution_id, timeout=None, poll_interval=10)  # Execution
uc.training.cancel(execution_id)                    # Execution
uc.training.rename(execution_id, "name")            # Execution
uc.training.delete(execution_id)

# 观测
uc.training.progress(execution_id, tail=20)         # list[str]
uc.training.metrics(execution_id, cursor=None, limit=None, keys=None)  # MetricsPage
uc.training.iter_metrics(execution_id, keys=["loss"])  # Iterator[MetricPoint]
uc.training.metric_keys(execution_id)               # list[MetricKey]
uc.training.result(execution_id)                    # ExecutionResult
uc.training.events(execution_id, page=None, page_size=None)  # Page[ExecutionEvent]
```

`config` 的字段由算法包定义，服务端按包的 schema 校验，SDK 不在本地校验。不传
`package_version` 时用该训练类型最新的 stable 包。

`base_version_id` 表示从已有模型（`uc.models` 里的 `version_id`）续训，这时可以不传
`base_model`。

`submit` 每次调用生成一个 `Idempotency-Key`，网络错误或 5xx 时用同一个 key 重试，不会
重复提交。用同一个 key 再调一次，也只会拿回同一个 run。

`list` 只返回训练记录（server 上 `kind=training` 的 execution），评测记录不在这里。

`Execution.state` 是调度器的状态：`pending` / `running` / `succeeded` / `failed` /
`cancelled`，`terminal` 属性判断是否已结束。训练成功、产出模型后，`version_id` 会有值。

`wait` 在 `failed` 或 `cancelled` 时不抛异常，只返回最终的 execution，由你检查 `state`；
超过 `timeout` 才抛 `FutureTimeoutError`。

`cancel` 返回时 run 可能还是 `running`（`cancel_requested=True`），之后由调度器收尾为
`cancelled`。

`metrics` 按 cursor 分页。`caught_up=True` 只表示"暂时没有更新的行"，不表示训练结束。
`iter_metrics` 会一直翻页到 caught up 为止。

`result()` 返回的是算法自己上报的结果，包括产物（`artifacts[].uri`，如 `mint://...`）、
数据摘要和错误。它的 `status` 不能代替 `Execution.state`。

`progress()` 是结构化的进度行，不是训练进程的原始 stdout/stderr。

## 模型

每次托管训练成功，服务端会登记一个模型（`version_id`）：记录它用哪个数据集、哪个基座、
哪次训练（`execution_id`）训出来，以及从哪个模型续训而来（`parent_id`）。部署时传的就是
这个 `version_id`，续训时 `base_version_id` 也填它。

```python
uc.models.list(page=None, page_size=None)     # Page[ModelVersion]
uc.models.get(version_id, owner_uid=None)     # ModelVersion
uc.models.lineage(version_id)                 # list[ModelVersion]，续训链：v3 ← v2 ← v1
uc.models.delete(version_id)
```

服务端接口叫它 version（`/versions`），SDK 里统一叫 models。

## 部署

```python
uc.deployments.create(version_id=..., display_name=None,
                      version_owner_uid=None, source=None)   # Deployment
uc.deployments.wait(deployment_id, until="ready", timeout=1800, poll_interval=10)
uc.deployments.list(state=None, page=None, page_size=None)   # Page[Deployment]
uc.deployments.get(deployment_id)                            # Deployment
uc.deployments.stop(deployment_id)                           # 释放资源并注销路由，保留 id
uc.deployments.start(deployment_id)                          # 重新启动已停止的部署
uc.deployments.delete(deployment_id)
```

`create` 立即返回，状态一般是 `pending`。`state` 有四种：`pending` / `ready` / `stopped` /
`unknown`；`desired_state` 是你要求的状态（`active` / `stopped`）。服务端要看到模型已加载、
健康、路由已注册才算 `ready`，要看到资源已释放、路由已注销才算 `stopped`；超时证明不了
任何一边时就是 `unknown`。

`wait(until="ready")` 遇到 `unknown`，或者部署已经被改成要停止，会抛 `DeploymentError`
（`err.deployment` 是最后一次读到的记录）；超时抛 `FutureTimeoutError`。刚调完 `start` 时
状态可能还停在 `stopped`，`wait` 会继续等。

`create` 只发一次、不重试：响应丢了再重发可能建出两个部署。不确定时先 `list` 看一下。

部署会持续占用 GPU，不用时记得 `stop`。`stop` 不保证立刻切断正在进行的请求。

## 推理

OpenAI 格式，写法和 OpenAI 官方 SDK 一样：

```python
resp = uc.chat.completions.create(
    model=dep.public_model_id,
    messages=[{"role": "user", "content": "你好"}],
    max_tokens=256,
)
resp.choices[0].message.content
```

参数都是关键字参数。`model`、`messages` 之外的参数原样放进请求体，比如 `max_tokens`、
`temperature`、`tools`；SDK 不校验、不补默认值。`timeout` 是
SDK 自己的参数，不进请求体，默认 600 秒。

请求发到公网推理入口 `{origin}/v1/chat/completions`，不走上面的平台入口。`model` 填部署的 `public_model_id`，
部署 `ready` 之后才有；key 就是同一个 `sk-` key。

返回值是上游 JSON 原样解码的结果，可以像官方 SDK 一样用点号访问，也可以当 dict 用：
`resp.choices[0].message.content` 和 `resp["choices"][0]["message"]["content"]` 等价。
不存在的字段用点号读时返回 `None`。和 dict 方法同名的字段（如 `items`）只能用 `[]` 读。
它不是官方 SDK 的类型，`isinstance(resp, openai.types.ChatCompletion)` 为假。

`stream=True` 时返回一个迭代器，每个元素是一条 SSE 事件：

```python
for chunk in uc.chat.completions.create(model=m, messages=msgs, stream=True):
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

推理请求从不重试，也不跟随重定向：响应丢失时生成可能已经计费，重放会再算一次。错误按
HTTP 状态映射成同样的异常类，`err.request_id` 是服务端的请求 id。

暂不提供的格式：

- Anthropic 格式（`messages.create`）：服务端暂不支持部署模型走 `/v1/messages`，支持后再加。
- OpenAI 纯文本补全（`completions.create`）和 Responses API（`responses.create`）：服务端
  暂无 `/v1/completions` 和 `/v1/responses`。

## 列表与分页

列表方法返回 `Page`：可以直接迭代，`page.items` 是这一页，`page.total` 是全部匹配的条数。
`page` 从 1 开始；不传时用服务端默认值（第 1 页，每页 20 条）。

## 错误

除了 [Future 与错误](futures-and-errors.md) 里的异常，平台调用还有这些情况：

| 情况 | 异常 | `err.code` |
|---|---|---|
| 网关拒绝凭证（`credential_*`） | `AuthenticationError` | 字符串，如 `"credential_rejected"` |
| 账号未准入（`access_forbidden` / `admission_denied`） | `PermissionDeniedError` | 字符串 |
| 服务端鉴权业务码（`100xxx`） | `AuthenticationError`（`100015` 是 `PermissionDeniedError`） | int |
| 数据集不存在（`140056`） | `NotFoundError` | `140056` |
| 数据集被训练引用，不能删（`140058`） | `ConflictError` | `140058` |
| 其他业务码 | `APIError` | int |
| 上传的数据集校验失败 | `DatasetValidationError` | — |

网关拒绝时还会带 `err.request_id`，排障时提供给平台。

服务端的业务错误也以 HTTP 200 返回，状态码看 envelope 里的 `code`。SDK 已经解开 envelope，
你拿到的只有数据或异常。

## 重试

GET 和带 `Idempotency-Key` 的写请求（`submit`、`retry`、上传的两步）在连接错误、429、5xx
时重试。其他写请求（`cancel`、`rename`、`delete`、`update_note`、部署的
`create` / `stop` / `start`）和所有推理请求只发一次，避免响应丢失时重复执行。
