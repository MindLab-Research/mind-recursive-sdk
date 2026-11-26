# ServiceClient

`ServiceClient` 是一个训练 session。构造它就创建 session，训练、采样和上报都挂在
这个 session 下。

```python
import mind_recursive as mr
with mr.ServiceClient(client_id="sft-run-1") as sc:
    trainer = sc.create_lora_trainer(base_model="Qwen/Qwen3.8-27B", rank=32)
    sampler = sc.create_sampler(base_model="Qwen/Qwen3.8-27B")
    sc.log("started")
```

## session 链接

构造成功后终端会打印这次运行的 Web 地址：

```text
Session created: <session_id>
Web console: https://mintrecursive.mindlab.im/v2train/records/sessions/<session_id>
```

前缀默认是 `https://mintrecursive.mindlab.im/v2train/records/sessions/`，可以用
`MIND_RECURSIVE_WEB_URL` 覆盖。这个地址只用于展示，SDK 不会向它发请求。

## 构造参数

```python
sc = mr.ServiceClient(
    api_key=None,
    base_url=None,
    timeout=600.0,
    connect_timeout=30.0,
    max_retries=3,
    heartbeat_seconds=10.0,
    client_id=None,
    trust_env=False,
    config=None,
    tags=None,
    user_metadata=None,
)
```

| 参数 | 作用 |
|---|---|
| `api_key` | 用户 key；默认读 `MIND_RECURSIVE_API_KEY` |
| `base_url` | 服务 origin；默认读 `MIND_RECURSIVE_BASE_URL`，都不设时用正式环境 |
| `timeout` | 数据面单请求超时，默认 600 秒 |
| `connect_timeout` | 建连超时，默认 30 秒 |
| `max_retries` | 可重试失败（连接错误、429、5xx）的重试次数 |
| `heartbeat_seconds` | session 保活间隔，默认 10 秒 |
| `client_id` | 关联同一进程多个 session 的标签，写进 `create_session` |
| `trust_env` | 是否读取系统代理环境变量，默认 `False` |
| `tags` | 随 `create_session` 发送的标签列表 |
| `user_metadata` | 随 `create_session` 发送的任意元数据 |
| `config` | 直接传一个已构造的 `Config`，覆盖上面的地址与凭证解析 |

凭证和地址的优先级是**构造参数 > 环境变量**。空字符串按未设置
处理。base URL 不设时默认连正式环境 `https://mintcn.macaron.xin`。

## base URL

`base_url` 接受两种写法：

```text
https://relay.example.com
https://relay.example.com/api/mindrsi/v1
```

SDK 统一追加固定根 `/api/mindrsi/v1`，再按用途分段：

```text
{BASE}/service/*   健康检查、模型能力、客户端行为协商
{BASE}/service/session/*   session 创建、心跳、结束
{BASE}/train/*     训练、权重与训练侧 future
{BASE}/sample/*    采样与采样侧 future
{BASE}/service/report/*    训练过程上报（只写）
```

带 `/train`、`/api/mint/cli/v1` 或其它路径的地址会在本地报 `ConfigurationError`。
需要绕过代理时保持默认的 `trust_env=False`。

## session 生命周期

构造 `ServiceClient` 时依次发生：

```text
POST {BASE}/service/client/config     协商客户端行为
POST {BASE}/service/session/create            创建 session
后台线程 POST {BASE}/service/session/{id}/heartbeat   每 heartbeat_seconds 一次
```

`session_id` 在构造成功后立即有值。一个 client 对应一个 session；trainer 和 sampler
从它创建，本身不持有连接。

关闭：

```python
sc.close()
```

顺序固定为**停心跳 → 结束 session → 关闭连接**。结束 session 是 best-effort，
失败只记日志。关闭后再调用训练、采样或上报会报 `SessionClosedError`。

`close()` 是唯一需要的清理，trainer / sampler 没有 `close()`。

属性：

| 属性 | 内容 |
|---|---|
| `session_id` | 当前 session id；`close()` 后是 `None` |
| `client_id` | 构造时传入的进程标签 |
| `config` | 解析后的 `Config` |
| `client_config` | 服务端协商结果（`ClientConfig`） |
| `transport` | 底层 `Transport`，一般不需要直接使用 |

## 一个 session，多个 trainer / sampler

```python
with mr.ServiceClient() as sc:
    a = sc.create_lora_trainer(base_model="Qwen/Qwen3.8-27B", rank=32)
    b = sc.create_lora_trainer(base_model="Qwen/Qwen3.8-27B", rank=16)

    s1 = a.create_sampler(model_path=path_from_a)
    s2 = sc.create_sampler(base_model="Qwen/Qwen3.8-27B")
```

每次 `create_lora_trainer` 在同一个 session 下创建一个独立模型；它们共享 session
和计费归属。需要完全隔离时，建两个 `ServiceClient`。

## Tinker 拼写

`ServiceClient` 上保留了 Tinker 的创建方法：

| Tinker 拼写 | 原生方法 |
|---|---|
| `create_lora_training_client` | `create_lora_trainer`（别名） |
| `create_sampling_client` | `create_sampler`（薄壳，接受上游的位置参数） |
| `create_training_client_from_state` | `create_lora_trainer_from_state`（薄壳） |
| `create_training_client_from_state_with_optimizer` | 同上，`load_optimizer=True` |

```python
sc.create_lora_training_client("Qwen/Qwen3.8-27B", 32).result()
sc.create_sampling_client(model_path="mint://run/adapters/step-1").result()
sc.create_training_client_from_state(
    "mint://run/checkpoints/step-1",
    base_model="Qwen/Qwen3.8-27B",
).result()
```

创建调用本身是同步的，返回对象上的 `.result()` 是兼容壳，返回同一个对象。
`retry_config`、`weights_access_token` 和非 Adam 优化器会被明确拒绝，不会静默忽略。
更完整的迁移说明见 [Tinker 迁移](tinker.md)。

## 异步

所有创建与训练方法都有 `*_async` 版本，见[异步调用](async.md)。`ServiceClient` 的构造函数仍是同步的；
异步结束时用 `await sc.aclose()`。

## 相关文档

- [训练](trainer.md)
- [Checkpoint 与续训](checkpoints.md)
- [采样](sampler.md)
- [上报](reporter.md)
- [UserClient](user-client.md)
