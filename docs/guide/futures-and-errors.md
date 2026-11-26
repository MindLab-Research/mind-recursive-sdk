# Future 与错误

训练和采样调用是异步的。它们返回 `APIFuture`，调用 `.result()` 才等待结果。

```python
out = trainer.forward_backward(batch).result()
```

## APIFuture

```python
future = trainer.forward_backward(batch)

future.request_id   # 服务端请求 id
future.done()       # 是否已经解析
future.result()     # 阻塞直到完成，返回结果
future.result(timeout=30)
```

- `.result()` 的结果会缓存，第二次调用不重新轮询。
- `.result(timeout=...)` 超时抛 `FutureTimeoutError`，但**服务端的请求仍在继续**。
  之后用更长的 timeout 再调用 `.result()` 仍然可以拿到结果。
- 不调用 `.result()` 就只是提交了请求。训练循环里漏掉它，下一步会基于不完整的梯度。

`forward_backward` 的 batch 超过 chunk 预算时，SDK 会在内部提交多个请求，返回一个
组合 future。它同样有 `request_id`、`done()` 和 `result()`；`.result()` 在所有分块完成
后合并结果。

## 异常层级

所有异常都继承 `MindRecursiveError`：

```text
MindRecursiveError
├── ConfigurationError        本地配置或参数错误，未发请求
├── APIError                  HTTP 请求失败
│   ├── APIConnectionError    DNS / TCP / TLS / 连接重置
│   │   └── APITimeoutError   请求超时
│   ├── AuthenticationError   401
│   ├── PermissionDeniedError 403
│   ├── NotFoundError         404
│   ├── BadRequestError       400
│   ├── ConflictError         409
│   ├── GoneError             410
│   ├── PayloadTooLargeError  413
│   ├── RateLimitError        429
│   └── ServerError           5xx
├── RequestFailedError        future 返回 request_failed
├── FutureTimeoutError        result(timeout=) 超时
└── SessionClosedError        client 已关闭
```

`APIError` 上有 `status`、`category`、`path`、`body` 属性。服务端错误信封里的
`category` 是 `"user"` 或 `"server"`，可以区分“我传错了”和“后端出问题”。

## 重试

以下错误由传输层自动重试，次数由 `max_retries` 控制：

- 连接错误、超时；
- 429（读取 `Retry-After`）；
- 5xx。

4xx 不重试。重试预算耗尽后异常才抛给调用方。

future 的 `request_failed` 不自动重试。它表示 HTTP 调用成功但工作失败，例如服务端
拒绝 Adam 参数：

```python
from mind_recursive import RequestFailedError

try:
    trainer.optim_step(mr.AdamParams(beta2=0.95)).result()
except RequestFailedError as exc:
    print(exc.category, exc.error)
    # server UNSUPPORTED: FSDP2 requires beta1=0.9, beta2=0.999, ...
```

`RequestFailedError` 携带服务端原文、`category` 和 `request_id`。

## 常见错误

| 现象 | 原因 |
|---|---|
| `ConfigurationError`，提示 base URL | `MIND_RECURSIVE_BASE_URL` 为空，或带了 `/train` 等上游前缀 |
| `AuthenticationError` | key 缺失、错误，或没有权限 |
| `NotFoundError` | 路径或资源不存在；先确认 base URL 没有带上游前缀 |
| `PayloadTooLargeError` | 单个 datum 超过 `fwdbwd` 预算；拆小数据或检查多模态 token 估算 |
| `FutureTimeoutError` | 训练或采样排队过久；请求仍在服务端继续，可稍后用更长 timeout 重试 |
| `SessionClosedError` | 调用了 `sc.close()` 之后又用 trainer / sampler / reporter |

## 日志

SDK 使用这些 logger：

| logger | 内容 |
|---|---|
| `mr.transport` | 请求失败与重试 |
| `mind_recursive.reporter` | 上报失败（只 warning） |
| `mr.trainer` | 分块与 Adam 参数 warning |
| `mr.service` | session 创建与关闭 |

```python
import logging
logging.getLogger("mind_recursive").setLevel(logging.INFO)
```

## 相关文档

- [训练](trainer.md)
- [采样](sampler.md)
- [API 参考](../reference/api.md)
