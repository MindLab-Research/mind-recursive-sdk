# UserClient

`UserClient` 做不需要 session 的用户级调用。构造它不发请求，也不创建任何服务端记录。

数据集、托管训练、训练记录和模型版本也挂在它上面（`uc.datasets` 等），见
[数据集、托管训练、部署与推理](platform.md)。

```python
import mind_recursive as mr
with mr.UserClient() as uc:
    if uc.health_check():
        for model in uc.get_server_capabilities().supported_models:
            print(model.model_name, model.max_context_length)
```

## 构造参数

```python
uc = mr.UserClient(
    api_key=None,
    base_url=None,
    timeout=60.0,
    connect_timeout=30.0,
    max_retries=3,
    trust_env=False,
    config=None,
)
```

凭证和地址的解析规则与 `ServiceClient` 相同：构造参数 > `MIND_RECURSIVE_*`。
默认 `timeout` 是 60 秒，因为这里是查询调用，不是训练 step。

构造只做本地解析。构造失败一定是配置问题；网络问题会出现在真正调用方法的那一行。

## 方法

```python
uc.health_check()               # bool
uc.get_server_capabilities()    # ServerCapabilities
uc.whoami()                     # Account：user_id / role / channel / 训练准入
uc.get_platform_capabilities()  # PlatformCapabilities：平台开了哪些功能
```

`whoami()` 和 `get_platform_capabilities()` 走平台入口 `/api/mint/cli/v1`，
另外两个走 `/api/mindrsi/v1/service`。

`health_check()` 请求 training 面的健康检查，返回 `True` / `False`，不抛异常。

`get_server_capabilities()` 返回服务端当前提供的模型：

```python
caps = uc.get_server_capabilities()
caps.names()                    # ["Qwen/Qwen3.8-27B", ...]
model = caps.find("Qwen/Qwen3.8-27B")
model.max_context_length
model.trainable
model.sampleable
```

它是最便宜的端到端自检：真实请求后端，证明 key 被接受，并且不创建 session。

## 关闭

```python
with mr.UserClient() as uc:
    ...
# 或者
uc.close()
```

`close()` 只关闭连接，没有 session 要结束。

属性：`config`、`transport`、`datasets`、`training`、`models`、`deployments`、`chat`（`chat.completions`）。

## 暂未提供的方法

SDK session 列表、`list_checkpoints` 和账单查询都属于用户作用域，但当前平台
还没有提供按用户 key 作用域的服务端接口。SDK 不提供这些方法，也不建议调用
上游的全局列表接口。

## 相关文档

- [数据集、托管训练、部署与推理](platform.md)
- [ServiceClient](service-client.md)
- [快速开始](quickstart.md)
