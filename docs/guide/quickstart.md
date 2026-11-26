# 快速开始

这篇文档从空环境开始，完成一次最小训练：检查连通性 → 建 session → 训练几步 →
保存权重 → 采样 → 关闭。

## 1. 安装与配置

```bash
uv add mind-recursive
# 或 pip install mind-recursive

export MIND_RECURSIVE_API_KEY=sk-...
# export MIND_RECURSIVE_BASE_URL=https://<origin>   # 可选，默认 https://mintcn.macaron.xin
# export MIND_RECURSIVE_WEB_URL=https://mintrecursive.mindlab.im/v2train/records/sessions/
```

`MIND_RECURSIVE_BASE_URL` 只写服务 origin，SDK 自己追加 `/api/mindrsi/v1`。完整的环境变量和
安装选项见 [docs/README.md](../README.md)。

`MIND_RECURSIVE_WEB_URL` 只影响创建 session 后打印的 Web 链接，不发请求。默认地址是
`https://mintrecursive.mindlab.im/v2train/records/sessions/`，拼上 session id 就是这次运行的页面。

## 2. 自检

先确认 key 和地址可用。`UserClient` 不创建 session：

```python
import mind_recursive as mr
with mr.UserClient() as uc:
    print(uc.get_server_capabilities().names())
```

`get_server_capabilities()` 会真实请求后端并在失败时抛异常；`health_check()` 只返回
`True` / `False`。

## 3. 建 session 和 trainer

```python
import mind_recursive as mr
sc = mr.ServiceClient(client_id="first-run")
trainer = sc.create_lora_trainer(
    base_model="Qwen/Qwen3.8-27B",
    rank=32,
    train_unembed=False,
)
```

`ServiceClient()` 会立刻创建 session。训练结束用 `sc.close()`，或者把整段放进
`with` 块。

## 4. 训练

下面用 `Datum.from_tokens` 构造 smoke 数据。它把 target 左移一位、mask 全设为 1，
适合验证链路；真实 SFT 要自己构造 loss mask，见 [数据与 batch](data.md)。

```python
token_batches = [
    [[11, 12, 13, 14], [21, 22, 23]],
    [[31, 32], [41, 42, 43, 44]],
]

for step, token_batch in enumerate(token_batches, start=1):
    batch = [mr.Datum.from_tokens(tokens) for tokens in token_batch]
    out = trainer.forward_backward(batch, loss_fn="cross_entropy").result()
    trainer.optim_step(mr.AdamParams(learning_rate=1e-4)).result()
    print(step, out.loss, out.num_tokens)
```

每个训练调用都返回 `APIFuture`。不调用 `.result()` 就只是提交了请求，没有等它完成。

## 5. 保存与采样

```python
path = trainer.save_weights_only("step-1").result()
print(path)   # mint://.../adapters/...

sampler = trainer.create_sampler(model_path=path)
resp = sampler.sample([11, 12, 13], num_samples=2).result()

for sequence in resp.sequences:
    print(sequence.tokens)
    print(sequence.logprobs)
```

`save_weights_only` 产出可采样、可部署的 adapter。要保存含优化器状态的训练状态，
用 `save_state`，见 [Checkpoint 与续训](checkpoints.md)。

## 6. 关闭

```python
sc.close()
```

或者从一开始就写：

```python
with mr.ServiceClient() as sc:
    trainer = sc.create_lora_trainer(base_model="Qwen/Qwen3.8-27B")
    ...
```

`close()` 会停心跳、结束 session、关闭连接。trainer 和 sampler 没有自己的
`close()`。

## 7. 切换到其他环境

默认连正式环境。要连其他环境，显式设地址和 key：

```bash
export MIND_RECURSIVE_BASE_URL=https://<origin>
export MIND_RECURSIVE_API_KEY=sk-...
uv run python recipes/00_smoke.py
```

## 下一步

- [ServiceClient](service-client.md)：认证、地址、session 生命周期
- [训练](trainer.md)：`forward_backward`、Adam 参数、自动分块
- [数据与 batch](data.md)：`Datum`、`ModelInput`、`TensorData`、loss mask
- [Checkpoint 与续训](checkpoints.md)：两种 save、`load_weights`、从 checkpoint 起训
- [采样](sampler.md)：解码参数和 logprob
- [上报](reporter.md)：log、metric、artifact、summary
