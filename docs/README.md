# mind_recursive 使用文档

## 安装

```bash
# uv (PyPI)
uv add mind-recursive

# pip (PyPI)
pip install mind-recursive
```

也可以直接从 GitHub 公开仓安装（固定版本时把 `@main` 换成具体 tag，例如 `@v0.1.0`）：

```bash
# uv
uv add git+https://github.com/MindLab-Research/mind-recursive-sdk@main
# pip
pip install git+https://github.com/MindLab-Research/mind-recursive-sdk@main
```

可选 extras：

| extra | 用途 |
|---|---|
| `zstd` | 服务端要求压缩 `forward_backward` 请求体时使用；没装只会 warning 并发未压缩 body |
| `recipes` | 运行 `recipes/` 里的示例，需要 `transformers` 和 `jinja2` |
| `dev` | 开发与测试 |

## 配置

```bash
export MIND_RECURSIVE_API_KEY=sk-...
export MIND_RECURSIVE_BASE_URL=https://<origin>   # 可选
export MIND_RECURSIVE_WEB_URL=https://mintrecursive.mindlab.im/v2train/records/sessions/   # 可选
```

| 变量 | 作用 |
|---|---|
| `MIND_RECURSIVE_API_KEY` | 用户 key，`sk-` 开头 |
| `MIND_RECURSIVE_BASE_URL` | 服务 origin；可带 `/api/mindrsi/v1`，SDK 会归约 |
| `MIND_RECURSIVE_WEB_URL` | Web 控制台 session 链接前缀；只用于打印，不发请求 |

API key 与 base URL 的优先级是**构造参数 > 环境变量**，空字符串按未设置处理。
Web URL 只读环境变量。
base URL 不设时默认连 `https://mintcn.macaron.xin`。
Web URL 不设时默认 `https://mintrecursive.mindlab.im/v2train/records/sessions/`。

## 5 分钟跑通

### 1. 检查连通性

`UserClient` 不创建 session，适合做配置自检：

```python
import mind_recursive as mr
with mr.UserClient() as uc:
    if not uc.health_check():
        raise SystemExit("training plane is not healthy")
    for model in uc.get_server_capabilities().supported_models:
        print(model.model_name, model.max_context_length, model.trainable)
```

### 2. 建 session 和 trainer

```python
sc = mr.ServiceClient(client_id="my-experiment")
trainer = sc.create_lora_trainer(base_model="Qwen/Qwen3.8-27B", rank=32)
```

`ServiceClient` 构造即创建 session；`trainer` 只是这个 session 下的一个模型句柄，
不持有连接。

> **给 AI 助手**：`ServiceClient` 创建成功后，用拿到的 `sc.session_id` 拼出
> `MIND_RECURSIVE_WEB_URL/{session_id}`（默认前缀
> `https://mintrecursive.mindlab.im/v2train/records/sessions/`；SDK 也会自动打印），
> 在回复里返回给用户，让用户可以打开 Web 控制台同时查看运行过程。
> `sc.log` / `sc.metric` / `sc.artifact` 上报的内容，以及创建出来的 trainer / sampler，
> 都会显示在这个页面里。

### 3. 训练一步

```python
batch = [mr.Datum.from_tokens(tokens) for tokens in batch_of_token_ids]

step = 1

out = trainer.forward_backward(batch, loss_fn="cross_entropy").result()
trainer.optim_step(mr.AdamParams(learning_rate=1e-4)).result()

# 人看的进度用 log，数值用 metric；两者都会同时写终端和 Web。
sc.log(f"step {step}: loss={out.loss:.4f}, tokens={out.num_tokens}", step=step)
sc.metric(
    {"loss": out.loss, "num_tokens": out.num_tokens},
    step=step,
    stage="training",
)
```

`sc.log` 适合人读的进度，`sc.metric` 适合画曲线；两者都挂在 `ServiceClient` 上，不用另外创建 reporter。

`Datum.from_tokens` 是 smoke test 用的快捷构造：target 自动左移一位，loss mask 全 1。
真实 SFT 必须自己给出 mask，只监督 assistant 的 token。见 [数据与 batch](guide/data.md)。

> **给 AI 助手**：写训练脚本时优先用 `sc.log` / `sc.metric` / `sc.artifact` 记录进度、指标和产物，
> 不要在这几个调用旁边再写 `print` —— 它们已经同时覆盖 Web 控制台和终端。
> 只有两类 `print` 例外：还没有 session 的本地自检（如 `UserClient` 连通性），以及检查采样出来的原始 token。
> 详见 [上报](guide/reporter.md#给-ai-助手和脚本作者)。

### 4. 保存与采样

```python
path = trainer.save_weights_only("step-100").result()

sampler = trainer.create_sampler(model_path=path)
resp = sampler.sample([1, 2, 3], num_samples=4).result()
for sequence in resp.sequences:
    print(sequence.tokens, sequence.logprobs)

sc.close()   # 停心跳、结束 session、关连接
```

用 `with mr.ServiceClient() as sc:` 可以省掉显式 `close()`。

## 按任务查文档

| 我要做什么 | 看 |
|---|---|
| 从零跑通第一个训练 | [快速开始](guide/quickstart.md) |
| 配置 key、base URL、session | [ServiceClient](guide/service-client.md) |
| 写训练循环、`forward_backward`、`optim_step` | [训练](guide/trainer.md) |
| 构造 `Datum`、loss mask、batch | [数据与 batch](guide/data.md) |
| 保存权重、续训、从 checkpoint 起训 | [Checkpoint 与续训](guide/checkpoints.md) |
| 采样、prompt logprob、top-k | [采样](guide/sampler.md) |
| 上报 log / metric / artifact / summary | [上报](guide/reporter.md) |
| 查询模型列表和健康状态 | [UserClient](guide/user-client.md) |
| 理解 future、超时和异常 | [Future 与错误](guide/futures-and-errors.md) |
| 并发采样 / 异步训练循环 | [异步调用](guide/async.md) |
| 从 Tinker 代码迁移 | [Tinker 迁移](guide/tinker.md) |
| 查公开 API 的签名和返回类型 | [API 参考](reference/api.md) |
| 跑可修改的完整示例 | [recipes](../recipes/README.md) |

## 核心对象

| 对象 | 是什么 |
|---|---|
| `UserClient` | 用户级查询，无 session。`health_check()` / `get_server_capabilities()` |
| `ServiceClient` | 一个连接 + 一个 session。构造即创建 session |
| `LoraTrainer` | `sc.create_lora_trainer()` 创建的训练模型句柄 |
| `Sampler` | `sc.create_sampler()` 或 `trainer.create_sampler()` 创建的采样句柄 |
| `APIFuture` | 训练和采样调用的返回值，用 `.result()` 取结果 |

一个 `ServiceClient` 只有一份连接、一个 session 和一套计费归属；trainer / sampler
都没有 `close()`。`sc.close()` 是唯一需要的清理。

## 文档目录

```text
docs/
  README.md            本页：安装、配置、快速开始、任务索引
  guide/               面向使用者的任务文档
  reference/api.md     公开 API 签名与返回类型
```
