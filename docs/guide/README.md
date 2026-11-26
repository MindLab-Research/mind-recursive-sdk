# 使用指南

按任务选择：

| 文档 | 内容 |
|---|---|
| [快速开始](quickstart.md) | 从空环境跑通一次训练、保存和采样 |
| [ServiceClient](service-client.md) | 认证、base URL、session 生命周期 |
| [训练](trainer.md) | `forward_backward`、`optim_step`、Adam 参数、自动分块 |
| [数据与 batch](data.md) | `Datum`、`ModelInput`、`TensorData`、loss mask |
| [Checkpoint 与续训](checkpoints.md) | 两种 save、加载、从 checkpoint 起训 |
| [采样](sampler.md) | token 级采样、logprob、top-k、解码参数 |
| [上报](reporter.md) | log / event / metric / artifact / summary |
| [UserClient](user-client.md) | 无 session 的模型与健康查询 |
| [数据集、托管训练、部署与推理](platform.md) | `uc.datasets` / `uc.training` / `uc.models` / `uc.deployments` / `uc.chat.completions` |
| [Future 与错误](futures-and-errors.md) | `APIFuture`、超时、异常与重试 |
| [异步调用](async.md) | `*_async` 方法、并发采样、`await sc.aclose()` |
| [Tinker 迁移](tinker.md) | Tinker 拼写别名：已覆盖、未覆盖与行为差异 |

API 签名速查见 [reference/api.md](../reference/api.md)。
