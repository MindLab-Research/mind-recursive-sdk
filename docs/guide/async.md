# 异步调用

SDK 同时提供同步和原生 async 两套调用方式，走同一组 HTTP 接口、同一套 future 协议；
后端不需要任何适配。

## 规则

- 同步方法直接返回结果或 `APIFuture`；`*_async` 方法返回 awaitable，`await` 后拿到最终结果。
- `ServiceClient` 的构造函数仍然是同步的（它要创建 session）；之后所有训练、采样、上报查询都有 async 版本。
- 异步调用结束后用 `await sc.aclose()`，它会结束 session 并关闭 async / sync 两个连接池。
  传 `status` / `detail` 时会和同步 `close()` 一样记入 `session.closing` 事件。
- 生命周期事件（`trainer.created` / `trainer.resumed` / `sampler.created` /
  `session.closing`）在 async 路径上同样自动上报；上报走工作线程，不阻塞事件循环。
- 心跳仍在后台线程运行，不阻塞事件循环。

## 示例

```python
import asyncio
import mind_recursive as mr
async def main():
    sc = mr.ServiceClient()
    try:
        trainer = await sc.create_lora_trainer_async(
            base_model="Qwen/Qwen3.8-27B", rank=32
        )
        out = await trainer.forward_backward_async(batch)
        await trainer.optim_step_async()

        # 临时导出：自动命名、24h 过期。要留下权重用 save_weights_only_async
        sampler = await trainer.save_weights_and_get_sampler_async()

        # 并发采样：所有请求同时发出，事件循环负责等待。
        results = await asyncio.gather(*[
            sampler.sample_async(prompt) for prompt in prompts
        ])
        return results
    finally:
        await sc.aclose()

asyncio.run(main())
```

## 覆盖范围

| 对象 | async 方法 |
|---|---|
| `ServiceClient` | `create_lora_trainer_async` / `create_lora_trainer_from_state_async` / `create_sampler_async` / `create_sampling_client_async` / `create_training_client_from_state(_with_optimizer)_async` / `get_server_capabilities_async` / `close_async` |
| `LoraTrainer` | `forward_async` / `forward_backward_async` / `optim_step_async` / `save_weights_only_async`（别名 `save_weights_for_sampler_async`）/ `save_state_async` / `load_weights_async` / `load_state_async` / `load_state_with_optimizer_async` / `get_info_async` / `create_sampler_async` / `create_sampling_client_async` / `save_weights_and_get_sampler_async` / `save_weights_and_get_sampling_client_async` |
| `Sampler` | `sample_async` / `compute_logprobs_async` / `get_base_model_async` |
| `UserClient` | `get_server_capabilities_async` / `health_check_async` / `close_async` |

自定义 loss（`forward_backward_custom`）仍不提供：那是服务端能力，不是 async 能解决的。
