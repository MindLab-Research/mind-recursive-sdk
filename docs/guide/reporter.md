# 上报：log / event / metric / artifact / summary

上报方法直接挂在 `ServiceClient` 上，没有单独的 reporter 对象：

```python
sc.log("starting SFT")
sc.metric({"loss": 0.42}, step=1, stage="training", total=800)
sc.summary(final_loss=0.4)
```

每次调用都会通过 client 自己的 transport POST 到
`{origin}/api/mindrsi/v1/service/report/{session_id}/...`，复用同一套凭证、连接和重试策略。

上报是 best-effort：失败只记 warning，不抛异常，不影响训练。SDK 只写不读；查看记录
用控制台或平台。

## 给 AI 助手和脚本作者

**优先用 `sc.log` / `sc.metric` / `sc.artifact` 上报，不要再在它们旁边额外写 `print`。**

这三个方法除了写 Web 控制台，也会把同一条信息打印到终端（走 `logging`，默认输出到 stderr）。
终端里看到的内容和 Web 控制台里看到的是同一份，不会出现“终端有、Web 没有”。

- 训练进度、阶段切换、异常说明：用 `sc.log`
- loss / lr / token 数 / 吞吐：用 `sc.metric`
- 保存出来的权重、checkpoint、评测产物：用 `sc.artifact`

如果调用方自己配置了 logging，SDK 不会再加 handler，终端格式和级别由调用方控制。
想关掉 SDK 的终端输出，提高 logger 级别即可：

```python
import logging
logging.getLogger("mind_recursive").setLevel(logging.WARNING)
```

创建 session 时 SDK 还会主动打印这次运行的 Web 地址：

```text
Session created: <session_id>
Web console: <MIND_RECURSIVE_WEB_URL>/<session_id>
```

## 五个方法

路由相对 `{origin}/api/mindrsi/v1/service/report/{session_id}`。

| 方法 | 路由 | 语义 |
|---|---|---|
| `sc.log(message, *, level="info", step=None)` | `/events` | 运行日志 |
| `sc.event(type, *, level="info", code=None, message=None, data=None, step=None)` | `/events` | 结构化事件 |
| `sc.metric(metrics, *, step, stage=None, total=None, timings=None, counters=None)` | `/log` | 逐 step 指标 |
| `sc.artifact(uri, *, type, name=None, version=None, description=None, metadata=None)` | `/artifact` | 产物引用，append-only |
| `sc.summary(mapping=None, **kwargs)` | `/summary` | 收尾标量，按 key 合并 |

`log(text)` 等价于 `event("log", message=text)`，两者共用 `/events` 和同一个 `seq`。

`level` 取 `debug` / `info` / `warning` / `error`；其它值会被修正为 `info`。

`event` 的 `type` 是点分名称，例如 `upstream.retrying`，要求匹配 `^[a-z0-9_.]+$`，
最多 128 字节。`data` 必须是 JSON object。

## metric

```python
sc.metric(
    {"loss": 0.42, "lr": 3e-5},
    step=step,
    stage="training",
    total=800,
)
```

| 参数 | 说明 |
|---|---|
| `metrics` | 这一步的指标 dict；值为 `None` 的 key 会被丢掉 |
| `step` | 必填；服务端按 step upsert，重复上报同一 step 是覆盖 |
| `stage` | 逐点标签，例如 `training` / `evaling` |
| `total` | 修正总步数，用于进度显示 |
| `timings` | 转成 `sys.step_seconds` distribution |
| `counters` | 每个名字转成 `sys.<name>` |

值为 `None` 的 key 不会发送，避免占用 key 并锁定类型。

### 值类型

服务端支持 3 种标量和 7 种复合类型。SDK 不解释、不校验，原样透传：

| 类型 | JSON 形态 | 约束 |
|---|---|---|
| `number` | `0.42` / `3` | 标量 |
| `string` | `"training"` | 标量 |
| `boolean` | `true` | 标量 |
| `distribution` | `{_type:"distribution", labels:[…], values:[…]}` | `labels` / `values` 等长且非空；`unit` 可选 |
| `histogram` | `{_type:"histogram", bins:[…], counts:[…]}` | `len(bins) == len(counts) + 1`；`bins` 递增；`counts` 非负 |
| `bar` | `{_type:"bar", labels:[…], values:[…]}` | `labels` / `values` 等长 |
| `table` | `{_type:"table", columns:[…], data:[[…]]}` | 每行长度等于 `columns` 长度 |
| `scatter` | `{_type:"scatter", points:[[x,y],…]}` | 每个点两个数 |
| `curve` | `{_type:"curve", kind:"pr\|roc", x:[…], y:[…]}` | `kind` 只能是 `pr` / `roc`；`x` / `y` 等长 |
| `heatmap` | `{_type:"heatmap", x:[…], y:[…], z:[[…]]}` | `z` 行数 = `len(y)`，每行长度 = `len(x)` |

前三种不带 `_type`，后七种必须带 `_type`。

```python
sc.metric({
    "loss": 0.42,
    "phase": "training",
    "converged": False,
    "time_breakdown": {
        "_type": "distribution",
        "labels": ["forward", "backward", "optim"],
        "values": [1.2, 0.8, 0.3],
        "unit": "s",
    },
    "grad": {
        "_type": "histogram",
        "bins": [-1, 0, 1, 2],
        "counts": [10, 25, 5],
    },
    "pr": {
        "_type": "curve",
        "kind": "pr",
        "x": [0, 0.5, 1],
        "y": [1, 0.7, 0.4],
    },
}, step=step)
```

裸数字数组 `[0.4, 0.6]` 等价于没有标签的 `distribution`。

### 类型锁

同一个 key 的类型由第一个非 null 值锁定。之后类型变化会被服务端以
`metric_type_conflict` 拒绝整条上报。复合对象内部的数组长度每步可变，不算类型变化。

复合对象可以带可选的 `_render` 指定渲染形式；服务端透传，不认识的值回退默认渲染。

## timings 和 counters

```python
sc.metric(
    {"loss": 0.42},
    step=step,
    timings={"forward": 1.2, "backward": 0.8},
    counters={"tokens": 8192},
)
```

`timings` 聚合成一个 `sys.step_seconds` distribution，`counters` 每个名字生成一条
`sys.<name>`。

`sys.` 是保留前缀。用户指标与内置指标撞名会触发类型锁，导致整条上报被拒。

## artifact

`save_weights_only()` / `save_state()` 成功后会自动登记 artifact，不需要手写。
`sc.artifact()` 用于额外产物：

```python
sc.artifact(
    "oss://bucket/eval-set.jsonl",
    type="dataset",
    name="eval-set",
    metadata={"rows": 2048},
)
```

只保存指针，不拉取字节。`uri` 需要带 scheme：`mint://`、`oss://`、`hf://`、
`https://` 等。

## summary

```python
sc.summary(final_loss=0.4)
sc.summary({"final_loss": 0.4})
```

两种写法等价，可以混用。按 key 合并，同 key 后写覆盖。

## 失败处理

五个方法都吞异常，只打 `mind_recursive.reporter` 的 warning：

```python
import logging
logging.getLogger("mind_recursive.reporter").setLevel(logging.WARNING)
```

要确认上报是否成功，看日志或控制台的请求列表。上报没有 `start` / `finish` /
`phase`；run 的生命周期由 `ServiceClient` 的 session 管理。

## 相关文档

- [ServiceClient](service-client.md)
- [训练](trainer.md)
- [Checkpoint 与续训](checkpoints.md)
