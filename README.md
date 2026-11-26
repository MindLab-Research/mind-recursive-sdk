# MIND-RECURSIVE SDK

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


使用前阅读文档： [mind-recursive SDK 文档](https://github.com/MindLab-Research/mind-recursive-sdk/blob/main/docs/README.md)。
