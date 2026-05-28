# club-3090 Docker 双卡极限性能方案

> 基于 [noonghunna/club-3090](https://github.com/noonghunna/club-3090) 的独立自包含 Docker 部署方案，专注于 **2× RTX 3090 PCIe 极限性能优化**。

## 项目定位

[club-3090](https://github.com/noonghunna/club-3090) 是一个面向 RTX 3090 的多引擎、多模型 LLM 本地部署项目，支持单卡/双卡、vLLM/llama.cpp/ik_llama 等多种配置。本项目（`docker/`）从中提取并优化了双卡 3090 的部署方案，形成可独立运行的轻量目录——`cp -r` 到任意机器即可使用。

**核心优化方向：**

- 针对 Qwen3.6-35B-A3B MoE（3B active/token）优化，替代原始的 27B dense 模型
- 引入 int8 KV cache 量化，KV pool 提升 86%
- MTP-3 speculative decoding 调优，代码 TPS 达 ~192
- 262K 上下文支持，4.4× 并发能力

## 相对于 club-3090 的改动

| 维度 | club-3090 上游 | 本项目 |
|---|---|---|
| **推荐模型** | Qwen3.6-27B (27B dense) | **Qwen3.6-35B-A3B** (3B active MoE) |
| **量化** | AutoRound INT4 (Lorbus) | **AWQ-INT4** (cyankiwi, compressed-tensors) |
| **KV Cache** | fp16 / fp8_e5m2 | **int8_per_token_head**（+86% KV pool） |
| **Drafter** | DFlash N=5 | **MTP N=3**（接受率 46-82% vs 10-59%） |
| **最大上下文** | 185K | **262K** |
| **代码 TPS** | ~115 (DFlash) | **~192** (MoE MTP + int8 KV) |
| **叙述 TPS** | ~41 (DFlash) | **~129** (MoE MTP + int8 KV) |
| **Thinking 模式** | 默认开启 | **默认关闭**（代码场景 +24% TPS） |
| **GPU 显存占用** | 0.95 | **0.85**（留更多余量） |
| **部署方式** | 依赖项目脚本 + PyYAML + .env 配置 | **纯 Docker Compose**，零外部依赖 |
| **Prefix Caching** | 开启 | **关闭**（MTP 场景下 +20-77% TPS） |

**关键性能洞察（来自 [tfriedel/qwen3.6-rtx3090-lab](https://github.com/tfriedel/qwen3.6-rtx3090-lab)）：**

MTP + `--no-enable-prefix-caching` 是双卡 MoE 的最佳组合。关闭 prefix caching 损失了缓存命中率，但 MTP 的 speculative decoding 带来 +20-77% 的 TPS 提升，净收益远大于缓存损失。

## 实测性能（2× RTX 3090 PCIe）

### 推荐方案：MoE MTP + int8 KV

当前默认配置，综合性能最优：

| 场景 | TPS | 说明 |
|---|---|---|
| 代码 (think OFF) | **~192** | 推荐日常使用 |
| 叙述 (think OFF) | **~129** | 默认配置 |
| 叙述 (think ON) | **~155** | 长文写作场景 |
| 代码 (think ON) | **~164** | 需要推理的编程任务 |

### 跨方案对比

| 指标 | MoE MTP + int8 (推荐) | DFlash | MoE (llama.cpp) |
|---|---|---|---|
| 模型 | 35B-A3B (3B active) | 27B dense | 35B-A3B (3B active) |
| 引擎 | vLLM | vLLM | ik_llama.cpp |
| KV Cache | int8 | fp16 | f16 |
| 代码 TPS | **~192** | 115 | — |
| 叙述 TPS | **~129** | 41 | ~131 |
| 最大上下文 | **262K** | 180K | 196K |
| KV Pool | **1.15M tokens** | 197K | — |
| 并发度 | **4.41x** | 1.10x | — |
| int8 KV | 支持 | 不支持 | — |
| Vision | 开启 | 开启 | 关闭 |
| VRAM/卡 | ~22.6 / 24 GB | ~23.6 / 24 GB | ~14 / 24 GB |

### KV Cache 类型选择

| KV 类型 | 启动 | MTP 效果 | TPS | KV Pool | 建议 |
|---|---|---|---|---|---|
| **int8** | 正常 | 正常 (58-60%) | 叙述 ~129 / 代码 ~192 | **1.15M** | **推荐** |
| fp16 | 正常 | 更好 (70-85%) | 叙述 ~179 / 代码 ~264 | 818K | 追求峰值 TPS |
| fp8_e4m3 | 正常 | **失效** (0%) | ~27 / ~66 | 1.53M | 不可用 |
| fp8_e5m2 | **崩溃** | — | — | — | 不可用 |

详细 benchmark 数据：[`banchmark/moe-awq-mtp-kv-comparison.md`](banchmark/moe-awq-mtp-kv-comparison.md)

## 使用说明

### 前提条件

| 项目 | 要求 |
|---|---|
| GPU | 2× RTX 3090 (24 GB) |
| 驱动 | NVIDIA 580.x+（CUDA 13） |
| 磁盘 | ~23 GB（MoE AWQ 模型） |
| 内存 | 32 GB+ 推荐 |
| Docker | Docker + NVIDIA Container Toolkit |
| 网络 | ModelScope 或 HuggingFace + Docker Hub |

### 1. 下载

```bash
git clone https://github.com/superlee-terry/club-3090-dual-performance.git
cd club-3090-dual-performance
```

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`，推荐配置（MoE MTP + int8 KV）：

```bash
# GPU 选择
GPUS=0,1

# MoE MTP（推荐方案）
MOE_AWQ_PORT=11434          # API 端口
MOE_AWQ_CTX=262144          # 最大上下文
MOE_AWQ_GPU_UTIL=0.85       # 显存占用（留余量）
MOE_AWQ_MTP_TOKENS=3        # MTP draft tokens 数
MOE_AWQ_MODEL_ALIAS=qwen3.6 # API 模型名
```

### 3. 下载模型

```bash
# 推荐：仅下载 MoE AWQ（~23 GB）
bash scripts/download.sh --moe-awq

# 下载全部（含 DFlash、MoE GGUF）
bash scripts/download.sh --all
```

下载策略：**ModelScope 优先，HuggingFace 自动降级**。

安装下载工具（二选一）：

```bash
pip install modelscope       # 推荐（国内快）
pip install huggingface_hub  # 备选
```

### 4. 启动

```bash
# 推荐：MoE MTP
bash scripts/start.sh start moe-mtp

# 其他方案
bash scripts/start.sh start dflash    # DFlash（27B dense）
bash scripts/start.sh start moe       # MoE (llama.cpp)
```

首次启动约 3-5 分钟（JIT 编译 + CUDA graph）。后续利用缓存约 10-15 秒。

查看日志：

```bash
docker logs -f vllm-qwen36-35b-a3b-mtp
```

等待 `Application startup complete.` 即可。

### 5. 测试

```bash
curl -sf http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.6","messages":[{"role":"user","content":"你好，介绍一下你自己"}],"max_tokens":200}'
```

### 6. 停止

```bash
bash scripts/start.sh stop moe-mtp
```

### 常用命令

```bash
bash scripts/start.sh status          # 查看所有服务状态
bash scripts/start.sh logs moe-mtp    # 查看日志
bash scripts/start.sh restart moe-mtp # 重启
bash scripts/start.sh stop moe-mtp    # 停止
```

## 高级配置

### 切换 KV Cache 类型

编辑 `compose/moe-awq-mtp.yml`，在 `--enable-chunked-prefill` 后面添加或删除：

```yaml
# int8 KV（推荐，KV pool +86%，TPS -25%）
- --kv-cache-dtype
- int8_per_token_head

# fp16 KV（不添加此参数即为 fp16，最高 TPS）
```

### 切换 Thinking 模式

```yaml
# 关闭（默认，代码更快）
- --default-chat-template-kwargs
- '{"enable_thinking": false}'

# 开启（叙述 +20% TPS，代码 -15% TPS）
- --default-chat-template-kwargs
- '{"enable_thinking": true}'
```

也可在请求级别临时切换：

```bash
# 单次请求开启 thinking
curl ... -d '{"...,"chat_template_kwargs":{"enable_thinking":true}}'
```

### 手动启动（跳过脚本）

```bash
docker compose --env-file .env -f compose/moe-awq-mtp.yml up -d
```

> **注意**：必须加 `--env-file .env`。

### Docker 镜像

预构建的 vLLM 镜像已推送到 Docker Hub，`docker compose up` 会自动拉取：

```bash
docker pull lee21321/vllm-openai:club-3090-performance
```

基于 `vllm/vllm-openai:nightly-aa2b56ffb0c`（v0.21.1rc1），已验证兼容 MoE AWQ + MTP + int8 KV。

## 目录结构

```
club-3090-dual-performance/
├── .env.example              # 配置模板
├── .env                      # 实际配置（gitignore）
├── LICENSE                   # Apache 2.0
├── README.md                 # 本文件
├── banchmark/                # 性能基准测试记录
├── models/                   # 模型权重（gitignore）
│   ├── qwen3.6-27b/
│   │   ├── autoround-int4/   # DFlash 主模型 ~14 GB
│   │   └── dflash/           # DFlash draft ~3.5 GB
│   └── qwen3.6-35b-a3b-awq-int4/  # MoE AWQ ~23 GB（推荐）
├── patches/                  # vLLM 补丁
│   ├── vllm-marlin-pad/      # Marlin INT4 TP=2 kernel 修复
│   ├── vllm-pr35936/         # tool_choice 修复（新 vLLM 已跳过）
│   ├── vllm-pr41800/         # truncate_prompt_tokens 修复
│   ├── tolist-cudagraph/     # CUDA graph tolist 修复
│   └── chat-template/        # Qwen 聊天模板 7 个 bug 修复
├── cache/                    # JIT 编译缓存
├── scripts/                  # 辅助脚本
│   ├── download.sh           # 模型下载
│   ├── start.sh              # 启动/停止/状态管理
│   ├── verify.sh             # DFlash 健康检查
│   └── detect_nvlink.sh      # NVLink 自动检测
└── compose/                  # Docker Compose 配置
    ├── dflash.yml            # DFlash（27B dense）
    ├── moe-llamacpp.yml      # MoE (llama.cpp)
    └── moe-awq-mtp.yml       # MoE MTP + int8 KV（推荐）
```

## 致谢

本项目基于以下工作：

- **[noonghunna/club-3090](https://github.com/noonghunna/club-3090)** — 原始项目框架、脚本工具、DFlash 配置、补丁和 benchmark 方法论
- **[tfriedel/qwen3.6-rtx3090-lab](https://github.com/tfriedel/qwen3.6-rtx3090-lab)** — MoE AWQ + MTP-3 的关键性能发现（MTP + no-prefix-caching 组合）
- **Qwen 团队** — Qwen3.6 系列模型和 MTP 架构
- **[cyankiwi](https://huggingface.co/cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit)** — AWQ-INT4 量化模型
- **[Lorbus](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound)** — AutoRound INT4 量化模型（DFlash 方案使用）
- **[z-lab](https://github.com/luce-spec)** — DFlash draft 模型
- **vLLM 项目** — 推理引擎
- **ik_llama.cpp** — MoE GGUF 推理引擎

## 开源协议

本项目衍生自 [club-3090](https://github.com/noonghunna/club-3090)，遵循其原始协议：

- **club-3090**: [Apache License 2.0](https://github.com/noonghunna/club-3090/blob/master/LICENSE)
- **vLLM**: [Apache License 2.0](https://github.com/vllm-project/vllm/blob/main/LICENSE)
- **llama.cpp**: [MIT License](https://github.com/ggerganov/llama.cpp/blob/master/LICENSE)

本目录下的配置文件、脚本和补丁同样以 **Apache License 2.0** 发布。可自由使用、修改和分发。

## 常见问题

**Q: 推荐哪个方案？**
A: **MoE MTP + int8 KV**（默认配置）。2×3090 上 TPS 最快，KV pool 最大，支持 262K 上下文。

**Q: int8 KV 和 fp16 KV 怎么选？**
A: 追求更长上下文/更高并发用 int8（KV pool 翻倍，TPS 降低约 25%）。追求最高 TPS 用 fp16（代码 ~264）。

**Q: DFlash 还值得用吗？**
A: 仅在需要 27B dense 模型质量时。DFlash 的 TPS 只有 MoE MTP 的 1/3 到 1/2，且不支持 int8 KV。

**Q: Thinking 开还是关？**
A: 代码场景关（+24% TPS），叙述场景开（+20% TPS）。默认关闭。

**Q: 两个服务能同时运行吗？**
A: 2×3090 VRAM 有限，建议一次只运行一个。

**Q: 容器 OOM 崩溃？**
A: 降低 `MOE_AWQ_GPU_UTIL`（如 0.80）。确认两张 3090 均被识别：`nvidia-smi -L`。

**Q: WSL2 启动失败？**
A: 取消 `.env` 中 `PYTORCH_CUDA_ALLOC_CONF` 和 `VLLM_ENFORCE_EAGER` 的注释。

**Q: 没有 NVLink？**
A: 完全可以。`detect_nvlink.sh` 自动检测。无 NVLink 性能损失约 10-15%。

**Q: 下载慢？**
A: 安装 `modelscope`（`pip install modelscope`），自动优先从 ModelScope 下载。

**Q: .env 没生效？**
A: compose 文件在 `compose/` 子目录下，必须 `--env-file .env`，或用 `bash scripts/start.sh`。
