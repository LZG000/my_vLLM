# my_vLLM

基于 [vLLM](https://github.com/vllm-project/vllm) 的个人分支，在原项目基础上增加了**多代理/多租户场景下的 LLM 服务质量监控**功能及部分 GPU 内核优化。

> 原项目 vLLM 是 UC Berkeley Sky Computing Lab 开发的高性能、内存高效的 LLM 推理与服务引擎，采用 PagedAttention 技术实现高效的 KV 缓存管理。

---

## 项目结构

### 核心 Python 包：`vllm/`

| 子目录 | 职责 |
|---|---|
| `vllm/core/` | 核心调度器、block 管理器、KV 缓存逻辑（块空间管理、驱逐策略） |
| `vllm/engine/` | 推理引擎（同步 `LLMEngine`、异步 `AsyncLLMEngine`、多进程引擎）、**指标系统（含本分支核心新增的 agent_metrics）** |
| `vllm/entrypoints/` | API 服务器（OpenAI 兼容 FastAPI）、CLI 入口、LLM 类 |
| `vllm/model_executor/` | 模型加载器、所有模型架构实现、量化层、引导解码 |
| `vllm/attention/` | 注意力机制后端（PagedAttention、FlashAttention 等） |
| `vllm/distributed/` | 分布式通信、并行状态、KV 传输 |
| `vllm/worker/` | 工作进程（CPU/GPU/HPU/TPU 等）、模型运行器、缓存引擎 |
| `vllm/v1/` | V1 架构升级（新一代推理执行器，含独立的 attention/core/engine） |
| `vllm/lora/` | LoRA 低秩适配支持 |
| `vllm/spec_decode/` | 推测解码（Medusa、MLP 推测器、ngram、MTP） |
| `vllm/multimodal/` | 多模态支持（图像、音频、视频输入） |
| `vllm/platforms/` | 硬件平台抽象层（CUDA/ROCm/CPU/HPU/Neuron/OpenVINO/TPU/XPU） |
| `vllm/compilation/` | Torch 编译优化（FX 图优化、算子融合、Inductor 后端） |
| `vllm/third_party/` | 第三方集成（DeepSeek DeepGEMM、FlashMLA、Triton 内核） |

### 原生 CUDA/ROCm/C++ 内核：`csrc/`

| 子目录 | 职责 |
|---|---|
| `attention/` | FlashAttention 内核 |
| `moe/` | MoE 内核（含 Marlin MoE WNA16） |
| `quantization/` | 量化内核（Marlin、AWQ、GPTQ 等） |
| `mamba/` | Mamba 架构内核 |
| `core/` | 核心算子 |
| `cpu/`, `rocm/` | 特定平台内核 |

### 其他目录

| 目录 | 职责 |
|---|---|
| `benchmarks/` | 性能基准测试（延迟、吞吐、服务、前缀缓存） |
| `tests/` | 测试套件（42+ 子目录，覆盖各功能模块） |
| `examples/` | 使用示例（离线推理、在线服务、Gradio Web UI） |
| `docs/` | Sphinx 文档源码 |
| `tools/` | 开发工具（lint、profiler） |
| `.buildkite/` | CI/CD 流水线（测试、发版、基准测试） |
| `.github/` | GitHub Actions 工作流、Issue 模板 |

### 构建与依赖

| 文件 | 用途 |
|---|---|
| `setup.py` / `CMakeLists.txt` | Python + CMake 混合构建，支持多平台编译 |
| `pyproject.toml` | 项目元数据、ruff/mypy/isort/pytest 配置 |
| `requirements-common.txt` | 通用依赖（transformers、fastapi、prometheus、aiohttp 等） |
| `requirements-cuda.txt` | NVIDIA GPU 依赖（torch、xformers、ray） |
| `requirements-rocm.txt` / `cpu.txt` / `hpu.txt` 等 | 各平台专用依赖 |
| `Dockerfile*` | 各平台 Docker 构建文件 |

---

## 本分支改动（相对上游 vLLM）

### 1. 代理编排指标系统（核心新增）

**`vllm/engine/agent_metrics.py`** — 基于 Prometheus 的指标模块，按 `agent_id` 和 `model_name` 追踪：

| 指标名 | 含义 |
|---|---|
| `vllm:agent_requests_total` | 按代理统计的已完成的请求数 |
| `vllm:agent_execution_seconds_total` | 按代理统计的执行耗时（秒） |
| `vllm:agent_tool_seconds_total` | 按代理统计的工具调用耗时（秒） |
| `vllm:agent_queue_seconds_total` | 按代理统计的排队耗时（秒） |
| `vllm:agent_wait_seconds_total` | 按代理统计的总等待耗时（秒） |
| `vllm:agent_tool_time_reports_total` | 工具耗时上报次数 |
| `vllm:agent_tool_time_duplicates_total` | 重复上报（被去重）次数 |

去重机制：维护滚动窗口（最多 20,000 个 session ID），避免同一 session 的工具耗时被重复计数。

### 2. 指标系统修改

- **`vllm/engine/metrics.py`** — Prometheus 清理周期中跳过 `vllm:agent_*` 指标，防止引擎重启时被错误注销。
- **`vllm/entrypoints/openai/api_server.py`** — `/metrics` 端点改为 FastAPI 路由方式实现，多进程模式下合并引擎指标与代理指标。

### 3. 示例改进

- `examples/offline_inference/chat_with_tools.py` — Mistral 模型工具调用演示
- `examples/offline_inference/basic/mychat.py` — 交互式聊天脚本（Qwen 模型）
- `examples/online_serving/gradio_openai_chatbot_webserver.py` — Gradio Web UI 升级

---

## 快速开始

```bash
# 安装依赖
pip install -r requirements-cuda.txt

# 启动 OpenAI 兼容 API 服务
python -m vllm.entrypoints.openai.api_server --model <model_name>
```

详见原项目文档：https://docs.vllm.ai

## 许可证

Apache 2.0，与原项目一致。
