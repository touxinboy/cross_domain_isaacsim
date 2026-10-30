# 复用已找到的 Qwen3-VL，建立自己的导航推理接口

这是自己的独立服务。2026-10-28 已部署并验证真实文字、图片及 MSGNav 决策调用，现有模型文件和 Python 环境只读使用。首次文字请求发现 Qwen3-VL 4.57 要求结构化文字块，已在自己的接口中修复并增加回归检查。

本次运行已获准占用一张 GPU、最多一小时。修复后的作业号是 1749，约 20:54 自动结束。下面的提交步骤用于后续部署；已有自己的运行中服务时不要重复提交。

2026-10-28 已验证 `epic-a100` 可经 `epic-controller` 登录，账号为 `wangjiong`；分区和节点名均为 `epic-cluster-compute-a100-01`。现有 LingBot-VLA 2.0 动作服务使用 Qwen3-VL-4B 骨干，监听 44560，健康检查和 WebSocket 元数据握手均通过。没有向它发送观测、推理或重置消息。

可复用的原始模型是 `/workspace/runs/individual/wangjiong/lingbo/models/Qwen3-VL-4B-Instruct`，两份 safetensors 权重共 8,875,719,344 字节。现有环境 `/workspace/runs/individual/wangjiong/lingbo/envs/lingbot-vla-v2` 包含 PyTorch 2.8.0、Transformers 4.57.3、Accelerate 1.7.0 等依赖。本次已实际加载和生成；`--check-only` 本身仍只检查配置和分片。

这个新接口读取原始视觉语言模型，输出文字供 MSGNav 使用。现成 LingBot 服务输出的是机器人动作，不能直接替代聊天接口。OmniNav 专用航点权重仍需另行准备。

## 后续部署位置

只在这个账号自己的 `/home/wangjiong/L17/qwen_service/` 内准备代码、独立环境、缓存和日志。共享模型目录可只读使用。不修改其他项目，也不停止现有作业。

后续把上传包内容解压到该目录，得到 `qwen_transformers_server.py`、`serve_qwen.slurm`、客户端检查程序和本说明。默认使用 Transformers 直接推理，当前不需要安装 vLLM。旧的 `serve_qwen_vllm_optional.py` 只是保留的另一条部署草案，不在本上传包中。

## 先检查，再提交一个新作业

下面是已经核对过的实际路径。仅执行现有 Python，不向它安装任何包：

```bash
export L17_QWEN_ROOT="$HOME/L17/qwen_service"
export QWEN_PYTHON="/workspace/runs/individual/wangjiong/lingbo/envs/lingbot-vla-v2/bin/python"
export QWEN_MODEL_DIR="/workspace/runs/individual/wangjiong/lingbo/models/Qwen3-VL-4B-Instruct"
"$QWEN_PYTHON" -B "$L17_QWEN_ROOT/qwen_transformers_server.py" --check-only
```

模型文件和环境检查通过、可安排 GPU 资源时，再执行：

```bash
mkdir -p "$L17_QWEN_ROOT/logs"
sbatch --chdir="$L17_QWEN_ROOT" --output="$L17_QWEN_ROOT/logs/%x-%j.out" "$L17_QWEN_ROOT/serve_qwen.slurm"
```

这会提交一个单独的一卡作业，可能排队。服务仅监听 A100 的 `127.0.0.1:18080`，使用模型 ID `l17-qwen3-vl-4b`。它不连接 44560、不调用 LingBot、不下载模型。自动生成的接口密钥只保存在本账号 `L17/qwen_service/credentials/api_key`，不会打印到日志或聊天。

当前模板将单请求上限设为 16 张图片、提示词与生成合计 16384 token、每张图片预处理上限为 512×512 像素总量。这是接口试验参数，会影响细小物体识别，不能当作原论文评测配置；若超过上限，应记录失败并调整压缩策略，不应静默丢图。服务用单请求推理锁，繁忙时返回 503，不同时堆叠多次模型推理。

## 从外部调用

服务真正启动后，建立 SSH 隧道，再设置 `QWEN_BASE_URL`、`QWEN_MODEL` 与 `QWEN_API_KEY`。先执行客户端的 `--models`、`--text`、`--image` 三步检查。上传包附带 `qwen_a100_connection.md`，工程中的对应文件在 `docs/`。

在 A100 的终端本机检查时，可直接读取自己的接口密钥到环境变量，命令不会显示密钥：

```bash
export QWEN_BASE_URL="http://127.0.0.1:18080/v1"
export QWEN_MODEL="l17-qwen3-vl-4b"
export QWEN_API_KEY="$(cat "$HOME/L17/qwen_service/credentials/api_key")"
"$QWEN_PYTHON" -B "$HOME/L17/qwen_service/check_qwen_api.py" --models
"$QWEN_PYTHON" -B "$HOME/L17/qwen_service/check_qwen_api.py" --text
```

再用自己的一张实际观测图片执行 `--image 图片路径 --timeout 120`，人工核对描述是否正确。仅握手或返回 HTTP 200 不算视觉推理验证。

推理代码依据：[Qwen3-VL 官方模型与 Transformers 示例](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)。
