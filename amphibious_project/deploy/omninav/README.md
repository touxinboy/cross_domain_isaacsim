# OmniNav 专用航点模型：在服务器部署

准备日期：2026-10-29。此次只准备本地代码和离线检查，没有重连服务器、启动 GPU 作业或安装服务器环境。

最新完整验收入口见 [上传前核验与服务器验收](../../docs/deployment_acceptance.md)。本地已经验证真实分词器/作者预处理与小结构前向、840 个模型状态键、完整场景组成和控制循环异常清理；目标服务器 GPU 与 PhysX 联调仍待执行。

## 空间与部署位置

| 内容 | 大小 | 应放在哪里 |
|---|---:|---|
| 我们自己的场景、机器人、脚本、配置 | 约 12 MB，随代码增加略有变化 | 本地保留；上传服务器 L17 |
| MSGNav / OmniNav 第三方源码 | 当前约 133 MB，包含源码仓库元数据 | 本地可保留；服务器用于加载原始函数和定制 Transformers |
| OmniNav_Slowfast 权重 | 8,167,451,146 字节，约 8.17 GB；另有约 16 MB 配置与分词器 | 服务器 `L17/amphibious_project/checkpoints/OmniNav_Slowfast` |
| 隔离 Python/CUDA 推理环境 | 估计 8–15 GB，取决于软件包和安装缓存 | 服务器项目 `.venvs/omninav`、`.runtime/model_cache` |
| RGB-D、日志、截图 | 随实验增长；当前最多约 45 次三视图采集，720×640 的 float32 深度约 249 MB，另有 RGB 和日志 | 服务器 `results`；只下载汇报所需结果 |
| Isaac Sim | 已安装的软件继续使用 | 不复制到本地，不另装一份 |
| Qwen 图文服务 | 现有 A100 上已有权重 | 通过 API 调用，不在本地复制一份 |

建议服务器为 OmniNav 环境、权重和早期实验留出 **30–50 GB 可用空间**。这个数字是容量预算，不是实测安装体积，也不含服务器上已有的 Isaac Sim。

2026-10-29 按用户“继续下载模型”的要求恢复断点下载。**两片权重与全部推理配置已完整下载，所有文件的固定大小和 SHA-256 均通过检查。** safetensors 头内已确认航点、角度、位置编码和到达预测参数存在。模型尚未加载到 GPU，也未进行真实航点推理。上传包排除整个 `checkpoints`；重连后可单独传输完整模型目录，服务器直接使用本地权重，无需再下载一遍。

## 模型选择与固定版本

使用作者发布的 [OmniNav_Slowfast](https://www.modelscope.ai/models/chongchongjj/OmniNav_Slowfast)，不使用普通 Qwen 对话权重替代航点头。

- 官方源码：[amap-cvlab/OmniNav](https://github.com/amap-cvlab/OmniNav)，固定提交 `e8b485953c6555e841a3283c3e3a4f7d8d0f24c3`。
- 权重仓库固定提交 `3159d6893927e7043b837b27eddca090ba47c143`。每个推理文件的大小与 SHA-256 见 `configs/omninav_checkpoint.lock.json`。
- 官方 config 已核对：`waypoint_number=5`、`predict_angle=true`、`use_arrive_list=true`、`cross_attention_flow_match=false`、Transformers 4.51.3、BF16。
- 官方权重索引包含 `wp_predictor`、`wp_predictor_angle`、`input_wp_encoder`、`query_action`、`query_multihead_attn` 和 `arrive_predictor`。完整下载后还要核对两片 safetensors 内的实际张量头。
- 推理环境使用作者定制的 Transformers 源码。Isaac 自带 Python、Qwen3 服务环境和其他人的环境不修改。
- 首次准备使用标准 `eager` attention，避免先编译 FlashAttention。显存和推理时延需要在目标 GPU 上实测；之后可独立验证 `sdpa` / FlashAttention。

## 服务器重新可用后的准备命令

以下命令均在用户自己的项目目录执行，不会启动 Isaac 或申请 GPU。

```bash
cd /home/ubuntu/sdc/L17/amphibious_project
python3 -B deploy/omninav/prepare_checkpoint.py --download
bash deploy/omninav/setup_environment.sh
.venvs/omninav/bin/python -B deploy/omninav/prepare_checkpoint.py --verify
.venvs/omninav/bin/python -B deploy/omninav/check_environment.py --output results/omninav_environment.json
```

安装脚本自动选择已有的 Python 3.10/3.11/3.12，也可通过 `OMNINAV_BASE_PYTHON` 指定。环境、缓存与日志均在自己项目目录中，不往 Isaac 自带 Python 中安装依赖。可先使用 `python3 deploy/omninav/setup_environment.py --dry-run` 查看安装计划。

## 启动独立航点进程

先核对 GPU 空闲情况。下面的 GPU 1 仅是原双 4090 机器的示例，不能把它当作当前可用资源的保证。启动命令不应放进自动安装脚本。

```bash
cd /home/ubuntu/sdc/L17/amphibious_project
CUDA_VISIBLE_DEVICES=1 .venvs/omninav/bin/python -B scripts/omninav_worker.py \
  --checkpoint checkpoints/OmniNav_Slowfast --device cuda:0 --attention eager --idle-exit-s 600 --startup-smoke
```

模型只加载一次；服务只读取项目内部的文件队列，不开放网络端口。空闲 10 分钟退出。`ready.json` 和 `heartbeat.json` 位于 `.runtime/navigation/worker`。模型加载失败保留原错误，不使用脚本路点伪装模型输出。

独立 Windows CPU 核验环境已安装并验证核心依赖。目标 Linux CUDA 环境安装、完整训练权重加载和 GPU 推理 **尚未执行**。
