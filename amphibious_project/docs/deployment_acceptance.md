# 上传前核验与服务器验收

核验日期：2026-10-29。此次没有连接服务器、申请 GPU 或运行新的 PhysX 导航实验。代码包与模型目录分别传输，目标均为自己的 `/home/ubuntu/sdc/L17/amphibious_project`。

## 本地已确认的内容

| 检查 | 结果与边界 |
|---|---|
| 模型文件 | 14 个推理文件的大小、SHA-256 与作者固定提交一致 |
| 模型结构 | 作者完整模型在 meta 设备上构造，840 个状态键与权重索引的键名、形状相符；15 个导航头张量均为有限值。没有把完整权重加载到本机 GPU |
| 实际前向接口 | 真实分词器、作者预处理、原版定制 Transformers 完成七图、六位置输入的小结构 CPU 前向；输出五个二维航点与五个到达值。这里的网络权重是小结构随机权重，不是完整模型导航结果 |
| 场景组成 | OpenUSD 实际打开 8 个层、1031 个 prim，无未解析资产；米制、Z-up、一个动态机器人、组合后的物理频率 120 Hz |
| 依赖 | 隔离核验环境使用 PyTorch 2.6、TorchVision 0.21 和计划中的核心依赖，`pip check` 无冲突；官方元数据中 12 项固定依赖均有 Python 3.10 / Linux x86_64 对应包。Linux CUDA 环境尚未实际安装 |
| 控制循环 | 全部 100 项离线测试通过。CPU 事件/运动替身运行真实异步编排，验证 shadow、反复短步、推理报错、错误场景、重复输出目录和导出报错后的退出清理；它不能替代目标机 PhysX 实验 |
| 上传包 | 重新打包后，从解压副本检查完整清单、场景、源文件与模型契约，并运行测试。具体结果见包外核验记录 |

## 已修复的部署问题

- 模型缓存和临时文件显式限定在项目 `.runtime/model_cache`，不依赖共享账号的默认 Hugging Face 缓存。
- GUI 入口在空闲状态刷新自己的相关模块，避免 Kit 缓存继续调用旧控制或旧传感器代码；其他项目的同名模块会被识别并拒绝刷新。
- 启动请求有效期 5 分钟；旧请求不会在重连时意外启动。Timeline Stop 会更换运行时 epoch 并清零需求，旧计划因此失效。
- 模型进程记录源码/相机配置指纹、模型配置哈希和停止状态。服务未准备好、已停止或版本不一致时，任务不会开始运动。
- 格式错误的队列请求会留下终态错误结果，避免无限重试；`--once` 推理失败返回非零状态。
- 退出先清零并释放本次物理回调，即使 CSV 导出失败也执行清理。已经存在的实验目录不会被覆盖。
- 场景路径要求与任务配置相符。到达点中的栅格文件改为项目相对路径；检查器同时检查 Linux 路径大小写。
- 导航相机改为作者 Slowfast 的 RGB 视图约定：720×640、水平视场 110°、左右 ±90°。安装位置适配自己的两栖机器人。此前 640×360 的静态标定只属于旧相机配置，新配置仍需在目标 Isaac 上重新标定。

## 第一次上传后的操作

将核验版 ZIP 解压到 L17，使其中 `amphibious_project` 对应服务器自己的项目目录。再单独上传完整的 `checkpoints/OmniNav_Slowfast`。ZIP 不含模型、虚拟环境、SSH 密钥或运行缓存；已存在的场景会话需要重新加载代码。

在该项目目录执行：

```bash
bash deploy/omninav/prepare_server.sh
```

该命令先核对上传清单、场景和模型 SHA-256，再选择 Python 3.10/3.11/3.12 建立独立环境，安装依赖并检查导入。日志在 `results/deployment/install_*.log`。它不启动 Isaac 或模型推理，不申请 GPU。

确认交互式 Isaac 机器上某张 GPU 可用后，例如 GPU 1：

```bash
bash deploy/omninav/start_worker.sh 1
```

该命令使用选定 GPU，在模型发布 `ready.json` 前实际做一次完整训练权重的七图前向。看到 `L17_WAYPOINT_WORKER_READY` 且 `startup_smoke_passed=true` 后再启动导航。结果记录在 `.runtime/navigation/worker/startup_smoke.json`，包括推理耗时和 GPU 峰值分配内存。

这是给 ToDesk 的双 4090 交互机器准备的示例。EPIC 集群须先有正常分配的 Slurm GPU 作业，并省略 GPU 参数以保留分配的 CUDA 掩码；本脚本不会提交 Slurm 作业。

在当前自己的 Isaac 4.5 GUI 打开 `scenes/river_robot_jackets_v2.usda`，手动执行器需求应为零。在 Script Editor 执行：

```python
from pathlib import Path
p = Path('/home/ubuntu/sdc/L17/amphibious_project/scripts/verify_isaac_gui.py')
exec(compile(p.read_text(), str(p), 'exec'), {'__file__': str(p), '__name__': '__main__'})
```

查看 `results/deployment/isaac_sensor_check.json` 的 `passed`。它会实际验证机载三视图 RGB-D、相机前方标定立方体、原生位姿匹配与控制时钟绑定；机器人位姿和速度不会被直接写入。

模型与传感器通过后，在服务器终端准备 shadow 任务：

```bash
python3 -B scripts/run_navigation_closed_loop.py --goal configs/navigation_goal.example.json --mode shadow
```

在自己的 GUI 的 Script Editor 执行：

```python
from pathlib import Path
p = Path('/home/ubuntu/sdc/L17/amphibious_project/scripts/start_navigation_closed_loop.py')
exec(compile(p.read_text(), str(p), 'exec'), {'__file__': str(p), '__name__': '__main__'})
```

检查本次 `results/navigation/run_.../summary.json`：应为 `reason=shadow_interface_recorded`、`error=null`、`cleanup_errors=[]`，并存在 `head_001.json`、`shadow_command.json`。shadow 不施加导航驱动力。

确认示例点是当前观测中的自由空间后，用新面板测试短程运动：

```bash
python3 -B scripts/run_navigation_closed_loop.py --goal configs/navigation_goal.example.json --mode execute --send
```

停止命令：

```bash
python3 -B scripts/run_navigation_closed_loop.py --stop
```

## 还不能据此承诺的结果

完整模型在目标 GPU 上的加载、真实七图推理、新相机配置的实际 RGB-D、以及新版控制回调与 PhysX 的组合，仍需上述目标机验收。验收脚本已经准备，不等于这些硬件测试已经通过。

示例任务是手动 PointGoal 接口试验。完整语义过河仍需在线观测地图、可通行性、语义记忆与目标到达点生成；Qwen 服务也需处于有效运行窗口。当前没有新的导航成功率、SPL 或实时性能结论，模式切换仍是接触/浮力/局部深度真值的基准策略。
