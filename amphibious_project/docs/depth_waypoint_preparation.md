# 深度定位、OmniNav 与物理闭环的本地准备

更新时间：2026-10-29。此次工作在本地完成；服务器处于断开状态。新代码已经过离线接口测试，尚未上传、加载真实 OmniNav 权重或运行新的 Isaac 导航实验。

本地模型结构、原版前向接口、场景组成和部署异常路径的核验结果，及上传后的简化入口，见 [服务器验收说明](deployment_acceptance.md)。

## 深度定位如何接到 MSGNav

1. MSGNav 选择一张观测图片，结果仍是图片编号和目标描述。
2. `resolve_navigation_depth.py` 找到这张图对应的 `frame.json`。必须是机载相机，并且 RGB 文件的 SHA-256 与深度帧一致。
3. 二维定位可以重放这张图片上保存的 Qwen 结果，也可通过 `--use-qwen` 明确发出一次新的图文请求。旧视口截图的像素不能套用到新的机载深度图。
4. `depth_localization.py` 检查米制深度、内外参、图像尺寸和同一时钟的时间戳，再在二维点附近提取深度簇、反投影得到目标可见表面的世界坐标。
5. `approach_planning.py` 在**外部提供的、已观测且经过能力检查的可通行栅格**上选择有机器人净空、与目标保持距离的到达点。未知格视为不可通行。它使用 Dijkstra 检查连通性，并返回到达点、朝向和参考栅格路径。

第 4 步的表面点与第 5 步的机器人到达点必须分别记录。表面点不能直接驱动机器人，也不能当作目标整体中心。第 5 步是我们补充的基础适配器，不是 MSGNav 或 OmniNav 原有的两栖能力。

**尚缺的输入**：从 Isaac 机载观测持续生成、融合可通行栅格的地图模块。当前没有从隐藏的 USD 对象或目标配置中偷取坐标来补这个缺口。水深、岸坡、地表支撑和水中障碍不能仅凭一张普通深度图完整恢复。

重放已保存结果的命令结构：

```bash
python3 -B scripts/resolve_navigation_depth.py \
  --observation results/your_run/observation.json \
  --decision results/your_run/msgnav_decision.json \
  --grounding results/your_run/grounding.json \
  --image-root . --output results/your_run/selected_surface.json
```

只有已有观测栅格时，才可生成停车姿态：

```bash
python3 -B scripts/approach_planning.py \
  --surface results/your_run/selected_surface.json \
  --grid results/your_run/observed_grid.json \
  --robot-xy -12 -23 --medium land --output results/your_run/arrival_goal.json
```

这些路径为接口示例，不代表本地已经存在对应真实观测。

## 相机与时间

机体安装左、前、右三台针孔相机，同一暂停状态采集。上传前核验时已对齐作者 Slowfast 的 RGB 约定：720×640、水平视场 110°、左右 ±90°；自己的安装位置保留。每帧保存 RGB、float32 米制轴向深度、内参、光学系到世界系变换、图像哈希与时间戳。固定总览/目标相机仅用于诊断，不进入机器人观测。旧的 640×360 标定属于此前配置，新配置仍需在目标机验收。

光学系为 X 向右、Y 向下、Z 向前；世界为米制、Z 向上。渲染时间与相机位姿时间都使用 Isaac 原生仿真时钟，另记录 `RobotRuntime` 的 `controller_time_s` 和 `controller_epoch`，防止把运行时重置前后的数据混用。机器人自身位姿当前来自模拟器，这是明确的基线假设，尚未使用视觉 SLAM。

上次断线前，服务器曾完成静态立方体几何标定：预期轴向深度 2.7 m，返回约 2.7000017 m。这只能验证理想渲染数据的几何接口，不能说明真实深度相机达到微米精度。该次原始捕获尚未下载到本地；本轮增加的运行时绑定字段仍需重连验证。

深度尚未模拟水下折射、浑浊或声呐。没有深度或语义点落在镂空背景时，应记录失败或重新观测。

## OmniNav 输入、输出与控制

`ObservationBuffer` 要求四帧真实更早的前视观测，以及当前左/前/右三视图。即使原地观察，各历史帧也必须来自不同的物理时间；不复制同一帧充当历史。模型接收六个位置项：四个历史自身位置、当前位置和目标到达点。坐标转换和 0.3 的空间缩放保留作者约定。

专用航点头返回五个二维预测、五个转向角以及到达 logits。适配器保留作者“第一个预测的半径 + 独立角度”的执行解释，执行短程目标，再用新观测重规划。到达 logits 转为概率供分析，不直接作为任务成功条件。

`omninav_worker.py` 与 Isaac 分进程运行。结果通过文件队列返回；`NavigationSession` 核对请求 ID、运行时 epoch、观测哈希、位姿锚点、仿真年龄和墙钟超时。模型报错、非有限值、过期或超长步均清空计划，输出零执行器需求。

物理更新使用现有 120 Hz PhysX，连续控制为 20 Hz。执行器分配输出左右履带速度和双推进器推力。水陆模式沿用接触、浮力和局部深度真值的基准切换器，并非学习到的安全切换算法。导航实验脚本不写入机器人位姿或速度。

首次闭环采用“暂停采图和推理，执行最多 1 秒短程运动，再暂停”的实验设置。这样可以先区分输入、模型预测和动力学错误；**不能把暂停推理的结果报告成实时闭环性能**。

## 重连后的操作

先按 `deploy/omninav/README.md` 在服务器 L17 安装独立环境并启动航点进程。新场景继续使用自己的 `river_robot_jackets_v2.usda`。现有 GUI 中手动指令应为零。

首先只准备一次 shadow 任务，记录模型输出和拟议执行器指令，不施加导航驱动力：

```bash
cd /home/ubuntu/sdc/L17/amphibious_project
python3 -B scripts/run_navigation_closed_loop.py --goal configs/navigation_goal.example.json --mode shadow
```

更新代码后，在当前**自己的** Isaac GUI 的 Script Editor 运行一次：

```python
from pathlib import Path
p = Path('/home/ubuntu/sdc/L17/amphibious_project/scripts/start_navigation_closed_loop.py')
exec(compile(p.read_text(), str(p), 'exec'), {'__file__': str(p), '__name__': '__main__'})
```

此后新面板支持从终端向既有 GUI 发送任务，无需再启动一个 Isaac 进程：

```bash
python3 -B scripts/run_navigation_closed_loop.py --goal configs/navigation_goal.example.json --mode shadow --send
python3 -B scripts/run_navigation_closed_loop.py --status
python3 -B scripts/run_navigation_closed_loop.py --stop
```

确认模型输出和路径后，`--mode execute --send` 才进入实际短程执行。示例目标是**手动 PointGoal 接口测试坐标**，应先在观测中确认它位于自由空间。它不是 Qwen 找到的语义目标，更不代表完整过河任务。语义导航应使用观测栅格生成的 `arrival_goal.json`，并提供可移植的 `grid_file` 项目内路径。

## 导出和未完成项

每次任务在自己的 `results/navigation/run_...` 保存：请求、观测 JSON、原始航点头输出、拟议/实加控制 CSV、PhysX 遥测 CSV、请求与拒绝事件、进度和摘要。原始 RGB-D 保留在 `results/rgbd`；观测 JSON 记录相对路径和哈希，无需在每次请求中复制全部图片。

目前可以准备并离线验证传感器绑定、几何定位、到达点接口、模型文件队列和执行器分配。真实 OmniNav 推理、机载目标图文定位与深度的一次完整串联、新版 Isaac 控制循环、在线地图及语义探索仍需后续实测。导航成功率、SPL、实时帧率和模型耗能尚无新结果。
