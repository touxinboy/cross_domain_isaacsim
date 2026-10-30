# 导管架导航目标

新场景：`scenes/river_robot_jacket_v1.usda`。它叠加原来的 `river_robot_v1.usda`，原河道、机器人和过河运动测试不改动。

导管架位于对岸 `(12, -23)` 米附近，四根外张主腿、三层 X 形支撑、黄色钢管、蓝色检修平台和爬梯。主框架高 6 米，底部腿间距 3.6 米，顶部腿间距 2.4 米，栏杆额外高 1.2 米。主要钢管、平台和四个混凝土基础带静态碰撞；爬梯和节点套筒为外观细节。这是用于感知与导航的简化模型。

机器人从 `(-12, -23)` 出发，参考停车点为 `(8.2, -23)`，到达半径 0.6 米，朝向 +X（导管架方向）。后续评价还应检查朝向误差不超过 20 度、速度不超过 0.05 m/s、持续 1 秒，并确认看见正确目标。这些判定参数已记录，自动导航与评价循环尚未接入。

在服务器双击自己的 `start_here.sh`，或运行：

```bash
cd /home/ubuntu/sdc/L17/amphibious_project
./start_here.sh
```

如果启动卡在外部资产库查询，可使用本项目的本地资产目录启动（不包含 NVIDIA 示例资产，当前场景全部使用自己的几何资源）：

```bash
python3 -B scripts/launch.py gui --local-assets-only --physics-gpu-index 0
```

默认打开新场景；右侧 `L17 Amphibious Robot` 面板默认选择导管架任务。`Target` 按钮切换目标近景，`Crossing` 查看过河区域，`Robot` 查看机器人。重置按钮返回本任务起点，Play 和执行器滑块用于手动运动；添加目标不会自动启动导航。

目标本体资源为 `assets/targets/jacket_4leg_v1.usda`，场景中的路径为 `/World/NavigationTargets/Jacket`，语义类别为 `jacket_structure`。没有可见的 Goal 旗标，停车点只是一个无渲染几何的 Xform。

`configs/jacket_target_v1.json` 中的真实目标坐标用于评测器或明确标注的 PointGoal 基线；视觉语言搜索算法只接收任务指令、传感器和允许的自身状态，不能直接读取目标真值坐标。`results/jacket_target_v1/verification.json` 记录碰撞数量、净空、原场景哈希和截图校验值。
