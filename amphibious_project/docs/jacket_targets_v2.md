# 橙色双导管架场景

默认场景为 `scenes/river_robot_jackets_v2.usda`，叠加 v1 场景，原河道、过河机器人和黄色导管架资源均保留。

陆地导管架保持原来的四腿平顶外形，主腿与下部支撑改为橙色，栏杆为黄色，上部斜撑为红色，平台为深橙色。水中导管架位于河道中心线 `y=-12 m` 附近，在 `Crossing` 视角中位于陆地目标左侧；它带有橙色支撑结构、黄色平台栏杆，以及红色开放式尖顶桁架。水中平台高出水面 1.5 米，尖顶额外高约 4.1 米。两座架体都是固定在地形上的静态碰撞目标，不会漂流。

右侧面板：`Target` 查看陆地目标；`Water target` 查看水中尖顶；`Crossing` 查看两者与机器人。任务下拉框包含陆地平顶、水中尖顶两个目标，选中后可重置机器人到各任务起点。目标选择只提供任务参考，尚未连接自动导航。

参考停车／观察点已分别记录在 `configs/jacket_targets_v2.json`，水中目标参考点位于架体下游，避免撞入支腿。真值坐标只供评测或 PointGoal 基线使用，不能直接输入视觉语言目标搜索策略。

启动：

```bash
cd /home/ubuntu/sdc/L17/amphibious_project
python3 -B scripts/launch.py gui --local-assets-only --physics-gpu-index 0
```

`results/jacket_targets_v2/verification.json` 记录目标碰撞数量、基础入水情况、观察点净空、左右关系、原文件哈希和三张原生渲染截图。
