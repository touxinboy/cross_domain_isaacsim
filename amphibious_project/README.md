# L17：自己的 Isaac Sim 场景和任务

服务器共用同一份 Isaac Sim 4.5 软件安装。这个文件夹是你自己的工程，
不会加载其他人的河流、机器人或任务场景，也不会修改 Isaac Sim 安装目录。

工程入口：`/home/ubuntu/L17/amphibious_project`。
真实路径：`/home/ubuntu/sdc/L17/amphibious_project`。

## 打开自己的实例

在 ToDesk 的 Linux 终端运行：

```bash
bash /home/ubuntu/L17/amphibious_project/start_here.sh
```

图形界面使用连接桌面的 GPU 0，打开配置指定的自己的场景；当前为 `scenes/river_robot_v1.usda`。
后续无界面实验可以使用 GPU 1。
这是自己的森林河道与双履带、双推进器机器人，未使用其他人的服务器场景或 CD-HSSRL 网格。
软件标题是 `L17 - Amphibious Navigation`。
首次启动的着色器编译可能需要几分钟。

## 当前机器人及组件验证

`scenes/river_robot_v1.usda` 引用原森林河道并加入约 1 米、48 kg 的原型。
打开工程会自动加载履带牵引、体积浮力、水阻和双推进器物理回调。
参数是设计初值，尚未标定；履带采用等效接触模型，没有逐节链条或土壤沉陷。

右侧 `L17 Amphibious Robot` 面板可手动控制、复位、切换视角，
或一键运行 land / float / thrust 三项组件检查。检查结果和 CSV 在 `results/robot_v1/`。
完整操作及物理模型范围见 `docs/robot_v1_guide.md`。

当前已实现机器人基本运动与漂浮，导航和学习式模式切换算法仍未实现。

## 保留的森林河道环境

文件为 `scenes/river_forest_v1.usda`。整体 32×72 米，沿 Y 方向布置弯曲河道，X 横穿两岸，Z 向上。
水面 Z=0 米，河床水深、岸高和两侧干岸坡宽度随 Y 变化。南段 y≈-23 米为浅滩与缓坡，
中央东岸 y≈0 米更陡，北段 y≈21 米另设较缓上下岸区；两处泥地使用较低摩擦参数。

配置文件 `configs/river_forest.json` 记录随机种子、尺寸、分辨率、材质和起终点。
当前为 160 棵树、76 块岩石。树干和岩石有静态碰撞；树冠和芦苇作为显示几何。
地形分辨率为 0.4 米，包含 28,800 个三角形；各类地面有独立物理摩擦材质。
这些几何和摩擦参数是可调整的试验初值，尚未与实地数据标定。

单独的森林环境文件没有机器人，水面不参与碰撞。
新机器人场景通过子层引用它，并由自己的运行程序加入浮力、水阻及推进器模型。

界面中的 `L17 River Task` 面板提供：

- Overview：全场总览；Crossing：南段浅滩；Riverbed：隐藏水面并查看河床。
- Show water surface：切换水面显示。查看河床后可重新勾选并回到总览。
- Mission setup：切换三种起终点定义。绿色标记为起点，红色为终点。

| 任务 | 起点 XY（米） | 终点 XY（米） | 用途 |
| --- | --- | --- | --- |
| Gentle crossing | (-12,-23) | (12,-23) | 在南段缓坡浅滩建立基础跨河验证 |
| Choose entry and exit | (-12,-29) | (12,23) | 后续比较不同上下岸选择与沿岸路线 |
| Steep bank detour | (-12,1) | (12,1) | 后续检验是否避开中央东岸陡坡 |

这些是任务定义和标记，不会自行规划或驱动机器人。
水面显示和任务标记切换存于临时会话层，查看操作不会自动覆盖场景文件。
检验结果和三张实际视口截图保存在 `results/river_forest_v1/`。

`scripts/build_river_forest.py` 是一次性场景构建脚本，已存在同名场景时会拒绝覆盖。
以后运行 `start_here.sh` 打开已保存的文件即可，场景面板会随之加载。

`docs/cd_hssrl_robot_review.md` 记录此前的机器人模型核查；
`configs/robot_prototype_plan.json` 是最初设计草案，当前运行参数以 `configs/robot_v1.json` 为准。

## 保留的入门河岸场景

单位为米，X 方向横穿河道，Y 方向沿河，Z 方向向上。整体为 20×12 米。

| 部分 | X 范围 | 顶面高度 Z |
| --- | --- | --- |
| LeftBank | -10 到 -6 | 0 |
| LeftRamp | -6 到 -2 | 从 0 连续降到 -1.5 |
| Riverbed | -2 到 2 | -1.5 |
| RightRamp | 2 到 6 | 从 -1.5 连续升到 0 |
| RightBank | 6 到 10 | 0 |

两条坡面各长 4 米（水平投影），坡度约 20.56°。五段均为封闭静态三角网格，
具有碰撞和基础摩擦材质，没有贯穿河道的隐藏碰撞地面。摩擦系数只是待标定的初值。
场景包含灯光和 OverviewCamera，暂时没有水面、浮力、水阻、机器人或导航代码。

已在 Isaac Sim 中通过两岸和河床的三次重力落地检查，以及五段地形的射线碰撞高度检查。
检查物体未保存到最终场景。结果在 `results/river_base_verification.json`，
视口截图在 `results/river_base_preview.png`。

`scripts/build_river_scene.py` 是本次替换测试场景的一次性构建脚本，
以后直接运行 `start_here.sh` 即可打开已完成的场景，无需再次构建。

## 以后在界面里新建其他场景

1. 若希望从完全空白开始，选择 File > New。
2. 选择 File > Save As，使用本工程 scenes 目录下的新文件名。
3. 先创建地面、灯光和一个立方体，给立方体添加刚体及碰撞。
4. 按 Play 验证立方体会落到地面，按 Stop 回到初始状态。
5. 保存后，再逐步把地形改成两岸、岸坡和河床。

每个新场景可以保存为独立的 USD 文件，不需要换账号或再安装软件。
如果另存成不同文件名，可用 File > Open 打开它。

## 场景与任务

- `scenes/*.usd`：地形、机器人、碰撞、质量和传感器等。
- `scripts/*.py`：之后实现起终点、控制接口、模式切换、结束条件和数据记录。
- `configs/`：可重复修改的任务参数。
- `results/`：之后实验产生的 CSV、JSON 和视频。
- `.runtime/`：当前工程自己的配置、缓存及日志。

目前已完成河道环境及机器人组件验证，还没有实现导航和学习式模式切换。

## 文件保护

启动器将 L17 之外的持久文件系统设为只读，只有 L17 内可保存和修改。
不修改、不升级、不清缓存、不结束其他人的 Isaac Sim 或实验进程。
NVIDIA 账号不是 Linux 文件权限隔离机制。

## 初始化和检查

初始化自己的空场景（若文件已存在，保留已有文件）：

```bash
python3 /home/ubuntu/L17/amphibious_project/scripts/launch.py prepare
```

检查只读隔离：

```bash
python3 /home/ubuntu/L17/amphibious_project/scripts/launch.py check
```

日志保存在 `.runtime/logs/`。启动新实例或实验前应确认目标显卡仍有可用资源。
