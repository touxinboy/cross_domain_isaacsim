# L17 机器人五地面交互仿真

使用 L17 的 48 kg 双履带、双推进器机器人，建立五块首尾相接的平地。
当前默认使用 SCM 式软土交互，支持分布式接触、滑移、沉陷和卸载后车辙；可在控制面板切回原刚性基线。
模型保留整车单刚体与履带整体轮廓，默认每侧 48 块履带板循环运行，履齿参与 SCM 接触，轮组同步转动。
传动采用简化带速响应，土壤参数尚未经过实物标定。履带细节见 [循环履带说明](docs/circulating_tracks.md)。
实现、参数、碰撞分工与验证见 [SCM 模型说明](docs/scm_model.md)。

沿 +X 方向依次为：

| 编号 | 地面 | X 范围 / m | 静摩擦系数 | 动摩擦系数 |
|---|---|---|---|---|
| 1 | 草地 Grass | -10 到 -6 | 0.95 | 0.75 |
| 2 | 干沙 DrySand | -6 到 -2 | 0.75 | 0.55 |
| 3 | 湿沙 WetSand | -2 到 2 | 0.65 | 0.45 |
| 4 | 泥地 Mud | 2 到 6 | 0.35 | 0.22 |
| 5 | 河床 Riverbed | 6 到 10 | 0.60 | 0.40 |

原刚性基线每块长 4 m、宽 6 m、厚 0.3 m，初始顶面统一为 Z=0，总尺寸 20×6 m。
SCM 模式保留布局，关闭原地面碰撞，接触由土壤模型计算；上表摩擦系数只用于刚性基线。
恢复系数全部为 0。地面之间没有间隙、台阶或重叠，也没有隐藏的公共碰撞地板。
表面白色数字对应表格编号；绿色线是起点，橙红色线是终点。
河床材质在这里作为第五种地面，不包含水体。

机器人初始位置为 (-8.5,0,0.24) m，朝向 +X；重力 9.81 m/s²，物理频率 120 Hz。
刚体求解沿用 CPU PhysX，GPU 0 用于渲染。推进器关闭，浮力和水阻为零。

## 启动可视化仿真（GUI）

在已连接桌面的终端中执行以下命令，打开 Isaac Sim 可视化窗口和控制面板：

```bash
cd /home/touxinboy/HUST/isaacsim/ground/five_surfaces
bash ./start_here.sh --local-assets-only
```

启动脚本直接使用已安装的 `~/miniconda3/envs/isaacsim4.5/bin/python`，无需提前激活环境。
`--local-assets-only` 使用本地资源目录，跳过外部 Nucleus 资源目录查询。可视化运行时不要添加 `--headless`。

也可以先激活已有的 Conda 环境后启动：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaacsim4.5
bash /home/touxinboy/HUST/isaacsim/ground/five_surfaces/start_here.sh --local-assets-only
```

首次启动时按提示确认 NVIDIA 软件许可。场景和控制面板会自动加载，初始处于停止状态，默认 SCM 模式。
加载完成后点击 **Forward → Play** 开始行驶，或点击 **Run across all five surfaces** 自动穿越五块地面。
单独打开原 USD 得到刚性基线；软土交互与车辙需要启动器挂接 Python 运行时。

启动器默认先检测 `bwrap`。若终端报 `setting up uid map: Permission denied`，会自动改用
Linux Landlock（需 ABI ≥ 3），无需 `sudo` 或修改 AppArmor / 系统用户命名空间设置。
日志会显示实际的 `Isolation: bwrap` 或 `Isolation: landlock`。也可显式选择：

```bash
bash /home/touxinboy/HUST/isaacsim/ground/five_surfaces/start_here.sh --isolation landlock
```

两种方式都限制工程外持久文件的创建、删除和内容写入；Landlock 不提供挂载或 IPC 隔离，
保留设备、`/dev/shm` 和受普通用户权限约束的 procfs 写访问，以支持 GPU 与 CUDA 线程命名。
临时文件通过 `TMPDIR` 放在 `.runtime/tmp`，Python 桌面快捷方式写到 `.runtime/applications`。
两种机制均不可用时启动器报错退出，不会无隔离启动。


- **Forward / Reverse / Zero**：设置两条履带的速度，按 **Play** 开始运动。
- 左右履带滑块：手动差速转向。
- **Pause / Reset**：暂停，或将车辆和土壤一起复位到起点。
- **Reset vehicle / keep ruts**：仅复位车辆，保留永久车辙。
- **Switch SCM / rigid baseline**：切换模型并复位，恢复所选模式对应的碰撞与受力。
- **Overview / Robot / Side**：切换总览、机器人跟随及侧面视角。
- **Run across all five surfaces**：复位后，按现有动力学沿直线穿过五块地面并停车。
- **Export telemetry CSV**：导出当前物理状态；自动直行结束后也会导出。

自动直行只是简单的方向保持与停车控制。记录保存在 `results/five_surfaces/`。
`results/validation/` 中保存场景检查、实际穿越记录以及总览截图。

## 滑移、沉陷与 RTF 观测

控制面板的 **LIVE TELEMETRY** 实时显示左右履带滑移率、纵向滑移速度、横向滑移速度 RMS、
几何沉陷深度（mm），以及最近约 1 秒和累计平均 RTF。窗口支持滚动。
每个物理步都把这些量加入内存中的历史记录；点击 **Export telemetry CSV** 或完成自动穿越后，
在相应 `results/five_surfaces/run_*/physics.csv` 保存记录，`summary.json` 保存指标定义和最终值。
SCM 模式额外导出 `soil_state.npz`，保存土壤变形历史与模型参数。
**Reset** 清空当前内存记录、累计滑移距离和 RTF；复位前如需保留手动实验，应先导出。

| CSV 字段（`left_` / `right_`） | 定义 |
|---|---|
| `slip_ratio` | 刚性基线为 `(u_belt - v_patch,x) / max(abs(u_belt), abs(v_patch,x), 0.05)`；循环履带 SCM 使用实际履带板接触点的纵向相对速度，按法向力加权，使用相同分母 |
| `slip_velocity_m_s` / `patch_vx_m_s` | 纵向滑移速度 / 每侧接触区平均纵向速度；包含刚体角速度引起的 `omega × r` |
| `slip_valid` / `ground_contact` | 法向载荷 > 0.5 N 且速度 ≥ 0.05 m/s 时滑移率有效 / 是否接触 |
| `lateral_slip_rms_m_s` | 每侧接触区横向速度的 RMS；SCM 模式按法向力加权 |
| `slip_distance_m` | 接触期间累计的纵向滑移距离；SCM 按局部滑移绝对值加权积分 |
| `sinkage_m` | 相对初始地表的最大下压；刚性模式查顶点，SCM 模式查分布式接触 |
| `ground_clearance_m` / `sinkage_valid` | 有符号离地间隙（正为离地）/ 沉陷是否有效；SCM 有效性按压缩接触 |
| `belt_distance_m` / `belt_phase_m` | 履带累计有符号行程 / 沿闭合路径的相位，SCM 循环履带模式 |
| `sprocket_angle_rad` / `grouser_contact_cells` | 根据行程换算的驱动轮角度 / 当前含履齿接触的网格单元数 |

履带速度使用参与现有受力计算的**滤波履带模型状态**，并非物理链节编码器。
滑移采用沿车体前向的有符号定义，倒车驱动打滑可为负，反向对转可能超过 100%；
静止、低速和离地时滑移率在面板显示 `N/A`，CSV 记为 `nan`，不记为无滑移。
离地时的滑移速度和横向 RMS 仍是运动学诊断值，应结合 `ground_contact` 筛选。

沉陷考虑姿态，参考地表为五块地面的 `top_z_m`；循环履带 SCM 使用移动的履带板和履齿几何，刚性基线使用原凸包轮廓。
初始落地前约 20 mm 的间隙不会计为沉陷。刚性模式中沉陷应接近零，微小数值穿透也会被记录。
SCM 模式中沉陷、载荷和滑移由分布式接触计算结果覆盖，并额外记录永久车辙深度与接触单元数；
永久车辙深度与车辆总沉陷分别记录，不能互相替代。

`rtf_average = rtf_sim_time_s / active_wall_time_s`；`rtf_window` 是最近约 1 秒活动墙钟时间的比值，
另记实际窗口长度 `rtf_window_wall_s`。使用单调时钟，暂停时间和 Play 前场景加载不计入；
恢复播放后重新积累窗口，最初 0.1 秒显示 `warming up` / CSV `nan`。
RTF 包含活动时段中渲染、UI、记录和物理步之间的开销，不能解释为纯物理求解速度；
`RTF >= 1` 表示达到实时，面板暂停时保留最后一次采样值。

## 当前物理范围

`robots/tracked_twin_thruster_v1.usda`、`configs/robot_v1.json` 和原动力学模块从 L17 原样复制，
来源与 SHA-256 记录在 `source_manifest.json`。
原履带模型通过真实接触载荷计算四个等效接触区的牵引力，读取对应地面的原有动摩擦系数。
新增的 `ground_environment.py` 只负责识别五块地面的材质，不改变受力公式。

SCM 模式已加入网格土壤、承载沉陷、剪切历史和残余车辙。原动力学模块原样保留供刚性基线使用；
新模型在 `scm_terrain.py` 中替换软土法向和切向力。没有完整三维土体流动、孔压、排水或实物标定。
循环履带板有履齿、导向齿和销轴外观，配有驱动轮、诱导轮、承重轮及托带轮；履带板沿固定路径运动，
没有独立链节刚体、铰链、悬架、张紧或脱轨动力学。SCM 计算履带板和履齿的压沉与剪切，不含完整履齿侧壁推土。

## 文件

- `scenes/five_surfaces.usda`：可直接在 Isaac Sim 中打开的场景；使用控制面板需运行启动器。
- `configs/five_surfaces.json`：布局与原地面材质参数。
- `scripts/build_scene.py`：使用配置重建场景，运行前先备份自己的场景修改。
- `scripts/ground_tools.py`：SCM／刚性模式切换、手动控制与直行验证。
- `configs/scm.json`：SCM 网格、显示和未标定的材料参数。
- `scripts/scm_terrain.py`：分布式接触、土壤法向与剪切状态、作用力积分。
- `scripts/scm_scene.py`：session layer 碰撞分工和变形网格。
- `configs/tracks.json`：履带块数量、尺寸、履齿、轮组与启用开关。
- `scripts/track_motion.py`：渲染与接触共用的循环路径、履带板位姿与接触点速度。
- `scripts/track_visuals.py`：实例化履带板与旋转轮组，30 Hz 更新显示。
- `scripts/validate_scm.py`：原生压沉、牵引、转向和重复通过验证。
- `scripts/ground_telemetry.py`：滑移、几何沉陷、活动时间 RTF 与显示格式；仅观测，不改变受力。
- `scripts/validate_scene.py`：在 Isaac Sim 内检查五块地面并实际穿越。
- `.runtime/`：本项目的独立缓存、设置与日志。

L17 原项目的场景、机器人与动力学文件保持不变。
