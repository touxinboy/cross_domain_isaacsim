# L17 机器人五地面交互仿真

使用 L17 的 48 kg 双履带、双推进器机器人，建立五块首尾相接的平地。
当前默认使用 SCM 式软土交互，支持分布式接触、滑移、沉陷和卸载后车辙；可在控制面板切回原刚性基线。
模型保留整车单刚体与履带整体轮廓，默认每侧 48 块履带板循环运行，履齿参与 SCM 接触，轮组同步转动。
传动采用简化带速响应，土壤参数尚未经过实物标定。履带细节见 [循环履带说明](docs/circulating_tracks.md)。
实现、参数、碰撞分工与验证见 [SCM 模型说明](docs/scm_model.md)。
完整的结构、坐标定义、运动学与履土力公式、参数表、数值算例及参考来源见
[履带结构与履土交互动力学详解](docs/track_soil_dynamics.md)，含 7 组可复现的尺寸图和受力示意图。
直线穿越、原地自转与绕点转动的实际曲线、数值比较、瞬态和误读说明见
[三种实验折线图解读](docs/experiment_curves_interpretation.md)，含 8 张原始图和可复算的 CSV 统计。

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
- **Export telemetry CSV + plots**：导出当前物理状态和折线图；自动直行结束后也会导出。

自动直行只是简单的方向保持与停车控制。记录保存在 `results/five_surfaces/`。
`results/validation/` 中保存场景检查、实际穿越记录以及总览截图。

## 直线穿越的折线图输出

点击 **Run across all five surfaces**，车辆完成穿越并停车后，自动保存 CSV、土壤状态和下列图表。
安全停止时也保留已采集的数据，图表标题注明实际结束原因。
绘图在后台进行，控制面板显示生成状态和结果目录；CSV 先保存，绘图失败不会丢失物理记录。

| `results/five_surfaces/run_*/plots/` 内的文件 | 内容 |
|---|---|
| `straight_timeseries.png/.svg` | 全程左右履带的有符号滑移率、总沉陷、平均压力、平均剪切应力 |
| `straight_surface_comparison.png/.svg` | 按质心进入地形的时刻对齐，叠加五段的滑移 RMS、最大总沉陷及应力均值 |
| `straight_stress_peaks.png/.svg` | 全程法向压力、剪切应力的接触面积加权均值与单元峰值 |
| `straight_index.json` | 图表清单及各地形在 CSV 中对应的样本范围和时间区间 |

全程图的背景色和竖线标出**车体质心**所在的地形与切换时刻；跨界时，分布式履带接触可能同时涉及两种材料。
分段对比来自一次连续行驶，各段进入状态和长度不同，不能视为五次相同初始条件的独立试验。
曲线使用原始物理样本，不作平滑；无效滑移或没有记录的应力留空，不填零。
刚性基线没有 SCM 应力记录，相应图中标记为不可用。

也可以不加载视口，直接执行同一车辆运行时和直行控制器，自动穿越、绘图并退出：

```bash
cd /home/touxinboy/HUST/isaacsim/ground/five_surfaces
bash ./run_straight.sh --local-assets-only
```

如需观看自动穿越，使用 `bash ./start_here.sh --local-assets-only --exec-script scripts/run_straight.py`。
对已有结果重新绘图：

```bash
python3 scripts/plot_straight.py results/five_surfaces/run_实际时间戳
```

2026-11-01 验证：完整应用的直行结束自动导出和无视口批处理均穿过全部五种地形并停车，
各记录 4730 个物理样本、生成 6 个图表文件；选定的 11 项物理字段完全一致。
验证详情见 [直行绘图验证记录](docs/straight_validation_20261101.json)。

## 五种地面转圈与曲线对比

控制面板新增 **Turning experiments / comparison plots**：

- **Spin on current surface**：在当前材质区域中心原地逆时针转一圈，控制器尽量维持质心位置。
- **Orbit on current surface**：绕当前材质区域中心转一圈，目标半径 0.85 m、前进速度 0.25 m/s。
- **Compare both turns on all five surfaces**：依次执行五地面原地转圈和五地面绕点转圈，共 10 次试验。
- **Stop turning experiment**：中止试验并保留已经采集的数据；手动驾驶、复位或切换模式也会中止。

试验自动切到 SCM，每次清空车辙和剪切历史，落地静置 2 s 后开始转圈。
原地转圈按实际车身累计偏航角判断完成；绕点转圈按车辆相对指定固定圆心的实际极角判断完成。
控制器只设置左右带速，不强制车辆位置、轨迹或转速。原地漂移和绕点半径误差会被记录。
配置位于 `configs/turning.json`；默认各转一圈，单次仿真最多 120 s。
越过材质边界的保护范围、姿态过大或触及隐藏刚性底层会提前停止，并记录原因。

也可直接运行完整批处理，完成后自动关闭 Isaac Sim：

```bash
cd /home/touxinboy/HUST/isaacsim/ground/five_surfaces
bash ./run_turning.sh --local-assets-only
```

这个批处理使用精简 Kit 应用，仅加载原生 PhysX 和 USD，不加载渲染视口；
车辆运行时、循环履带接触、土壤公式和转圈控制器与 GUI 共用。
它记录的 RTF 不包含 GUI 渲染与地形网格显示开销，不能当作可视化实时性能。
如需观看整组试验，执行 `bash ./start_here.sh --local-assets-only --exec-script scripts/run_turning.py`。
结果位于 `results/turning/run_*/`：

| 文件 | 内容 |
|---|---|
| `experiments.json` | 配置快照、固定圆心、各次完成状态、实际转角、指标定义与图表清单 |
| `Grass_spin.csv` 等 | 各次物理步的左右滑移、总沉陷、压力、剪切应力、实际轨迹及控制记录 |
| `Grass_spin_soil.npz` 等 | 各次结束后的土壤变形历史 |
| `plots/spin_comparison.png/.svg` | 五地面原地转圈的四项指标对比折线图 |
| `plots/orbit_comparison.png/.svg` | 五地面绕点转圈的四项指标对比折线图 |
| `plots/trajectories.png/.svg` | 原地漂移和绕指定圆心的实际轨迹 |
| `plots/Grass_spin.png/.svg` 等 | 单次试验左右履带曲线，包括压力及剪切应力均值、单元峰值 |

对比图的滑移采用左右有效滑移率的**法向载荷加权 RMS**，避免原地对转时正负抵消；
单次图及 CSV 保留左右有符号滑移率。总沉陷取两侧当前最大值，不等于永久车辙。
压力均值为 `sum(p_i*A_i)/sum(A_i)`，剪切应力均值为
`sum(norm(tau_i)*A_i)/sum(A_i)`，使用最终阻尼和限幅后的实际应力。
后者不是剪切向量平均值的模，因此左右方向相反时不会抵消。
没有接触时应力记为 NaN；滑移率无效时留断点，不填零。曲线使用原始样本、不作平滑。
比较时间从转圈指令开始计，排除落地静置阶段；未完成的曲线在图例和报告中注明原因。

图表自动使用 Matplotlib 生成 PNG 和 SVG；如果 Isaac Sim 环境缺少 Matplotlib，CSV 仍会保存，
并在报告中记录绘图错误。可在装有 NumPy、Matplotlib 的 Python 环境重新生成：

```bash
python3 scripts/plot_turning.py results/turning/run_实际时间戳
```

2026-11-01 原生验证：完整应用与无视口批处理均完成 10/10 次试验，
共用的 24638 个物理样本在位置、偏航、滑移、沉陷、压力和剪切应力等选定字段上完全一致；
每组生成 13 张图，分别保存 PNG 与 SVG。原地最大质心漂移约 7.3 mm，
绕点最大半径误差约 5.9 cm。详细数值见 [转圈验证记录](docs/turning_validation_20261101.json)。
两次验证同时运行，记录的 RTF 不作为单独运行的性能基准；这些检查也不等同于实物标定。

## 滑移、沉陷与 RTF 观测

控制面板的 **LIVE TELEMETRY** 实时显示左右履带滑移率、纵向滑移速度、横向滑移速度 RMS、
几何沉陷深度（mm），以及最近约 1 秒和累计平均 RTF。窗口支持滚动。
每个物理步都把这些量加入内存中的历史记录；点击 **Export telemetry CSV + plots** 或完成自动穿越后，
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
| `pressure_mean_pa` / `pressure_max_pa` | 每侧压缩接触面积加权平均法向压力 / 接触单元峰值，Pa，仅 SCM |
| `shear_stress_mean_pa` / `shear_stress_max_pa` | 每侧最终剪切应力模的面积加权均值 / 单元峰值，Pa，仅 SCM |
| `shear_longitudinal_mean_pa` / `shear_lateral_mean_pa` | 每侧剪切应力沿车身前向、横向的有符号面积均值，Pa，仅 SCM |

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
- `scripts/ground_runtime.py`：GUI 与无视口批处理共用的原生 PhysX 车辆运行时。
- `configs/scm.json`：SCM 网格、显示和未标定的材料参数。
- `scripts/scm_terrain.py`：分布式接触、土壤法向与剪切状态、作用力积分。
- `scripts/scm_scene.py`：session layer 碰撞分工和变形网格。
- `configs/tracks.json`：履带块数量、尺寸、履齿、轮组与启用开关。
- `scripts/track_motion.py`：渲染与接触共用的循环路径、履带板位姿与接触点速度。
- `scripts/track_visuals.py`：实例化履带板与旋转轮组，30 Hz 更新显示。
- `scripts/validate_scm.py`：原生压沉、牵引、转向和重复通过验证。
- `configs/turning.json`：原地与绕点转圈的目标、控制和终止条件。
- `scripts/turning_experiments.py`：共享转圈控制器与指标定义，不依赖 Kit。
- `scripts/run_turning.py`：原生 PhysX 五地面转圈批处理入口。
- `run_turning.sh` / `configs/turning_physics.kit`：使用相同物理模型的无视口批处理启动器。
- `scripts/plot_turning.py`：从实测 CSV 重建对比折线图和轨迹图。
- `run_straight.sh` / `scripts/run_straight.py`：无视口原生直线穿越与自动绘图。
- `scripts/ground_export.py`：先保存物理记录，再异步生成图表并记录状态。
- `scripts/plot_straight.py`：直线穿越全程曲线、地形分段对比、压力和剪切应力峰值。
- `scripts/ground_telemetry.py`：滑移、几何沉陷、活动时间 RTF 与显示格式；仅观测，不改变受力。
- `scripts/validate_scene.py`：在 Isaac Sim 内检查五块地面并实际穿越。
- `.runtime/`：本项目的独立缓存、设置与日志。

L17 原项目的场景、机器人与动力学文件保持不变。
