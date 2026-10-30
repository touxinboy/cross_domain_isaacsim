# 双履带＋双推进器原型 v1

该原型用于两栖导航研究的组件验证。机体约 1 米、总质量 48 kg、两套履带和两个尾部水平推进器。
尺寸、惯量、摩擦和水动力系数都是设计初值，尚未用实物数据标定。

## 文件及打开方式

- `robots/tracked_twin_thruster_v1.usda`：独立机器人资产，含质量、惯量、显示网格和碰撞。
- `scenes/river_robot_v1.usda`：机器人加入自己的森林河道；以 USD 子层引用原环境。
- `scenes/robot_validation_land.usda`、`robot_validation_water.usda`：独立组件测试台。
- `configs/robot_v1.json`：质量、惯量、履带接触、浮力和推进器参数。
- `scripts/robot_runtime.py`：从 PhysX 读取位姿、速度和接触，向 PhysX 施加力与力矩。
- `scripts/amphibian_dynamics.py`：浮力、阻力、牵引力及推力模型。
- `scripts/robot_validation.py`：测试时序、通过标准、CSV 和汇总输出。

仍用 `bash /home/ubuntu/L17/amphibious_project/start_here.sh` 打开自己的实例。
默认打开 `river_robot_v1.usda`，自动加载控制器，初始停在岸上。
单独在其他 Isaac Sim 实例中打开 USD 不会自动安装这些 Python 物理回调。

原 `river_forest_v1.usda` 和 `river_base.usda` 保留。

## 如何操作

右侧面板为 `L17 Amphibious Robot`：

1. `Robot` 查看机器人，`Overview` 看全场，`Riverbed` 隐藏水面查看河床。
2. 任务下拉框选择起终点，`Reset to mission start` 停止仿真并复位到所选起点。
3. `LAND` 启用履带命令；`WATER` 启用推进器命令；`HYBRID` 两套执行器都允许工作。
4. 两个履带滑块单位为 m/s，两个推进器滑块范围 -1 到 1。陆地首次手动试验可把两条履带均设为 0.3，再点 `Play`。
5. `Zero` 下发零指令，执行器仍按响应时滞衰减；水中会继续滑行。`Stop` 停止并回到初始状态。
6. `Run land / float / thrust checks` 自动进入独立测试台，依次完成三项检查并返回河道。
7. `Export current telemetry CSV` 保存当前手动试验的数据。

推进器出水时有效推力为零，因此停在岸上的机器人不会因选择 WATER 而自动进入水中。
目前没有导航策略或学习式切换策略，起终点标记不会自行驱动机器人。
顶部传感器外形是占位结构，目前位姿和速度记录使用仿真真值，尚未接入传感器仿真。

模式只决定执行器是否接受命令。浮力、水阻和地面接触始终按实际浸水和接触状态计算，
不会因为切换模式就强行切换机器人的位置或运动学方程。

## 履带模型的范围

机器人整体是一个有质量和惯量的六自由度刚体，两侧履带有独立凸碰撞包络。
PhysX 负责法向碰撞及刚体运动。接触报告给出的冲量除以时间步，得到每侧接触载荷；
牵引模型按载荷与地面摩擦系数限制驱动力，并计算四个接触斑块的纵向滑移和横向阻力。
左右履带速度不同产生实际偏航力矩。离地时牵引力为零。

履带环、履齿和滚轮是显示结构，没有逐节链条关节，也没有土壤沉陷、履带张紧和结构柔性。
这属于双履带等效接触模型，不能用于验证履带链条机构本身的性能。

## 浮力及推进器

密封浮体采用 0.94×0.54×0.30 米的盒体，用 8×6×4 个体积单元计算浸水体积。
每个单元的浮力向上，合力及其相对重心的力矩一起交给 PhysX。
在河道里，只有真正位于水域、河床以上的浸水部分才产生浮力。

静水参考值：

- 淡水密度 1000 kg/m³，重力 9.81 m/s²；
- 48 kg 对应排水量 0.048 m³、自重 470.88 N；
- 理论吃水约 0.09456 m，机体原点理论高度约为水面上 0.03544 m；
- 上述高度是机体原点，不是河床高度或水深。

水阻使用相对水流速度的线性和二次阻力，并有角阻尼。当前没有附加质量、波浪或 CFD。
履带和外部附件的额外排水体积暂未计入。

两个推进器安装在尾部左右，最大推力设计值每个 45 N，采用 0.20 s 的一阶响应。
浸水比例决定有效推力；左右不同的推力在其真实安装位置产生偏航力矩。
目前未标定电机电流、空化、尾流及桨间相互作用。

## 三项组件检查

测试中只在每个独立试验开始时设置初始位姿，运行阶段不直接改机器人坐标或速度。
位姿和速度来自原生 PhysX Tensor 刚体接口；地面接触来自原生 PhysX 接触报告。

| 检查 | 内容与主要通过标准 |
| --- | --- |
| 陆地 | 等速直行 4 s 前进超过 2 m，横向漂移小于 0.15 m；差速转向超过 40°；制动后速度小于 0.08 m/s |
| 漂浮 | 从有横摇、纵摇的初态落水，15 s 后接近平衡高度；浮力与自重误差小于 5%，无河床支撑 |
| 水中推进 | 双推进器前进超过 2 m、速度超过 0.6 m/s；差动转向超过 35°；关推后速度显著下降，无地面接触 |

另做三个反向检查：履带离地不牵引、推进器出水不推进、关闭浮力后穿过可视水面下沉。
蓝色水面没有碰撞，漂浮不依赖隐藏平台。

当前检查通过表示该原型在所设参数下满足这些组件响应，不代表真实机器人已标定，
也不代表已验证自动上下岸、完整过河或导航/安全切换算法。

## 数据导出

每次检查创建 `results/robot_v1/run_时间/`，包括：

- `land.csv`、`floating.csv`、`propulsion.csv`：120 Hz 物理步记录；
- `water_without_buoyancy.csv`、`dry_thrusters.csv`：反向检查；
- `summary.json`：所有通过标准、实测值、单位、样本数；
- `land.png`、`floating.png`、`propulsion.png`：各测试的实际视口截图。

最近一次通过的汇总另存为 `results/robot_v1/latest_summary.json`。
物理状态包括位置、四元数、速度、姿态、接触载荷；同时记录命令、履带速度、有效推力、
浸水量、浮力、外力和力矩。机械履带功率与理想推进器功率是能量代理量，不是电池实测能耗。

后续算法在 Kit 内调用 `get_robot_runtime().set_commands(mode, tracks, thrusters)` 即可发控制命令。
例如 `mode="HYBRID", tracks=[0.3,0.3], thrusters=[0,0]`。
策略更新频率可以低于 120 Hz 物理步，导航算法和模式切换模型须另行实现。

底层接口参考：[PhysX Tensor API](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html)、
[PhysX Contact Reports](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/extensions/runtime/source/omni.physx/docs/dev_guide/contact_reports.html)。
