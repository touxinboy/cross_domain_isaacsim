# SCM 式履土模型：实现与验证

本实现是用于交互仿真的工程原型。保留 48 kg 整车单刚体、原左右履带整体尺寸、原质量惯量；
由 PhysX 以 120 Hz 积分刚体。SCM 在 CPU 上用 NumPy 计算软土作用力，不依赖 PyChrono。
它不是 Chrono SCM 的逐行移植；材料参数尚未经过实物标定。

## 受力分工

- SCM 模式：五块原地面网格的碰撞在 USD session layer 中关闭；变形网格只有显示功能。
- SCM 提供履带—土壤法向与切向力。旧四区牵引模型收到零载荷，不再产生履带力。
- 原机器人、刚性地面 USD 和原动力学文件不改写。切换回刚性模式恢复地面碰撞及旧受力模型。
- Z=-0.18 m 的隐藏刚性底层提供深陷保护。正常验证中 `native_track_normal_n` 应为零；
  若它上升，说明刚性接触已经参与支承，不能再将总支承全部解释成 SCM。
- 机体碰撞仍存在，但首版软土接触只覆盖履带，不计算车腹—软土接触。

## 分布式接触

土壤网格固定在世界 XY 坐标，默认间距 0.02 m。20×6 m 范围包含 1000×300 个单元。
每个物理步只访问履带附近的候选单元，用每单元四个子采样做竖直射线，估计覆盖面积和底部高度。
默认循环履带模式中，原凸包用于定位附近链节，再与当前位姿的履带板和履齿盒体求交。
关闭 `configs/tracks.json` 的 `enabled` 后，SCM 使用原光滑凸包交点。两种模式均考虑车身姿态。
射线前先做保守局部包围盒筛选，并合并凸包的重复共面约束。循环履带细节见 [履带说明](circulating_tracks.md)。

单元面积为 `grid_spacing_m² × 覆盖比例`。压力乘以该面积得到力，力矩按实际交点到质心的力臂累计。
Bekker 中的接触宽度 b 使用物理履带宽度 0.16 m，不使用数值网格边长。

## 法向承载、卸载与车辙

每个单元保存永久沉陷 `plastic`、弹性下压 `elastic`、压力 `pressure`、历史最大压沉 `peak_sinkage`。
初始表面为 h0，履带底部射线交点高度为 z_track，总压沉 z=h0-z_track。

采用以下工程弹塑性近似（单元独立，法向沿世界 Z）：

1. 若 z 不超过已有永久沉陷，无压缩接触，也不产生土壤拉力。
2. 弹性试算压力 p_trial=ke·(z-zp)。
3. 新土加载包络 p_B=(kc/b+kphi)·z^n。
4. 静态压力 p_static=min(p_trial,p_B)，永久沉陷更新为 max(zp,z-p_static/ke)。
5. 弹性下压 ze=p_static/ke；接触压力 p=max(0,p_static-R·v_normal)。
6. 接触脱离后压力和弹性下压清零，永久沉陷保留。

卸载地表 h_rest=h0-zp；受载显示高度 h=h0-zp-ze。
同一强制压沉深度重复加载不会每步无限加深。重新从较高位置落下会增加冲击载荷，因此可能继续产生塑性沉陷。
当前不加入与空转时间直接关联的挖掘项，也没有孔压、排水、土体搬运或完整压实密度模型。

## 剪切与等效带速

光滑轮廓模式中，底部接触相对速度取车辆点速度减去履带相对车体的带速：

`v_rel,t = (v_COM + omega × r)_XY - u_belt × forward_XY`

forward 是车体前向在水平地面的单位投影。默认循环履带模式中，改用
`v_contact = v_COM + omega_body × r + R_body × (v_shoe_center + omega_shoe × r_shoe)`，
包括弧段履带板的转动，不再额外减一次等效带速。下方直线段退化为原带速关系。
等效带速仍采用原 0.15 s 一阶响应，没有电机转矩—转速闭环。
每个土壤单元保存世界 XY 中的接触剪切位移向量 j，连续接触时积分 `j += v_rel,t × dt`。
新建接触或重新进入已脱离的单元时，接触剪切状态重新建立；永久沉陷不清除。

剪切强度上限 `tau_max=c+p·tan(phi)`，剪应力主体采用向量 Janosi 形式：

`tau = -unit(j) × tau_max × (1-exp(-|j|/K))`

为了处理低速保持、反转与数值振荡，j 的长度限制为 5K，另加切向速度阻尼，并将最终剪应力向量限幅到 tau_max。
这是有界剪切弹簧的工程近似，不是经过验证的三维土壤循环剪切本构。
剪切的两个平面分量参与合力和力矩，所以包含前后接触区横向擦滑形成的转向阻力。

## 参数

`configs/scm.json` 独立保存 SCM 参数，`configs/robot_v1.json` 的旧刚性基线参数保持不变。
五个材料名是参数分区，不能理解为已经匹配真实草地、干沙、湿沙、泥土和河床。

| 字段 | 含义 / 单位 |
|---|---|
| kc, kphi, n | Bekker 系数及指数，SI 压力—沉陷关系；系数量纲随 n 改变 |
| cohesion_pa | 剪切黏聚项 / Pa |
| friction_deg | 内摩擦角 / 度 |
| janosi_m | 剪切变形长度 / m |
| elastic_pa_per_m | 弹性卸载刚度 / Pa/m |
| damping_pa_s_per_m | 法向阻尼 / Pa·s/m |
| shear_damping_pa_s_per_m | 切向阻尼 / Pa·s/m |

默认所有材料 ke=600000 Pa/m、法向阻尼 5000 Pa·s/m、切向阻尼 1000 Pa·s/m。
这些值用于当前步长下的机制验证，不能替代压板、加载卸载、剪切和履带土槽标定。

## 显示、复位和保存

- 默认启动 SCM；切换按钮重置车辆与土壤，然后恢复所选模型的碰撞分工。
- `Reset` / 自动穿越：车辆和土壤一起复位。
- `Reset vehicle / keep ruts`：车辆回到起点，保留永久沉陷和历史最大压沉；清除接触弹性和驱动瞬态。
- 暂停保持车辆载荷对应的显示变形；停止并卸载后显示残余车辙。
- 显示网格间距 0.02 m、最多 20 Hz 更新，只更新受扰的材料分区。几何凹槽与扰动变暗均由同一土壤状态产生。
- 导出目录含 `physics.csv`、`summary.json` 与 `soil_state.npz`。NPZ 保存永久沉陷、弹性下压、压力、
  历史压沉、累计剪切路程以及模型配置，便于分析；目前没有界面导入状态功能。

## 观测

SCM 模式中左右载荷、牵引、接触面积来自分布式土壤积分；滑移用法向力加权的接触区速度计算。
`left/right_sinkage_m` 为当前受压土壤单元的最大总压沉，`left/right_rut_depth_m` 为当前接触区的最大永久沉陷。
`soil_rut_depth_max_m` 为全场历史最大永久沉陷；`scm_reentered_rut_cells` 计数本步重新接触已有车辙的单元。
`native_track_normal_n` 单独记录原生刚性履带接触；`scm_step_ms` 记录土壤计算耗时。
RTF 仍包含活动时段的物理、显示和记录开销，排除暂停。

## 验证入口

```bash
/home/touxinboy/miniconda3/envs/isaacsim4.5/bin/python -B -m unittest discover \
  -s /home/touxinboy/HUST/isaacsim/ground/five_surfaces/tests -v

OMNI_KIT_ACCEPT_EULA=YES bash /home/touxinboy/HUST/isaacsim/ground/five_surfaces/start_here.sh \
  --headless --local-assets-only --exec-script scripts/validate_scm.py
```

原生验证包括单刚体与碰撞分工、静置压沉与车重平衡、卸载、重复加载、直行牵引、沿原轨迹再次驶过、
差速转向、五地面穿越、状态导出、基线恢复和原始文件哈希。默认循环履带结果按时间戳放在
`results/tracks_validation/`；关闭循环履带时放在 `results/scm_validation/`。
这些检查证明实现的机制和软件链路，不构成与实物对照的精度验证。

## 当前边界

地面是水平分区的 2.5D 土柱近似，支承法向沿世界 Z，没有完整的车辙侧壁法向力和土体横向流动。
履带不弯曲、没有承重轮悬架自由度，也不计算履齿独立推土、脱轨或张紧。
未标定的参数不能用于真实车辆设计或不同土质间定量优劣比较。

## 2026-10-29 光滑履带基线验证记录（改造前）

23 项单元检查和 38 项 Isaac Sim 原生检查通过。原生报告见 [验证 JSON](scm_validation_20261029.json)，
曲线见 [验证曲线](images/scm_validation_curves_20261029.png)，卸载地表见 [车辙截图](images/scm_ruts_20261029.png)。

- 干沙静置总沉陷约 20.9 mm，残余塑性沉陷约 18.2 mm，支承力 470.88 N，与 48 kg 车重平衡。
- 静置工况使用 828 个接触单元；原地面刚性支承为零。
- 完成直行、原地转向、同轨迹重复通过与五地面自动穿越。
- 五地面仿真推进 39.57 s，活动墙钟耗时 40.58 s，平均 RTF=0.975；接近实时，尚未证明稳定快于实时。
- 单步 SCM 平均耗时 2.66 ms、95 分位 3.12 ms。硬件为 RTX 3070 Ti Laptop / CPU PhysX，启用无窗口渲染。
- 完成车辆复位后跨材料分区车辙重建、顶点颜色长度和原始场景文件哈希检查。

上述速度只适用于本次地图、分辨率与机器，不是大地图、多车或实物精度承诺。
循环履带加入后的验证结果另见 [履带说明](circulating_tracks.md)。
