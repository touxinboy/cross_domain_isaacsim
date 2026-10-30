# Isaac Sim 水陆机器人与履土交互工程

当前工作区包含两个独立工程：`amphibious_project` 保存水陆机器人、森林河道与导航集成；
`ground/five_surfaces` 用于五种地面上的 SCM 履土交互仿真。
两者有各自的配置、场景、脚本、测试、启动入口和运行记录。

## 目录结构

```text
isaacsim/
├── README.md                         # 工作区结构与版本管理说明
├── .gitignore                        # 两个工程共用的忽略规则
├── ground/
│   ├── IsaacSim_地面与轮土交互仿真调研.md
│   ├── .backups/                     # 修改前的本地备份，不提交
│   └── five_surfaces/                # SCM 与循环履带工程
│       ├── start_here.sh             # 可视化仿真启动入口
│       ├── environment.yml           # Isaac Sim 环境依赖声明
│       ├── configs/                  # 地面布局、机器人、SCM 与履带参数
│       ├── scenes/                   # 五地面 USD 场景
│       ├── robots/                   # 引用的机器人 USD 资产
│       ├── scripts/                  # 启动、运行时、接触、显示与验证
│       ├── tests/                    # 脱离图形界面的基础测试
│       ├── docs/                     # 模型说明、保留的验证报告与截图
│       ├── source_manifest.json      # 原始模块来源与文件哈希
│       ├── results/                  # 自动生成的实验结果，不提交
│       └── .runtime/                 # 缓存、日志与临时状态，不提交
└── amphibious_project/               # 水陆机器人、河道与导航工程
    ├── start_here.sh                 # 原服务器启动入口，尚含旧路径
    ├── configs/                      # 机器人、河道、跨河及导航配置
    ├── scenes/                       # 河道、机器人与目标场景
    ├── robots/                       # 机器人 USD 资产
    ├── assets/targets/               # 导航目标资产
    ├── scripts/                      # 动力学、跨河、RGB-D 与导航接口
    ├── tests/                        # 动力学、导航与部署接口测试
    ├── deploy/                       # OmniNav 与 Qwen 部署脚本和依赖
    ├── third_party/                  # MSGNav、OmniNav 与定制 Transformers 源码
    ├── docs/                         # 操作、模型与部署说明
    ├── project_manifest.json         # 工程清单
    ├── PACKAGED_FILES.json           # 打包文件清单
    ├── checkpoints/                  # 下载的模型权重，不提交
    ├── .venvs/                       # 本地模型虚拟环境，不提交
    ├── results/                      # 实验数据与采集图像，不提交
    └── .runtime/                     # 运行缓存、服务状态与日志，不提交
```

## 五地面工程的运行关系

`start_here.sh → scripts/launch.py → scripts/open_stage.py → GroundTools / GroundRuntime`。
启动器打开 `scenes/five_surfaces.usda`，加载配置并挂接物理回调和控制面板。

| 模块 | 用途 |
| --- | --- |
| `ground_tools.py` | 模式切换、控制、复位、自动穿越与导出 |
| `robot_runtime.py` | 单车刚体状态读取、物理回调与力的施加 |
| `track_motion.py` / `track_visuals.py` | 循环履带板共用位姿、接触点速度与轮组显示 |
| `scm_terrain.py` | 分布式履带接触、土壤承载、剪切与永久沉陷 |
| `scm_scene.py` | SCM 模式的碰撞分工与变形地面显示 |
| `ground_telemetry.py` | 滑移、沉陷和 RTF 指标及面板格式 |
| `amphibian_dynamics.py` | 保留的原动力学模块，供刚性基线和驱动响应使用 |

在已连接桌面的终端中启动：

```bash
cd /home/touxinboy/HUST/isaacsim/ground/five_surfaces
bash ./start_here.sh --local-assets-only
```

加载后点击 **Forward → Play**，或 **Run across all five surfaces**。
详细操作见 [五地面 README](ground/five_surfaces/README.md)。
循环履带改造的基础测试已通过；此前原生验证停在场景加载阶段，新履带的实际画面、稳定性及 RTF 尚待完成验证。

## 水陆与导航工程

`amphibious_project` 包含机器人运动、浮力、水阻、河道环境、跨河控制，以及 RGB-D、语义目标和航点导航接口。
`deploy/` 负责模型环境与服务部署，`third_party/` 提供已接入的原始函数与定制 Transformers，
`configs/` 中的来源、提交号、权重锁定文件用于复现，不能用下载的权重替代这些文件。

此工程的 `scripts/launch.py` 仍硬编码原服务器的 `/home/ubuntu/sdc/L17` 和
`/home/ubuntu/qsq/isaacsim`，其 README 也保留旧服务器入口。迁移到当前机器后需要适配这些路径才能使用该启动器。
功能范围与历史验证见 [水陆工程说明](amphibious_project/README.md) 和该工程的 `docs/`。

## Git 管理范围

截至 2026-10-30，工作区根目录尚未初始化 Git 仓库；根目录 `.gitignore` 已按现有结构配置。

- 提交源码、配置、USD 场景与资产、测试、部署脚本、依赖声明、来源清单和文档。
- 保留 `third_party/` 中项目实际依赖的源码，特别是 OmniNav 的定制 Transformers。
- 忽略 `.runtime/`、`.backups/`、`results/`、`checkpoints/`、`.venvs/`、模型权重、Python 缓存、临时文件和本地环境文件。
- `results/` 中的原始实验输出不提交；需要随版本保留的报告、曲线或截图放入 `docs/`。
- 保留 `.env.example`、`.env.template` 和 `*.lock.json`；根目录 `vln/` 仍按先前约定整体忽略。

忽略规则只影响 Git 文件选择，不会删除本地环境、模型、备份或实验数据。
