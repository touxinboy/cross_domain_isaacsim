# 从 Isaac Sim 机器调用 A100 上的 Qwen

更新日期：2026-10-28。已通过 `epic-a100` 登录并找到正在运行的 **LingBot-VLA 2.0**。它使用 **Qwen3-VL-4B** 骨干，提供 WebSocket 动作接口。已验证健康检查和元数据握手，没有发送推理或重置请求。

当前现成动作服务：A100 本机 `ws://127.0.0.1:44560`。它使用 MessagePack 接收图片、机器人状态和指令，返回机器人动作；不能直接填入下面的 `QWEN_BASE_URL` 当作聊天接口。现有服务共用动作缓存和步数计数，不应插入我们的测试请求。

另已找到原始 `Qwen3-VL-4B-Instruct` 权重，目录为 `/workspace/runs/individual/wangjiong/lingbo/models/Qwen3-VL-4B-Instruct`，两份 safetensors 分片共 8,875,719,344 字节。原有 Python 环境的 PyTorch 2.8.0、Transformers 4.57.3 等依赖只读复用。独立服务已部署到自己的 `/home/wangjiong/L17/qwen_service`，真实文字、场景图片及 MSGNav 决策调用已通过；机器人感知和运动闭环仍未接通。

后续建议让独立服务读取这个原始 Qwen3-VL 模型，为 MSGNav 提供图片到语义决策的接口。它与现成的 LingBot 动作服务分别运行，不改动原项目代码或权重。

2026-10-28 这次获准的单卡服务使用 `127.0.0.1:18080` 和模型 ID `l17-qwen3-vl-4b`。修复文字编码格式后，作业号为 1749，时限保持在本次授权的一小时内，约 20:54 自动结束。之后须重新确认资源并启动服务，连接命令不会自行申请 GPU。

## 在当前 Windows 工程中直接调用

`scripts/use_qwen_api.ps1` 会建立必要的本机隧道、在内存中配置密钥并调用客户端；退出时关闭它自己建立的隧道，不会停止服务器模型或别人的连接。例如：

```powershell
& E:/dt/crossdomian/server_workspace/L17/amphibious_project/scripts/use_qwen_api.ps1 -Action text
& E:/dt/crossdomian/server_workspace/L17/amphibious_project/scripts/use_qwen_api.ps1 -Action image -ImagePath E:/dt/crossdomian/analysis/2026-10-28/jackets_v2/verified/results/jacket_targets_v2/both_jackets.png
```

一次 MSGNav 决策用 `-Action decision -ObservationPath 观测JSON路径`。这只验证记录观测上的决策，不启动机器人。下文是手工连接或部署到 Linux 时的通用说明。

## 已经有兼容 API 的情况

在运行导航程序的机器上设置环境变量。下面的地址是占位示例，必须替换为实际地址；8000 也只是常见端口。

```bash
export QWEN_BASE_URL="http://实际A100地址:实际端口/v1"
```

如果服务要求鉴权，用下面的方式在当前终端输入，不把密钥写入命令历史：

```bash
read -rsp 'Qwen API key: ' QWEN_API_KEY; echo
export QWEN_API_KEY
```

先列出模型，不发起推理：

```bash
python3 /home/ubuntu/L17/amphibious_project/scripts/check_qwen_api.py --models
```

把返回的模型 ID 填入：

```bash
export QWEN_MODEL="返回的完整模型ID"
python3 /home/ubuntu/L17/amphibious_project/scripts/check_qwen_api.py --text
```

再发一张自己工程内的照片检查视觉能力：

```bash
python3 /home/ubuntu/L17/amphibious_project/scripts/check_qwen_api.py --image /home/ubuntu/L17/amphibious_project/results/robot_v1/robot_in_river_scene.png --timeout 120
```

模型应描述照片中的机器人、履带和河岸等可见内容。仅返回 HTTP 200 不能证明服务真的支持图像。使用 base64 图片数据，因此 A100 无需访问 Isaac 机器上的图片路径。

## 服务只监听 A100 的 localhost

从运行导航程序的机器建立隧道，保持该终端运行。用户名、服务器地址和端口都用实际值替换：

```bash
ssh -N -L 18000:127.0.0.1:18080 A100的SSH别名
```

另开一个终端，设置：

```bash
export QWEN_BASE_URL="http://127.0.0.1:18000/v1"
```

然后执行相同的 `--models`、`--text`、`--image` 检查。隧道必须从调用模型的机器建立；只在自己的 Windows 电脑建立隧道，不会自动让另一台 Isaac 服务器获得这个接口。

对于准备好的新服务，目标端口使用 `18080`。在当前 Windows 电脑上，经过实际连通验证的专用 SSH 配置是：

```powershell
ssh -F E:/dt/crossdomian/analysis/2026-10-28/qwen_connection/ssh_config -N -o ExitOnForwardFailure=yes -L 127.0.0.1:18000:127.0.0.1:18080 epic-a100
```

该终端需要保持运行。此配置的 Wi-Fi 地址和网卡编号是当次网络的值，换网络后需重新核对。自动调用脚本会按需管理自己的隧道。

如果导航客户端放在 Isaac Linux 服务器上，而它没有 A100 的 SSH 登录条件，可以通过 Windows 中转：先建立上面的 A100 隧道，再从 Windows 开第二个隧道，`ISAAC_HOST` 替换为那台 Isaac 服务器的实际地址或已有 SSH 别名。

```powershell
ssh -N -o ExitOnForwardFailure=yes -R 127.0.0.1:18000:127.0.0.1:18000 ubuntu@ISAAC_HOST
```

此后 Isaac 服务器的客户端地址也是 `http://127.0.0.1:18000/v1`。两条连接均需保持，服务启动后再验证，密钥通过自己的终端配置，不需要贴到聊天中。

## 不知道服务有没有启动

在 A100 机器的终端只读检查：

```bash
ss -lntp
curl --max-time 5 http://127.0.0.1:18080/healthz
```

18080 是自己的接口端口，有效时间取决于 Slurm 作业是否仍在运行。其他服务应结合其启动命令或日志确认。

## 与导航拼接的关系

- MSGNav 的高层视觉语义决策可以通过这个接口请求 Qwen-VL。当前回调 `QwenClient.msgnav_completion` 已按 MSGNav 的文本／图片输入形式准备。
- OmniNav 的专用航点预测需要作者训练的模型及定制航点头。一个普通 Qwen-VL 的聊天 API 不能直接替代它。
- Isaac Sim 的接触、浮力和运动控制仍留在仿真机器；A100 只接收必要的观测并返回推理结果。
- 目前的双履带、双推进器已有基础执行代码，尚需与导航闭环集成；不能把模型给出的坐标直接写到机器人位置来验证运动能力。

参考：[Qwen3-VL 官方模型与推理示例](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)、[LingBot-VLA 2.0 官方动作服务](https://github.com/Robbyant/lingbot-vla-v2/blob/main/deploy/websocket_policy_server.py)。
