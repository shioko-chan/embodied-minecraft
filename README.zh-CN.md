# MCSociety

[English](README.md) | [简体中文](README.zh-CN.md)

面向具身 AI 研究的 Minecraft 仿真环境。复用 **CraftGround 2.7.4 + Fabric + Minecraft 1.21 + Java 21**，把真实游戏传感器、动作、任务、轨迹和评测接入同一个 Python 接口。后端与研究模块分离，可替换其他开放世界引擎。

当前处于端到端构建阶段，完整目标尚未完成。没有用替代游戏或生成图片冒充 Minecraft 运行结果。

实施优先级是整合已有开源功能。CraftGround 的原生 RGB、深度与移动已通过实机检查；Mindcraft 原有寻路、合成、放置和箱子存取技能已通过 Python 接口接入，两个 Agent 的通信与 MineCollab 双人合成任务在真实游戏中通过验证。本项目为 LLCL 等外部自主智能体提供游戏环境，不要求在此仓库重新实现世界模型、Actor/Critic 或训练循环。Mindcraft 的完整 LLM Agent 和完整 MineCollab 基准尚未接入；见 [复用决策记录](docs/reuse-plan.md)。

## 安装与运行

要求 Python 3.11–3.13、Java **21**、CMake，以及 OpenGL/GLEW 开发库；Linux 无显示器时需要 Xvfb。首次启动需要下载 Minecraft 和 Gradle 依赖。使用独立实验世界。

```sh
uv sync --extra minecraft
bash scripts/bootstrap_minecraft.sh
source .runtime/env.sh
uv run --extra minecraft mcsociety doctor
uv run --extra minecraft mcsociety rollout --scenario scenarios/exploration_forest.yaml
```

本机是 NixOS，需要开发环境提供动态库搜索路径；详细说明见 [运行时文档](docs/runtime.md)。原生构建产物保存在 `.runtime/`，不修改全局 Minecraft 安装。`doctor` 只检查配置，不代表游戏已通过运行验证。

Python 使用方法：

```python
from mcsociety.worker_backend import SupervisedBackend
from mcsociety.environment import Simulation
from mcsociety.models import Action, WorldEdit
from mcsociety.scenarios import generate_scenario

with Simulation(SupervisedBackend(), record_dir="runs") as env:
    observation, info = env.reset(generate_scenario("exploration", seed=7))
    rgb = observation["rgb"]
    state = observation["state"]
    observation, reward, terminated, truncated, info = env.step(
        Action(forward=True, yaw=30, ticks=5)
    )
    if not (terminated or truncated):
        # 实验者的世界干预单独标注，不能作为 Agent 的生存技能。
        env.intervene(WorldEdit(operation="weather", value="rain"))
```

Gymnasium：

```python
import gymnasium as gym
import mcsociety.gym_env

with gym.make("MCSociety-v0") as env:
    observation, info = env.reset(seed=7)
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`observation` 包含 RGB、可选深度和 11 维本体状态；完整地形/背包/实体状态在 `info["state"]`。`camera` 是 `[pitch_delta, yaw_delta]`，单位为度；重复动作只在第一个 tick 改变相机。`ticks` 为 1–200，终止时提前停止。

## API、任务与数据

```sh
uv run --extra minecraft mcsociety serve --port 8765
uv run --extra minecraft mcsociety generate exploration --seed 7 --biome desert --output /tmp/desert.yaml
uv run --extra minecraft mcsociety inspect-dataset runs
```

API 默认只监听 `127.0.0.1`；文档位于 `http://127.0.0.1:8765/docs`。REST 提供 `/reset`、`/step`、`/world`、`/observation`，WebSocket 位于 `/ws`。接口按完整操作串行执行。Python 客户端在 `mcsociety.client`。

启用共享世界的 Mindcraft 高层技能时，先运行 `bash scripts/bootstrap_mindcraft.sh`，再用 `mcsociety serve --lan-port 55916` 启动服务。重置场景后，`POST /agents` 加入 Agent，`POST /agent-action` 调用 Mindcraft 的寻路、合成、放置、箱子存取或聊天技能；`GET /agents` 读取各 Bot 状态。Python 客户端对应 `start_agents()`、`agent_action()` 与 `observe_agents()`。技能执行期间的每个 CraftGround tick 都记录 `peer_action`；放置结果若在主视角 3×3×3 方块范围内，会等待原生观测确认，并返回 `observed_by_craftground`。

`POST /task` 接受固定 Mindcraft 源码内的 MineCollab 任务文件和任务 ID，例如 `{"source":"multiagent_crafting_tasks.json","task_id":"multiagent_techtree_1_shears"}`。当前仅支持 techtree 合成任务：接口按上游数据初始化各 Bot 背包，世界修改标注为 `assisted_setup`。`GET /task` 调用上游 `Task` 评测器，再用 Minecraft 服务端只计数的背包命令核对目标物品，分别返回 `success` 与 `server_success`；只有完整读到服务器回执时 `server_verified` 才为 `true`。Python 客户端提供 `start_task()` 和 `evaluate_task()`。

LLCL 接入使用 [`LLCLGameAdapter`](src/mcsociety/llcl_adapter.py)：将真实 RGB 转成其既有的 64×64 图像输入，把有限的离散动作映射到 Minecraft 原语，并将任务成功、死亡和时间限制转换为按动作对齐的稀疏奖励与结束标记。评估坐标、目标距离、背包和地图不进入策略传感器；LLCL 现有 `AgentEpisode` 负责持续记忆、Actor/Critic 与训练。接法及当前限制见 [LLCL 接入说明](docs/llcl-integration.md)。

场景 YAML 涵盖生存、探索、建造、社会协作，支持天气、资源、生物数量以及定时结冰事件。森林定义为训练域，沙漠定义为测试域；生成器输出 Minecraft 命令改变实际实验场地。当前内置场景是有限测试场地，不能代表整个开放世界任务分布。

每个 episode 的目录包含元数据、观测和逐 tick 的 JSONL transition。RGB、depth、segmentation 数组以无损 NumPy 文件存储；动作重复展开为多个真实 tick。读取时检查相邻观测、终止边界和传感器形状。深度目前是 **OpenGL 非线性 0–1 缓冲值**，不是米制距离。

```python
from mcsociety.dataset import TrajectoryDataset
from mcsociety.memory import MemoryStore

for sequence in TrajectoryDataset("runs").windows(length=8):
    # sequence 中每项都有 observation、action、next_observation
    pass

with MemoryStore("runs/memory.sqlite") as memory:
    memory.remember_place("explorer", "world-7", "forest", (200, 64, 50), ["wood"])
    memory.add_episode("explorer", "world-7", "found village", tick=120)
    memory.save_skill("walk_forward", "Move forward for five ticks", [{"forward": True, "ticks": 5}])
```

空间和经历记忆按 agent/world 隔离。技能是声明式动作序列，`Controller.run_skill()` 可执行；不会动态执行任意 Python。`Controller.navigate()` 提供闭环朝向控制基线，不保证避障最优。

## 能力边界与验证

| 模块 | 已实现接口 | 尚需完成的目标 |
|---|---|---|
| 世界 | 原生状态转换、地形高度/3×3×3 方块、实体、物品、时刻、世界干预；森林/沙漠真实生物群系实测 | 实测干预回执、天气传感器、任意区域查询 |
| 感知 | 真实 RGB、原生 depth 适配及实机检查、无损多模态数据格式 | 语义分割传感器和标定 |
| 动作 | 移动/转向/跳跃/攻击/使用/快捷栏；Mindcraft 现成寻路/合成/放置/箱子存取/聊天技能的 Python 与 REST 接口、逐帧轨迹；LLCL 离散动作适配 | 建造任务所需的可靠游戏结果回执；规划循环由外部 Agent 提供 |
| 记忆 | SQLite 空间、经历、技能持久化 | 开放世界 Agent 中的长期运行验证 |
| World model | 对齐的 transition、序列窗口、可恢复日志；实机高层技能逐 tick 采集验收 | 多主体同时动作的因果归因与批量数据质量评估 |
| 场景 | 四类模板、种子、域划分、定时事件 | 社会任务真实多 Agent 执行 |
| 多 Agent | 两个 Mindcraft bot 同世界行动；游戏内通信和经箱子交接资源实机验证 | 同步多主体步进与冲突评测 |
| 评测 | 观测驱动的成功率、规划效率、域与适应统计；MineCollab techtree 任务数据、原评测器及服务端物品计数的实机接入 | 完整 MineCollab 与真实游戏批量基准结果 |

评测不会将“请求建造”当作建造成功：必须观测所有目标位置和内部空气。社会任务需要确认的物品转交及通信记录。没有可信最优解时，规划效率为 `null`；未确认规则事件发生时，适应后的指标为 `null`。相同 seed 控制世界与任务生成，不宣称 Minecraft 运行具有逐位确定性。

```sh
uv run pytest -q
uv run ruff check src tests
```

普通测试使用仅存在于测试目录中的后端替身，覆盖 API、Gym、轨迹和评测逻辑；它们不证明 Minecraft 本体可运行。真实游戏测试必须显式启用，见运行时文档。

## 架构与来源

```text
LLM / VLM / RL policy
        ↓
Python Simulation / Gymnasium / REST + WebSocket
        ↓
CraftGround socket IPC (isolated game worker)
        ↓
Fabric client + integrated Minecraft 1.21 server
```

复用上游 socket 协议和同步 tick 实现，隔离进程限制游戏启动和动作等待时间；外部调用仍可使用 WebSocket。`Backend` 协议不依赖 Minecraft 具体类，研究侧轨迹和记忆可由其他引擎接入。

许可证为 GPL-3.0-only。上游依赖、许可证证据和版本选择见 [THIRD_PARTY.md](THIRD_PARTY.md)。
