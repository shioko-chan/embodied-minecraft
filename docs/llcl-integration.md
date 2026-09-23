# 将 Minecraft 接入 LLCL

LLCL 已有独立的 `game_learning.py`：上游 DreamerV3 的世界模型、Actor/Critic、
`Learner` 与每回合持续状态都在 LLCL 内。本仓库只提供真实游戏适配；不复制其模型或
训练算法，也不要求运行 Mindcraft 的 LLM Agent。

## 边界与时序

`LLCLGameAdapter` 接受 `Simulation` 或 `SimulationClient`。首次 `reset()` 返回
`step=0, reward=0`；每次 `step(action_name)` 在真实 Minecraft 中执行一个 tick，
返回动作后的图像、该动作产生的奖励、真正终止或预算截断。终帧只送入
`AgentEpisode.observe()` 一次，不再执行动作。`Simulation` 的轨迹仍按真实 tick 记录。

策略传感器默认仅有真实 RGB 下采样的 `image`（64×64×3、`uint8`）。若 LLCL 已把
公开任务指令编码成固定的一维 `float32` 向量，可传 `task_features`，成为第二个
`task` 传感器。适配器绝不把 `state`、世界坐标、目标距离、背包、地图或评估指标
传给模型；完整评估结果仅供实验侧通过 `last_evaluation` 读取。不要把这个属性
并入 `sensors`。需要合法的本体反馈时，应逐项审查后显式增加，不能直接传完整
`info["state"]` 或 Gym 的现有 11 维向量（其中包含绝对坐标）。

动作名由 `ACTION_NAMES` 固定：等待、前后左右移动、跳跃、左右转向、上下看、
攻击和使用。转向/视角每次改变 15 度。当前都是一个游戏 tick；需要改变持续时间
时，应同步记录实验配置并检查奖励、动作预算和模型训练的时间尺度。

奖励是每个动作 `-0.001`，经任务评测器确认成功时加 `1`，死亡终止时减 `1`；
时间预算耗尽只设置 `truncated=True`。这是给 LLCL 的稀疏训练反馈，**不使用**
环境默认的目标距离进展塑形，因此不会通过奖励持续泄露隐藏目标距离。
`Scenario` 定义目标和最大 tick 数，`TaskEvaluator` 只从动作后的真实状态判断结果；
按下 `use` 或 `attack` 本身不等于任务成功。

## 与 LLCL 现有循环连接

在已配置好 LLCL 的模型、`Learner` 和公开任务向量之后，执行器只需如下调用。
`AgentEpisode` 和 `ObservationSpec` 直接使用 LLCL 中的现有实现：

```python
from llcl_train.game_learning import AgentEpisode, ObservationSpec
from mcsociety.llcl_adapter import ACTION_NAMES, LLCLGameAdapter

game = LLCLGameAdapter(simulation, scenario, task_features=task_features)
observations = {
    "image": ObservationSpec("uint8", (64, 64, 3)),
    "task": ObservationSpec("float32", task_features.shape),
}
# 用 observations 和 ACTION_NAMES 创建 LLCL agent/learner；每次 reset 后创建新 AgentEpisode。
frame = game.reset()
episode = AgentEpisode(agent, ACTION_NAMES, learner=learner)
while True:
    decision = episode.observe(
        frame.sensors,
        step=frame.step,
        reward=frame.reward,
        terminated=frame.terminated,
        truncated=frame.truncated,
    )
    if decision is None:
        break
    frame = game.step(decision["action"])
```

若任务不需要指令向量，则省略 `task_features` 和 `observations["task"]`。调用
`create_agent()` 时，LLCL 的观测声明必须与实际传感器集合完全一致。这个连接
只调用 LLCL 的策略决策，不在本仓库内重写持续记忆、replay、世界模型或优化器。

## 目前能证明什么

CraftGround 的真实 RGB、动作与逐 tick 轨迹，以及部分 Mindcraft 技能和一个
MineCollab 双人任务已有实机验证；此适配器的防泄漏、时序和奖励映射有自动测试。
本机另用真实 Minecraft/Fabric 客户端完成了两次适配器动作：初始观测、前进、
等待之后，服务端任务评测确认为成功，LLCL 帧收到 `step=2`、`reward=0.999`、
`terminated=True`，策略传感器只有图像。还用 LLCL 原有 `AgentEpisode` 接口和
测试策略检查了两步决策及终帧消费时序。尚未运行受训 LLCL 策略在 Minecraft
中的真实训练、冻结评估或迁移对照，不能据此
声称世界模型改善了行为。

现有 `Scenario` 主要是有限场地的生存、到达、建造及协作任务。LLCL 目标要求的
“找到对象—前置交互—进入新区域”等多阶段任务，还需要可靠的游戏结果回执、
任务阶段奖励，以及独立的保留场景。建造和社交成功评测若缺少对应真实传感器，
会保持未确认；当前语义分割和同步多主体 `step` 也未提供。实现这些任务证据时，
应优先复用 CraftGround/Mindcraft 或其他维护中的公开接口，再添加最薄的桥接。
