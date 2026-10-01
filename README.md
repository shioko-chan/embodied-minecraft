# MCSociety

[English](README.md) | [简体中文](README.zh-CN.md)

A Minecraft simulation environment for embodied AI research. It reuses **CraftGround 2.7.4 + Fabric + Minecraft 1.21 + Java 21** to bring real game sensors, actions, tasks, trajectories, and evaluation into a single Python interface. The backend is separate from the research modules and can be replaced with another open-world engine.

The project is currently being built end to end, and the full set of goals has not yet been completed. No substitute games or generated images have been presented as Minecraft execution results.

The implementation prioritizes integrating existing open-source capabilities. CraftGround's native RGB, depth, and movement have passed checks in the actual game. Mindcraft's existing pathfinding, crafting, placement, and chest access skills are exposed through the Python interface, and communication between two agents and a two-player MineCollab crafting task have been verified in the real game. This project provides a game environment for external autonomous agents such as LLCL; it does not require reimplementing world models, Actor/Critic, or training loops in this repository. Mindcraft's full LLM agent and the complete MineCollab benchmark have not yet been integrated; see the [reuse decision record](docs/reuse-plan.md).

## Installation and Usage

Requires Python 3.11–3.13, Java **21**, CMake, and OpenGL/GLEW development libraries. Headless Linux environments also need Xvfb. The first launch downloads Minecraft and Gradle dependencies. Use a separate experimental world.

```sh
uv sync --extra minecraft
bash scripts/bootstrap_minecraft.sh
source .runtime/env.sh
uv run --extra minecraft mcsociety doctor
uv run --extra minecraft mcsociety rollout --scenario scenarios/exploration_forest.yaml
```

The local development machine runs NixOS and requires the development environment to provide dynamic library search paths; see the [runtime documentation](docs/runtime.md) for details. Native build artifacts are stored in `.runtime/`, without modifying the global Minecraft installation. `doctor` checks configuration only; it does not mean the game has passed runtime verification.

Python usage:

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
        # Researcher interventions are labeled separately and cannot count as agent survival skills.
        env.intervene(WorldEdit(operation="weather", value="rain"))
```

Gymnasium:

```python
import gymnasium as gym
import mcsociety.gym_env

with gym.make("MCSociety-v0") as env:
    observation, info = env.reset(seed=7)
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

`observation` contains RGB, optional depth, and an 11-dimensional proprioceptive state; full terrain, inventory, and entity state is in `info["state"]`. `camera` is `[pitch_delta, yaw_delta]` in degrees; repeated actions change the camera only on the first tick. `ticks` ranges from 1 to 200, with early stopping on termination.

## API, Tasks, and Data

```sh
uv run --extra minecraft mcsociety serve --port 8765
uv run --extra minecraft mcsociety generate exploration --seed 7 --biome desert --output /tmp/desert.yaml
uv run --extra minecraft mcsociety inspect-dataset runs
```

By default, the API listens only on `127.0.0.1`; documentation is at `http://127.0.0.1:8765/docs`. REST provides `/reset`, `/step`, `/world`, and `/observation`, with WebSocket at `/ws`. The interface serializes complete operations. The Python client is in `mcsociety.client`.

To enable Mindcraft high-level skills in a shared world, first run `bash scripts/bootstrap_mindcraft.sh`, then start the service with `mcsociety serve --lan-port 55916`. After resetting the scenario, use `POST /agents` to add agents, `POST /agent-action` to invoke Mindcraft pathfinding, crafting, placement, chest access, or chat skills, and `GET /agents` to read each bot's state. The corresponding Python client methods are `start_agents()`, `agent_action()`, and `observe_agents()`. Every CraftGround tick during skill execution records `peer_action`. If the placement result is within the main viewpoint's 3×3×3 block region, the operation waits for native observation confirmation and returns `observed_by_craftground`.

`POST /task` accepts a MineCollab task file from the pinned Mindcraft source and a task ID, for example `{"source":"multiagent_crafting_tasks.json","task_id":"multiagent_techtree_1_shears"}`. Currently, only techtree crafting tasks are supported: the interface initializes each bot's inventory from upstream data, and world modifications are labeled `assisted_setup`. `GET /task` calls the upstream `Task` evaluator, then cross-checks target items using count-only inventory commands on the Minecraft server, returning `success` and `server_success` separately. `server_verified` is `true` only when the complete server response has been read. The Python client provides `start_task()` and `evaluate_task()`.

LLCL integration uses [`LLCLGameAdapter`](src/mcsociety/llcl_adapter.py): it converts real RGB into LLCL's existing 64×64 image input, maps a finite set of discrete actions to Minecraft primitives, and converts task success, death, and time limits into sparse rewards and termination flags aligned with actions. Evaluation coordinates, goal distance, inventory, and maps are excluded from policy sensor inputs. LLCL's existing `AgentEpisode` handles persistent memory, Actor/Critic, and training. See the [LLCL integration notes](docs/llcl-integration.md) for setup and current limitations.

Scenario YAML covers survival, exploration, construction, and social cooperation, with support for weather, resources, creature counts, and timed freezing events. Forests are defined as the training domain and deserts as the test domain; the generator emits Minecraft commands to change the actual experimental area. Current built-in scenarios are limited test areas and do not represent the full distribution of open-world tasks.

Each episode directory contains metadata, observations, and per-tick JSONL transitions. RGB, depth, and segmentation arrays are stored as lossless NumPy files; repeated actions are expanded into multiple actual ticks. Reading checks adjacent observations, termination boundaries, and sensor shapes. Depth currently consists of **nonlinear OpenGL buffer values from 0 to 1**, not metric distances.

```python
from mcsociety.dataset import TrajectoryDataset
from mcsociety.memory import MemoryStore

for sequence in TrajectoryDataset("runs").windows(length=8):
    # Each item in sequence contains observation, action, and next_observation.
    pass

with MemoryStore("runs/memory.sqlite") as memory:
    memory.remember_place("explorer", "world-7", "forest", (200, 64, 50), ["wood"])
    memory.add_episode("explorer", "world-7", "found village", tick=120)
    memory.save_skill("walk_forward", "Move forward for five ticks", [{"forward": True, "ticks": 5}])
```

Spatial and episodic memories are isolated by agent and world. Skills are declarative action sequences executed by `Controller.run_skill()`; arbitrary Python is not dynamically executed. `Controller.navigate()` provides a closed-loop heading-control baseline and does not guarantee optimal obstacle avoidance.

## Capabilities, Limitations, and Verification

| Module | Implemented Interfaces | Remaining Goals |
|---|---|---|
| World | Native state conversion; terrain height/3×3×3 blocks, entities, items, time, and world interventions; real-game checks of forest/desert biomes | Real-game checks of intervention acknowledgments, weather sensors, and arbitrary-region queries |
| Perception | Real RGB, native depth adaptation and real-game checks, lossless multimodal data format | Semantic segmentation sensors and calibration |
| Actions | Movement/turning/jumping/attacking/using/hotbar; Python and REST interfaces for Mindcraft's existing pathfinding/crafting/placement/chest access/chat skills, per-frame trajectories; LLCL discrete action adaptation | Reliable game-result acknowledgments required for construction tasks; planning loops are provided by external agents |
| Memory | SQLite persistence for spatial, episodic, and skill memory | Long-running verification in open-world agents |
| World model | Aligned transitions, sequence windows, recoverable logs; real-game acceptance checks for per-tick collection during high-level skills | Causal attribution for simultaneous multi-agent actions and batch data quality evaluation |
| Scenarios | Four template categories, seeds, domain splits, and timed events | Actual multi-agent execution of social tasks |
| Multi-agent | Two Mindcraft bots acting in the same world; in-game communication and resource handoffs through chests verified in the real game | Synchronous multi-agent stepping and conflict evaluation |
| Evaluation | Observation-driven success rates, planning efficiency, domain and adaptation statistics; real-game integration of MineCollab techtree task data, the original evaluator, and server-side item counts | Full MineCollab and batch benchmark results in the real game |

Evaluation does not treat a “build request” as successful construction: all target positions and interior air blocks must be observed. Social tasks require confirmed item transfers and communication records. Without a trustworthy optimal solution, planning efficiency is `null`; when a rule event has not been confirmed to occur, post-adaptation metrics are `null`. The same seed controls world and task generation; no claim is made that Minecraft execution is bitwise deterministic.

```sh
uv run pytest -q
uv run ruff check src tests
```

Regular tests use backend test doubles that exist only in the test directory, covering API, Gym, trajectory, and evaluation logic. They do not prove that Minecraft itself can run. Real-game tests must be explicitly enabled; see the runtime documentation.

## Architecture and Sources

```text
LLM / VLM / RL policy
        ↓
Python Simulation / Gymnasium / REST + WebSocket
        ↓
CraftGround socket IPC (isolated game worker)
        ↓
Fabric client + integrated Minecraft 1.21 server
```

The project reuses the upstream socket protocol and synchronous tick implementation. An isolated process bounds the time spent waiting for game startup and actions; external callers can still use WebSocket. The `Backend` protocol does not depend on Minecraft-specific classes, and the research-side trajectory and memory modules can be connected to other engines.

Licensed under GPL-3.0-only. See [THIRD_PARTY.md](THIRD_PARTY.md) for upstream dependencies, license evidence, and version choices.
