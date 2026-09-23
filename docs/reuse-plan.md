# Open-source integration first

The user reaffirmed on 2026-09-23 that this project should integrate existing
open-source functionality and should not reimplement every missing feature in a
custom Minecraft mod. It is a game environment for LLCL and other external
autonomous agents; LLCL owns the world model, persistent policy state,
Actor/Critic and training loop.

## Current integration

CraftGround 2.7.4 provides the actual Fabric/Minecraft 1.21 runtime, rendered RGB,
depth capture, state protocol, tick control and primitive actions. These systems
are upstream code, not a new engine. This repository adds Python interfaces,
experiment records, task definitions and evaluation, and now exposes existing
Mindcraft skills through Python and REST in that world. The full user objective is not yet met.
CraftGround's native RGB, depth and movement pass the
[real-engine smoke](../scripts/smoke_raw.py) at 320×180. Three source-verified
upstream fixes in [patch_runtime.py](../scripts/patch_runtime.py) correct
framebuffer sizing, read depth before the renderer clears its buffer, and
report the biome from the loaded client world after `/fillbiome`.

## Implementation rule

Before implementing another capability, identify and run the relevant existing
project or plugin. Prefer importing/pinning its public API or running its existing
service over copying and modifying its algorithms. Keep local code limited to
integration, experiment-specific definitions, and verified upstream fixes.
Replace redundant local implementations once their upstream replacements work
end to end; do not retain parallel compatibility paths.

The proposed custom native bridge for weather, voxel queries, receipts and
segmentation is paused. No such bridge is currently integrated. Three small
verified upstream fixes make CraftGround's framebuffer, depth and current biome
observation work; none adds a new sensor.

## Next integration candidates

| Requirement | Reuse target | Validation still required |
|---|---|---|
| Optional LLM-agent baseline, skills, shared-world collaboration | [Mindcraft](https://github.com/mindcraft-bots/mindcraft) | Python/REST skill actions, two-bot communication, and a MineCollab techtree task pass in the shared world; its full LLM lifecycle is not required for LLCL's learning loop |
| Resource-correct crafting, block access, inventory, communication | [Mineflayer](https://github.com/PrismarineJS/mineflayer) | Block queries, crafting and chat work through Mindcraft's integrated Mineflayer bot; resolve observed packet/inventory issues before using bot inventory as benchmark ground truth |
| Navigation and resource collection | [mineflayer-pathfinder](https://github.com/PrismarineJS/mineflayer-pathfinder), [mineflayer-collectblock](https://github.com/PrismarineJS/mineflayer-collectblock) | Navigation works through Mindcraft; resource collection remains to verify in the shared world |
| Construction/cooperation tasks and evaluation | Mindcraft's [MineCollab](https://github.com/mindcraft-bots/mindcraft/blob/develop/minecollab.md) | Upstream techtree task data and `Task` validator are now connected, with server-side count verification for target items; cooking, construction and batch results remain |
| Segmentation | Existing model or simulator sensor integration | Distinguish model-predicted masks from engine ground truth and verify image alignment |

These are candidates under integration review, not claims that their capabilities
are already connected. Selecting a different upstream foundation is acceptable
if it satisfies more of the objective with less custom maintenance. Minecraft
version preference must be weighed against actual end-to-end support rather than
forcing incompatible projects together.

## Foundation comparison (2026-09-23)

| Project | Reusable foundation | Selection limit |
|---|---|---|
| [CraftGround](https://github.com/yhs0602/CraftGround) | Fabric/Minecraft 1.21 simulator, Gym-style step, engine RGB/depth, structured player/world observation, commands for world setup. The installed, pinned version is 2.7.4. Upstream documents 26.2 support starting with 2.7.8, but [PyPI currently publishes 2.7.4](https://pypi.org/project/craftground/). | Its [action API](https://github.com/yhs0602/CraftGround/blob/main/docs/action_space.md) does not implement high-level craft/equip/place/destroy. Synchronized multi-agent control and semantic segmentation remain unverified. |
| [Mindcraft](https://github.com/mindcraft-bots/mindcraft) with [Mineflayer](https://github.com/PrismarineJS/mineflayer) | Existing LLM agent, skills, memory, navigation, crafting, and multi-bot support on Minecraft up to 1.21.11. [MineCollab](https://github.com/mindcraft-bots/mindcraft/blob/develop/minecollab.md) already provides collaborative crafting, cooking, construction tasks and evaluators. | Runs bots against a LAN/dedicated server in real time. It does not document engine-framebuffer RGB/depth or a synchronized Gym step. Its rendered bot view must not be treated as ground-truth game RGB. |
| [MineStudio](https://github.com/CraftJarvis/MineStudio) | Strongest ready-made Python research stack for trajectories, model training, inference and batch benchmarks. | MineRL-derived simulator requires JDK 8; a recent Minecraft version and synchronous multi-agent control were not established by this review. |
| [MineLand](https://github.com/cocacola-lab/MineLand) | Gym-style shared world for up to 48 agents. | Uses Minecraft 1.19, has no game client for native first-person capture, and was archived in June 2026. |

**Decision:** retain CraftGround for game-rendered perception and trajectories;
reuse Mindcraft/Mineflayer for skills and multi-agent tasks. The shared-world
feasibility test passed with Mineflayer 4.39.0: CraftGround published its
integrated world over vanilla `/publish`, two bots joined, one moved, queried a
block, placed a stone that CraftGround observed, and crafted four planks. The
server confirmed that the log was consumed and four planks were present. See
[the probe](../scripts/probe_shared_world.py) and its local report at
`.runtime/evidence/shared_world/report.json`.

The [Mindcraft skill probe](../scripts/probe_mindcraft.mjs) now imports the pinned
Mindcraft source revision `5f3acc87b479864124173de444f31fa5538f94a6` and
calls its existing `initBot`, `goToPosition`, `craftRecipe`, and `placeBlock`
functions. In the same real world, the bot navigated five blocks, crafted four
planks from one log and placed a stone block observed by CraftGround. The server
confirmed that the log was consumed and four planks existed. The dependency
subset is locked in [`integrations/mindcraft/package-lock.json`](../integrations/mindcraft/package-lock.json),
and [bootstrap_mindcraft.sh](../scripts/bootstrap_mindcraft.sh) fetches the pinned
source. See `.runtime/evidence/mindcraft/report.json` after running the probe.

The [Python bridge](../src/mcsociety/mindcraft.py) and
[REST control plane](../src/mcsociety/api.py) now expose those same skills. A
real-game integration test drove two bots, crafted planks, placed a block that
CraftGround subsequently observed, recorded the active skill on each tick, and
confirmed game-chat delivery between bots through REST. The initial observation
now waits for setup and local chunks; a third verified runtime patch reports the
actual client-world biome after `/fillbiome`. Forest and desert were both
verified against their native biome and floor readings.

The bridge now loads a pinned MineCollab techtree task and calls Mindcraft's
original `Task` validator rather than porting its evaluation logic. The upstream
two-agent shears task passed in the real game: each bot started with one ingot,
the bots transferred one through a chest using Mindcraft's `putInChest` and
`takeFromChest` skills, and one crafted shears using `craftRecipe`. Its setup is
logged as privileged intervention. The upstream validator uses Mineflayer's
occasionally stale inventory, so the task API separately queries the server's
count-only inventory command and reports `server_counts`, `server_success`, and
`server_verified`. The two-agent shears result was confirmed server-side in a
real-game test. Mindcraft's `giveToPlayer` failed to deliver the item in this
runtime, so it was removed from the exposed actions.

This proves those open-source skills and one upstream task run together, **not** a synchronized
multi-agent `step` contract or Mindcraft's full Agent/MineCollab runtime. The
skill probe and bridge use a minimal mode adapter because the
full agent scheduler is outside its scope. Two `PartialReadError` messages still
occur during login; the bot's inventory can temporarily report a consumed log
while the server reports none. Use server-authoritative checks for evaluation
until that upstream/protocol issue is resolved. The immediate LLCL connection uses
[the game adapter](../src/mcsociety/llcl_adapter.py), which maps discrete actions
and sparse outcomes to LLCL's existing `AgentEpisode` interface while withholding
evaluator state from policy sensors. Future tasks should extend real-game outcome
evidence and compare frozen LLCL agents, rather than reproduce its training core
or require Mindcraft's full LLM lifecycle. MineCollab's full
benchmark also requires separately distributed server/world data; its task
definitions alone are not a runnable benchmark here.
