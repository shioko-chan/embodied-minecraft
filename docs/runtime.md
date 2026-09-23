# Minecraft runtime

The runtime is **Minecraft Java 1.21 / Fabric**, provided by the pinned
`craftground==2.7.4` and `craftground-runtime-mc121==0.1.0` packages. It launches a
real Minecraft client and integrated server. The Python-only tests do not prove
that this runtime works; use the opt-in live checks below.

## Install and build

Use Python 3.11–3.13, `uv`, **JDK 21**, CMake 3.28+, a C++ compiler, Git, and
OpenGL/GLEW development libraries. Linux rendering needs an X11 display or Xvfb.
On Debian/Ubuntu, the corresponding native packages include
`openjdk-21-jdk build-essential cmake git libgl1-mesa-dev libegl1-mesa-dev
libglew-dev libpng-dev zlib1g-dev xvfb`. Package installation is a host setup step;
the project bootstrap does not modify system configuration.

```sh
bash scripts/bootstrap_minecraft.sh
source .runtime/env.sh
.venv/bin/mcsociety doctor
```

On NixOS the bootstrap enters a development shell using the configured
`<nixpkgs>` and exports the required native library paths into `.runtime/env.sh`.
This also handles the actual JDK location under `lib/openjdk`. Python package
versions are locked in `uv.lock`; native packages follow the configured nixpkgs.

The bootstrap copies the installed upstream runtime into `.runtime/minecraft`,
applies three source-hash-verified fixes for framebuffer sizing, depth capture
timing and current-biome observation, then compiles Java/Kotlin and the native framebuffer library there. Gradle
downloads, game assets, build outputs and evidence stay under `.runtime`.
The first run needs network access to PyPI, the CPU PyTorch index, Gradle, Maven,
Fabric, Mojang assets and GitHub. Minecraft game assets are fetched from upstream
and are not committed to this repository.

After the Python extra is already installed, `--skip-sync` rebuilds only the
runtime. `JAVA_HOME` may be supplied explicitly. Source `.runtime/env.sh` in each
terminal used to run Minecraft. Use `.venv/bin/python` or `uv run --extra minecraft`;
plain `uv sync` can remove the optional Minecraft dependencies.

## Real engine checks

With an existing display:

```sh
source .runtime/env.sh
.venv/bin/python scripts/smoke_raw.py
```

The smoke check requests native RGB and raw OpenGL depth, sets a flat scenario,
executes movement and asserts that position and center depth change while approaching
a stone wall. It saves the rendered frame, depth array and measurement report
under `.runtime/evidence/raw`. It raises on blank frames, missing depth or lack
of movement instead of replacing observations. This check passed on the current
host at the default 320×180 resolution after the verified upstream fixes.

On a headless Ubuntu installation, prefix the command with
`xvfb-run -a -s '-screen 0 1280x720x24'`. On systems with only the `Xvfb` binary,
start a dedicated Xvfb display and set `DISPLAY` to it before running the command.
Software OpenGL rendering is supported by Mesa but may be slower than a GPU.

The application integration tests are opt-in because they start a real game:

```sh
source .runtime/env.sh
MCSOCIETY_RUN_MINECRAFT=1 .venv/bin/pytest -q -m minecraft tests/test_minecraft_live.py
MCSOCIETY_RUN_MINECRAFT=1 .venv/bin/pytest -q -m minecraft tests/test_mindcraft_live.py
```

The shared-world feasibility probe uses Mindcraft's underlying Mineflayer library.
Install its tested version only into the ignored runtime directory, then run the
real-game probe:

```sh
npm install --prefix .runtime/probe-node --no-audit --no-fund mineflayer@4.39.0
source .runtime/env.sh
.venv/bin/python scripts/probe_shared_world.py
```

The probe opens the integrated Minecraft world on LAN, joins two Mineflayer bots,
moves one, places a block, crafts planks and checks the server's inventory count.
It checks that CraftGround observes the bot-placed block and saves a report under
`.runtime/evidence/shared_world/`. It does not establish synchronized multi-agent
Gym stepping. Mineflayer currently logs packet parsing errors on join and can
temporarily retain a consumed item in its local inventory view; the server count
is used as the authoritative crafting check.

To run Mindcraft's actual high-level skills in the same world, install its pinned
source and a locked subset of its dependencies, then run the second probe:

```sh
bash scripts/bootstrap_mindcraft.sh
source .runtime/env.sh
.venv/bin/python scripts/probe_shared_world.py --mindcraft
```

This calls Mindcraft's `initBot`, `goToPosition`, `craftRecipe` and `placeBlock`
directly; it does not reimplement those algorithms. It passed locally with a
five-block path, four crafted planks, and a placed stone block observed by
CraftGround. The game server confirmed the inventory transaction. The report is
`.runtime/evidence/mindcraft/report.json`. The same two login parse warnings
remain. This focused skill probe does not start the full Mindcraft LLM agent or
the MineCollab benchmark; the latter needs additional server/world data.

The product interface also exposes these skills, including Mindcraft's chest
deposit and withdrawal. Run
`mcsociety serve --lan-port 55916` after the two bootstrap scripts. Then reset a
scenario, start one to eight bots with `POST /agents`, issue `POST /agent-action`,
and inspect `GET /agents`. The Python client and direct `MindcraftBridge` expose
the same calls. CraftGround advances while an action is pending and records each
tick with `peer_action`; a placement in the primary player's local sensor range
waits up to three seconds for the native block observation to catch up. This is
an asynchronous shared world, not a synchronized multi-agent Gym `step`.

For a pinned MineCollab techtree task, submit `POST /task` after starting the
bots, with `source` relative to the upstream `tasks/` directory and `task_id`
from that JSON file. `GET /task` asks the original Mindcraft `Task` validator to
score each bot. The two-agent `multiagent_techtree_1_shears` task passed locally
using a shared chest and Mindcraft's existing `putInChest`, `takeFromChest`, and
`craftRecipe` skills. Task setup uses logged privileged inventory commands.
The validator reads Mineflayer's possibly stale inventory. The task API also
uses vanilla `/clear <player> <item> 0` to count each bot's target item without
removing it, records these inspection ticks, and returns separate
`server_counts`/`server_success`. `server_verified` is true only if all command
replies were parsed; the shears task passed this server-side check in the real
game. This is one verified task, not a complete batch benchmark. The
upstream `giveToPlayer` skill dropped the item without delivering it in our
local test; it is not exposed as a successful transfer path.

## Sensor interpretation

RGB is captured from the game renderer. Depth is the native **nonlinear OpenGL
depth buffer in [0, 1]**, vertically aligned with RGB; it is not a metric distance
map. The adapter labels this encoding explicitly. Upstream's built-in conversion
assumes a projection far plane, so it is disabled. Native semantic segmentation,
weather observation and a synchronized shared-world multi-agent environment are
not present in the pinned upstream package and must not be inferred from passing
single-agent tests.

## Upstream provenance

- [CraftGround source and installation](https://github.com/yhs0602/CraftGround)
- [Published Python package](https://pypi.org/project/craftground/2.7.4/)
- [Minecraft runtime package](https://pypi.org/project/craftground-runtime-mc121/0.1.0/)
- [Observation specification](https://github.com/yhs0602/CraftGround/blob/main/docs/observation_space/index.md)

The upstream LICENSE is GPL-3.0, despite conflicting README/package classifiers.
Preserve its license and notices with any redistributed upstream source or binary.
The three derived source patches are documented by
`.runtime/patches/framebuffer-sizing.diff` and
`.runtime/patches/depth-capture-timing.diff` and
`.runtime/patches/biome-current-world.diff` after bootstrap. The current-biome
fix reads the client world's loaded biome; the upstream nearby-biome scan still
uses the generator prediction, so only `state.biome` has been verified against
world edits. The reset adapter waits for the initialization marker and loaded
nearby chunks before recording observation zero.
