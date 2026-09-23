# Third-party foundations

- [CraftGround](https://github.com/yhs0602/CraftGround), PyPI package **2.7.4** and **craftground-runtime-mc121 0.1.0**: reused as the actual renderer, tick controller, Fabric mod and state/action protocol. Runtime source is copied by the bootstrap into ignored `.runtime/minecraft` for a local build; the project does not redistribute Minecraft binaries.
- CraftGround's installed `craftground-2.7.4.dist-info/licenses/LICENSE` contains **GNU GPL version 3**. Its README says LGPL and package classifiers say MIT; this project follows the bundled license text, retains upstream notices in copied runtime sources, and does not relabel it MIT. The root `LICENSE` contains the unmodified GPLv3 license text.
- [Gymnasium](https://gymnasium.farama.org/api/env/) supplies the standard environment interface and checker. NumPy, Pillow, Pydantic, PyYAML, FastAPI and Uvicorn provide array storage, image encoding, schema validation and the API server; resolved versions are in `uv.lock`.
- [Mindcraft](https://github.com/mindcraft-bots/mindcraft), source revision `5f3acc87b479864124173de444f31fa5538f94a6` (MIT): its existing Mineflayer bot initialization, navigation, crafting, placement, chest skills, MineCollab techtree task data and `Task` validator are run from a local source checkout. The checkout is ignored and fetched by `scripts/bootstrap_mindcraft.sh`; the project does not copy its skill or validator implementations.
- [Mineflayer](https://github.com/PrismarineJS/mineflayer) 4.39.0 and Mindcraft's required bot plugins are installed from the npm lockfile in `integrations/mindcraft/`. These drive the published integrated server over the standard Minecraft protocol. The full Mindcraft LLM app and complete MineCollab benchmark have not been installed or claimed as integrated. MineCollab results preserve Mindcraft's Mineflayer-based score and separately report server-side target-item counts when available.
- Minecraft 1.21 is selected because it is the runtime shipped with the pinned published CraftGround package. Upstream `main` advertises 26.2; its source and release distribution differ, so 26.2 is not presented as verified here. The inspected upstream main commit was `18eba01a87a8481bc4fbb54b42fee1428c0c3719`.

Primary technical references:

- [CraftGround configuration](https://github.com/yhs0602/CraftGround/blob/main/docs/configuration/initial_environment.md)
- [CraftGround observations](https://github.com/yhs0602/CraftGround/blob/main/docs/observation_space/index.md)
- [CraftGround actions](https://github.com/yhs0602/CraftGround/blob/main/docs/action_space.md)
- [CraftGround PyPI release](https://pypi.org/project/craftground/2.7.4/)

Source inspection and runtime validation take precedence over examples that describe unreleased functionality. CraftGround's primitive crafting arguments are documented as unsupported; granting an item with `/give` is not an implementation of agent crafting.
