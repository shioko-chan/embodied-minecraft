"""Apply verified sensor fixes to the pinned CraftGround MC 1.21 runtime.

Upstream: craftground-runtime-mc121 0.1.0, EnvironmentInitializer.kt.
The upstream LICENSE and this derived change are GPL-3.0. No Minecraft source is
included here. Unknown source files are rejected rather than patched heuristically.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.metadata
import importlib.resources
from pathlib import Path

FRAMEBUFFER_PATH = Path("src/main/java/com/kyhsgeekcode/minecraftenv/EnvironmentInitializer.kt")
FRAMEBUFFER_SHA256 = "475d46a165d2a56695eb0040c90cb525ef66d71b4fb2215a5a4da0c0268e1a37"
DEPTH_PATH = Path("src/main/java/com/kyhsgeekcode/minecraftenv/mixin/GameRendererDepthCaptureMixin.java")
DEPTH_SHA256 = "4765941b61f5d780ad994fc4fa0001a3ead92762702925402a4423be252dcd09"
BIOME_PATH = Path("src/main/java/com/kyhsgeekcode/minecraftenv/MinecraftEnv.kt")
BIOME_SHA256 = "e96a2853f87196f7c38a74a1d4823764e66d178dfe143cd8291a16503961ac48"
START = "        val windowSizeGetter = (window as WindowSizeAccessor)"
END = "        if (!hasMinimizedWindow)"
REPLACEMENT = """        // Framebuffer pixels per window unit are not GLFW's UI content scale.
        // X11 can report a 2x content scale with a 1x framebuffer pixel ratio.
        val windowWidth = IntArray(1)
        val windowHeight = IntArray(1)
        val framebufferWidth = IntArray(1)
        val framebufferHeight = IntArray(1)
        GLFW.glfwGetWindowSize(window.handle, windowWidth, windowHeight)
        GLFW.glfwGetFramebufferSize(window.handle, framebufferWidth, framebufferHeight)
        val desiredWindowWidth =
            (initialEnvironment.imageSizeX.toDouble() * windowWidth[0] /
                framebufferWidth[0].coerceAtLeast(1)).toInt().coerceAtLeast(1)
        val desiredWindowHeight =
            (initialEnvironment.imageSizeY.toDouble() * windowHeight[0] /
                framebufferHeight[0].coerceAtLeast(1)).toInt().coerceAtLeast(1)
        if (framebufferWidth[0] != initialEnvironment.imageSizeX ||
            framebufferHeight[0] != initialEnvironment.imageSizeY
        ) {
            GLFW.glfwSetWindowSize(window.handle, desiredWindowWidth, desiredWindowHeight)
            client.onResolutionChanged()
        }
"""

DEPTH_METHOD_START = "    private void afterRenderWorld("
DEPTH_METHOD_END = "    @Override"
DEPTH_METHOD = """    private void afterRenderWorld(
            RenderTickCounter tickCounter,
            CallbackInfo ci
    ) {
        if (!FramebufferCapturer.INSTANCE.getShouldCaptureDepth()) {
            return;
        }
        // This callback already runs on the render thread. Deferring it until
        // the next render call reads a cleared depth buffer instead of the world.
        if (!RenderSystem.isOnRenderThread()) {
            throw new IllegalStateException("Call on render thread");
        }
        if (!FramebufferCapturer.INSTANCE.checkGLEW()) {
            throw new RuntimeException("GLEW not initialized");
        }
        MinecraftClient client = MinecraftClient.getInstance();
        Window window = client.getWindow();
        org.lwjgl.opengl.GL.createCapabilities();
        lastDepthBuffer = FramebufferCapturer.INSTANCE.captureDepthImpl(
                client.getFramebuffer().fbo,
                window.getFramebufferWidth(),
                window.getFramebufferHeight(),
                FramebufferCapturer.INSTANCE.getRequiresDepthConversion(),
                0.05f,
                client.options.getViewDistance().getValue() * 4.0f
        );
    }

"""

BIOME_QUERY = """                        val currentPlayerBiome =
                            serverWorld.getGeneratorStoredBiome(
                                BiomeCoords.fromBlock(player.blockPos.x),
                                BiomeCoords.fromBlock(player.blockPos.y),
                                BiomeCoords.fromBlock(player.blockPos.z),
                            )
"""
BIOME_QUERY_FIXED = """                        // Read the loaded client chunk: /fillbiome changes it without
                        // changing the generator's original biome prediction.
                        // Querying ServerWorld.getBiome here can block the render thread.
                        val currentPlayerBiome = world.getBiome(player.blockPos)
"""
BIOME_RESULT = """                        if (biomeCenter != null) {
                            biomeInfo =
                                biomeInfo {
                                    centerX = biomeCenter.x
                                    centerY = biomeCenter.y
                                    centerZ = biomeCenter.z
                                    biomeName = currentPlayerBiome.idAsString
                                }
                        }
"""
BIOME_RESULT_FIXED = """                        biomeInfo =
                            biomeInfo {
                                // A world-edited biome may have no center in the
                                // unchanged generator map. Still report its name.
                                if (biomeCenter != null) {
                                    centerX = biomeCenter.x
                                    centerY = biomeCenter.y
                                    centerZ = biomeCenter.z
                                }
                                biomeName = currentPlayerBiome.idAsString
                            }
"""


def _write_verified_patch(runtime: Path, relative: Path, digest: str,
                          patched: str, diff_name: str) -> None:
    source = importlib.resources.files("craftground_runtime_mc121") / relative
    upstream_bytes = source.read_bytes()
    if hashlib.sha256(upstream_bytes).hexdigest() != digest:
        raise RuntimeError(f"Installed runtime source differs from audited version: {relative}")
    original = upstream_bytes.decode()
    path = runtime / relative
    current = path.read_text()
    if current not in (original, patched):
        raise RuntimeError(f"Refusing to overwrite unknown runtime edits: {path}")
    path.write_text(patched)
    patch_dir = runtime.parent / "patches"
    patch_dir.mkdir(exist_ok=True)
    (patch_dir / diff_name).write_text("".join(difflib.unified_diff(
        original.splitlines(keepends=True), patched.splitlines(keepends=True),
        fromfile=f"upstream/{relative}", tofile=f"mcsociety/{relative}",
    )))


def apply(runtime: Path) -> None:
    if importlib.metadata.version("craftground-runtime-mc121") != "0.1.0":
        raise RuntimeError("Framebuffer sizing patch requires runtime 0.1.0")
    source = importlib.resources.files("craftground_runtime_mc121")
    framebuffer_original = (source / FRAMEBUFFER_PATH).read_text()
    begin = framebuffer_original.index(START)
    end = framebuffer_original.index(END, begin)
    framebuffer_patched = framebuffer_original[:begin] + REPLACEMENT + framebuffer_original[end:]
    framebuffer_patched = framebuffer_patched.replace(
        "import com.kyhsgeekcode.minecraftenv.mixin.WindowSizeAccessor\n", ""
    )
    _write_verified_patch(runtime, FRAMEBUFFER_PATH, FRAMEBUFFER_SHA256,
                          framebuffer_patched, "framebuffer-sizing.diff")

    depth_original = (source / DEPTH_PATH).read_text()
    begin = depth_original.index(DEPTH_METHOD_START)
    end = depth_original.index(DEPTH_METHOD_END, begin)
    depth_patched = depth_original[:begin] + DEPTH_METHOD + depth_original[end:]
    depth_patched = depth_patched.replace(
        "import static com.kyhsgeekcode.minecraftenv.PrintWithTimeKt.printWithTime;\n", ""
    )
    _write_verified_patch(runtime, DEPTH_PATH, DEPTH_SHA256,
                          depth_patched, "depth-capture-timing.diff")

    biome_original = (source / BIOME_PATH).read_text()
    if biome_original.count(BIOME_QUERY) != 1 or biome_original.count(BIOME_RESULT) != 1:
        raise RuntimeError("Audited biome observation source changed")
    biome_patched = biome_original.replace(BIOME_QUERY, BIOME_QUERY_FIXED)
    biome_patched = biome_patched.replace(BIOME_RESULT, BIOME_RESULT_FIXED)
    _write_verified_patch(runtime, BIOME_PATH, BIOME_SHA256,
                          biome_patched, "biome-current-world.diff")
    print("Verified CraftGround framebuffer, depth and current-biome patches")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runtime", type=Path)
    apply(parser.parse_args().runtime)
