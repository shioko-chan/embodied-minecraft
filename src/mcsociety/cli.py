"""Small commands for running and inspecting genuine Minecraft experiments."""

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="mcsociety")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Check dependencies without launching the game")
    serve = sub.add_parser("serve", help="Serve the local Python and WebSocket API")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--record-dir", default="runs")
    serve.add_argument("--lan-port", type=int, default=None,
                       help="Publish the game for Mindcraft peers on this port")
    generate = sub.add_parser("generate", help="Write a seeded scenario YAML")
    generate.add_argument("kind", choices=["exploration", "survival", "building", "social"])
    generate.add_argument("--seed", type=int, default=0)
    generate.add_argument("--biome", choices=["forest", "desert"], default="forest")
    generate.add_argument("--adaptation", action="store_true")
    generate.add_argument("--output", required=True)
    rollout = sub.add_parser(
        "rollout", help="Run a genuine Minecraft episode with a navigation/no-op baseline"
    )
    rollout.add_argument("--scenario", default="scenarios/exploration_forest.yaml")
    rollout.add_argument("--record-dir", default="runs")
    rollout.add_argument("--steps", type=int, default=None, help="Override maximum primitive ticks")
    dataset = sub.add_parser("inspect-dataset")
    dataset.add_argument("path")
    args = parser.parse_args(argv)

    if args.command == "doctor":
        checks = {name: shutil.which(name) for name in ("java", "javac", "cmake", "Xvfb")}
        checks.update(
            craftground=importlib.util.find_spec("craftground") is not None,
            runtime=os.environ.get("MCSOCIETY_MINECRAFT_PATH"),
            display=os.environ.get("DISPLAY"),
            java_home=os.environ.get("JAVA_HOME"),
        )
        print(json.dumps(checks, indent=2))
    elif args.command == "serve":
        import uvicorn

        from .api import create_app
        from .models import EnvironmentConfig

        config = EnvironmentConfig(lan_port=args.lan_port)
        uvicorn.run(
            create_app(config=config, record_dir=args.record_dir),
            host="127.0.0.1", port=args.port,
        )
    elif args.command == "generate":
        from .scenarios import generate_scenario

        scenario = generate_scenario(
            args.kind, args.seed, biome=args.biome, adaptation=args.adaptation
        )
        scenario.to_yaml(args.output)
        print(str(Path(args.output).resolve()))
    elif args.command == "inspect-dataset":
        from .dataset import TrajectoryDataset

        transitions = 0
        episodes = set()
        for item in TrajectoryDataset(args.path):
            transitions += 1
            episodes.add(item["episode_id"])
        print(json.dumps({"episodes": len(episodes), "transitions": transitions}))
    else:
        from PIL import Image

        from .controller import navigation_action, run_episode
        from .environment import Simulation
        from .models import Action
        from .scenarios import Scenario
        from .worker_backend import SupervisedBackend

        scenario = Scenario.from_yaml(args.scenario)
        if args.steps is not None:
            scenario = Scenario.model_validate({**scenario.model_dump(), "max_steps": args.steps})

        def policy(observation):
            if scenario.kind == "exploration":
                return navigation_action(observation["state"], scenario.goal["target"])
            return Action()

        with Simulation(SupervisedBackend(), record_dir=args.record_dir) as simulation:
            metrics = run_episode(simulation, policy, scenario)
            output = Path(args.record_dir)
            output.mkdir(parents=True, exist_ok=True)
            Image.fromarray(simulation.observation["rgb"]).save(output / "last-frame.png")
            (output / "last-result.json").write_text(json.dumps(metrics, indent=2) + "\n")
            print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
