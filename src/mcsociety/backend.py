"""Small simulation boundary; policies, memory and datasets do not depend on CraftGround."""

from typing import Protocol

from .models import Action


class Backend(Protocol):
    def reset(self, *, seed: int, commands: list[str], biome: str) -> dict: ...

    def step(
        self, action: Action, *, first_tick: bool = True, commands: list[str] = ()
    ) -> dict: ...

    def close(self) -> None: ...
