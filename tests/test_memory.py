import math

import pytest

from mcsociety.memory import MemoryStore


def test_spatial_memory_is_persistent_and_isolated(tmp_path):
    path = tmp_path / "memory.sqlite"
    with MemoryStore(path) as store:
        store.remember_place("alice", "forest", "grove", (3, 4, 0), ["oak"])
        store.remember_place("alice", "forest", "outside sphere", (5, 5, 0), ["birch"])
        store.remember_place("alice", "desert", "grove", (0, 0, 0), ["cactus"])
        store.remember_place("bob", "forest", "grove", (0, 0, 0), ["spruce"])
    with MemoryStore(path) as store:
        places = store.nearby_places("alice", "forest", (0, 0, 0), 5)
        assert [place["name"] for place in places] == ["grove"]
        assert places[0]["distance"] == 5
        assert places[0]["resources"] == ["oak"]
        store.remember_place("alice", "forest", "grove", (1, 0, 0), ["oak", "animals"])
        assert len(store.nearby_places("alice", "forest", (0, 0, 0), 5)) == 1
        assert store.nearby_places("bob", "forest", (0, 0, 0), 0)[0]["resources"] == ["spruce"]


def test_episode_order_limits_and_isolation(tmp_path):
    with MemoryStore(tmp_path / "memory.sqlite") as store:
        first = store.add_episode("alice", "world", "found village", 50, {"location": [1, 2, 3]})
        second = store.add_episode("alice", "world", "traded iron", 50, {"quantity": 2})
        store.add_episode("alice", "world", "spawned", 0)
        store.add_episode("bob", "world", "secret", 500)
        store.add_episode("alice", "other world", "secret", 500)
        assert [row["id"] for row in store.episodes("alice", "world", limit=2)] == [second, first]
        assert store.episodes("alice", "world")[0]["details"] == {"quantity": 2}
        assert store.episodes("missing", "world") == []


def test_skill_recipes_are_json_and_persist_without_execution(tmp_path):
    path = tmp_path / "memory.sqlite"
    marker = tmp_path / "should-not-exist"
    skill_name = "'; DROP TABLE skills;--"
    malicious_looking_string = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    actions = [{"type": "say", "text": malicious_looking_string}]
    with MemoryStore(path) as store:
        store.save_skill(skill_name, "A literal recipe", actions)
    with MemoryStore(path) as store:
        assert store.get_skill(skill_name)["actions"] == actions
        assert len(store.list_skills()) == 1
        assert store.get_skill("missing") is None
        store.save_skill(skill_name, "Updated recipe", [{"type": "jump"}])
        assert store.get_skill(skill_name)["description"] == "Updated recipe"
        assert len(store.list_skills()) == 1
    assert not marker.exists()


@pytest.mark.parametrize("position", [(math.nan, 0, 0), (math.inf, 0, 0), (True, 0, 0), (0, 0)])
def test_invalid_spatial_values_are_rejected(position):
    with MemoryStore(":memory:") as store, pytest.raises(ValueError):
        store.remember_place("a", "w", "place", position, [])


def test_invalid_memory_records_do_not_persist():
    with MemoryStore(":memory:") as store:
        with pytest.raises(ValueError):
            store.add_episode("a", "w", "bad", 0, {"value": math.nan})
        with pytest.raises(ValueError):
            store.add_episode("a", "w", "bad", -1)
        with pytest.raises(ValueError):
            store.save_skill("skill", "bad", [{"value": object()}])
        with pytest.raises(ValueError):
            store.save_skill("skill", "bad", [{1: "ambiguous key"}])
        with pytest.raises(ValueError):
            store.nearby_places("a", "w", (0, 0, 0), -1)
        assert store.episodes("a", "w") == []
        assert store.list_skills() == []
