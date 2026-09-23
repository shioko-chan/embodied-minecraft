import pytest

from mcsociety.evaluation import TaskEvaluator, aggregate_results
from mcsociety.scenarios import ScheduledEvent, generate_scenario


def observation(**overrides):
    return {
        "agent_id": "explorer",
        "position": [0, 64, 0],
        "health": 20,
        "tick": 0,
        "inventory": [],
        "nearby_blocks": [],
        "messages": [],
        "transfers": [],
        **overrides,
    }


def test_navigation_uses_observed_position_and_reports_unknown_optimum():
    scenario = generate_scenario("exploration", 1)
    evaluator = TaskEvaluator(scenario)
    before = observation()
    result = evaluator.update(
        before, observation(position=scenario.goal["target"]), {"type": "wait"}
    )
    assert result["success"] and result["terminated"]
    assert result["metrics"]["target_distance"] == 0
    assert result["metrics"]["planning_efficiency"] is None
    assert result["metrics"]["path_length"] > 0
    with pytest.raises(RuntimeError, match="ended"):
        evaluator.update(before, before, {})


def test_missing_position_and_requested_move_do_not_prove_navigation():
    scenario = generate_scenario("exploration", 1)
    result = TaskEvaluator(scenario).update(
        observation(), {}, {"type": "navigate", "target": scenario.goal["target"]}
    )
    assert not result["success"]
    assert result["metrics"]["navigation_success"] is None
    assert result["metrics"]["path_length"] is None


def test_certified_oracle_only_controls_planning_metric():
    scenario = generate_scenario("exploration", 1)
    scenario.goal.update(
        oracle_steps=2, oracle_source="exhaustive shortest path over fixed macro action graph"
    )
    evaluator = TaskEvaluator(scenario)
    evaluator.update(observation(), observation(position=[1, 64, 0]), {})
    result = evaluator.update(
        observation(position=[1, 64, 0]), observation(position=scenario.goal["target"]), {}
    )
    assert result["metrics"]["planning_efficiency"] == 1


def test_building_requires_current_coordinate_evidence():
    scenario = generate_scenario("building", 1)
    evaluator = TaskEvaluator(scenario)
    required = scenario.goal["required_blocks"]
    air = [
        {"position": position, "block": "minecraft:air"}
        for position in scenario.goal["required_empty"]
    ]
    initial = evaluator.update(
        observation(),
        observation(inventory=[{"item": "oak_planks", "count": 64}]),
        {"type": "build"},
    )
    assert not initial["success"]
    partial = evaluator.update(observation(), observation(nearby_blocks=required[:-1]), {})
    assert not partial["success"]
    # Seeing only the final block later must not reuse stale earlier evidence.
    stale = evaluator.update(observation(), observation(nearby_blocks=required[-1:]), {})
    assert not stale["success"]
    wrong = [dict(block) for block in required]
    wrong[-1] = {**wrong[-1], "block": "minecraft:air"}
    destroyed = evaluator.update(observation(), observation(goal_blocks=wrong + air), {})
    assert not destroyed["success"]
    assert destroyed["metrics"]["evidence_complete"]
    solid = evaluator.update(
        observation(),
        observation(
            goal_blocks=required + [{**block, "block": "minecraft:oak_planks"} for block in air]
        ),
        {},
    )
    assert not solid["success"]
    complete = evaluator.update(observation(), observation(goal_blocks=required + air), {})
    assert complete["success"]


def test_survival_requires_elapsed_simulation_ticks_and_health():
    scenario = generate_scenario("survival", 1)
    evaluator = TaskEvaluator(scenario)
    assert not evaluator.update(observation(tick=20), observation(tick=419), {})["success"]
    result = evaluator.update(observation(tick=419), observation(tick=420), {})
    assert result["success"]
    assert result["metrics"]["survived_ticks"] == 400
    assert not TaskEvaluator(scenario).update({}, {"tick": 500}, {})["success"]


def test_default_survival_is_achievable_before_primitive_tick_limit():
    scenario = generate_scenario("survival", 7)
    evaluator = TaskEvaluator(scenario)
    for tick in range(1, scenario.goal["duration_ticks"] + 1):
        result = evaluator.update(observation(tick=tick - 1), observation(tick=tick), {})
        assert not result["truncated"]
    assert result["success"] and result["terminated"]


def test_death_and_time_limit_are_distinct():
    scenario = generate_scenario("exploration", 1)
    scenario.max_steps = 1
    timed_out = TaskEvaluator(scenario).update(observation(), observation(), {})
    assert timed_out["truncated"] and not timed_out["terminated"]
    died = TaskEvaluator(scenario).update(observation(), observation(health=0), {})
    assert died["terminated"] and not died["truncated"] and not died["success"]
    flagged_dead = TaskEvaluator(scenario).update(
        observation(), observation(dead=True, health=20, position=scenario.goal["target"]), {}
    )
    assert flagged_dead["terminated"] and not flagged_dead["success"]


def test_social_requires_delivered_message_and_verified_transfer():
    scenario = generate_scenario("social", 1)
    evaluator = TaskEvaluator(scenario)
    action = {"type": "transfer", "recipient": "trader", "item": "bread", "count": 2}
    assert not evaluator.update(observation(), observation(), action)["success"]
    message = {"sender": "explorer", "recipient": "trader", "content": "Here is food."}
    transfer = {
        "id": "delivery-1",
        "sender": "explorer",
        "recipient": "trader",
        "item": "bread",
        "count": 1,
        "success": True,
    }
    partial = evaluator.update(
        observation(), observation(messages=[message], transfers=[transfer]), {}
    )
    assert not partial["success"]
    duplicate = evaluator.update(observation(), observation(transfers=[transfer]), {})
    assert duplicate["metrics"]["transferred_count"] == 1
    failed = evaluator.update(
        observation(),
        observation(transfers=[{**transfer, "id": "delivery-2", "success": False}]),
        {},
    )
    assert not failed["success"]
    complete = evaluator.update(
        observation(), observation(transfers=[{**transfer, "id": "delivery-3"}]), {}
    )
    assert complete["success"]


def test_social_does_not_accept_other_actors_or_unidentified_transfers():
    scenario = generate_scenario("social", 1)
    transfer = {
        "sender": "explorer",
        "recipient": "trader",
        "item": "bread",
        "count": 2,
        "success": True,
    }
    after = observation(
        messages=[{"sender": "other", "recipient": "trader", "content": "x"}], transfers=[transfer]
    )
    result = TaskEvaluator(scenario).update(observation(), after, {})
    assert not result["success"]
    assert not result["metrics"]["message_delivered"]
    assert result["metrics"]["transferred_count"] == 0


def test_adaptation_requires_applied_event_and_before_after_evidence():
    scenario = generate_scenario("survival", 1, adaptation=True)
    scenario.events = [ScheduledEvent(step=2)]
    evaluator = TaskEvaluator(scenario)
    first = evaluator.update(observation(), observation(tick=10), {})
    assert first["metrics"]["adaptation"]["after"] is None
    second = evaluator.update(observation(tick=10), observation(tick=20), {})
    assert second["metrics"]["adaptation"]["difference"] is None
    third = evaluator.update(
        observation(tick=20),
        observation(tick=30, events_applied=[{"kind": "water_freezes", "step": 2}]),
        {},
    )
    assert third["metrics"]["adaptation"]["difference"] == pytest.approx(0)
    assert third["metrics"]["adaptation_event_applied"]


def test_aggregation_reports_coverage_and_forest_desert_generalization():
    train = generate_scenario("exploration", 1)
    test = generate_scenario("exploration", 1, "desert")
    test.max_steps = 1
    success = TaskEvaluator(train).update(
        observation(), observation(position=train.goal["target"]), {}
    )
    failure = TaskEvaluator(test).update(observation(), observation(), {})
    result = aggregate_results([success, failure])
    assert result["navigation"]["success_rate"] == 0.5
    assert result["generalization"]["success_rates"] == {"train": 1, "test": 0}
    assert result["generalization"]["test_minus_train"] == -1
    assert result["planning"]["mean_actual_over_optimal_steps"] is None
    assert result["adaptation"]["mean_difference"] is None
    empty = aggregate_results([])
    assert empty["success_rate"] is None
    assert empty["navigation"]["success_rate"] is None


def test_aggregation_excludes_privileged_assistance_from_all_benchmark_rates():
    scenario = generate_scenario("exploration", 1)
    legitimate = TaskEvaluator(scenario).update(observation(), observation(), {})
    assisted = TaskEvaluator(scenario).update(
        observation(), observation(position=scenario.goal["target"]), {}
    )
    assisted["metrics"]["assisted"] = True
    assisted["metrics"]["planning_efficiency"] = 1.0
    assisted["metrics"]["adaptation"] = {"before": 0.0, "after": 1.0, "difference": 1.0}
    result = aggregate_results([legitimate, assisted])
    assert result["total_episodes"] == 2
    assert result["episodes"] == 1
    assert result["excluded_assisted"] == 1
    assert result["success_rate"] == 0
    assert result["navigation"]["episodes"] == 1
    assert result["navigation"]["success_rate"] == 0
    assert result["planning"]["evaluated"] == 0
    assert result["adaptation"]["evaluated"] == 0
    assert result["generalization"]["success_rates"]["train"] == 0
    only_assisted = aggregate_results([assisted])
    assert only_assisted["episodes"] == 0
    assert only_assisted["excluded_assisted"] == 1
    assert only_assisted["success_rate"] is None
