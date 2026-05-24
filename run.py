"""
run.py

Command-line entry point for the Factorio agent.

Usage
-----
Run with a goals file:
    python run.py --goals goals/collect_iron.json

Run with a single inline goal:
    python run.py --goal-type collection \
                  --success "inventory('iron-ore') >= 50" \
                  --failure "tick > 36000" \
                  --description "Collect 50 iron ore"

Run the default smoke-test goal sequence:
    python run.py

Options
-------
--goals PATH        Load goal queue from a JSON file.
--goal-type STR     Goal type for an inline goal (default: collection).
--success STR       Success condition for an inline goal.
--failure STR       Failure condition for an inline goal.
--description STR   Description for an inline goal.
--loop              Loop the goal queue indefinitely (default: run once).
--save-goals PATH   Save the goal queue to a JSON file after the run.
--log-level STR     Logging level: DEBUG, INFO, WARNING (default: INFO).
--host STR          RCON host (default: config.RCON_HOST).
--port INT          RCON port (default: config.RCON_PORT).
--password STR      RCON password (default: config.RCON_PASSWORD).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the Factorio agent against a live game.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--goals", metavar="PATH",
                   help="Load goal queue from a JSON file")
    p.add_argument("--goal-type", default="collection",
                   help="Goal type for an inline goal")
    p.add_argument("--success", metavar="CONDITION",
                   help="Success condition for an inline goal")
    p.add_argument("--failure", metavar="CONDITION",
                   help="Failure condition for an inline goal")
    p.add_argument("--description", default="Agent goal",
                   help="Description for an inline goal")
    p.add_argument("--loop", action="store_true",
                   help="Loop the goal queue indefinitely")
    p.add_argument("--save-goals", metavar="PATH",
                   help="Save goal queue (with outcomes) after the run")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                   help="Logging verbosity")
    p.add_argument("--host", default=config.RCON_HOST)
    p.add_argument("--port", type=int, default=config.RCON_PORT)
    p.add_argument("--password", default=config.RCON_PASSWORD)
    return p

from llm.goal_source import GoalQueue, GoalQueueEntry
def _build_goal_queue(args) -> "GoalQueue":
    if args.goals:
        return GoalQueue.from_file(args.goals, loop_forever=args.loop)

    if args.success:
        # Single inline goal.
        entry = GoalQueueEntry(
            description=args.description,
            success_condition=args.success,
            failure_condition=args.failure or f"tick > {36000}",
            goal_type=args.goal_type,
        )
        return GoalQueue([entry], loop_forever=args.loop)

    # Default smoke-test sequence.
    log.info("run.py: no goal specified — using default smoke-test sequence")
    entries = [
        GoalQueueEntry(
            description="Collect 50 iron ore",
            success_condition="inventory('iron-ore') >= 50",
            failure_condition="tick > 36000",
            goal_type="collection",
        ),
        GoalQueueEntry(
            description="Explore until 10 chunks charted",
            success_condition="charted_chunks >= 10",
            failure_condition="tick > 54000",
            goal_type="exploration",
        ),
    ]
    return GoalQueue(entries, loop_forever=args.loop)


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Build all components ---

    from bridge.rcon_client import RconClient
    from bridge.state_parser import StateParser
    from bridge.action_executor import ActionExecutor
    from bridge.prototype_query import make_prototype_query_fn
    from bridge.world_poller import WorldPoller
    from agent.blackboard import Blackboard
    from agent.subtask import SubtaskLedger
    from agent.self_model import SelfModel
    from agent.memory.behavioral import SQLiteBehavioralMemory
    from agent.network.registry import AgentRegistry
    from agent.network.agents.navigation import NavigationAgent
    from agent.network.agents.mining import MiningAgent
    from agent.network.agents.crafting import CraftingAgent
    from agent.network.coordinator import RuleBasedCoordinator
    from agent.loop import FactorioLoop, LoopConfig
    from planning.reward_evaluator import RewardEvaluator
    from world.knowledge import KnowledgeBase

    log.info("run.py: connecting to Factorio at %s:%d", args.host, args.port)
    client = RconClient(
        host=args.host,
        port=args.port,
        password=args.password,
        timeout_s=config.RCON_TIMEOUT_S,
    )

    parser_inst = StateParser()
    executor = ActionExecutor(client)

    # Knowledge base — persistent across runs, wired to Factorio via RCON.
    kb = KnowledgeBase(data_dir="data/knowledge")
    kb._query_fn = make_prototype_query_fn(client)

    # Agent network
    nav_agent   = NavigationAgent()
    mine_agent  = MiningAgent()
    craft_agent = CraftingAgent()

    registry = AgentRegistry()
    registry.register(nav_agent)
    registry.register(mine_agent)
    registry.register(craft_agent)

    blackboard = Blackboard()
    ledger     = SubtaskLedger()
    sm         = SelfModel()
    mem        = SQLiteBehavioralMemory(db_path="data/behavioral.db")

    coordinator = RuleBasedCoordinator(
        registry=registry,
        blackboard=blackboard,
        ledger=ledger,
        self_model=sm,
        kb=kb,
    )

    goal_queue = _build_goal_queue(args)
    evaluator  = RewardEvaluator()

    poller = WorldPoller(
        client=client,
        local_scan=config.LOCAL_SCAN_RADIUS,
        resource_scan=config.RESOURCE_SCAN_RADIUS,
        item_scan=config.GROUND_ITEM_SCAN_RADIUS,
    )

    loop = FactorioLoop(
        client=client,
        parser=parser_inst,
        poller=poller,
        executor=executor,
        coordinator=coordinator,
        goal_source=goal_queue,
        behavioral_mem=mem,
        self_model=sm,
        evaluator=evaluator,
        cfg=LoopConfig(),
    )

    # --- Run ---
    try:
        stats = loop.run()
    finally:
        if args.save_goals:
            goal_queue.save(args.save_goals)
            log.info("run.py: goal queue saved to %s", args.save_goals)

    sys.exit(0 if stats.goals_failed == 0 and stats.goals_stuck == 0 else 1)


log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()