"""
agent/network/registry.py

AgentRegistry — maps goal types to the agents responsible for handling them.

The registry is populated at startup by the main loop. When a goal arrives,
the coordinator queries the registry for the relevant agents and activates them.

Rules
-----
- Pure data structure. No LLM calls. No game interaction.
- An agent may be registered for multiple goal types.
- An unknown goal type returns an empty list (not an error) — the coordinator
  handles the missing-agent case by escalating to STUCK.
- Registration is append-order-stable; agents_for_goal preserves insertion order.
"""

from __future__ import annotations

from agent.network.agent_protocol import AgentProtocol


class AgentRegistry:
    """
    Registry mapping goal type strings to AgentProtocol instances.

    Goal types are arbitrary strings matching the 'type' field conventions
    used by the LLM layer when producing Goal objects (e.g. "production",
    "exploration", "research", "construction").

    Usage
    -----
        registry = AgentRegistry()
        registry.register(navigation_agent, ["exploration", "construction"])
        registry.register(production_agent, ["production"])

        agents = registry.agents_for_goal("production")
        # -> [production_agent]
    """

    def __init__(self) -> None:
        # goal_type -> ordered list of agents (preserves registration order)
        self._by_goal_type: dict[str, list[AgentProtocol]] = {}
        # all agents (deduped, preserves first-registration order)
        self._all: list[AgentProtocol] = []
        self._all_set: set[int] = set()   # id() of registered agents

    def register(
        self,
        agent: AgentProtocol,
        goal_types: list[str],
    ) -> None:
        """
        Register an agent for one or more goal types.

        If the agent has already been registered (by object identity) it is
        not added to all_agents() again, but it will be added to the bucket
        for any new goal types.

        Parameters
        ----------
        agent      : Agent instance satisfying AgentProtocol.
        goal_types : Non-empty list of goal type strings this agent handles.

        Raises ValueError if goal_types is empty.
        """
        if not goal_types:
            raise ValueError("goal_types must be non-empty")

        # Track agent in all_agents list (dedup by identity)
        if id(agent) not in self._all_set:
            self._all.append(agent)
            self._all_set.add(id(agent))

        for goal_type in goal_types:
            bucket = self._by_goal_type.setdefault(goal_type, [])
            # Dedup within bucket by identity
            if not any(a is agent for a in bucket):
                bucket.append(agent)

    def agents_for_goal(self, goal_type: str) -> list[AgentProtocol]:
        """
        Return the agents registered for the given goal type.

        Returns an empty list (not an error) if no agent is registered for
        the type. The coordinator handles this by returning STUCK.

        Returns a shallow copy — modifying the result does not affect the
        registry.
        """
        return list(self._by_goal_type.get(goal_type, []))

    def all_agents(self) -> list[AgentProtocol]:
        """
        Return all registered agents in first-registration order.

        Returns a shallow copy.
        """
        return list(self._all)

    def registered_goal_types(self) -> list[str]:
        """Return all goal types that have at least one registered agent."""
        return list(self._by_goal_type.keys())

    def __repr__(self) -> str:
        type_count = len(self._by_goal_type)
        agent_count = len(self._all)
        return f"AgentRegistry({agent_count} agents, {type_count} goal types)"
