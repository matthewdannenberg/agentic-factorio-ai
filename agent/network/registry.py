"""
agent/network/registry.py

AgentRegistry — a simple ordered registry of AgentProtocol instances.

Agents are registered directly without goal-type annotations. The coordinator
selects the appropriate agent for each subtask using the agent_hint mechanism
(a string matching the agent's AGENT_ID class attribute). This keeps routing
logic in the coordinator where it belongs and keeps the registry free of any
knowledge about goals or subtask types.

Rules
-----
- Pure data structure. No LLM calls. No game interaction.
- Registration is append-order-stable; all_agents() preserves insertion order.
- Duplicate registration (by object identity) is silently ignored.
"""

from __future__ import annotations

from agent.network.agent_protocol import AgentProtocol


class AgentRegistry:
    """
    Registry of AgentProtocol instances available to the coordinator.

    Agents are registered once at startup. The coordinator calls all_agents()
    to get the full list and selects the appropriate one based on the active
    subtask's agent_hint and each agent's AGENT_ID class attribute.

    Usage
    -----
        registry = AgentRegistry()
        registry.register(navigation_agent)
        registry.register(mining_agent)

        # Coordinator selects from registry.all_agents() using agent_hint.
    """

    def __init__(self) -> None:
        self._agents: list[AgentProtocol] = []
        self._agent_ids: set[int] = set()   # id() of registered agents

    def register(self, agent: AgentProtocol) -> None:
        """
        Register an agent.

        Duplicate registration by object identity is silently ignored.
        Preserves insertion order.
        """
        if id(agent) not in self._agent_ids:
            self._agents.append(agent)
            self._agent_ids.add(id(agent))

    def all_agents(self) -> list[AgentProtocol]:
        """
        Return all registered agents in insertion order.

        Returns a shallow copy — modifying the result does not affect the
        registry.
        """
        return list(self._agents)

    def agent_by_id(self, agent_id: str) -> AgentProtocol | None:
        """
        Return the first registered agent whose AGENT_ID class attribute
        matches *agent_id*, or None if not found.

        This is the primary lookup used by the coordinator when routing
        subtasks via agent_hint.
        """
        for agent in self._agents:
            if getattr(type(agent), "AGENT_ID", None) == agent_id:
                return agent
        return None

    def __len__(self) -> int:
        return len(self._agents)

    def __repr__(self) -> str:
        return f"AgentRegistry({len(self._agents)} agents)"