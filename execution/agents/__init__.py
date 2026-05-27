"""
execution/agents/

Agent implementations. Each agent satisfies AgentProtocol (base.py),
owns one active Task, and emits Actions each tick.

base.py          AgentProtocol — the interface every agent must satisfy.
navigation.py    NavigationAgent — rule-based movement.
mining.py        MiningAgent — resource gathering and terrain clearing.
crafting.py      CraftingAgent — hand-crafting item queuing.
"""
