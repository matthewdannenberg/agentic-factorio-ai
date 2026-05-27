"""
execution/coordinator/

Goal-to-task translation and agent routing.

coordinator.py   RuleBasedCoordinator — derives Tasks from Goals using the
                 SelfModel and KnowledgeBase. Routes each Task to the
                 appropriate agent via agent_hint. Applies SelfModelPatches.
registry.py      AgentRegistry — ordered registry of AgentProtocol instances.
"""
