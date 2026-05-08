"""
world/tech_tree.py

TechTree — queries the KnowledgeBase for research dependency data.

The agent starts with no knowledge of the
tech tree and learns it through:
  1. ResearchState.unlocked — techs the game says are already researched
  2. ensure_tech() — called for each tech encountered; queries Factorio for
     prerequisites and unlock effects if the tech is unknown

All query methods remain safe for unknown tech names (return [] / False / etc.).
The TechTree is not stateful itself — it reads from the KnowledgeBase on every
call. The KnowledgeBase handles caching and persistence.
"""

from __future__ import annotations

from world.knowledge import KnowledgeBase, TechRecord
from world.state import ResearchState


class TechTree:
    """
    Research dependency graph backed by KnowledgeBase.

    Parameters
    ----------
    kb : KnowledgeBase
        Shared knowledge store. TechTree reads and writes through it.
    """

    def __init__(self, kb: KnowledgeBase) -> None:
        self._kb = kb

    def known(self, tech: str) -> bool:
        return self._kb.get_tech(tech) is not None

    def ensure(self, tech: str) -> TechRecord:
        return self._kb.ensure_tech(tech)

    def is_unlocked(self, tech: str, research: ResearchState) -> bool:
        return tech in research.unlocked

    def is_reachable(self, tech: str, research: ResearchState) -> bool:
        record = self._kb.ensure_tech(tech)
        if record.is_placeholder:
            return False
        return all(p in research.unlocked for p in record.prerequisites)

    def prerequisites(self, tech: str) -> list[str]:
        record = self._kb.get_tech(tech)
        return list(record.prerequisites) if record and not record.is_placeholder else []

    def all_prerequisites(self, tech: str) -> set[str]:
        return self._kb.all_prerequisites(tech)

    def unlocks_entity(self, tech: str) -> list[str]:
        record = self._kb.get_tech(tech)
        return list(record.unlocks_entities) if record else []

    def unlocks_recipe(self, tech: str) -> list[str]:
        record = self._kb.get_tech(tech)
        return list(record.unlocks_recipes) if record else []

    def path_to(self, tech: str, research: ResearchState) -> list[str]:
        if self._kb.get_tech(tech) is None:
            raise ValueError(f"Unknown technology: {tech!r}")
        if tech in research.unlocked:
            return []

        needed: set[str] = set()
        stack = [tech]
        while stack:
            current = stack.pop()
            if current in research.unlocked or current in needed:
                continue
            needed.add(current)
            record = self._kb.get_tech(current)
            if record and not record.is_placeholder:
                stack.extend(record.prerequisites)

        in_degree: dict[str, int] = {t: 0 for t in needed}
        dependents: dict[str, list[str]] = {t: [] for t in needed}
        for t in needed:
            record = self._kb.get_tech(t)
            if record and not record.is_placeholder:
                for prereq in record.prerequisites:
                    if prereq in needed:
                        in_degree[t] += 1
                        dependents[prereq].append(t)

        queue = [t for t in needed if in_degree[t] == 0]
        order: list[str] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for dep in dependents.get(current, []):
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)
        return order

    def next_researchable(self, research: ResearchState) -> list[str]:
        result = []
        for name, record in self._kb.all_techs().items():
            if record.is_placeholder:
                continue
            if name in research.unlocked:
                continue
            if all(p in research.unlocked for p in record.prerequisites):
                result.append(name)
        result.sort(key=lambda t: len(self.all_prerequisites(t)))
        return result

    def absorb_research_state(self, research: ResearchState) -> None:
        """Ensure every tech mentioned in research state is known to the KB."""
        for tech in research.unlocked:
            self._kb.ensure_tech(tech)
        for tech in research.queued:
            self._kb.ensure_tech(tech)
        if research.in_progress:
            self._kb.ensure_tech(research.in_progress)