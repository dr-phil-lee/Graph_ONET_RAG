"""GraphRAG variant for chat: union of user-forced seeds and linker output.

Free-form chat questions often defeat the exact-match entity linker (it was
built for benchmark phrasings), so the UI lets the user pin an occupation;
those pins come first so seed-composition heuristics see them.
"""
from __future__ import annotations

import bootstrap  # noqa: F401  (sys.path setup; must precede onetrag imports)
from onetrag.systems.graph_rag import GraphRAG


class HybridGraphRAG(GraphRAG):
    name = "s2_graph_hybrid"

    def seeds_for(self, item: dict) -> list[str]:
        forced = [s for s in item.get("forced_seeds") or [] if s in self.G]
        linked = super().seeds_for(item)
        return list(dict.fromkeys(forced + linked))[:6]
