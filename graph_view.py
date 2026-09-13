"""Interactive graph rendering (pyvis) for the chatbot UI.

Two uses: the per-answer evidence subgraph (cited edges highlighted red) and
the ego-network Graph Explorer. HTML is generated fully in memory with vis.js
inlined, so it works offline and writes no files.
"""
from __future__ import annotations

import json
from pathlib import Path

import bootstrap
from onetrag.graph import query as q
from onetrag.graph.serialize import LevelAnchors, edge_line

LABEL_COLORS = {
    "Job": "#d62728", "Task": "#1f77b4", "DWA": "#2ca02c", "IWA": "#98df8a",
    "GWA": "#17becf", "Technology": "#9467bd", "EssentialSkill": "#ff7f0e",
    "TransferableSkill": "#ffbb78", "Ability": "#e377c2", "Knowledge": "#bcbd22",
    "WorkStyle": "#8c564b", "ElementGroup": "#7f7f7f",
}
# labels rendered on the node itself (others are hover-only to reduce clutter)
_LABELED = {"Job", "DWA", "IWA", "GWA", "EssentialSkill", "TransferableSkill",
            "Ability", "Knowledge", "WorkStyle", "ElementGroup"}

_OPTIONS = """{
  "physics": {"barnesHut": {"gravitationalConstant": -12000, "springLength": 140},
              "stabilization": {"iterations": 150}},
  "edges": {"smooth": false, "font": {"size": 9, "color": "#666666"}},
  "nodes": {"font": {"size": 12}},
  "interaction": {"hover": true, "tooltipDelay": 120}
}"""


def _short(text: str, limit: int = 28) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_edges_html(G, edges: list[q.Edge], anchors: LevelAnchors | None = None,
                      cited_edge_ids: set[int] | frozenset = frozenset(),
                      height: int = 550) -> str:
    """Standalone HTML for a subgraph given as an edge list."""
    from pyvis.network import Network

    net = Network(directed=True, height=f"{height}px", width="100%",
                  cdn_resources="in_line", notebook=False)
    seen: set[str] = set()

    def add_node(n: str):
        if n in seen:
            return
        seen.add(n)
        a = G.nodes[n] if n in G else {}
        label = a.get("label", "?")
        name = a.get("name", n)
        net.add_node(n,
                     label=_short(name) if label in _LABELED else " ",
                     title=f"{label}: {name} — click for details",
                     color=LABEL_COLORS.get(label, "#999999"),
                     size=22 if label == "Job" else 12)

    for u, v, k, d in edges:
        add_node(u)
        add_node(v)
        cited = int(k) in cited_edge_ids
        net.add_edge(u, v,
                     title=edge_line(G, (u, v, k, d), anchors),
                     label=d.get("type", ""),
                     color="#d62728" if cited else "#c9c9c9",
                     width=3 if cited else 1,
                     eid=str(int(k)))
    net.set_options(_OPTIONS)
    html = net.generate_html()
    return _inject_click_panel(html, G, edges, seen, anchors, cited_edge_ids)


def _node_details_html(G, n: str) -> str:
    from html import escape

    a = G.nodes[n] if n in G else {}
    label = a.get("label", "?")
    name = a.get("name", n)
    color = LABEL_COLORS.get(label, "#999999")
    rows = []
    if a.get("description"):
        rows.append(("Description", escape(str(a["description"]))))
    jz = a.get("job_zone")
    if jz is not None and jz == jz:
        rows.append(("Job Zone", f"{jz:.0f} (1 = little preparation … "
                                 "5 = extensive preparation)"))
    alts = a.get("alt_titles")
    if alts is not None and len(alts) > 0:
        sample = ", ".join(escape(str(x)) for x in list(alts)[:8])
        more = f" … (+{len(alts) - 8} more)" if len(alts) > 8 else ""
        rows.append((f"Alternate titles ({len(alts)})", sample + more))
    if a.get("tech_category"):
        rows.append(("Software category", escape(str(a["tech_category"]))))
    if a.get("task_type"):
        rows.append(("Task type", escape(str(a["task_type"]))))
    # connectivity profile in the FULL graph, by edge type
    counts: dict[str, int] = {}
    for _, _, d in G.out_edges(n, data=True):
        t = d.get("type", "?")
        counts[t] = counts.get(t, 0) + 1
    for _, _, d in G.in_edges(n, data=True):
        t = d.get("type", "?") + " (in)"
        counts[t] = counts.get(t, 0) + 1
    if counts:
        conn = ", ".join(f"{t}: {c}" for t, c in
                         sorted(counts.items(), key=lambda x: -x[1]))
        rows.append(("Edges in full graph", conn))

    body = "".join(
        f"<div class='ir'><span class='ik'>{k}</span><br>{v}</div>"
        for k, v in rows)
    return (f"<div class='ih' style='border-left:5px solid {color}'>"
            f"<b>{escape(str(name))}</b><br>"
            f"<span class='im'>{escape(label)} · {escape(str(n))}</span></div>"
            + body)


def _edge_details_html(G, edge, anchors) -> str:
    from html import escape

    u, v, k, d = edge
    rows = [("Statement", escape(edge_line(G, edge, anchors)))]
    skip = {"type"}
    pretty = {"importance": "Importance (1–5)", "level": "Level (0–7)",
              "w": "Retrieval weight w", "tier": "Relatedness tier",
              "index": "Relatedness rank", "task_type": "Task type",
              "hot_technology": "Hot technology", "in_demand": "In demand",
              "not_relevant": "Marked not relevant"}
    for key, val in d.items():
        if key in skip or val is None or val != val:
            continue
        label = pretty.get(key, key)
        if isinstance(val, float):
            val = f"{val:.2f}"
        rows.append((label, escape(str(val))))
    rows.append(("Source → target",
                 f"{escape(str(G.nodes[u].get('name', u)))} → "
                 f"{escape(str(G.nodes[v].get('name', v)))}"))
    body = "".join(
        f"<div class='ir'><span class='ik'>{key}</span><br>{val}</div>"
        for key, val in rows)
    return (f"<div class='ih' style='border-left:5px solid #d62728'>"
            f"<b>Edge E{k}</b><br>"
            f"<span class='im'>{escape(str(d.get('type', '?')))}</span></div>"
            + body)


def _inject_click_panel(html: str, G, edges, nodes, anchors,
                        cited_edge_ids) -> str:
    """Add a click-to-inspect details panel inside the pyvis page."""
    node_info = {str(n): _node_details_html(G, n) for n in nodes}
    edge_info = {str(int(k)): _edge_details_html(G, (u, v, k, d), anchors)
                 for u, v, k, d in edges}
    panel = """
<style>
#infopanel {position:absolute; top:10px; right:10px; width:300px;
  max-height:88%; overflow-y:auto; display:none; z-index:20;
  background:#ffffff; color:#222; border:1px solid #d5d9de;
  border-radius:10px; box-shadow:0 4px 16px rgba(0,0,0,.18);
  font-family:'Segoe UI',system-ui,sans-serif; font-size:12px;
  line-height:1.45; padding:0 0 6px 0;}
#infopanel .ih {padding:8px 10px; background:#f4f6f8;
  border-radius:10px 10px 0 0; font-size:12.5px;}
#infopanel .im {color:#67707a; font-size:11px;}
#infopanel .ir {padding:6px 12px 2px 12px;}
#infopanel .ik {color:#1D4ED8; font-weight:600; font-size:11px;
  text-transform:uppercase; letter-spacing:.3px;}
#infopanel .ix {position:sticky; top:4px; float:right; margin:4px 6px 0 0;
  cursor:pointer; color:#8a9199; font-size:14px; font-weight:700;}
#clickhint {position:absolute; bottom:8px; left:10px; z-index:20;
  font-family:'Segoe UI',system-ui,sans-serif; font-size:11px;
  color:#8a9199; background:rgba(255,255,255,.85); padding:2px 8px;
  border-radius:6px;}
</style>
<div id="infopanel"></div>
<div id="clickhint">Click a node or edge to inspect its attributes</div>
<script type="text/javascript">
  var NODE_INFO = __NODE_INFO__;
  var EDGE_INFO = __EDGE_INFO__;
  (function attach() {
    if (typeof network === "undefined" || network === null) {
      setTimeout(attach, 200); return;
    }
    var panel = document.getElementById("infopanel");
    function show(html) {
      panel.innerHTML = "<span class='ix' onclick=\\"this.parentNode.style.display='none'\\">&#10005;</span>" + html;
      panel.style.display = "block";
    }
    network.on("click", function (p) {
      if (p.nodes.length > 0) {
        var info = NODE_INFO[String(p.nodes[0])];
        if (info) { show(info); return; }
      }
      if (p.edges.length > 0) {
        var e = network.body.data.edges.get(p.edges[0]);
        var info = e && e.eid ? EDGE_INFO[String(e.eid)] : null;
        if (info) { show(info); return; }
      }
      panel.style.display = "none";
    });
  })();
</script>
"""
    panel = (panel.replace("__NODE_INFO__", json.dumps(node_info))
                  .replace("__EDGE_INFO__", json.dumps(edge_info)))
    return html.replace("</body>", panel + "\n</body>")


def ego_edges(G, center: str, radius: int = 1,
              edge_types: set[str] | None = None,
              max_edges: int = 150) -> list[q.Edge]:
    return q.khop_edges(G, [center], depth=radius, edge_types=edge_types,
                        max_edges=max_edges)


def graph_stats() -> dict:
    meta = json.loads((Path(bootstrap.RESEARCH_DIR) / "data" / "graph" /
                       "meta.json").read_text(encoding="utf-8"))
    return meta
