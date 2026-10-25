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

# layout.randomSeed makes vis.js place the nodes deterministically, so the
# same records always give the same picture (whichever model answered)
_OPTIONS = """{
  "layout": {"randomSeed": 7, "improvedLayout": true},
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
                      height: int = 550, *,
                      extra_nodes: list[str] | tuple = (),
                      focus_nodes: set[str] | frozenset = frozenset(),
                      exposure: dict[str, dict] | None = None,
                      edge_colors: dict[int, str] | None = None,
                      legend: list[tuple[str, str]] | None = None) -> str:
    """Standalone HTML for a subgraph given as an edge list.

    extra_nodes:  nodes to draw even when no edge touches them (e.g. an
                  occupation whose profile was cited).
    focus_nodes:  drawn larger with a thick dark border (the user's job).
    exposure:     soc -> AI-exposure dict; Job nodes are then colored on a
                  continuous green-red scale and labelled with the score.
    edge_colors:  edge id -> color (e.g. one color per conversation turn);
                  overrides the cited/retrieved coloring.
    legend:       [(color, text)] drawn in the bottom-left corner.
    """
    from pyvis.network import Network

    net = Network(directed=True, height=f"{height}px", width="100%",
                  cdn_resources="in_line", notebook=False)
    seen: set[str] = set()
    exposure = exposure or {}
    edge_colors = edge_colors or {}

    def add_node(n: str):
        if n in seen:
            return
        seen.add(n)
        a = G.nodes[n] if n in G else {}
        label = a.get("label", "?")
        name = a.get("name", n)
        color = LABEL_COLORS.get(label, "#999999")
        text = _short(name) if label in _LABELED else " "
        title = f"{label}: {name} — click for details"
        x = exposure.get(n) if label == "Job" else None
        if x:
            color = x.get("color", color)
            text = f"{_short(name, 26)}\nAI {x['composite']:.2f}"
            title = (f"{label}: {name} — AI exposure score {x['composite']:.2f} "
                     f"— click for details")
        elif exposure and label == "Job":
            color = "#9CA3AF"      # exposure view, but the index has no row
        focus = n in focus_nodes
        kw = {}
        if focus:
            kw = {"borderWidth": 4, "shape": "star" if label == "Job" else "dot"}
            color = {"background": color, "border": "#111827",
                     "highlight": {"background": color, "border": "#111827"}}
        net.add_node(n, label=text, title=title, color=color,
                     size=(34 if focus else 22) if label == "Job" else 12,
                     font={"size": 14 if focus else 12,
                           "face": "Inter, Segoe UI, sans-serif"}, **kw)

    for u, v, k, d in edges:
        add_node(u)
        add_node(v)
        cited = int(k) in cited_edge_ids
        color = edge_colors.get(int(k)) or ("#d62728" if cited else "#c9c9c9")
        net.add_edge(u, v,
                     title=edge_line(G, (u, v, k, d), anchors),
                     label=d.get("type", ""),
                     color=color,
                     width=3 if (cited or int(k) in edge_colors) else 1,
                     dashes=not cited and int(k) not in edge_colors,
                     eid=str(int(k)))
    for n in extra_nodes:
        if n in G:
            add_node(n)
    net.set_options(_OPTIONS)
    html = net.generate_html()
    html = _inject_click_panel(html, G, edges, seen, anchors, cited_edge_ids,
                               exposure=exposure)
    if legend:
        html = html.replace("</body>", _legend_html(legend) + "\n</body>")
    return html


def _legend_html(items: list[tuple[str, str]]) -> str:
    from html import escape

    rows = "".join(
        f"<div><span style='display:inline-block;width:11px;height:11px;"
        f"border-radius:50%;background:{c};margin-right:6px;vertical-align:-1px'>"
        f"</span>{escape(t)}</div>" for c, t in items)
    return ("<div style=\"position:absolute;bottom:30px;left:10px;z-index:20;"
            "background:rgba(255,255,255,.92);border:1px solid #e3e7ec;"
            "border-radius:8px;padding:6px 10px;font:11px 'Segoe UI',system-ui,"
            f"sans-serif;color:#374151;line-height:1.7\">{rows}</div>")


def exposure_legend() -> list[tuple[str, str]]:
    import ai_exposure

    lo, mid, hi = ai_exposure.COLOR_STOPS
    return [(lo[1], f"AI exposure score {lo[0]:.2f} or lower"),
            (mid[1], f"AI exposure score {mid[0]:.2f}"),
            (hi[1], f"AI exposure score {hi[0]:.2f} or higher"),
            ("#9CA3AF", "Job not in the AI Exposure Index"),
            ("#111827", "★ = occupation you asked about")]


def _node_details_html(G, n: str, exposure: dict | None = None) -> str:
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
    if exposure:
        rows.append(("AI exposure (Opportunity Data index)",
                     f"<b style='color:{exposure.get('color', '#333')}'>"
                     f"{exposure['composite']:.2f}</b> (0 = protected, 1 = exposed)<br>"
                     f"digital {exposure['digital']:.2f} · human contact "
                     f"{exposure['human_contact']:.2f} · physical {exposure['physical']:.2f}<br>"
                     f"= {escape(exposure.get('calculation', ''))}"))
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
                        cited_edge_ids, exposure: dict | None = None) -> str:
    """Add a click-to-inspect details panel inside the pyvis page."""
    exposure = exposure or {}
    node_info = {str(n): _node_details_html(G, n, exposure.get(n)) for n in nodes}
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
    // graphs inside a hidden tab are laid out at zero size: re-fit the view
    // whenever the frame gets a real size (tab opened, window resized)
    var box = document.getElementById("mynetwork");
    var lastW = 0;
    function refit() {
      var w = box ? box.clientWidth : 0;
      if (w > 0 && w !== lastW) { lastW = w; network.redraw(); network.fit({animation: false}); }
    }
    if (box && typeof ResizeObserver !== "undefined") { new ResizeObserver(refit).observe(box); }
    network.once("stabilizationIterationsDone", function () { lastW = 0; refit(); });
    document.addEventListener("visibilitychange", refit);
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
    meta = json.loads((Path(bootstrap.DATA_DIR) / "graph" /
                       "meta.json").read_text(encoding="utf-8"))
    return meta
