#!/usr/bin/env python3
"""Make SGLang's bundled Grafana dashboard render on someone else's Grafana.

``examples/monitoring/grafana/dashboards/json/sglang-dashboard.json`` in the
SGLang repo is written against the author's own Grafana. Imported elsewhere, all
of its panels are blank, for two independent reasons:

1. **Datasource uid.** Every panel targets ``ddyfngn31dg5cf``. A provisioned
   Prometheus gets whatever uid Grafana assigns, so the panels have no
   datasource to query.
2. **Metric names.** Its queries use ``sglang_gen_throughput``; SGLang exposes
   ``sglang:gen_throughput``. The colon survives exposition -- it is not
   rewritten to an underscore -- so these match nothing.

A blank panel looks the same as a broken one, which is why this is worth a
script rather than a note.

It also appends the four panels the upstream dashboard lacks entirely: prefill
and decode speed, aggregated and per request. Those are derived from
``prompt_tokens_total``, ``generation_tokens_total``, the TTFT histogram and the
inter-token-latency histogram -- all of which are already exposed once
``--enable-metrics`` is set.

Usage::

    python3 fix_grafana_dashboard.py \\
        --input sglang-dashboard.json \\
        --output sglang-dashboard.fixed.json \\
        --datasource-uid "$(curl -s -u admin:admin localhost:3001/api/datasources \\
            | python3 -c 'import json,sys;print(json.load(sys.stdin)[0]["uid"])')"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re

#: Rate window used when a panel is rewritten outside a Grafana session, where
#: ``$__rate_interval`` is meaningless.
FALLBACK_RATE = "5m"

SPEED_PANELS = (
    ("Prefill speed — prompt tokens/s (aggregate)",
     "sum(rate(sglang:prompt_tokens_total[$__rate_interval]))"),
    ("Decode speed — output tokens/s (aggregate)",
     "sum(rate(sglang:generation_tokens_total[$__rate_interval]))"),
    ("Prefill speed per request (tokens/s)",
     "sum(rate(sglang:uncached_prompt_tokens_histogram_sum[$__rate_interval]))\n"
     "/ sum(rate(sglang:uncached_prompt_tokens_histogram_count[$__rate_interval]))\n"
     "/ (sum(rate(sglang:time_to_first_token_seconds_sum[$__rate_interval]))\n"
     "   / sum(rate(sglang:time_to_first_token_seconds_count[$__rate_interval])))"),
    ("Decode speed per request (tokens/s)",
     "1 / (sum(rate(sglang:inter_token_latency_seconds_sum[$__rate_interval]))\n"
     "     / sum(rate(sglang:inter_token_latency_seconds_count[$__rate_interval])))"),
)


def rewrite(dash: dict, uid: str, add_speed: bool) -> tuple[int, int]:
    """Point panels at ``uid`` and translate metric names. Returns (exprs, panels)."""
    datasource = {"type": "prometheus", "uid": uid}
    rewritten = 0

    def walk(panels: list[dict]) -> None:
        nonlocal rewritten
        for panel in panels:
            if panel.get("datasource") is not None:
                panel["datasource"] = datasource
            for target in panel.get("targets") or []:
                if target.get("datasource") is not None:
                    target["datasource"] = datasource
                expr = target.get("expr")
                if expr:
                    new = re.sub(r"\bsglang_", "sglang:", expr)
                    if new != expr:
                        rewritten += 1
                    target["expr"] = new
            walk(panel.get("panels") or [])

    walk(dash.get("panels") or [])

    added = 0
    if add_speed:
        ids = [p.get("id", 0) for p in dash.get("panels") or []]
        next_id = (max(ids) + 1) if ids else 1
        bottom = max(
            (p.get("gridPos", {}).get("y", 0) + p.get("gridPos", {}).get("h", 8))
            for p in dash.get("panels") or [{}]
        )
        for i, (title, expr) in enumerate(SPEED_PANELS):
            dash.setdefault("panels", []).append({
                "id": next_id + i,
                "type": "timeseries",
                "title": title,
                "datasource": datasource,
                "gridPos": {"h": 8, "w": 12, "x": (i % 2) * 12,
                            "y": bottom + (i // 2) * 8},
                "fieldConfig": {"defaults": {"unit": "none", "custom": {
                    "drawStyle": "line", "fillOpacity": 10}}, "overrides": []},
                "options": {"legend": {"displayMode": "list", "placement": "bottom"},
                            "tooltip": {"mode": "multi", "sort": "desc"}},
                "targets": [{"datasource": datasource, "editorMode": "code",
                             "expr": expr, "legendFormat": title.split(" — ")[0],
                             "range": True, "refId": "A"}],
            })
            added += 1
    return rewritten, added


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=pathlib.Path, required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    ap.add_argument("--datasource-uid", required=True,
                    help="uid of the Prometheus datasource in the target Grafana")
    ap.add_argument("--no-speed-panels", action="store_true",
                    help="only repair datasource/metrics, do not add the speed panels")
    args = ap.parse_args()

    dash = json.loads(args.input.read_text())
    exprs, added = rewrite(dash, args.datasource_uid, not args.no_speed_panels)
    args.output.write_text(json.dumps(dash, indent=2))
    print(f"{args.output}: {exprs} expressions translated to sglang:, "
          f"{added} speed panels added, datasource -> {args.datasource_uid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
