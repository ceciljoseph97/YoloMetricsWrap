import base64
import os
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# Define base stems (without extension). We'll expand these to include box-prefixed variants.
BASE_METRIC_STEMS = {
    "PR": ["pr_curve", "pr-curve", "prcurve", "precision_recall_curve", "precision-recall", "pr"],
    "P": ["p_curve", "p-curve", "pcurve", "precision_curve", "precision", "boxpcurve"],
    "R": ["r_curve", "r-curve", "rcurve", "recall_curve", "recall"],
    "F1": ["f1_curve", "f1-curve", "f1curve", "f1"],
    "CM": ["confusion_matrix", "confusion-matrix", "cm"],
    "CM_N": [
        "confusion_matrix_normalized",
        "confusion-matrix-normalized",
        "normalized_confusion_matrix",
        "confusion_matrix_norm",
        "confusion-matrix-norm",
        "confusionmatrix_normalized",
        "confusionmatrix-normalized",
        "confusion_matrix_normalised",   # UK spelling
        "confusion-matrix-normalised",  # UK spelling
        "cm_normalized",
        "cm_norm",
        "cm-normalized",
        "cm-norm",
    ],
}

def expand_with_box(stems: List[str]) -> List[str]:
    variants: List[str] = []
    seen = set()
    for s in stems:
        for v in (s, f"box{s}", f"box_{s}", f"box-{s}"):
            low = v.lower()
            if low not in seen:
                seen.add(low)
                variants.append(v)
    return variants

# Expanded aliases including box-prefixed possibilities
METRIC_ALIASES = {k: expand_with_box(v) for k, v in BASE_METRIC_STEMS.items()}

ALLOWED_EXTS = [".png", ".jpg", ".jpeg", ".webp", ".svg"]

def guess_mime_type(path: Path) -> str:
    suf = path.suffix.lower()
    if suf == ".png":
        return "image/png"
    if suf in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suf == ".webp":
        return "image/webp"
    if suf == ".svg":
        return "image/svg+xml"
    return "application/octet-stream"


@dataclass
class ConfigImages:
    config_name: str
    images: Dict[str, Any]  # value can be data URL string or list of data URLs (for galleries)


@dataclass
class RunGroup:
    run_name: str
    run_path: Path
    configs: List[ConfigImages]


def encode_image_to_data_url(image_path: Path) -> Optional[str]:
    if not image_path.exists():
        return None
    try:
        with image_path.open("rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        mime = guess_mime_type(image_path)
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def collect_config_images(root: Path) -> List[ConfigImages]:
    config_dirs = [
        p for p in root.iterdir()
        if p.is_dir() and p.name.endswith("_config")
    ]
    config_dirs.sort(key=lambda p: p.name.lower())

    collected: List[ConfigImages] = []
    for cfg_dir in config_dirs:
        images: Dict[str, Any] = {}
        # Snapshot of existing files lowercased for quick lookups
        present = {f.name.lower(): f for f in cfg_dir.iterdir() if f.is_file()}
        # For each metric, pick the first existing alias + ext
        for metric_key, stems in METRIC_ALIASES.items():
            data_url: Optional[str] = None
            found_ext: Optional[str] = None
            for stem in stems:
                for ext in ALLOWED_EXTS:
                    candidate = stem + ext
                    f = present.get(candidate.lower())
                    if f is not None:
                        data_url = encode_image_to_data_url(f)
                        found_ext = ext
                        break
                if data_url:
                    break
            # Store under .png key for stable lookups, but if we discovered a different ext,
            # also store under that exact key to allow flexible consumers.
            canonical_png_key = stems[0] + ALLOWED_EXTS[0]
            images[canonical_png_key] = data_url
            if found_ext and found_ext != ALLOWED_EXTS[0]:
                images[stems[0] + found_ext] = data_url
        # Fallback: if normalized CM missing, reuse raw CM
        cm_key = METRIC_ALIASES["CM"][0] + ALLOWED_EXTS[0]
        cmn_key = METRIC_ALIASES["CM_N"][0] + ALLOWED_EXTS[0]
        if images.get(cmn_key) is None and images.get(cm_key) is not None:
            images[cmn_key] = images.get(cm_key)

        # Collect galleries: val batches and generic images
        val_labels: List[str] = []
        val_preds: List[str] = []
        all_imgs: List[str] = []
        for name, file_path in present.items():
            if not any(name.endswith(ext) for ext in ALLOWED_EXTS):
                continue
            lower = name.lower()
            url = encode_image_to_data_url(file_path)
            if not url:
                continue
            if lower.startswith("val_batch") and "_labels" in lower:
                val_labels.append(url)
            elif lower.startswith("val_batch") and "_pred" in lower:
                val_preds.append(url)
            else:
                all_imgs.append(url)
        if val_labels:
            # Convert to mapping by batch name for categorical subtabs
            labels_map: Dict[str, List[str]] = {}
            for item in val_labels:
                # item is data URL string; infer we already filtered
                # We don't have the filename here; keep sequential keys
                key = "batch_" + str(len(labels_map) + 1)
                labels_map.setdefault(key, []).append(item)
            images["VAL_LABELS"] = labels_map if labels_map else val_labels
        if val_preds:
            preds_map: Dict[str, List[str]] = {}
            for item in val_preds:
                key = "batch_" + str(len(preds_map) + 1)
                preds_map.setdefault(key, []).append(item)
            images["VAL_PRED"] = preds_map if preds_map else val_preds
        collected.append(ConfigImages(config_name=cfg_dir.name, images=images))
    return collected


def discover_runs(root: Path) -> List[RunGroup]:
    runs: List[RunGroup] = []

    def is_run_dir(p: Path) -> bool:
        try:
            return any(c.is_dir() and c.name.endswith("_config") for c in p.iterdir())
        except Exception:
            return False

    # Include root itself if it contains *_config
    candidates: List[Tuple[str, Path]] = []
    if is_run_dir(root):
        candidates.append((root.name or str(root), root))
    # Also scan immediate subdirectories
    for child in root.iterdir():
        if child.is_dir() and is_run_dir(child):
            candidates.append((child.name, child))

    # De-duplicate by path
    seen = set()
    for run_name, run_path in candidates:
        if run_path in seen:
            continue
        seen.add(run_path)
        configs = collect_config_images(run_path)
        if configs:
            runs.append(RunGroup(run_name=run_name, run_path=run_path, configs=configs))
    # Stable sort by name
    runs.sort(key=lambda r: r.run_name.lower())
    return runs


def generate_html_multi(runs: List[RunGroup], title: str) -> str:
    # Minimal, readable CSS; responsive grid per config + tabs + lightbox
    css = """
    :root { --gap: 14px; --card-bg: #0b0f14; --ink: #e5e7eb; --muted: #9ca3af; --accent: #60a5fa; }
    * { box-sizing: border-box; }
    body { margin: 0; padding: 24px; background: #0a0a0a; color: var(--ink); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial, "Apple Color Emoji", "Segoe UI Emoji"; }
    h1 { margin: 0 0 8px 0; font-size: 24px; }
    .subtitle { color: var(--muted); margin-bottom: 22px; }
    .tabs { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 14px; }
    .tab-btn { appearance: none; border: 1px solid #1f2937; background: #0b1220; color: var(--ink); padding: 8px 12px; border-radius: 8px; cursor: pointer; font-size: 13px; }
    .tab-btn[aria-selected="true"] { background: #11203a; border-color: #25406b; color: #a8c7ff; }
    .tab-panel { display: none; position: relative; }
    .tab-panel.active { display: block; }
    .configs { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: var(--gap); }
    .card { background: var(--card-bg); border: 1px solid #111827; border-radius: 10px; padding: 14px; position: relative; }
    .card h2 { font-size: 16px; margin: 0 0 10px 0; color: var(--accent); word-break: break-all; }
    .imgwrap { background: #0a0a0a; border: 1px solid #1f2937; border-radius: 8px; padding: 8px; }
    .imgwrap h3 { margin: 0 0 6px 0; font-size: 13px; color: var(--muted); }
    .imgwrap img { width: 100%; height: auto; display: block; border-radius: 6px; cursor: zoom-in; }
    .missing { border: 1px dashed #374151; color: #6b7280; display: grid; place-items: center; height: 200px; border-radius: 6px; font-size: 13px; }
    .foot { margin-top: 24px; color: var(--muted); font-size: 12px; }
    @media (max-width: 920px) { .configs { grid-template-columns: 1fr; } }

    /* Lightbox */
    .lightbox { position: fixed; inset: 0; background: rgba(0,0,0,0.9); display: none; align-items: center; justify-content: center; z-index: 1000; }
    .lightbox.show { display: flex; }
    .lightbox-content { position: relative; max-width: 95vw; max-height: 95vh; overflow: hidden; }
    .lightbox-img { user-select: none; -webkit-user-drag: none; transform-origin: 0 0; cursor: grab; }
    .lightbox-controls { position: absolute; left: 50%; transform: translateX(-50%); bottom: 10px; display: flex; gap: 8px; }
    .lb-btn { background: #111827; color: var(--ink); border: 1px solid #374151; padding: 6px 10px; border-radius: 6px; font-size: 13px; cursor: pointer; }
    .lb-close { position: absolute; top: 10px; right: 10px; }

    /* JSON viewer */
    .json-toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 18px 0 10px; }
    .json-toolbar input[type="text"] { background: #0b1220; color: var(--ink); border: 1px solid #1f2937; padding: 6px 10px; border-radius: 6px; }
    .json-box { background: #0b0f14; border: 1px solid #111827; border-radius: 10px; padding: 10px; white-space: pre; overflow: auto; max-height: 60vh; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; color: #e5e7eb; }
    .json-status { color: var(--muted); font-size: 12px; margin-left: 6px; }
    .src-ref { text-decoration: underline dotted; color: #93c5fd; cursor: pointer; }
    .src-ref:hover { background: rgba(96,165,250,0.18); }

    /* Source explorer */
    .code-toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 14px 0 10px; }
    .code-toolbar input[type="text"] { background: #0b1220; color: var(--ink); border: 1px solid #1f2937; padding: 6px 10px; border-radius: 6px; min-width: 340px; }
    .code-box { background: #0b0f14; border: 1px solid #111827; border-radius: 10px; padding: 10px; overflow: auto; max-height: 60vh; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; font-size: 12px; color: #e5e7eb; }
    .code-line { white-space: pre; }
    .code-gutter { display: inline-block; width: 54px; color: #6b7280; user-select: none; }
    .code-hit { background: rgba(96,165,250,0.18); }
    /* Two-column layout for JSON + Source explorer */
    .inspectors { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }
    @media (max-width: 1100px) { .inspectors { grid-template-columns: 1fr; } }

    /* Compare UI */
    .cmp-toggle { width: 18px; height: 18px; accent-color: #60a5fa; cursor: pointer; }
    .cmp-header { position: absolute; top: 8px; right: 12px; }
    .cmp-btn { position: fixed; right: 18px; bottom: 18px; background:#2563eb; color:#fff; border:0; padding:10px 14px; border-radius:8px; font-size:13px; cursor:pointer; opacity:0.85; z-index: 1200; }
    .cmp-btn[disabled] { opacity:0.4; cursor:not-allowed; }
    .cmp-modal { position: fixed; inset:0; background: rgba(0,0,0,0.92); display:none; z-index:1100; }
    .cmp-modal.show { display:block; }
    .cmp-wrap { position:absolute; top:40px; bottom:40px; left:40px; right:40px; overflow:auto; border:1px solid #1f2937; border-radius:10px; background:#0b0f14; padding:14px; }
    .cmp-close { position:absolute; top:10px; right:14px; background:#111827; color:#e5e7eb; border:1px solid #374151; border-radius:8px; padding:6px 10px; cursor:pointer; }
    .cmp-row { display:grid; grid-template-columns: 1fr 1fr; gap:12px; margin-bottom:16px; }
    .cmp-cell { background:#0a0a0a; border:1px solid #1f2937; border-radius:8px; padding:8px; }
    .cmp-title { color:#a8c7ff; font-size:13px; margin:0 0 8px 0; }
    """

    def render_img(label: str, data_url: Optional[str]) -> str:
        if data_url:
            return (
                f'<div class="imgwrap">\n'
                f'  <h3>{label}</h3>\n'
                f'  <img loading="lazy" src="{data_url}" alt="{label}" />\n'
                f'</div>'
            )
        return (
            f'<div class="imgwrap">\n'
            f'  <h3>{label}</h3>\n'
            f'  <div class="missing">Missing</div>\n'
            f'</div>'
        )

    parts: List[str] = []
    parts.append("<" + "!DOCTYPE html>")
    parts.append("<html lang=\"en\">")
    parts.append("<head>")
    parts.append("  <meta charset=\"utf-8\" />")
    parts.append("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />")
    parts.append("  <title>YOLOmetrics Report</title>")
    parts.append("  <style>" + css + "</style>")
    parts.append("</head>")
    parts.append("<body>")
    parts.append(f"  <h1>{title}</h1>")
    parts.append("  <div class=\"subtitle\">Precision/Recall curves and confusion matrices across runs and configs.</div>")
    # Top-level navigation
    parts.append("  <div class=\"tabs\" role=\"tablist\" id=\"topnav\">")
    parts.append("    <button class=\"tab-btn\" role=\"tab\" aria-controls=\"homePage\" aria-selected=\"true\">Home</button>")
    parts.append("    <button class=\"tab-btn\" role=\"tab\" aria-controls=\"evalPage\" aria-selected=\"false\">Evaluation</button>")
    parts.append("    <button class=\"tab-btn\" role=\"tab\" aria-controls=\"analysisPage\" aria-selected=\"false\">Calltrace Analysis</button>")
    parts.append("  </div>")
    # Home Page
    parts.append("  <section id=\"homePage\" class=\"tab-panel active\" aria-label=\"Home\">")
    parts.append("    <div class=\"card\" style=\"display:grid; gap:10px;\">")
    parts.append("      <div><strong>YOLOmetrics</strong></div>")
    parts.append("      <div>Two tabs:</div>")
    parts.append("      <ul style=\"margin:0 0 6px 18px; line-height:1.6\">")
    parts.append("        <li><em>Evaluation</em>: view PR/P/R/F1 curves, confusion matrices, and validation batches for each *_config. Use the small checkboxes to compare two entries side-by-side.</li>")
    parts.append("        <li><em>Calltrace Analysis</em>: load/search JSON traces and open source files from GitHub or local .py files.</li>")
    parts.append("      </ul>")
    parts.append("      <div class=\"tabs\" role=\"tablist\" style=\"margin:0\">")
    parts.append("        <button class=\"tab-btn\" onclick=\"document.querySelector('#topnav .tab-btn[aria-controls=\\'evalPage\\']').click()\">YOLO Metrics Evaluation</button>")
    parts.append("        <button class=\"tab-btn\" onclick=\"document.querySelector('#topnav .tab-btn[aria-controls=\\'analysisPage\\']').click()\">Calltrace Analysis</button>")
    parts.append("      </div>")
    parts.append("    </div>")
    parts.append("  </section>")
    # Begin Evaluation Page wrapper
    parts.append("  <section id=\"evalPage\" class=\"tab-panel\" aria-label=\"Evaluation\">")
    parts.append("  <div style=\"margin:8px 0 16px\">"
                 "<input id=\"dirPicker\" type=\"file\" webkitdirectory directory multiple style=\"display:none\" />"
                 "<button id=\"addRunBtn\" class=\"tab-btn\">+ Add Run (folder)</button>"
                 "</div>")

    # Embed alias/extension data for client-side loader
    import json as _json
    alias_payload = {
        "alias": {k: [s for s in METRIC_ALIASES[k]] for k in METRIC_ALIASES},
        "exts": ALLOWED_EXTS,
        "canonical": {k: BASE_METRIC_STEMS[k][0] + ALLOWED_EXTS[0] for k in BASE_METRIC_STEMS},
    }
    parts.append("  <script id=\"aliasData\" type=\"application/json\">" + _json.dumps(alias_payload) + "</script>")

    # Run tab bar (starts empty; load runs via + Add Run)
    parts.append("  <div class=\"tabs\" role=\"tablist\" id=\"run-tabs\"></div>")

    # For each run, render a panel with metric-level tabs
    def render_run_panel(idx: int, run: RunGroup) -> None:
        # Tabs: one per metric
        metric_specs = [
            ("tab-pr", "Precision-Recall", METRIC_ALIASES["PR"][0] + ALLOWED_EXTS[0]),
            ("tab-p", "Precision vs Confidence", METRIC_ALIASES["P"][0] + ALLOWED_EXTS[0]),
            ("tab-r", "Recall vs Confidence", METRIC_ALIASES["R"][0] + ALLOWED_EXTS[0]),
            ("tab-f1", "F1 vs Confidence", METRIC_ALIASES["F1"][0] + ALLOWED_EXTS[0]),
            ("tab-cm", "Confusion Matrix", METRIC_ALIASES["CM"][0] + ALLOWED_EXTS[0]),
            ("tab-cm-norm", "Confusion Matrix (Normalized)", METRIC_ALIASES["CM_N"][0] + ALLOWED_EXTS[0]),
            ("tab-val-labels", "Validation Batches (Labels)", "VAL_LABELS"),
            ("tab-val-pred", "Validation Batches (Pred)", "VAL_PRED"),
        ]

        run_panel_id = f"run-{idx}"
        panel_active = " active" if (not multi_run and idx == 0) or (multi_run and idx == 0) else ""
        parts.append(f"  <section id=\"{run_panel_id}\" class=\"tab-panel{panel_active}\" role=\"tabpanel\" aria-label=\"{run.run_name}\">")

        # Metric tab bar for this run
        parts.append("    <div class=\"tabs\" role=\"tablist\">")
        for i, (tab_id, label, _fname) in enumerate(metric_specs):
            selected = "true" if i == 0 else "false"
            scoped_id = f"{run_panel_id}-{tab_id}"
            parts.append(
                f"      <button class=\"tab-btn\" role=\"tab\" aria-controls=\"{scoped_id}\" aria-selected=\"{selected}\">{label}</button>"
            )
        parts.append("    </div>")

        # Metric panels
        for i, (tab_id, label, fname) in enumerate(metric_specs):
            scoped_id = f"{run_panel_id}-{tab_id}"
            active_class = "active" if i == 0 else ""
            parts.append(f"    <section id=\"{scoped_id}\" class=\"tab-panel {active_class}\" role=\"tabpanel\" aria-label=\"{label}\">")
            parts.append("      <div class=\"configs\">")
            for cfg in run.configs:
                parts.append("        <article class=\"card\">")
                parts.append(f"          <h2>{cfg.config_name}</h2>")
                img_label = label
                data_entry = cfg.images.get(fname)
                if isinstance(data_entry, dict):
                    # Categorical sub-tabs per batch
                    parts.append("          <div class=\"tabs\" role=\"tablist\">")
                    batch_keys = sorted(data_entry.keys())
                    for bi, bkey in enumerate(batch_keys):
                        sel = "true" if bi==0 else "false"
                        parts.append(f'            <button class="tab-btn" role="tab" aria-controls="{scoped_id}-{cfg.config_name}-{bkey}" aria-selected="{sel}">{bkey}</button>')
                    parts.append("          </div>")
                    for bi, bkey in enumerate(batch_keys):
                        bactive = "active" if bi==0 else ""
                        parts.append(f'          <section id="{scoped_id}-{cfg.config_name}-{bkey}" class="tab-panel {bactive}" aria-label="{bkey}">')
                        parts.append("            <div class=\"cmp-header\"><input type=\"checkbox\" class=\"cmp-toggle\" data-run=\"" + run.run_name + "\" data-config=\"" + cfg.config_name + "\" data-metric=\"" + tab_id + "\" data-cat=\"" + bkey + "\" /></div>")
                        parts.append("            <div class=\"grid\" style=\"grid-template-columns: repeat(2, 1fr); gap: 10px;\">")
                        for url in data_entry[bkey]:
                            parts.append(
                                "              <div class=\"imgwrap\">\n"
                                f"                <img loading=\"lazy\" src=\"{url}\" alt=\"{img_label}\" data-lb=\"1\" />\n"
                                "              </div>"
                            )
                        parts.append("            </div>")
                        parts.append("          </section>")
                elif isinstance(data_entry, list):
                    if data_entry:
                        parts.append("          <div class=\"cmp-header\"><input type=\"checkbox\" class=\"cmp-toggle\" data-run=\"" + run.run_name + "\" data-config=\"" + cfg.config_name + "\" data-metric=\"" + tab_id + "\" data-cat=\"panel\" /></div>")
                        parts.append("          <div class=\"imgwrap\">\n" f"            <h3>{img_label}</h3>\n" "          </div>")
                        # gallery grid
                        parts.append("          <div class=\"grid\" style=\"grid-template-columns: repeat(2, 1fr); gap: 10px;\">")
                        for url in data_entry:
                            parts.append(
                                "            <div class=\"imgwrap\">\n"
                                f"              <img loading=\"lazy\" src=\"{url}\" alt=\"{img_label}\" data-lb=\"1\" />\n"
                                "            </div>"
                            )
                        parts.append("          </div>")
                    else:
                        parts.append(
                            "          <div class=\"imgwrap\">\n"
                            f"            <h3>{img_label}</h3>\n"
                            "            <div class=\"missing\">Missing</div>\n"
                            "          </div>"
                        )
                else:
                    data_url = data_entry
                    # If not found directly and fname looks like a .png metric, try alternate extensions
                    if not data_url and isinstance(fname, str) and fname.endswith('.png'):
                        base = fname[:-4]
                        for ext in ALLOWED_EXTS:
                            alt = base + ext
                            if alt == fname:
                                continue
                            data_url = cfg.images.get(alt)
                            if data_url:
                                break
                    if data_url:
                        parts.append("          <div class=\"cmp-header\"><input type=\"checkbox\" class=\"cmp-toggle\" data-run=\"" + run.run_name + "\" data-config=\"" + cfg.config_name + "\" data-metric=\"" + tab_id + "\" data-cat=\"single\" /></div>")
                        parts.append("          <div class=\"imgwrap\">\n" f"            <h3>{img_label}</h3>\n" f"            <img loading=\"lazy\" src=\"{data_url}\" alt=\"{img_label}\" data-lb=\"1\" />\n" "          </div>")
                    else:
                        parts.append(
                            "          <div class=\"imgwrap\">\n"
                            f"            <h3>{img_label}</h3>\n"
                            "            <div class=\"missing\">Missing</div>\n"
                            "          </div>"
                        )
                parts.append("        </article>")
            parts.append("      </div>")
            parts.append("    </section>")

        parts.append("  </section>")

    # Defer initial rendering; prompt user to load
    parts.append("  <div id=\"noRunsMsg\" class=\"missing\">No runs loaded. Use + Add Run (folder) to load results.</div>")

    # Lightbox HTML
    parts.append(
        "  <div class=\"lightbox\" id=\"lightbox\" aria-hidden=\"true\">\n"
        "    <div class=\"lightbox-content\">\n"
        "      <img class=\"lightbox-img\" id=\"lbImg\" alt=\"zoomed\" />\n"
        "      <button class=\"lb-btn lb-close\" id=\"lbClose\">Close</button>\n"
        "      <div class=\"lightbox-controls\">\n"
        "        <button class=\"lb-btn\" id=\"lbZoomIn\">+</button>\n"
        "        <button class=\"lb-btn\" id=\"lbZoomOut\">-</button>\n"
        "        <button class=\"lb-btn\" id=\"lbReset\">Reset</button>\n"
        "      </div>\n"
        "    </div>\n"
        "  </div>"
    )

    parts.append("  <div class=\"foot\">Click an image to zoom (drag to pan, wheel to zoom).</div>")
    parts.append("  <div class=\"foot\">Generated locally. All images embedded; the file is portable.</div>")
    parts.append("  </section>")

    # Analysis Page wrapper with inspectors
    parts.append("  <section id=\"analysisPage\" class=\"tab-panel\" aria-label=\"Calltrace Analysis\">")
    parts.append("  <div class=\"inspectors\">")
    # Left: JSON viewer
    parts.append("    <section>")
    parts.append("      <h2 style=\"margin:0 0 8px 0; font-size:18px; color:#a8c7ff\">JSON Viewer</h2>")
    parts.append("      <div class=\"json-toolbar\">")
    parts.append("        <input id=\"jsonPicker\" type=\"file\" accept=\"application/json\" multiple style=\"position:absolute; left:-9999px; width:1px; height:1px; opacity:0; pointer-events:none;\" />")
    parts.append("        <label id=\"jsonLoadBtn\" for=\"jsonPicker\" class=\"tab-btn\" style=\"display:inline-block; cursor:pointer;\">Browse JSON</label>")
    parts.append("        <button id=\"jsonBeautify\" class=\"tab-btn\">Beautify</button>")
    parts.append("        <button id=\"jsonCopy\" class=\"tab-btn\">Copy</button>")
    parts.append("        <input id=\"jsonSearch\" type=\"text\" placeholder=\"Search...\" />")
    parts.append("        <button id=\"jsonPrev\" class=\"tab-btn\">Prev</button>")
    parts.append("        <button id=\"jsonNext\" class=\"tab-btn\">Next</button>")
    parts.append("        <span id=\"jsonStatus\" class=\"json-status\"></span>")
    parts.append("      </div>")
    parts.append("      <div class=\"tabs\" role=\"tablist\" id=\"jsonTabs\"></div>")
    parts.append("      <pre id=\"jsonBox\" class=\"json-box\"></pre>")
    parts.append("    </section>")
    # Right: Source explorer (fetch-only)
    parts.append("    <section>")
    parts.append("      <h2 style=\"margin:0 0 8px 0; font-size:18px; color:#a8c7ff\">Source Explorer</h2>")
    parts.append("      <div class=\"code-toolbar\">")
    parts.append("        <input id=\"srcPath\" type=\"text\" placeholder=\"ultralytics/models/yolo/detect/val.py:80\" />")
    parts.append("        <button id=\"srcOpen\" class=\"tab-btn\">Open</button>")
    parts.append("        <input id=\"srcFilePicker\" type=\"file\" accept=\".py\" multiple style=\"position:absolute; left:-9999px; width:1px; height:1px; opacity:0; pointer-events:none;\" />")
    parts.append("        <label id=\"srcBrowseBtn\" for=\"srcFilePicker\" class=\"tab-btn\" style=\"display:inline-block; cursor:pointer;\">Browse .py</label>")
    parts.append("        <span id=\"srcStatus\" class=\"json-status\"></span>")
    parts.append("      </div>")
    parts.append("      <div class=\"tabs\" role=\"tablist\" id=\"srcTabs\"></div>")
    parts.append("      <pre id=\"codeBox\" class=\"code-box\"></pre>")
    parts.append("    </section>")
    parts.append("  </div>")
    parts.append("  </section>")
    # Tabs & Lightbox JS
    js = """
    (function(){
      // Scoped tabs: each tablist controls its sibling/descendant panels by matching aria-controls
      document.querySelectorAll('.tabs').forEach(tablist => {
        const buttons = Array.from(tablist.querySelectorAll('.tab-btn'));
        const root = tablist.parentElement || document;
        function activateButton(btn){
          const targetId = btn.getAttribute('aria-controls');
          const panels = Array.from(document.querySelectorAll('.tab-panel'))
            .filter(p => buttons.some(b => p.id === b.getAttribute('aria-controls')));
          buttons.forEach(b => b.setAttribute('aria-selected', String(b===btn)));
          panels.forEach(p => p.classList.toggle('active', p.id === targetId));
          // If switching topnav, ensure nested tablists activate first child
          if (tablist.id === 'topnav'){
            const section = document.getElementById(targetId);
            if (section){
              const innerLists = Array.from(section.querySelectorAll('.tabs'));
              innerLists.forEach(list=>{
                const first = list.querySelector('.tab-btn');
                if (first) first.click();
              });
              // Toggle compare button visibility based on visible section
              const evalSection = document.getElementById('evalPage');
              const cmpBtn = document.querySelector('.cmp-btn');
              if (cmpBtn) cmpBtn.style.display = (evalSection && evalSection.classList.contains('active')) ? 'block' : 'none';
            }
          }
        }
        buttons.forEach((btn, i) => {
          btn.addEventListener('click', () => activateButton(btn));
          if (i === 0) activateButton(btn);
        });
      });

      // Lightbox
      const lb = document.getElementById('lightbox');
      const lbImg = document.getElementById('lbImg');
      const lbClose = document.getElementById('lbClose');
      const zoomIn = document.getElementById('lbZoomIn');
      const zoomOut = document.getElementById('lbZoomOut');
      const reset = document.getElementById('lbReset');
      let scale = 1;
      let offsetX = 0, offsetY = 0;
      let dragging = false; let startX = 0; let startY = 0;

      function apply(){
        lbImg.style.transform = `translate(${offsetX}px, ${offsetY}px) scale(${scale})`;
      }
      function open(src){
        lbImg.src = src;
        scale = 1; offsetX = 0; offsetY = 0; apply();
        lb.classList.add('show');
        lb.setAttribute('aria-hidden','false');
      }
      function close(){
        lb.classList.remove('show');
        lb.setAttribute('aria-hidden','true');
      }

      document.addEventListener('click', (e)=>{
        const t = e.target;
        if (t instanceof HTMLImageElement && t.dataset.lb === '1') {
          open(t.src);
        }
      });
      lbClose.addEventListener('click', close);
      lb.addEventListener('click', (e)=>{ if (e.target === lb) close(); });
      document.addEventListener('keydown', (e)=>{ if (e.key==='Escape') close(); });

      zoomIn.addEventListener('click', ()=>{ scale = Math.min(10, scale*1.2); apply(); });
      zoomOut.addEventListener('click', ()=>{ scale = Math.max(0.2, scale/1.2); apply(); });
      reset.addEventListener('click', ()=>{ scale=1; offsetX=0; offsetY=0; apply(); });

      lbImg.addEventListener('mousedown', (e)=>{ dragging=true; startX=e.clientX-offsetX; startY=e.clientY-offsetY; lbImg.style.cursor='grabbing'; });
      window.addEventListener('mouseup', ()=>{ dragging=false; lbImg.style.cursor='grab'; });
      window.addEventListener('mousemove', (e)=>{ if(!dragging) return; offsetX=e.clientX-startX; offsetY=e.clientY-startY; apply(); });
      lbImg.addEventListener('wheel', (e)=>{ e.preventDefault(); const delta = Math.sign(e.deltaY); const factor = delta>0? 1/1.1 : 1.1; scale = Math.min(10, Math.max(0.2, scale*factor)); apply(); }, { passive: false });

      // Compare selection logic (max 2)
      const cmpModal = document.createElement('div');
      cmpModal.className = 'cmp-modal';
      cmpModal.innerHTML = '<button class="cmp-close">Close</button><div class="cmp-wrap" id="cmpWrap"></div>';
      document.body.appendChild(cmpModal);
      const cmpWrap = cmpModal.querySelector('#cmpWrap');
      const cmpClose = cmpModal.querySelector('.cmp-close');
      cmpClose.addEventListener('click', ()=> cmpModal.classList.remove('show'));
      const cmpBtn = document.createElement('button');
      cmpBtn.className = 'cmp-btn';
      cmpBtn.textContent = 'Compare (0/2)';
      cmpBtn.disabled = true;
      document.body.appendChild(cmpBtn);
      function updateCompareVisibility(){
        const evalSection = document.getElementById('evalPage');
        const isEval = !!(evalSection && evalSection.classList.contains('active'));
        cmpBtn.style.display = isEval ? 'block' : 'none';
      }
      updateCompareVisibility();

      const selected = [];
      function updateCmpBtn(){
        cmpBtn.textContent = `Compare (${selected.length}/2)`;
        cmpBtn.disabled = selected.length !== 2;
      }
      document.addEventListener('change', (e)=>{
        const t = e.target;
        if (!(t instanceof HTMLInputElement) || t.type !== 'checkbox' || !t.classList.contains('cmp-toggle')) return;
        const meta = { run: t.dataset.run, config: t.dataset.config, metric: t.dataset.metric, cat: t.dataset.cat || null };
        if (t.checked){
          if (selected.length >= 2){ t.checked = false; return; }
          if (selected.length === 1){
            // enforce same run and same metric category
            const a = selected[0];
            if (a.metric !== meta.metric){
              t.checked = false; alert('Select two entries from the same metric tab (e.g., Precision vs Confidence).'); return;
            }
            if (a.run !== meta.run){
              t.checked = false; alert('Select two configs from the same run to compare.'); return;
            }
            if ((a.cat||'') !== (meta.cat||'')){
              t.checked = false; alert('Select from the same categorical subtab.'); return;
            }
          }
          selected.push(meta);
        } else {
          const idx = selected.findIndex(x=> x.run===meta.run && x.config===meta.config && x.metric===meta.metric);
          if (idx>=0) selected.splice(idx,1);
        }
        updateCmpBtn();
      });

      function findPanelImgs(meta){
        // Locate the specific RUN panel via its aria-label (run name)
        const runPanel = Array.from(document.querySelectorAll('[id^="run-"].tab-panel'))
          .find(p => p.getAttribute('aria-label') === meta.run);
        if (!runPanel) return [];
        // Within that run, metric panel id is `${runPanel.id}-${meta.metric}`
        const metricPanelId = `${runPanel.id}-${meta.metric}`;
        const metricPanel = document.getElementById(metricPanelId) || runPanel;
        const cards = Array.from(metricPanel.querySelectorAll('.card'));
        const match = cards.find(c=> c.querySelector('h2')?.textContent === meta.config);
        return match ? Array.from(match.querySelectorAll('img')).map(i=> i.src) : [];
      }

      cmpBtn.addEventListener('click', ()=>{
        if (selected.length !== 2) return;
        cmpWrap.innerHTML = '';
        const [a,b] = selected;
        const aImgs = findPanelImgs(a);
        const bImgs = findPanelImgs(b);
        const maxLen = Math.max(aImgs.length, bImgs.length);
        for (let i=0;i<maxLen;i++){
          const row = document.createElement('div'); row.className='cmp-row';
          const ca = document.createElement('div'); ca.className='cmp-cell'; ca.innerHTML = `<div class="cmp-title">${a.run} / ${a.config} / ${a.metric}</div>`;
          const cb = document.createElement('div'); cb.className='cmp-cell'; cb.innerHTML = `<div class="cmp-title">${b.run} / ${b.config} / ${b.metric}</div>`;
          const ia = document.createElement('img'); ia.loading='lazy'; ia.src = aImgs[i]||''; ia.style.width='100%'; ia.style.height='auto';
          const ib = document.createElement('img'); ib.loading='lazy'; ib.src = bImgs[i]||''; ib.style.width='100%'; ib.style.height='auto';
          if (ia.src) ca.appendChild(ia); if (ib.src) cb.appendChild(ib);
          row.appendChild(ca); row.appendChild(cb);
          cmpWrap.appendChild(row);
        }
        cmpModal.classList.add('show');
      });

      // Add Run button: client-side load of *_config images
      const addRunBtn = document.getElementById('addRunBtn');
      const dirPicker = document.getElementById('dirPicker');
      const runTabBar = document.getElementById('run-tabs');
      const noRunsMsg = document.getElementById('noRunsMsg');
      if (addRunBtn && dirPicker) {
        addRunBtn.addEventListener('click', ()=> dirPicker.click());
        dirPicker.addEventListener('change', async (e)=>{
          const files = Array.from(dirPicker.files || []);
          if (!files.length) return;
          // Group by top-level selected folder name (run name)
          const runName = (files[0].webkitRelativePath || files[0].name || 'Run').split('/')[0] || 'Run';

          // Aliases (stems) and extensions from embedded payload
          const aliasEl = document.getElementById('aliasData');
          let alias = {PR:[],P:[],R:[],F1:[],CM:[],CM_N:[]}, exts = [".png"], canonical = {};
          try {
            const data = JSON.parse(aliasEl?.textContent || '{}');
            alias = data.alias || alias;
            exts = data.exts || exts;
            canonical = data.canonical || {};
          } catch {}

          // Index selected files by lowercased name for quick lookup
          const present = new Map();
          for (const f of files) {
            const name = (f.webkitRelativePath || f.name).split('/').pop();
            if (name) present.set(name.toLowerCase(), f);
          }

          // Build a map: configDirName -> { canonicalFilename -> blobUrl | Array<blobUrl> }
          const byConfig = new Map();
          for (const f of files) {
            const rel = f.webkitRelativePath || f.name;
            if (!rel) continue;
            const parts = rel.split('/');
            if (parts.length < 2) continue; // need at least <Run>/<Config>/...
            const cfg = parts[1];
            if (!cfg.endsWith('_config')) continue;

            // Try to resolve metric for this file by alias stems
            const base = parts[parts.length-1].toLowerCase();
            // Quick skip if it's not an allowed image type
            if (!exts.some(ext => base.endsWith(ext))) continue;

            let matchedKey = null; let canonicalName = null;
            for (const [key, stems] of Object.entries(alias)) {
              for (const stem of stems) {
                if (base.startsWith(stem.toLowerCase())) { matchedKey = key; break; }
              }
              if (matchedKey) { canonicalName = canonical[matchedKey]; break; }
            }
            const m = byConfig.get(cfg) || {};
            const url = URL.createObjectURL(f);
            if (matchedKey && canonicalName) {
              if (!m[canonicalName]) m[canonicalName] = url;
            } else {
              // Not a known metric image; add to galleries
              if (base.startsWith('val_batch') && base.includes('_labels')) {
                m.VAL_LABELS = m.VAL_LABELS || [];
                m.VAL_LABELS.push(url);
              } else if (base.startsWith('val_batch') && base.includes('_pred')) {
                m.VAL_PRED = m.VAL_PRED || [];
                m.VAL_PRED.push(url);
              } else {
                m.ALL_IMAGES = m.ALL_IMAGES || [];
                m.ALL_IMAGES.push(url);
              }
            }
            byConfig.set(cfg, m);
          }
          // Fallback: if normalized CM missing, reuse raw CM
          for (const [cfgName, m] of byConfig.entries()){
            if (!m[canonical.CM_N] && m[canonical.CM]) m[canonical.CM_N] = m[canonical.CM];
          }
          if (byConfig.size === 0) return;

          // Create a new run panel DOM mirroring server-rendered structure
          const container = document.body; // root
          const runIdx = document.querySelectorAll('[id^="run-"][role="tabpanel"]').length;

          noRunsMsg?.remove();
          const runBtn = document.createElement('button');
          runBtn.className = 'tab-btn';
          runBtn.setAttribute('role','tab');
          runBtn.setAttribute('aria-controls', `run-${runIdx}`);
          runBtn.setAttribute('aria-selected', 'false');
          runBtn.textContent = runName;
          runTabBar.appendChild(runBtn);
          const runClose = document.createElement('button');
          runClose.className = 'tab-btn';
          runClose.style.background = '#1f2937'; runClose.style.borderColor = '#374151'; runClose.style.color = '#9ca3af';
          runClose.setAttribute('aria-controls', `run-${runIdx}`);
          runClose.title = 'Remove';
          runClose.textContent = '×';
          runTabBar.appendChild(runClose);

          const runPanel = document.createElement('section');
          runPanel.id = `run-${runIdx}`;
          runPanel.className = 'tab-panel';
          runPanel.setAttribute('role','tabpanel');
          runPanel.setAttribute('aria-label', runName);

          // Metric tabs
          const metricTabBar = document.createElement('div');
          metricTabBar.className = 'tabs';
          metricTabBar.setAttribute('role','tablist');
          runPanel.appendChild(metricTabBar);

          const metrics = [
            ['tab-pr','Precision-Recall', canonical.PR],
            ['tab-p','Precision vs Confidence', canonical.P],
            ['tab-r','Recall vs Confidence', canonical.R],
            ['tab-f1','F1 vs Confidence', canonical.F1],
            ['tab-cm','Confusion Matrix', canonical.CM],
            ['tab-cm-norm','Confusion Matrix (Normalized)', canonical.CM_N],
            ['tab-val-labels','Validation Batches (Labels)','VAL_LABELS'],
            ['tab-val-pred','Validation Batches (Pred)','VAL_PRED'],
            ['tab-all-imgs','All Images','ALL_IMAGES']
          ];
          const metricPanels = [];
          metrics.forEach(([mid, label, fname], i) => {
            const btn = document.createElement('button');
            btn.className = 'tab-btn';
            btn.setAttribute('role','tab');
            const scopedId = `run-${runIdx}-${mid}`;
            btn.setAttribute('aria-controls', scopedId);
            btn.setAttribute('aria-selected', i===0 ? 'true' : 'false');
            btn.textContent = label;
            metricTabBar.appendChild(btn);

            const panel = document.createElement('section');
            panel.id = scopedId;
            panel.className = 'tab-panel' + (i===0 ? ' active' : '');
            panel.setAttribute('role','tabpanel');
            panel.setAttribute('aria-label', label);
            const grid = document.createElement('div');
            grid.className = 'configs';
            panel.appendChild(grid);

            Array.from(byConfig.keys()).sort((a,b)=> a.localeCompare(b)).forEach(cfgName => {
              const card = document.createElement('article');
              card.className = 'card';
              const h2 = document.createElement('h2');
              h2.textContent = cfgName;
              card.appendChild(h2);
              // compare checkbox
              const cmpHeader = document.createElement('div');
              cmpHeader.className = 'cmp-header';
              const cb = document.createElement('input');
              cb.type = 'checkbox'; cb.className = 'cmp-toggle';
              cb.dataset.run = runName; cb.dataset.config = cfgName; cb.dataset.metric = mid;
              cmpHeader.appendChild(cb);
              card.appendChild(cmpHeader);
              const wrap = document.createElement('div');
              wrap.className = 'imgwrap';
              const h3 = document.createElement('h3');
              h3.textContent = label;
              wrap.appendChild(h3);
              const srcMap = byConfig.get(cfgName) || {};
              const entry = srcMap[fname];
              if (Array.isArray(entry) && entry.length){
                const gallery = document.createElement('div');
                gallery.style.display = 'grid';
                gallery.style.gridTemplateColumns = 'repeat(2, 1fr)';
                gallery.style.gap = '10px';
                entry.forEach(u => {
                  const w = document.createElement('div'); w.className='imgwrap';
                  const img = document.createElement('img'); img.loading='lazy'; img.src=u; img.alt=label; img.dataset.lb='1';
                  w.appendChild(img); gallery.appendChild(w);
                });
                card.appendChild(gallery);
              } else if (typeof entry === 'string') {
                const img = document.createElement('img');
                img.loading = 'lazy';
                img.src = entry;
                img.alt = label;
                img.dataset.lb = '1';
                wrap.appendChild(img);
              } else {
                const missing = document.createElement('div');
                missing.className = 'missing';
                missing.textContent = 'Missing';
                wrap.appendChild(missing);
              }
              card.appendChild(wrap);
              grid.appendChild(card);
            });

            metricPanels.push(panel);
            runPanel.appendChild(panel);
          });

          runTabBar.parentElement.insertBefore(runPanel, runTabBar.nextSibling);

          function initTablist(tablist){
            const buttons = Array.from(tablist.querySelectorAll('.tab-btn'));
            const root = tablist.parentElement || document;
            function activateButton(btn){
              const targetId = btn.getAttribute('aria-controls');
              const panels = Array.from(root.querySelectorAll('.tab-panel'))
                .filter(p => p.id && buttons.some(b => p.id === b.getAttribute('aria-controls')));
              buttons.forEach(b => b.setAttribute('aria-selected', String(b===btn)));
              panels.forEach(p => p.classList.toggle('active', p.id === targetId));
            }
            buttons.forEach((b, i)=>{
              b.addEventListener('click', ()=> activateButton(b));
              if (i===0) activateButton(b);
            });
          }
          initTablist(runTabBar);
          initTablist(metricTabBar);

          // Bind remove for this run
          runClose.addEventListener('click', ()=>{
            const targetId = runClose.getAttribute('aria-controls');
            const panel = document.getElementById(targetId);
            runBtn.remove(); runClose.remove(); panel?.remove();
            const first = runTabBar.querySelector('.tab-btn');
            first?.dispatchEvent(new Event('click'));
            if (!runTabBar.querySelector('.tab-btn')) {
              const msg = document.createElement('div');
              msg.id = 'noRunsMsg'; msg.className = 'missing';
              msg.textContent = 'No runs loaded. Use + Add Run (folder) to load results.';
              runTabBar.parentElement?.insertBefore(msg, runTabBar.nextSibling);
            }
          });
        });
      }
    })();
    """
    parts.append("  <script>" + js + "</script>")
    # Append JSON viewer logic (kept separate to avoid interfering with existing features)
    js2 = """
      // JSON viewer logic
      const jPicker = document.getElementById('jsonPicker');
      const jBeaut = document.getElementById('jsonBeautify');
      const jCopy = document.getElementById('jsonCopy');
      const jSearch = document.getElementById('jsonSearch');
      const jPrev = document.getElementById('jsonPrev');
      const jNext = document.getElementById('jsonNext');
      const jStatus = document.getElementById('jsonStatus');
      const jBox = document.getElementById('jsonBox');

      // Source explorer elements
      const sPicker = document.getElementById('srcPicker'); // may be null (folder upload removed)
      const sOpen = document.getElementById('srcOpen');
      const sPath = document.getElementById('srcPath');
      const sStatus = document.getElementById('srcStatus');
      const codeBox = document.getElementById('codeBox');
      const sFilePicker = document.getElementById('srcFilePicker');
      const sTabs = document.getElementById('srcTabs');

      // Large-file friendly settings
      const MAX_PREVIEW_CHARS = 1000000; // 1MB preview to keep UI responsive
      const SNIPPET_RADIUS = 20000; // render +-20k chars around match

      let jActive = '';     // current text used for search/render (raw or pretty)
      let jMode = 'raw';    // 'raw' | 'pretty'

      let jMatches = [];
      let jIdx = -1;

      function setJStatus(msg){ if (jStatus) jStatus.textContent = msg || ''; }
      function setSStatus(msg){ if (sStatus) sStatus.textContent = msg || ''; }

      function beautifyText(txt){
        try { return JSON.stringify(JSON.parse(txt), null, 2); }
        catch { return txt; }
      }

      function renderPreviewAround(startIndex){
        if (!jBox) return;
        const total = jActive.length;
        if (typeof startIndex !== 'number' || isNaN(startIndex)) startIndex = 0;
        const begin = Math.max(0, startIndex - SNIPPET_RADIUS);
        const end = Math.min(total, startIndex + SNIPPET_RADIUS);
        const slice = jActive.slice(begin, end);
        jBox.innerHTML = buildLinkifiedHTML(slice);
        if (begin > 0 || end < total) {
          setJStatus(`Showing ${slice.length.toLocaleString()} of ${total.toLocaleString()} chars (${jMode}). Use search or Beautify to navigate. (` + (begin>0?`…`:'') + `${begin}-${end}` + (end<total?`…`:'') + `)`);
        } else {
          setJStatus(`Showing full ${total.toLocaleString()} chars (${jMode})`);
        }
      }

      function loadJsonText(txt){
        jActive = txt || '';
        jMode = 'raw';
        // Render preview only
        renderPreviewAround(0);
        runJSearch();
      }

      jPicker?.addEventListener('change', async ()=>{
        const f = jPicker.files?.[0]; if(!f) return;
        setJStatus(`Loading ${f.name}...`);
        try { const txt = await f.text(); loadJsonText(txt); }
        catch { setJStatus('Failed to load file'); }
      });
      jBeaut?.addEventListener('click', ()=>{
        if (!jActive) return;
        setJStatus('Beautifying...');
        setTimeout(()=>{
          jActive = beautifyText(jActive);
          jMode = 'pretty';
          renderPreviewAround(0);
          runJSearch();
        }, 0);
      });
      jCopy?.addEventListener('click', async ()=>{ try { await navigator.clipboard.writeText(jBox?.textContent||''); setJStatus('Copied'); setTimeout(()=>setJStatus(''), 1200);} catch{} });

      function clearJHighlights(){
        const txt = jBox?.textContent || jActive || '';
        if (jBox) jBox.innerHTML = buildLinkifiedHTML(txt);
      }
      function highlightJAt(start, end){
        if (!jBox) return;
        // Render a small snippet around the match to avoid reflow on huge docs
        const begin = Math.max(0, start - SNIPPET_RADIUS);
        const afterPos = Math.min(jActive.length, end + SNIPPET_RADIUS);
        const before = jActive.slice(begin, start);
        const hit = jActive.slice(start, end);
        const after = jActive.slice(end, afterPos);
        jBox.innerHTML = '';
        const notePrefix = document.createElement('div');
        if (begin > 0 || afterPos < jActive.length) {
          notePrefix.style.color = '#9ca3af';
          notePrefix.style.fontSize = '11px';
          notePrefix.textContent = `Snippet ${begin}-${afterPos} of ${jActive.length}`;
          jBox.appendChild(notePrefix);
        }
        const spanBefore = document.createElement('span');
        spanBefore.innerHTML = buildLinkifiedHTML(before);
        const mark = document.createElement('mark');
        mark.style.background = '#3b82f6';
        mark.style.color = '#0b0f14';
        mark.innerHTML = buildLinkifiedHTML(hit);
        const spanAfter = document.createElement('span');
        spanAfter.innerHTML = buildLinkifiedHTML(after);
        jBox.appendChild(spanBefore); jBox.appendChild(mark); jBox.appendChild(spanAfter);
        mark.scrollIntoView({block:'center'});
      }
      function runJSearch(){
        const q = (jSearch?.value||'');
        jMatches = []; jIdx = -1; clearJHighlights();
        if (!q) { setJStatus(''); return; }
        const text = jActive || '';
        try {
          const rx = new RegExp(q.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\$&'),'gi');
          let m; while ((m = rx.exec(text))){ jMatches.push([m.index, m.index+m[0].length]); if (jMatches.length>5000) break; }
          if (jMatches.length){ jIdx = 0; const [s,e] = jMatches[0]; highlightJAt(s,e); setJStatus(`${jMatches.length} match(es)`); }
          else setJStatus('0 matches');
        } catch { setJStatus('Invalid regex'); }
      }
      jSearch?.addEventListener('input', ()=> runJSearch());
      jNext?.addEventListener('click', ()=>{ if(!jMatches.length) return; jIdx=(jIdx+1)%jMatches.length; const [s,e]=jMatches[jIdx]; highlightJAt(s,e); });
      jPrev?.addEventListener('click', ()=>{ if(!jMatches.length) return; jIdx=(jIdx-1+jMatches.length)%jMatches.length; const [s,e]=jMatches[jIdx]; highlightJAt(s,e); });

      // ---- Source Explorer ----
      const sourceFiles = new Map(); // rel path (lowercase) -> File

      function normalizePath(p){
        if (!p) return '';
        p = String(p).replace(/\\\\/g,'/');
        // Try to keep from project roots such as ultralytics/, torch/, etc.
        const anchors = ['ultralytics/','torch/','/site-packages/'];
        let best = p;
        for (const a of anchors){
          const idx = p.toLowerCase().indexOf(a);
          if (idx >= 0) { best = p.slice(idx); break; }
        }
        return best.replace(/^\/+/, '');
      }

      sPicker?.addEventListener('change', ()=>{
        const files = Array.from(sPicker.files||[]);
        let added = 0;
        for (const f of files){
          const rel = normalizePath(f.webkitRelativePath || f.name);
          if (!rel) continue;
          const key = rel.toLowerCase();
          if (!sourceFiles.has(key)) { sourceFiles.set(key, f); added++; }
        }
        setSStatus(`Indexed ${added} files (total ${sourceFiles.size})`);
      });

      async function readLocalFile(relPath){
        const key = normalizePath(relPath).toLowerCase();
        const f = sourceFiles.get(key);
        if (!f) return null;
        try { return await f.text(); } catch { return null; }
      }

      async function fetchFromGithub(relPath){
        // Primary: ultralytics main repo
        const ulBase = 'https://raw.githubusercontent.com/ultralytics/ultralytics/main/';
        // For torch, try pytorch repo (best effort)
        const torchBase = 'https://raw.githubusercontent.com/pytorch/pytorch/main/';
        let url = '';
        const norm = normalizePath(relPath);
        if (norm.startsWith('ultralytics/')) url = ulBase + norm;
        else if (norm.startsWith('torch/')) url = torchBase + norm.slice('torch/'.length);
        else return null;
        try {
          const res = await fetch(url);
          if (!res.ok) return null;
          return await res.text();
        } catch { return null; }
      }

      function renderCode(text, hlLine){
        if (!codeBox) return;
        codeBox.innerHTML = '';
        const lines = (text||'').split('\\n');
        lines.forEach((ln, i)=>{
          const row = document.createElement('div');
          row.className = 'code-line' + ((i+1)===hlLine ? ' code-hit' : '');
          const gut = document.createElement('span'); gut.className = 'code-gutter'; gut.textContent = String(i+1).padStart(4,' ') + ' | ';
          const txt = document.createElement('span'); txt.textContent = ln || '';
          row.appendChild(gut); row.appendChild(txt);
          codeBox.appendChild(row);
        });
        if (hlLine && hlLine>=1 && hlLine<=lines.length){
          const el = codeBox.children[hlLine-1]; el?.scrollIntoView({block:'center'});
        }
      }

      async function openSource(pathSpec){
        if (!pathSpec) return;
        let path = pathSpec; let line = null;
        const m = String(pathSpec).match(/^(.*?):(\d+)$/);
        if (m){ path = m[1]; line = parseInt(m[2],10)||null; }
        const normPath = normalizePath(path);
        if (!/\.py$/i.test(normPath)) { setSStatus('Only Python files (.py) are supported'); return; }
        if (normPath.startsWith('torch/')) { setSStatus('Torch source fetch disabled'); return; }
        setSStatus('Opening...');
        let text = await fetchFromGithub(path);
        if (text==null){ setSStatus('Not found on GitHub'); return; }
        renderCode(text, line);
        setSStatus(normalizePath(path) + (line?`:${line}`:''));
      }

      sOpen?.addEventListener('click', ()=> openSource(sPath?.value||''));

      // Local .py browse with subtabs
      sFilePicker?.addEventListener('change', async ()=>{
        const files = Array.from(sFilePicker.files||[]);
        if (!files.length) return;
        files.forEach((f, idx)=>{
          if (!/\.py$/i.test(f.name)) return;
          const btn = document.createElement('button');
          btn.className = 'tab-btn';
          btn.setAttribute('role','tab');
          btn.setAttribute('aria-selected','false');
          btn.textContent = f.name;
          btn.addEventListener('click', async ()=>{
            setSStatus(`Opening ${f.name}...`);
            try {
              const txt = await f.text();
              renderCode(txt, null);
              Array.from(sTabs.children).forEach(b=>b.setAttribute('aria-selected','false'));
              btn.setAttribute('aria-selected','true');
              setSStatus(f.name);
            } catch { setSStatus('Failed to read local file'); }
          });
          sTabs?.appendChild(btn);
          if (idx === 0) btn.click();
        });
      });

      function extractPathAndLineFromContext(){
        const mark = jBox?.querySelector('mark');
        const context = mark ? mark.parentElement?.textContent || '' : jBox?.textContent || '';
        return findFirstPyRef(context);
      }

      function findFirstPyRef(text){
        if (!text) return null;
        const norm = String(text).replace(/\\\\/g,'/');
        // Accept absolute paths, site-packages, or repo-relative ultralytics/... patterns
        const patterns = [
          /(ultralytics\/[\w\/\.-]+\.py):(\d+)/i,
          /(.*site-packages\/.+?\/(?:ultralytics|torch)\/[\w\/\.-]+\.py):(\d+)/i,
          /(.*ultralytics\/[\w\/\.-]+\.py):(\d+)/i,
          /(torch\/[\w\/\.-]+\.py):(\d+)/i,
          /(.*torch\/[\w\/\.-]+\.py):(\d+)/i
        ];
        for (const rx of patterns){
          const m = rx.exec(norm);
          if (m) {
            const src = m[1];
            const low = src.toLowerCase();
            const relIdxU = low.indexOf('ultralytics/');
            const relIdxT = low.indexOf('torch/');
            const relIdx = relIdxU >= 0 ? relIdxU : relIdxT;
            const rel = relIdx >= 0 ? src.slice(relIdx) : src;
            return `${rel}:${m[2]}`;
          }
        }
        return null;
      }

      function escapeHtml(s){
        return String(s)
          .replace(/&/g,'&amp;')
          .replace(/</g,'&lt;')
          .replace(/>/g,'&gt;')
          .replace(/\"/g,'&quot;')
          .replace(/'/g,'&#39;');
      }
      function buildLinkifiedHTML(text){
        const srcPatterns = [
          /(ultralytics\/[\w\/\.-]+\.py):(\d+)/gi,
          /(torch\/[\w\/\.-]+\.py):(\d+)/gi,
          /(.*site-packages\/.+?\/(?:ultralytics|torch)\/[\w\/\.-]+\.py):(\d+)/gi,
          /(.*ultralytics\/[\w\/\.-]+\.py):(\d+)/gi,
          /(.*torch\/[\w\/\.-]+\.py):(\d+)/gi
        ];
        let html = '';
        let cursor = 0;
        const norm = String(text).replace(/\\\\/g,'/');
        function appendRef(src, line, display){
          const relIdx = src.toLowerCase().indexOf('ultralytics/');
          const rel = relIdx >= 0 ? src.slice(relIdx) : src;
          const ref = `${rel}:${line}`;
          return `<span class="src-ref" data-ref="${escapeHtml(ref)}">${escapeHtml(display)}</span>`;
        }
        // Merge all matches by scanning with a unified regex
        const master = /(ultralytics\/[\w\/\.-]+\.py):(\d+)|(torch\/[\w\/\.-]+\.py):(\d+)|(.*site-packages\/.+?\/(?:ultralytics|torch)\/[\w\/\.-]+\.py):(\d+)|(.*ultralytics\/[\w\/\.-]+\.py):(\d+)|(.*torch\/[\w\/\.-]+\.py):(\d+)/gi;
        let m;
        while ((m = master.exec(norm))) {
          const idx = m.index;
          html += escapeHtml(norm.slice(cursor, idx));
          if (m[1]) html += appendRef(m[1], m[2], m[0]);
          else if (m[3]) html += appendRef(m[3], m[4], m[0]);
          else if (m[5]) html += appendRef(m[5], m[6], m[0]);
          else if (m[7]) html += appendRef(m[7], m[8], m[0]);
          else if (m[9]) html += appendRef(m[9], m[10], m[0]);
          cursor = master.lastIndex;
        }
        html += escapeHtml(norm.slice(cursor));
        return html;
      }
      const _origHighlight = highlightJAt;
      highlightJAt = function(start, end){
        _origHighlight(start, end);
        const guess = extractPathAndLineFromContext();
        if (guess && sPath) sPath.value = guess;
      }

      // Interactive JSON viewer: click/hover to drive Source Explorer
      jBox?.addEventListener('click', (e)=>{
        const refEl = (e.target && e.target.closest) ? e.target.closest('.src-ref') : null;
        if (refEl) {
          const ref = refEl.getAttribute('data-ref');
          if (ref) { if (sPath) sPath.value = ref; openSource(ref); }
          return;
        }
        const txt = jBox?.textContent || '';
        const guess = findFirstPyRef(txt);
        if (guess){ if (sPath) sPath.value = guess; if (e.shiftKey) openSource(guess); }
      });
      jBox?.addEventListener('mouseenter', ()=>{
        const txt = jBox?.textContent || '';
        const guess = findFirstPyRef(txt);
        if (guess) setJStatus(`Detected source ref: ${guess} (Click to open)`);
      });
      jBox?.addEventListener('mouseleave', ()=>{ setJStatus(''); });
    """
    parts.append("  <script>" + js2 + "</script>")
    parts.append("</body>")
    parts.append("</html>")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLO metrics HTML report from *_config folders.")
    parser.add_argument("--root", type=str, default=os.getcwd(), help="Root directory to scan (default: cwd)")
    parser.add_argument("--output", type=str, default="YOLOmetrics_report.html", help="Output HTML filename or path")
    parser.add_argument("--title", type=str, default="YOLOmetrics Report", help="Report title")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    runs = discover_runs(root)
    html = generate_html_multi(runs, title=args.title)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()


