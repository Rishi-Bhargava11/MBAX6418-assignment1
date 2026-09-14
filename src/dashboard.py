"""Results dashboard generator (Steps 3-7).

Reads a scored-results JSON (schema: {"meta": {...}, "summary": {...},
"rows": [...]}) and renders a single self-contained HTML page with everything
baked in:

  * headline numbers (accuracy, correct / mismatched)
  * star-rating distribution (descriptive)
  * confusion matrix (truth vs predicted), driven by the run's classes
  * per-class accuracy
  * class distribution (truth vs predicted)
  * emotion comparison (LLM vs NRC), when rows carry llm_emotion/nrc_emotion
  * an explorable review table with live filtering by class and verdict

Works fully offline — no CDN, no server. Theme is driven by CSS custom
properties so the host can recolor it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from . import config

DEFAULT_CLASSES = ["POSITIVE", "NEGATIVE"]
EMOTION_PALETTE = {
    "anger": "#d93025", "anticipation": "#8e6b12", "disgust": "#6a0dad",
    "fear": "#7a4b94", "joy": "#0f9d58", "sadness": "#1a56db",
    "surprise": "#f08600", "trust": "#2563eb",
}


def _kpi_html(rows, summary):
    n = summary["n"]
    acc = summary["accuracy"]
    wrong = n - summary["correct"]
    return f"""
    <div class="kpi-row">
      <div class="kpi"><div class="kpi-value">{n}</div><div class="kpi-label">Reviews scored</div></div>
      <div class="kpi accent"><div class="kpi-value">{acc:.1%}</div><div class="kpi-label">Overall accuracy</div></div>
      <div class="kpi"><div class="kpi-value">{summary['correct']}</div><div class="kpi-label">Correct</div></div>
      <div class="kpi kpi-bad"><div class="kpi-value">{wrong}</div><div class="kpi-label">Mismatched</div></div>
    </div>"""


def _star_dist_html(rows):
    counts = Counter(int(round(r["rating"])) for r in rows)
    mx = max(counts.values()) if counts else 1
    bars = ""
    for star in range(1, 6):
        c = counts.get(star, 0)
        bars += f"""
        <div class="bar-row">
          <span class="bar-label">{'★'*star}</span>
          <div class="bar-track"><div class="bar-fill star s{star}" style="width:{c/mx*100:.1f}%"></div></div>
          <span class="bar-val">{c}</span>
        </div>"""
    return f"""
    <div class="card">
      <h3>Star-rating distribution</h3>
      <p class="sub">How reviews in this set spread across 1–5 stars.</p>
      {bars}
    </div>"""


def _confusion_html(rows, classes):
    value = {(t, p): 0 for t in classes for p in classes}
    seen = set()
    for r in rows:
        t = r["truth"] if r["truth"] in classes else None
        p = r["predicted"] if r["predicted"] in classes else None
        if t is None:
            t = "OTHER"; p = p or "OTHER"
        if (t, p) not in value:
            value.setdefault((t, p), 0)
            seen.add((t, p))
        value[(t, p)] += 1
    head = "".join(f"<th>→ {p[:4]}</th>" for p in classes)
    body = ""
    for t in classes:
        cells = ""
        for p in classes:
            v = value.get((t, p), 0)
            cls = "hit" if t == p else "miss"
            cells += f'<td><div class="cm-cell {cls}"><b>{v}</b><span>{"match" if t==p else "mis"}</span></div></td>'
        body += f"<tr><th>{t} (truth)</th>{cells}</tr>"
    return f"""
    <div class="card">
      <h3>Confusion matrix</h3>
      <p class="sub">Truth rows vs predicted columns. Off-diagonal cells show how errors move.</p>
      <table class="cm"><thead><tr><th></th>{head}</tr></thead><tbody>{body}</tbody></table>
    </div>"""


def _classacc_html(summary, classes):
    bars = ""
    for cls in classes:
        c = summary["per_class"].get(cls, {"n": 0, "correct": 0, "accuracy": 0.0})
        pct = c["accuracy"]
        bars += f"""
        <div class="bar-row">
          <span class="bar-label">{cls}</span>
          <div class="bar-track"><div class="bar-fill classy" style="width:{pct*100:.1f}%"></div></div>
          <span class="bar-val">{c['correct']}/{c['n']} · {pct:.0%}</span>
        </div>"""
    return f"""
    <div class="card">
      <h3>Per-class accuracy</h3>
      <p class="sub">How often the model is right on each class.</p>
      {bars}
    </div>"""


def _distribution_html(rows, classes):
    truth = Counter(r["truth"] for r in rows)
    pred = Counter(r["predicted"] for r in rows)
    mx = max([truth.get(l, 0) for l in classes] +
             [pred.get(l, 0) for l in classes] + [1])
    charts = ""
    for meta, counter, tone in (("Truth (from stars)", truth, "truth"),
                                ("Predicted", pred, "pred")):
        bars = ""
        for l in classes:
            cnt = counter.get(l, 0)
            bars += f"""
            <div class="bar-row">
              <span class="bar-label">{l}</span>
              <div class="bar-track"><div class="bar-fill {tone}" style="width:{cnt/mx*100:.1f}%"></div></div>
              <span class="bar-val">{cnt}</span>
            </div>"""
        charts += f'<div class="dist"><h4>{meta}</h4>{bars}</div>'
    return f"""
    <div class="card">
      <h3>Class distribution</h3>
      <p class="sub">Truth vs predicted side by side.</p>
      <div class="dist-grid">{charts}</div>
    </div>"""


def _emotion_html(rows):
    llm = Counter(r.get("llm_emotion") for r in rows if r.get("llm_emotion"))
    nrc = Counter(r.get("nrc_emotion") for r in rows if r.get("nrc_emotion"))
    both = [r for r in rows if r.get("llm_emotion") and r.get("nrc_emotion")]
    agree = sum(1 for r in both if r["llm_emotion"] == r["nrc_emotion"])
    rate = (agree / len(both) * 100) if both else 0
    all_emo = sorted(set(list(llm) + list(nrc)))
    mx = max(list(llm.values()) + list(nrc.values()) + [1])
    emo_bars = ""
    for e in all_emo:
        col = EMOTION_PALETTE.get(e, "#999")
        emo_bars += f"""
        <div class="bar-row">
          <span class="bar-label emo" style="color:{col}">● {e}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width:{llm.get(e,0)/mx*100:.1f}%;background:{col};opacity:.9"></div>
          </div>
          <span class="bar-val">LLM {llm.get(e,0)} · NRC {nrc.get(e,0)}</span>
        </div>"""
    return f"""
    <div class="card">
      <h3>Primary emotion — LLM vs word list</h3>
      <p class="sub">Agreement: {agree}/{len(both)} reviews ({rate:.0f}%) where both methods had an answer.
      Dots show predicted emotion; bars are LLM counts; NRC counts in the label.</p>
      {emo_bars}
    </div>"""


def _table_html(rows, classes):
    body = ""
    for i, r in enumerate(rows):
        is_right = r["predicted"] == r["truth"]
        cls = r["truth"] if r["truth"] in classes else "OTHER"
        emo = r.get("llm_emotion") or "—"
        nrc = r.get("nrc_emotion") or "—"
        body += f"""
        <tr class="{'' if is_right else 'row-wrong'}">
          <td class="idx">{i}</td>
          <td>{'⭐'*int(round(r['rating']))}</td>
          <td class="title">{r['title']}</td>
          <td class="text">{r['text']}</td>
          <td><span class="tag truth">{r['truth']}</span></td>
          <td><span class="tag {"ok" if is_right else "bad"}">{r['predicted']}</span></td>
          <td><span class="tag {"hit" if is_right else "miss"}">{'match' if is_right else 'mismatch'}</span></td>
          <td class="emo">{emo}</td>
          <td class="emo">{nrc}</td>
        </tr>"""
    return f"""
    <div class="card table-card">
      <div class="table-head">
        <h3>Review detail</h3>
        <div class="controls">
          <label>Class
            <select id="fclass">
              <option value="all">All</option>
              {''.join(f'<option value="{c}">{c}</option>' for c in classes)}
            </select>
          </label>
          <label>Verdict
            <select id="fverdict">
              <option value="all">All</option>
              <option value="match">Matches</option>
              <option value="mismatch">Mismatches</option>
            </select>
          </label>
          <span class="count" id="count"></span>
        </div>
      </div>
      <div class="tbl-scroll">
        <table class="rev">
          <thead><tr><th>#</th><th>Rating</th><th>Title</th><th>Text</th><th>Truth</th><th>Model</th><th>Verdict</th><th>Emotion (LLM)</th><th>Emotion (NRC)</th></tr></thead>
          <tbody id="tbody">{body}</tbody>
        </table>
      </div>
    </div>"""


def build_html(payload: dict) -> str:
    summary = payload["summary"]
    rows = payload["rows"]
    classes = payload.get("meta", {}).get("classes") or DEFAULT_CLASSES
    total = len(rows)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sentiment Results · MBAX 6418</title>
<style>
:root {{
  --bg:#fbfaf7; --surface:#fff; --ink:#1c1b18; --muted:#7a7468; --border:#e7e2d7;
  --accent:#2563eb; --accent-ink:#fff; --pos:#0f9d58; --neg:#d93025; --warn:#f4b400;
  --radius:14px; --shadow:0 1px 3px rgba(28,27,24,.06),0 8px 24px rgba(28,27,24,.06);
  --font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);font-family:var(--font);-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1220px;margin:0 auto;padding:40px 24px 80px}}
header.hero{{margin-bottom:28px}}
header.hero .eyebrow{{text-transform:uppercase;letter-spacing:.16em;font-size:12px;color:var(--accent);font-weight:600}}
header.hero h1{{font-size:30px;margin:8px 0 6px;letter-spacing:-.02em}}
header.hero p{{color:var(--muted);margin:0;max-width:66ch}}
.kpi-row{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
.kpi{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:20px}}
.kpi .kpi-value{{font-size:34px;font-weight:700;letter-spacing:-.02em;line-height:1}}
.kpi .kpi-label{{color:var(--muted);font-size:13px;margin-top:8px}}
.kpi.accent{{background:var(--accent);color:var(--accent-ink);border:none}}
.kpi.accent .kpi-label{{color:rgba(255,255,255,.8)}}
.kpi.kpi-bad .kpi-value{{color:var(--neg)}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);box-shadow:var(--shadow);padding:22px}}
.card h3{{margin:0 0 4px;font-size:16px}}
.card p.sub{{color:var(--muted);font-size:13px;margin:0 0 16px}}
table.cm{{width:100%;border-collapse:collapse}}
table.cm th,table.cm td{{padding:10px;text-align:center;font-size:13px}}
table.cm th{{color:var(--muted);font-weight:500}}
.cm-cell{{border-radius:10px;padding:12px;color:var(--ink)}}
.cm-cell b{{font-size:22px;display:block}}
.cm-cell span{{font-size:10px;text-transform:uppercase;letter-spacing:.05em}}
.cm-cell.hit{{background:color-mix(in srgb,var(--pos) 14%,transparent)}}
.cm-cell.miss{{background:color-mix(in srgb,var(--neg) 12%,transparent)}}
.bar-row{{display:grid;grid-template-columns:150px 1fr 130px;gap:12px;align-items:center;margin-bottom:12px}}
.bar-label{{font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.bar-track{{background:#f0ede4;border-radius:999px;height:16px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:999px}}
.bar-fill.truth{{background:var(--pos)}}
.bar-fill.pred{{background:var(--warn)}}
.bar-fill.classy{{background:var(--accent)}}
.bar-fill.star{{background:var(--warn)}}
.bar-fill.star.s1{{background:#d93025}}.bar-fill.star.s2{{background:#e67c2f}}
.bar-fill.star.s3{{background:#f4b400}}.bar-fill.star.s4{{background:#7da04a}}
.bar-fill.star.s5{{background:#0f9d58}}
.bar-val{{font-size:12px;color:var(--muted);text-align:right;white-space:nowrap}}
.dist-grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.dist h4{{margin:0 0 12px;font-size:14px;color:var(--muted);font-weight:600}}
.bar-label.emo{{font-weight:500}}
.table-card{{padding:0;overflow:hidden}}
.table-head{{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:20px 22px;border-bottom:1px solid var(--border);flex-wrap:wrap}}
.table-head h3{{margin:0}}
.controls{{display:flex;align-items:center;gap:14px;flex-wrap:wrap}}
.controls label{{font-size:13px;color:var(--muted)}}
.controls select{{font:inherit;padding:6px 10px;border:1px solid var(--border);border-radius:8px;background:var(--surface);color:var(--ink)}}
.count{{font-size:13px;font-weight:600;background:#f0ede4;padding:4px 10px;border-radius:999px}}
.tbl-scroll{{max-height:560px;overflow:auto}}
table.rev{{width:100%;border-collapse:collapse;font-size:13px}}
table.rev th{{position:sticky;top:0;background:var(--surface);text-align:left;padding:10px 12px;font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);border-bottom:1px solid var(--border);z-index:1}}
table.rev td{{padding:11px 12px;border-bottom:1px solid var(--border);vertical-align:top}}
table.rev .idx{{color:var(--muted)}}
table.rev .title{{font-weight:600;max-width:190px}}
table.rev .text{{color:var(--ink);max-width:400px;line-height:1.45}}
table.rev .emo{{color:var(--muted);font-size:12px}}
.row-wrong{{background:color-mix(in srgb,var(--neg) 5%,transparent)}}
.tag{{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}}
.tag.truth{{background:#f0ede4;color:var(--muted)}}
.tag.ok{{color:var(--pos)}}.tag.bad{{color:var(--neg)}}
.tag.hit{{background:color-mix(in srgb,var(--pos) 14%,transparent);color:var(--pos)}}
.tag.miss{{background:color-mix(in srgb,var(--neg) 13%,transparent);color:var(--neg)}}
@media (max-width:900px){{
 .grid,.dist-grid{{grid-template-columns:1fr}}.kpi-row{{grid-template-columns:repeat(2,1fr)}}
 .bar-row{{grid-template-columns:90px 1fr 100px}}
}}
</style></head><body><div class="wrap">
<header class="hero">
  <div class="eyebrow">MBAX 6418 · Assignment 1 · {payload.get('meta',{}).get('mode','')} run</div>
  <h1>Sentiment &amp; emotion classification results</h1>
  <p>Classifier on the Amazon Reviews '23 Gift Cards category via the course OpenAI-compatible endpoint,
  scored against the star rating ({summary['n']} reviews, {len(classes)} classes). The model never saw ratings.</p>
</header>

{_kpi_html(rows, summary)}

<div class="grid">
  {_star_dist_html(rows)}
  {_confusion_html(rows, classes)}
</div>
<div class="grid">
  {_classacc_html(summary, classes)}
  {_distribution_html(rows, classes)}
</div>

{_emotion_html(rows)}

{_table_html(rows, classes)}
</div>
<script>
(function(){{
  const fc = document.getElementById('fclass');
  const fv = document.getElementById('fverdict');
  const tbody = document.getElementById('tbody');
  const count = document.getElementById('count');
  const rows = Array.from(tbody.querySelectorAll('tr'));
  const isMatch = tr => !tr.classList.contains('row-wrong');
  function clsOf(tr){{ return tr.querySelector('.tag.truth').textContent; }}
  function apply(){{
    const cc = fc.value, vv = fv.value;
    const show = rows.filter(tr => {{
      const okc = cc==='all' || clsOf(tr)===cc;
      const okv = vv==='all' ? true : (vv==='match' ? isMatch(tr) : !isMatch(tr));
      return okc && okv;
    }});
    rows.forEach(tr => tr.style.display = show.includes(tr) ? '' : 'none');
    count.textContent = show.length + ' / ' + rows.length + ' reviews';
  }}
  apply();
  [fc, fv].forEach(el => {{ el.addEventListener('change', apply); el.addEventListener('input', apply); }});
}})();
</script>
</body></html>"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.dashboard")
    parser.add_argument("--input", type=str,
                        default=str(config.RESULTS_DIR / "balanced_50_per_class.json"))
    parser.add_argument("--output", type=str,
                        default=str(config.RESULTS_DIR / "dashboard.html"))
    args = parser.parse_args(argv)
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    html = build_html(payload)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {len(html):,} bytes -> {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
