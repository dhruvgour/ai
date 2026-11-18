#!/usr/bin/env python
"""Build the Dhruv AI tutorial.

    .venv/bin/python build.py                 # everything -> PDF + animated HTML
    .venv/bin/python build.py --only L02 L05  # only these chapters (fast check, no PDF)
    .venv/bin/python build.py --force         # ignore the render cache

Each chapter is a folder ``chapters/Lxx_name/`` holding
    theory.md   Hindi theory (Markdown, $inline$ and $$display$$ math)
    anims.py    five or more 3D animations (see lib/anim.py)
    example.py  runnable Python example; may save PNG figures in the working directory
"""
from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent
CHAPTERS = ROOT / "chapters"
OUT = ROOT / "out"
ANIM = OUT / "anim"
EXAMPLES = OUT / "examples"
CONTACT = OUT / "contact"
CACHE = OUT / "cache.json"
KATEX = ROOT / "vendor" / "katex" / "package" / "dist"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PDF_NAME = "Dhruv_AI_tutorial.pdf"
HTML_NAME = "index.html"
MIN_ANIMS = 5
MAX_OUTPUT_LINES = 70

sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------- chapters

def chapter_dirs(only=None):
    dirs = sorted(d for d in CHAPTERS.iterdir() if d.is_dir() and re.match(r"L\d\d_", d.name))
    if only:
        dirs = [d for d in dirs if any(d.name.startswith(o) for o in only)]
    return dirs


def load_anims(ch_dir: Path):
    name = f"anims_{ch_dir.name}"
    spec = importlib.util.spec_from_file_location(name, ch_dir / "anims.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, "ANIMATIONS", [])


def file_hash(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def stem(ch_dir: Path, idx: int) -> str:
    return f"{ch_dir.name[:3]}_{idx + 1}"


# ---------------------------------------------------------------- animations

def _render_one(task):
    ch_path, idx = task
    ch_dir = Path(ch_path)
    t0 = time.time()
    try:
        from lib.anim import render

        spec = load_anims(ch_dir)[idx]
        render(spec, ANIM, stem(ch_dir, idx))
        return ch_dir.name, idx, time.time() - t0, None
    except Exception:
        return ch_dir.name, idx, time.time() - t0, traceback.format_exc()


def render_animations(dirs, cache, force, jobs):
    tasks, specs_by_ch, errors = [], {}, []
    for d in dirs:
        try:
            specs = load_anims(d)
        except Exception:
            errors.append(f"{d.name}/anims.py failed to import:\n{traceback.format_exc()}")
            specs_by_ch[d.name] = []
            continue
        specs_by_ch[d.name] = specs
        if len(specs) < MIN_ANIMS:
            errors.append(f"{d.name}: only {len(specs)} animations (need >= {MIN_ANIMS})")
        h = file_hash(d / "anims.py", ROOT / "lib" / "anim.py")
        done = all((ANIM / f"{stem(d, i)}.gif").exists() and (ANIM / f"{stem(d, i)}_k2.png").exists()
                   for i in range(len(specs)))
        if force or not done or cache.get(f"anim:{d.name}") != h:
            tasks += [(str(d), i) for i in range(len(specs))]
            cache[f"anim:{d.name}"] = h
    if tasks:
        print(f"rendering {len(tasks)} animations with {jobs} workers ...", flush=True)
        failed = set()
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            for name, idx, secs, err in pool.map(_render_one, tasks):
                if err:
                    failed.add(name)
                    errors.append(f"{name} animation #{idx + 1} failed:\n{err}")
                print(f"  {name} #{idx + 1}: {'FAILED' if err else 'ok'} ({secs:.1f}s)", flush=True)
        for name in failed:
            cache.pop(f"anim:{name}", None)
    return specs_by_ch, errors


def contact_sheet(d: Path, n: int):
    """One QA image per chapter: every animation's three key frames in a row."""
    from PIL import Image

    rows = []
    for i in range(n):
        ims = [ANIM / f"{stem(d, i)}_k{k}.png" for k in range(3)]
        if all(p.exists() for p in ims):
            rows.append([Image.open(p).convert("RGB") for p in ims])
    if not rows:
        return
    w, h = rows[0][0].size
    w, h = w // 2, h // 2
    sheet = Image.new("RGB", (3 * w, len(rows) * h), "white")
    for r, ims in enumerate(rows):
        for c, im in enumerate(ims):
            sheet.paste(im.resize((w, h), Image.LANCZOS), (c * w, r * h))
    CONTACT.mkdir(parents=True, exist_ok=True)
    sheet.save(CONTACT / f"{d.name}.png")


# ---------------------------------------------------------------- examples

def run_example(d: Path, cache, force):
    src = d / "example.py"
    work = EXAMPLES / d.name
    work.mkdir(parents=True, exist_ok=True)
    out_file = work / "stdout.txt"
    h = file_hash(src)
    err = None
    if force or cache.get(f"ex:{d.name}") != h or not out_file.exists():
        for old in work.glob("*.png"):
            old.unlink()
        env = dict(os.environ, MPLBACKEND="Agg", PYTHONHASHSEED="0")
        t0 = time.time()
        try:
            p = subprocess.run([sys.executable, str(src)], cwd=work, env=env, capture_output=True,
                               text=True, timeout=300)
            out_file.write_text(p.stdout)
            if p.returncode != 0:
                err = f"{d.name}/example.py exited with {p.returncode}:\n{p.stderr[-3000:]}"
        except subprocess.TimeoutExpired:
            err = f"{d.name}/example.py timed out"
        print(f"  {d.name} example: {'FAILED' if err else 'ok'} ({time.time() - t0:.1f}s)", flush=True)
        if err:
            cache.pop(f"ex:{d.name}", None)
        else:
            cache[f"ex:{d.name}"] = h
    stdout = out_file.read_text() if out_file.exists() else ""
    return {"code": src.read_text(), "stdout": stdout, "images": sorted(work.glob("*.png")), "error": err}


# ---------------------------------------------------------------- markdown

_MATH = re.compile(r"\$\$(.+?)\$\$|(?<![\\$])\$(?!\s)([^$\n]+?)(?<![\s\\])\$", re.S)


def md_to_html(text: str, inline: bool = False) -> str:
    """Markdown -> HTML, keeping $math$ untouched for KaTeX."""
    import markdown

    stash = []

    def hide(m):
        stash.append(m)
        return f"MATHSTASH{len(stash) - 1}X"

    body = markdown.markdown(_MATH.sub(hide, text), extensions=["tables", "fenced_code", "sane_lists"])

    def show(m):
        g = stash[int(m.group(1))]
        if g.group(1) is not None:
            return f'<span class="math-display">\\[{html.escape(g.group(1).strip())}\\]</span>'
        return f'<span class="math-inline">\\({html.escape(g.group(2))}\\)</span>'

    body = re.sub(r"MATHSTASH(\d+)X", show, body)
    if inline:
        body = re.sub(r"^<p>|</p>$", "", body.strip())
    return body


def check_math(sources):
    """sources: list of (label, text). Returns error strings for bad LaTeX or stray '$'."""
    items, labels, errors = [], [], []
    for label, text in sources:
        for m in _MATH.finditer(text):
            display = m.group(1) is not None
            items.append({"tex": (m.group(1) if display else m.group(2)).strip(), "display": display})
            labels.append(label)
        rest = re.sub(r"```.*?```|`[^`\n]*`", "", _MATH.sub("", text), flags=re.S)
        for line in rest.splitlines():
            if "$" in line.replace("\\$", ""):
                errors.append(f"{label}: stray or unbalanced '$' near: {line.strip()[:90]}")
    if items:
        p = subprocess.run(["node", str(ROOT / "lib" / "check_math.js"), str(KATEX / "katex.js")],
                           input=json.dumps(items), capture_output=True, text=True, timeout=120)
        if p.returncode != 0:
            errors.append(f"math checker failed to run: {p.stderr[-500:]}")
        else:
            for e in json.loads(p.stdout):
                errors.append(f"{labels[e['i']]}: LaTeX error in `{items[e['i']]['tex'][:80]}`: {e['msg']}")
    return errors


def split_title(md: str):
    lines = md.strip().splitlines()
    if lines and lines[0].startswith("# "):
        return lines[0][2:].strip(), "\n".join(lines[1:])
    return "", md


# ---------------------------------------------------------------- html

CSS = r"""
@page { size: A4; margin: 16mm 13mm 17mm 13mm;
        @bottom-center { content: counter(page); font: 9pt "Helvetica Neue", sans-serif; color: #6b7280; } }
@page :first { @bottom-center { content: none; } }
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: "Kohinoor Devanagari", "Devanagari Sangam MN", "Helvetica Neue", sans-serif;
       font-size: 10.6pt; line-height: 1.72; color: #111827; margin: 0; }
body.screen { max-width: 900px; margin: 0 auto; padding: 24px 28px 80px; font-size: 16px; background: #fff; }
h1, h2, h3, h4 { line-height: 1.35; break-after: avoid; }
h2 { font-size: 13.5pt; margin: 1.3em 0 0.4em; color: #1e3a8a; }
h3 { font-size: 11.5pt; margin: 1.1em 0 0.3em; color: #1f2937; }
p { margin: 0.45em 0; orphans: 2; widows: 2; }
ul, ol { margin: 0.4em 0; padding-left: 1.5em; }
li { margin: 0.18em 0; }
a { color: #1d4ed8; text-decoration: none; }
code { font-family: "SF Mono", Menlo, Consolas, "Kohinoor Devanagari", monospace; font-size: 0.88em;
       background: #f1f5f9; padding: 0.08em 0.3em; border-radius: 3px; }
blockquote { margin: 0.7em 0; padding: 0.5em 0.9em; border-left: 4px solid #2563eb; background: #eff6ff;
             border-radius: 0 6px 6px 0; break-inside: avoid; }
blockquote p { margin: 0.2em 0; }
table { border-collapse: collapse; margin: 0.7em 0; width: 100%; font-size: 0.93em; break-inside: avoid; }
th, td { border: 1px solid #d1d5db; padding: 0.3em 0.6em; text-align: left; vertical-align: top; }
th { background: #eef2ff; }
.math-display { display: block; margin: 0.55em 0; overflow-x: auto; break-inside: avoid; }
.katex { font-size: 1.04em; }

.cover { height: 255mm; display: flex; flex-direction: column; justify-content: center; text-align: center;
         break-after: page; }
body.screen .cover { height: auto; padding: 70px 0 50px; }
.cover .kicker { letter-spacing: 0.22em; font-size: 10pt; color: #2563eb; font-weight: 700; }
.cover h1 { font-size: 34pt; margin: 0.25em 0 0.1em; color: #111827; }
.cover .sub { font-size: 15pt; color: #374151; }
.cover .pill { display: inline-block; margin: 2.2em auto 0; padding: 0.5em 1.4em; border-radius: 999px;
               background: #1e3a8a; color: #fff; font-size: 11.5pt; }
.cover .note { margin-top: 3em; font-size: 9.5pt; color: #6b7280; }

.front { break-after: page; }
.toc ol { list-style: none; padding: 0; margin: 0.4em 0; }
.toc li { display: flex; gap: 0.7em; padding: 0.22em 0; border-bottom: 1px dotted #cbd5e1; }
.toc .n { flex: 0 0 5.6em; color: #2563eb; font-weight: 700; }

.chapter { break-before: page; }
.ch-head { background: linear-gradient(135deg, #1e3a8a, #2563eb); color: #fff; padding: 14px 18px;
           border-radius: 10px; margin-bottom: 0.9em; break-inside: avoid; }
.ch-head .n { font-size: 9.5pt; letter-spacing: 0.18em; opacity: 0.85; font-weight: 700; }
.ch-head h1 { font-size: 17.5pt; margin: 0.1em 0 0; }
.part { display: flex; align-items: center; gap: 0.6em; margin: 1.5em 0 0.6em; padding-bottom: 0.25em;
        border-bottom: 2px solid #1e3a8a; break-after: avoid; }
.part .tag { background: #1e3a8a; color: #fff; border-radius: 5px; padding: 0.05em 0.6em; font-size: 9.5pt;
             font-weight: 700; }
.part h2 { margin: 0; font-size: 14pt; }

.anim { border: 1px solid #e5e7eb; border-radius: 9px; padding: 8px 7px 6px; margin: 0.8em 0;
        break-inside: avoid; background: #fff; }
.anim-head { font-weight: 700; margin-bottom: 0.3em; }
.anim-head .badge { background: #ea580c; color: #fff; border-radius: 4px; padding: 0.05em 0.5em;
                    font-size: 9pt; margin-right: 0.5em; }
.frames { display: flex; gap: 3px; }
.frames figure { margin: 0; flex: 1; text-align: center; }
.frames img { width: 100%; display: block; border: 1px solid #eef0f3; border-radius: 5px; }
.frames figcaption { font-size: 8pt; color: #6b7280; line-height: 1.3; margin-top: 1px; }
.anim .gif { display: block; margin: 0 auto; width: 480px; max-width: 100%; }
.anim .caption { font-size: 0.95em; margin: 0.4em 0 0.15em; }
.anim .file { font-size: 8pt; color: #6b7280; font-family: "SF Mono", Menlo, monospace; }

.code { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 9px 12px; margin: 0.6em 0; }
.code pre, pre.output { margin: 0; font-family: "SF Mono", Menlo, Consolas, "Kohinoor Devanagari", monospace;
                        font-size: 8.3pt; line-height: 1.5; white-space: pre-wrap; word-break: break-word; }
body.screen .code pre, body.screen pre.output { font-size: 13px; }
pre.output { background: #0f172a; color: #e2e8f0; border-radius: 8px; padding: 9px 12px; margin: 0.6em 0; }
.label { font-size: 9pt; font-weight: 700; color: #475569; margin: 1em 0 0.2em; break-after: avoid; }
.result-img { display: block; max-width: 88%; margin: 0.6em auto; border: 1px solid #e5e7eb; border-radius: 6px;
              break-inside: avoid; }
.run { font-size: 0.93em; }
"""

FRAME_LABELS = ["शुरुआत", "बीच में", "अंत"]


def build_html(dirs, specs_by_ch, examples, animated: bool, base: str, katex: str) -> str:
    from pygments import highlight
    from pygments.formatters import HtmlFormatter
    from pygments.lexers import PythonLexer

    fmt = HtmlFormatter(style="friendly", cssclass="code")
    parts, toc = [], []
    for d in dirs:
        num = int(d.name[1:3])
        title, theory = split_title((d / "theory.md").read_text())
        toc.append(f'<li><span class="n">Lecture {num}</span><a href="#{d.name}">{html.escape(title)}</a></li>')
        sec = [f'<section class="chapter" id="{d.name}">',
               f'<div class="ch-head"><div class="n">LECTURE {num}</div><h1>{html.escape(title)}</h1></div>',
               '<div class="part"><span class="tag">भाग 1</span><h2>Theory</h2></div>',
               md_to_html(theory),
               '<div class="part"><span class="tag">भाग 2</span><h2>3D Animations</h2></div>']
        for i, spec in enumerate(specs_by_ch.get(d.name, [])):
            st = stem(d, i)
            if animated:
                media = f'<img class="gif" loading="lazy" src="{base}anim/{st}.gif" alt="{html.escape(spec.title)}">'
            else:
                media = '<div class="frames">' + "".join(
                    f'<figure><img src="{base}anim/{st}_k{k}.png"><figcaption>{FRAME_LABELS[k]}</figcaption></figure>'
                    for k in range(3)) + "</div>"
            sec.append(
                f'<div class="anim"><div class="anim-head"><span class="badge">Animation {num}.{i + 1}</span>'
                f'{md_to_html(spec.title, inline=True)}</div>{media}'
                f'<p class="caption"><b>क्या देखें:</b> {md_to_html(spec.caption, inline=True)}</p>'
                f'<div class="file">चलती animation: cs229_hindi/out/anim/{st}.gif</div></div>')
        ex = examples[d.name]
        sec.append('<div class="part"><span class="tag">भाग 3</span><h2>Python Example</h2></div>')
        sec.append(f'<p class="run">इसे खुद चलाएँ: <code>cs229_hindi/.venv/bin/python '
                   f'cs229_hindi/chapters/{d.name}/example.py</code></p>')
        sec.append(highlight(ex["code"], PythonLexer(), fmt))
        lines = ex["stdout"].rstrip().splitlines()
        if len(lines) > MAX_OUTPUT_LINES:
            lines = lines[:MAX_OUTPUT_LINES] + ["..."]
        if lines:
            sec.append('<div class="label">OUTPUT</div>')
            sec.append(f'<pre class="output">{html.escape(chr(10).join(lines))}</pre>')
        for img in ex["images"]:
            sec.append(f'<img class="result-img" src="{base}examples/{d.name}/{img.name}">')
        sec.append("</section>")
        parts.append("\n".join(sec))

    front_md = (CHAPTERS / "_front.md").read_text() if (CHAPTERS / "_front.md").exists() else ""
    cover = ('<section class="cover"><div class="kicker">DHRUV AI TUTORIAL &middot; SELF-STUDY GUIDE</div>'
             '<h1>Dhruv AI tutorial<br>Machine Learning हिंदी में</h1>'
             '<div class="sub">Lecture 1 से Lecture 20 तक</div>'
             '<div class="pill">Theory &nbsp;&bull;&nbsp; 3D Animations &nbsp;&bull;&nbsp; Python Examples</div>'
             '<div class="note">यह Machine Learning के मूल सिद्धांतों से लेकर Reinforcement Learning तक की self-study guide है।</div></section>')
    front = (f'<section class="front">{md_to_html(front_md)}'
             f'<h2>विषय-सूची</h2><div class="toc"><ol>{"".join(toc)}</ol></div></section>')
    return f"""<!doctype html>
<html lang="hi"><head><meta charset="utf-8">
<title>Dhruv AI tutorial | Machine Learning हिंदी में</title>
<link rel="stylesheet" href="{katex}katex.min.css">
<style>{CSS}{fmt.get_style_defs(".code")}</style>
</head><body class="{'screen' if animated else 'print'}">
{cover}
{front}
{''.join(parts)}
<script src="{katex}katex.min.js"></script>
<script src="{katex}contrib/auto-render.min.js"></script>
<script>
renderMathInElement(document.body, {{
  delimiters: [{{left: "\\\\[", right: "\\\\]", display: true}}, {{left: "\\\\(", right: "\\\\)", display: false}}],
  throwOnError: false, ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
}});
</script>
</body></html>"""


def to_pdf(html_path: Path, pdf_path: Path):
    cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--hide-scrollbars",
           "--allow-file-access-from-files", "--virtual-time-budget=60000",
           f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=600)


# ---------------------------------------------------------------- main

def load_cache():
    try:
        return json.loads(CACHE.read_text())
    except (OSError, ValueError):
        return {}


def save_cache(updates):
    """Merge our keys into the cache file atomically (several builds may run at once)."""
    merged = load_cache()
    merged.update(updates)
    tmp = CACHE.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(merged, indent=1))
    os.replace(tmp, CACHE)


def rasterise(pdf: Path, out_dir: Path, dpi=75):
    """PNG of every PDF page, for visual QA."""
    import pymupdf

    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("p*.png"):
        old.unlink()
    doc = pymupdf.open(pdf)
    for i, page in enumerate(doc):
        page.get_pixmap(dpi=dpi).save(out_dir / f"p{i + 1:02d}.png")
    return len(doc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="chapter prefixes, e.g. L02 L05")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()

    if args.jobs is None:
        args.jobs = 3 if args.only else max(1, (os.cpu_count() or 4) - 1)
    OUT.mkdir(exist_ok=True)
    cache = load_cache()
    dirs = chapter_dirs(args.only)
    if not dirs:
        sys.exit("no chapters matched")

    errors = []
    for d in dirs:
        for f in ("theory.md", "anims.py", "example.py"):
            if not (d / f).exists():
                errors.append(f"{d.name}: missing {f}")
    if errors:
        sys.exit("\n".join(errors))

    specs_by_ch, errs = render_animations(dirs, cache, args.force, args.jobs)
    errors += errs
    for d in dirs:
        contact_sheet(d, len(specs_by_ch.get(d.name, [])))
    print("running examples ...", flush=True)
    examples = {}
    for d in dirs:
        examples[d.name] = run_example(d, cache, args.force)
        if examples[d.name]["error"]:
            errors.append(examples[d.name]["error"])
    save_cache(cache)

    sources = []
    for d in dirs:
        sources.append((f"{d.name}/theory.md", (d / "theory.md").read_text()))
        for i, spec in enumerate(specs_by_ch.get(d.name, [])):
            sources.append((f"{d.name}/anims.py animation #{i + 1} caption", spec.caption))
            sources.append((f"{d.name}/anims.py animation #{i + 1} title", spec.title))
    errors += check_math(sources)

    partial = bool(args.only)
    tag = "_".join(d.name[:3] for d in dirs)
    print_html = OUT / (f"preview_{tag}.html" if partial else "print.html")
    print_html.write_text(build_html(dirs, specs_by_ch, examples, animated=False, base="",
                                     katex="../vendor/katex/package/dist/"))
    if partial:
        pdf = OUT / f"preview_{tag}.pdf"
    else:
        pdf = WORKSPACE / PDF_NAME
        (WORKSPACE / HTML_NAME).write_text(
            build_html(dirs, specs_by_ch, examples, animated=True, base="cs229_hindi/out/",
                       katex="cs229_hindi/vendor/katex/package/dist/"))
    if not args.no_pdf:
        to_pdf(print_html, pdf)
        print(f"PDF: {pdf} ({pdf.stat().st_size / 1e6:.1f} MB)")
        if partial:
            qa = OUT / "qa" / tag
            print(f"QA page images: {qa}/pNN.png ({rasterise(pdf, qa)} pages)")
            print(f"QA contact sheets: {CONTACT}/<chapter>.png")

    if errors:
        print("\n==== ERRORS ====")
        print("\n\n".join(errors))
        sys.exit(1)
    print("build ok")


if __name__ == "__main__":
    main()
