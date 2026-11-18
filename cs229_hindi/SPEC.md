# Chapter spec (Dhruv AI tutorial)

Read first: `lib/anim.py` (animation API), `chapters/L02_linear_regression/` (reference chapter: theory.md, anims.py, example.py), `build.py` docstring.

Only edit files inside your own chapter directory. Do not edit `lib/`, `build.py`, `_front.md`, or other chapters. Do not install packages. Never run `build.py` without `--only`.

## Deliverables per chapter
Directory `chapters/<Lxx_name>/` with exactly: `theory.md`, `anims.py`, `example.py`. Some files may already exist; check them and complete or fix them. Keep existing good files.

### theory.md
- Line 1: `# <title>` (title given in your task).
- Then `> **इस lecture में:**` blockquote listing coverage.
- Hindi in Devanagari, addressing reader as "आप", short sentences, `।` as full stop. AI/maths words stay in English Latin script (gradient, likelihood, kernel, variance, policy, reward, algorithm names). Do not transliterate them. No em dashes.
- Sections: `## 1. एक line में idea`, then setup/notation, maths with derivation steps, algorithm step by step, `## Practical बातें और आम गलतियाँ`, `## खुद से पूछें` (three questions).
- About 3-5 printed pages. Maths must be correct; double-check derivations.
- Math: `$...$` inline, `$$...$$` display on its own paragraph. KaTeX-compatible LaTeX only.

### anims.py
- At least five `@animation(title, caption=r"""...""", frames=30..48, keys=(a,b,c))` functions, each `def f(s, n)` that calls `s.ax3d(...)` once, builds static artists, returns `update(i)` using `s.clear()`, `s.add(...)`, `s.status(...)`, `s.spin(...)`.
- Each animation must be genuinely 3D and show a process from the theory (iteration, parameter sweep, morphing). Five different ideas.
- Text inside figure: English or mathtext only. Titles <= 36 chars. Status line <= 45 chars.
- Caption: 3-5 Hindi sentences, raw string, `$...$` math.
- Surfaces translucent (alpha <= 0.5), grids <= 40x40, `depthshade=False` on scatter, explicit axis limits. Deterministic RNG seeds. Each render well under 20 s.
- Animate real computation (run the algorithm, then animate its history).

### example.py
- Self-contained, NumPy from scratch (scipy/sklearn for datasets or reference only). Hindi docstring and comments, technical words in English. 80-150 lines.
- Prints <= 40 lines of meaningful output demonstrating the theory. Must run under 20 s, no network.
- Saves `fig.savefig("result.png", dpi=130)`. No `plt.show()`.

## Verify
From project dir:
`cd "/Users/dhruv/Desktop/AI FOLDER/cs229_hindi" && .venv/bin/python build.py --only <Lxx>`
Must end with `build ok`. Fix every error. Then Read `out/contact/<Lxx_dir>.png` and check each animation row is clean. Read `out/examples/<Lxx_dir>/stdout.txt` and check the numbers match the theory.

## Report back (short)
Directory, animation titles in order (one line each), example headline numbers, `build ok` confirmed yes/no, anything unsure.
