"""Dhruv AI tutorial: 3D animation helpers.

Every chapter's ``anims.py`` declares its animations like this::

    from lib.anim import animation, C, surface, ease
    import numpy as np

    @animation("Gradient Descent on J(theta)",
               caption="Hindi caption shown under the animation in the book.",
               frames=40)
    def gd_surface(s, n):
        ax = s.ax3d(labels=(r"$\\theta_0$", r"$\\theta_1$", r"$J$"))
        surface(ax, lambda a, b: a**2 + b**2, (-2, 2), (-2, 2))   # static artists

        def update(i):
            s.clear()                              # drop last frame's dynamic artists
            s.add(ax.scatter([..], [..], [..]))    # register this frame's dynamic artists
            s.status(f"iteration {i}")             # small text line under the title
            s.spin(i, n)                           # slow camera rotation
        return update

Text drawn inside the figure must be English / math only: matplotlib cannot
shape Devanagari. Hindi belongs in ``caption``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

C = {
    "blue": "#2563eb",
    "red": "#dc2626",
    "green": "#16a34a",
    "orange": "#ea580c",
    "purple": "#7c3aed",
    "teal": "#0d9488",
    "pink": "#db2777",
    "yellow": "#ca8a04",
    "gray": "#6b7280",
    "dark": "#111827",
}
CLASS_COLORS = [C["blue"], C["red"], C["green"], C["orange"], C["purple"], C["teal"]]

FIGSIZE = (4.2, 3.8)
GIF_DPI = 120
KEY_DPI = 190
TITLE_SIZE = 12.5
TITLE_CHARS = 36   # longer titles are shrunk to fit

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.labelsize": 12,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 9.5,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


@dataclass
class AnimSpec:
    name: str
    title: str
    caption: str
    frames: int
    fps: int
    keys: Sequence[float]
    func: Callable


def animation(title: str, caption: str, frames: int = 36, fps: int = 10,
              keys: Sequence[float] = (0.0, 0.5, 1.0)):
    """Register a 3D animation in the calling module's ``ANIMATIONS`` list.

    title   : English title drawn on the figure (keep it under ~36 characters).
    caption : Hindi explanation printed in the book ("what to watch for").
              May contain $inline math$.
    frames  : number of frames (30-48 is a good range).
    keys    : three fractions in [0, 1] picking the still frames printed in the PDF.
    """

    def deco(func):
        mod = sys.modules[func.__module__]
        mod.__dict__.setdefault("ANIMATIONS", []).append(
            AnimSpec(func.__name__, title, " ".join(caption.split()), frames, fps, tuple(keys), func)
        )
        return func

    return deco


class Stage:
    """One figure with a single 3D axes, plus bookkeeping for per-frame artists."""

    def __init__(self, title: str):
        self.fig = plt.figure(figsize=FIGSIZE, dpi=GIF_DPI)
        plain = len(title.replace("$", "").replace("\\", ""))
        size = TITLE_SIZE * min(1.0, TITLE_CHARS / max(1, plain))
        self.fig.suptitle(title, fontsize=size, fontweight="bold", y=0.975)
        self._status = self.fig.text(0.5, 0.895, "", ha="center", va="center",
                                     fontsize=10.5, color="#374151")
        self.ax = None
        self._dyn = []
        self._elev = 24.0
        self._azim = -58.0

    def ax3d(self, labels=("x", "y", "z"), elev=24, azim=-58,
             xlim=None, ylim=None, zlim=None, zoom=0.84, ticks=True):
        """Create the 3D axes. Call exactly once per animation."""
        ax = self.fig.add_axes([0.0, 0.0, 1.0, 0.88], projection="3d")
        ax.set_xlabel(labels[0], labelpad=-4)
        ax.set_ylabel(labels[1], labelpad=-4)
        ax.set_zlabel(labels[2], labelpad=-4)
        ax.tick_params(pad=-2)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if zlim is not None:
            ax.set_zlim(*zlim)
        if not ticks:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])
        ax.set_box_aspect(None, zoom=zoom)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((0.97, 0.97, 0.98, 1.0))
            axis.pane.set_edgecolor((0.85, 0.85, 0.88, 1.0))
            axis._axinfo["grid"]["color"] = (0.88, 0.88, 0.9, 1.0)
            axis._axinfo["grid"]["linewidth"] = 0.6
        self._elev, self._azim = float(elev), float(azim)
        ax.view_init(elev=elev, azim=azim)
        self.ax = ax
        return ax

    def add(self, *artists):
        """Register artists that must disappear on the next ``clear()``. Returns the first one."""
        first = None
        for a in artists:
            if isinstance(a, (list, tuple)):
                for b in a:
                    self.add(b)
                    first = first if first is not None else b
            elif a is not None:
                self._dyn.append(a)
                first = first if first is not None else a
        return first

    def clear(self):
        """Remove every artist registered with ``add()``."""
        for a in self._dyn:
            try:
                a.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass
        self._dyn = []

    def status(self, text: str):
        """Set the small grey text line under the title (English / math only)."""
        self._status.set_text(text)

    def spin(self, i: int, n: int, sweep: float = 50.0, tilt: float = 0.0):
        """Rotate the camera slowly: ``sweep`` degrees of azimuth over the whole animation."""
        t = i / max(1, n - 1)
        self.ax.view_init(elev=self._elev + tilt * t, azim=self._azim + sweep * t)


def ease(t):
    """Smoothstep easing for t in [0, 1]."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return (1 - t) * np.asarray(a, dtype=float) + t * np.asarray(b, dtype=float)


def grid(xlim, ylim, n=40):
    """Meshgrid over a rectangle."""
    xs = np.linspace(xlim[0], xlim[1], n)
    ys = np.linspace(ylim[0], ylim[1], n)
    return np.meshgrid(xs, ys)


def surface(ax, f, xlim, ylim, n=40, cmap="viridis", alpha=0.55, wire=False, **kw):
    """Plot z = f(x, y). Returns (artist, X, Y, Z)."""
    X, Y = grid(xlim, ylim, n)
    Z = f(X, Y)
    if wire:
        art = ax.plot_wireframe(X, Y, Z, rstride=max(1, n // 14), cstride=max(1, n // 14),
                                color=kw.pop("color", C["gray"]), linewidth=0.5, alpha=alpha, **kw)
    else:
        art = ax.plot_surface(X, Y, Z, cmap=cmap, alpha=alpha, linewidth=0,
                              antialiased=True, rstride=1, cstride=1, **kw)
    return art, X, Y, Z


def plane(ax, f, xlim, ylim, color=C["blue"], alpha=0.3, n=8):
    """Plot a flat coloured plane z = f(x, y). Returns the artist."""
    X, Y = grid(xlim, ylim, n)
    return ax.plot_surface(X, Y, f(X, Y), color=color, alpha=alpha, linewidth=0, shade=False)


def arrow(ax, start, vec, color=C["dark"], lw=2.0, ratio=0.14):
    """3D arrow from ``start`` along ``vec``. Returns the artist."""
    s = np.asarray(start, dtype=float)
    v = np.asarray(vec, dtype=float)
    return ax.quiver(s[0], s[1], s[2], v[0], v[1], v[2], color=color, linewidth=lw,
                     arrow_length_ratio=ratio)


def render(spec: AnimSpec, out_dir: Path, stem: str) -> None:
    """Render one animation to ``<stem>.gif`` plus three key frames ``<stem>_k0..2.png``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(0)
    stage = Stage(spec.title)
    update = spec.func(stage, spec.frames)
    if stage.ax is None or not callable(update):
        raise RuntimeError(f"{spec.name}: must call s.ax3d(...) and return update(i)")
    n = spec.frames
    key_idx = [min(n - 1, int(round(k * (n - 1)))) for k in spec.keys]
    frames = []
    for i in range(n):
        update(i)
        stage.fig.canvas.draw()
        rgba = np.asarray(stage.fig.canvas.buffer_rgba())
        frames.append(Image.fromarray(rgba).convert("RGB"))
        for k, idx in enumerate(key_idx):
            if idx == i:
                stage.fig.savefig(out_dir / f"{stem}_k{k}.png", dpi=KEY_DPI)
    pal = [f.quantize(colors=96, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
           for f in frames]
    durations = [int(1000 / spec.fps)] * n
    durations[-1] = 1600
    pal[0].save(out_dir / f"{stem}.gif", save_all=True, append_images=pal[1:],
                duration=durations, loop=0, optimize=False, disposal=1)
    plt.close(stage.fig)
