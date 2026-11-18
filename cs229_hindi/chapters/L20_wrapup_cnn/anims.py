"""Lecture 20: Wrap-up and Convolutional Neural Networks: 3D animations."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib.anim import C, CLASS_COLORS, animation, ease, grid, lerp


def _conv2d_single(img, ker, pad=0, stride=1):
    """Valid/padded 2D cross-correlation of one 2D image with one 2D kernel."""
    if pad > 0:
        img = np.pad(img, pad, mode="constant")
    f = ker.shape[0]
    nh = (img.shape[0] - f) // stride + 1
    nw = (img.shape[1] - f) // stride + 1
    out = np.zeros((nh, nw))
    for r in range(nh):
        for c in range(nw):
            out[r, c] = np.sum(img[r * stride:r * stride + f, c * stride:c * stride + f] * ker)
    return out, img


def _floor_grid(ax, x0, y0, rows, cols, z=0.0, face=(0.95, 0.96, 0.98, 0.55),
                edge=(0.55, 0.58, 0.65, 0.85), lw=0.6):
    """Draw a rows x cols tile grid on plane z with cell centres at (x0 + c, y0 + rows - 1 - r)."""
    polys = []
    for r in range(rows):
        yc = y0 + (rows - 1 - r)
        for c in range(cols):
            xc = x0 + c
            polys.append([(xc - 0.5, yc - 0.5, z), (xc + 0.5, yc - 0.5, z),
                          (xc + 0.5, yc + 0.5, z), (xc - 0.5, yc + 0.5, z)])
    col = Poly3DCollection(polys, facecolors=face, edgecolors=edge, linewidths=lw)
    ax.add_collection3d(col)
    return col


def _wire_rect(ax, x_lo, x_hi, y_lo, y_hi, z, color, lw=2.0, ls="-"):
    xs = [x_lo, x_hi, x_hi, x_lo, x_lo]
    ys = [y_lo, y_lo, y_hi, y_hi, y_lo]
    return ax.plot(xs, ys, [z] * 5, color=color, lw=lw, ls=ls)


def _box_faces(x0, x1, y0, y1, z0, z1):
    return [
        [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0)],
        [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],
        [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],
        [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],
        [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],
    ]


# ------------------------------------------------------------------ 1. sliding filter

@animation("Filter sliding to build feature map",
           caption=r"""बाएँ फ़र्श पर $6 \times 6$ input तस्वीर के खंभे हैं और उनके ऊपर नारंगी चौखट $3 \times 3$
           filter $W$ है। हर कदम पर वही एक filter आगे खिसकता है, ढके हुए $3 \times 3$ टुकड़े से गुणा-जोड़
           करता है, और दाएँ तरफ़ $4 \times 4$ feature map $Z$ का संगत खंभा उभर आता है। जहाँ तस्वीर का पैटर्न
           filter की खड़ी किनारी से मेल खाता है वहाँ हरे-पीले ऊँचे खंभे उठते हैं। एक ही $9$ weights सारे $16$
           स्थानों पर साझा होकर पूरा नक्शा भर देते हैं।""",
           frames=40, keys=(0.12, 0.52, 1.0))
def sliding_filter(s, n):
    img = np.array([
        [0.1, 0.9, 0.9, 0.2, 0.1, 0.1],
        [0.1, 0.9, 0.9, 0.2, 0.1, 0.1],
        [0.1, 0.9, 0.9, 0.8, 0.8, 0.1],
        [0.1, 0.9, 0.9, 0.8, 0.8, 0.1],
        [0.1, 0.9, 0.9, 0.2, 0.1, 0.1],
        [0.1, 0.2, 0.2, 0.1, 0.1, 0.1],
    ])
    ker = np.array([[1.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0],
                    [1.0, 0.0, -1.0]]) / 2.2
    fmap, _ = _conv2d_single(img, ker, pad=0, stride=1)
    fmap_pos = np.maximum(0.0, fmap)                     # ReLU feature map

    ax = s.ax3d(labels=("space x", "row y", "intensity / activation"), elev=31, azim=-62,
                xlim=(-0.8, 11.4), ylim=(-0.8, 5.8), zlim=(0, 1.45), ticks=False)
    ax.set_box_aspect((2.15, 1.15, 0.72), zoom=0.92)

    _floor_grid(ax, 0.0, 0.0, 6, 6)
    _floor_grid(ax, 7.2, 1.0, 4, 4, face=(0.93, 0.96, 1.0, 0.6))
    ax.text(2.5, -0.9, 0.0, "6x6 input X", fontsize=8.5, ha="center", color=C["dark"])
    ax.text(8.7, 0.1, 0.0, "4x4 feature map", fontsize=8.5, ha="center", color=C["blue"])

    rr, cc = np.meshgrid(np.arange(6), np.arange(6), indexing="ij")
    ix = cc.ravel().astype(float)
    iy = (5 - rr.ravel()).astype(float)
    ih = img.ravel()
    icols = plt.cm.Blues(0.25 + 0.65 * ih)
    ax.bar3d(ix - 0.36, iy - 0.36, np.zeros_like(ih), 0.72, 0.72, np.maximum(ih, 0.02),
             color=icols, alpha=0.68, shade=True, edgecolor=(0, 0, 0, 0.22), linewidth=0.25)

    positions = [(r, c) for r in range(4) for c in range(4)]
    zf = 1.12

    def update(i):
        s.clear()
        step = min(15, int(i * 16 / max(1, n - 4)))
        r, c = positions[step]
        x_lo, x_hi = c - 0.48, c + 2.48
        y_hi, y_lo = (5 - r) + 0.48, (5 - r - 2) - 0.48
        s.add(_wire_rect(ax, x_lo, x_hi, y_lo, y_hi, zf, C["orange"], lw=2.2))
        # show the 3 columns of the vertical-edge filter (+, 0, -) on the floating window
        tiles, tcols = [], []
        for u in range(3):
            for v in range(3):
                xc, yc = c + v, 5 - (r + u)
                tiles.append([(xc - 0.44, yc - 0.44, zf), (xc + 0.44, yc - 0.44, zf),
                              (xc + 0.44, yc + 0.44, zf), (xc - 0.44, yc + 0.44, zf)])
                tcols.append(to_rgba(C["green"], 0.35) if v == 0 else
                             (to_rgba(C["gray"], 0.16) if v == 1 else to_rgba(C["red"], 0.35)))
        s.add(ax.add_collection3d(Poly3DCollection(tiles, facecolors=tcols,
                                                   edgecolors=C["orange"], linewidths=0.6)))

        ox, oy, oh, ocols = [], [], [], []
        for idx in range(step + 1):
            pr, pc = positions[idx]
            val = fmap_pos[pr, pc]
            ox.append(7.2 + pc)
            oy.append(1.0 + (3 - pr))
            oh.append(max(val, 0.03))
            ocols.append(to_rgba(C["orange"], 0.95) if idx == step else
                         plt.cm.viridis(0.2 + 0.75 * min(1.0, val / 1.25)))
        ox, oy, oh = np.array(ox), np.array(oy), np.array(oh)
        s.add(ax.bar3d(ox - 0.36, oy - 0.36, np.zeros_like(oh), 0.72, 0.72, oh,
                       color=ocols, alpha=0.88, shade=True, edgecolor=(0, 0, 0, 0.3), linewidth=0.3))

        tx, ty, tz = 7.2 + c, 1.0 + (3 - r), max(fmap_pos[r, c], 0.05)
        s.add(ax.plot([c + 1.0, tx], [5 - r - 1.0, ty], [zf, tz + 0.08],
                      color=C["orange"], lw=1.4, ls="--"))
        s.status(f"pos ({r},{c})  ReLU(W.patch) = {fmap_pos[r, c]:.2f}  ({step + 1}/16)")
        s.spin(i, n, sweep=28)

    return update


# ------------------------------------------------------------------ 2. vertical-edge response

@animation("Vertical-edge filter on a step edge",
           caption=r"""नीली पारदर्शी चादर एक step-edge तस्वीर $X(x, y)$ है: बाईं ओर ऊँचा उजाला ($1$) और $x = 0$ पर
           अचानक नीचे अँधेरा ($0$)। हरा $(+1)$ और लाल $(-1)$ पट्टियों वाला vertical-edge kernel बाएँ से दाएँ
           गुज़रता है। जब दोनों पट्टियाँ सपाट हिस्से पर होती हैं तो $+1$ और $-1$ एक-दूसरे को काटकर response $0$
           देते हैं, पर ठीक ढलान $x = 0$ पर नारंगी feature-map की ऊँची दीवार उभर आती है।""",
           frames=42, keys=(0.15, 0.5, 0.95))
def edge_response(s, n):
    ax = s.ax3d(labels=("x (across edge)", "y (along edge)", "value"), elev=27, azim=-60,
                xlim=(-3.6, 3.6), ylim=(-2.5, 2.5), zlim=(-0.15, 1.65))
    X, Y = grid((-3.5, 3.5), (-2.4, 2.4), 36)
    step_surf = 1.0 / (1.0 + np.exp(7.0 * X))            # crisp vertical step edge at x = 0
    ax.plot_surface(X, Y, step_surf, color=C["blue"], alpha=0.28, linewidth=0, shade=True)
    ax.plot_wireframe(X, Y, step_surf, rstride=5, cstride=5, color=C["blue"], linewidth=0.45, alpha=0.4)

    hw = 0.55                                            # filter half-width: compares x - hw with x + hw
    resp_full = 1.45 * (1.0 / (1.0 + np.exp(7.0 * (X + hw))) - 0.0
                        + 0.0)                           # placeholder overwritten cleanly below
    # exact continuous response of [ +1 at x - hw , 0 at x , -1 at x + hw ] on step_surf:
    resp_full = 1.42 * (1.0 / (1.0 + np.exp(7.0 * (X - hw))) - 1.0 / (1.0 + np.exp(7.0 * (X + hw))))
    xs_1d = X[0, :]
    resp_1d = resp_full[0, :]
    ax.plot([], [], [], color=C["blue"], lw=2, label="input step X")
    ax.plot([], [], [], color=C["orange"], lw=2, label="filter response Z")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        t = ease(i / (n - 1))
        xc = lerp(-2.7, 2.7, t)
        # draw the 3-strip kernel footprint (+1 green on left, -1 red on right)
        z_k = 1.52
        strips = [
            (xc - 1.5 * hw, xc - 0.5 * hw, to_rgba(C["green"], 0.42)),
            (xc - 0.5 * hw, xc + 0.5 * hw, to_rgba(C["gray"], 0.18)),
            (xc + 0.5 * hw, xc + 1.5 * hw, to_rgba(C["red"], 0.42)),
        ]
        polys = [[(a, -2.2, z_k), (b, -2.2, z_k), (b, 2.2, z_k), (a, 2.2, z_k)] for a, b, _ in strips]
        s.add(ax.add_collection3d(Poly3DCollection(polys, facecolors=[c for *_, c in strips],
                                                   edgecolors=C["dark"], linewidths=0.7)))

        mask = X <= xc
        if np.any(mask[0, :]):
            k_end = int(np.searchsorted(xs_1d, xc, side="right"))
            k_end = max(2, min(len(xs_1d), k_end))
            s.add(ax.plot_surface(X[:, :k_end], Y[:, :k_end], resp_full[:, :k_end],
                                  cmap="YlOrRd", alpha=0.48, linewidth=0, vmin=0.0, vmax=1.45))
            s.add(ax.plot(xs_1d[:k_end], np.full(k_end, -2.4), resp_1d[:k_end],
                          color=C["orange"], lw=2.3))

        cur_resp = float(1.42 * (1.0 / (1.0 + np.exp(7.0 * (xc - hw)))
                                 - 1.0 / (1.0 + np.exp(7.0 * (xc + hw)))))
        ys_line = np.linspace(-2.2, 2.2, 20)
        s.add(ax.plot(np.full_like(ys_line, xc), ys_line, np.full_like(ys_line, cur_resp),
                      color=C["red"], lw=2.4))
        s.status(f"filter x = {xc:+.2f}   response = {cur_resp:+.2f}")
        s.spin(i, n, sweep=32)

    return update


# ------------------------------------------------------------------ 3. max pooling shrinking bars

@animation("2x2 Max Pooling shrinks 4x4 to 2x2",
           caption=r"""यहाँ $4 \times 4$ activation नक्शे को चार रंगीन $2 \times 2$ इलाकों में बाँटा गया है ($f_p = 2, s_p = 2$)।
           पहले चरण में हर $2 \times 2$ खिड़की के तीन छोटे खंभे पिघलकर शून्य हो जाते हैं और केवल सबसे ऊँचा विजेता
           खंभा बचता है। दूसरे चरण में चारों विजेता खंभे सिमटकर बीच में $2 \times 2$ pooled नक्शा बना लेते हैं।
           बिना एक भी नया parameter सीखे स्थानिक आकार चौथाई रह गया और सबसे मज़बूत संकेत सुरक्षित बचे।""",
           frames=40, keys=(0.0, 0.48, 1.0))
def max_pooling_bars(s, n):
    vals = np.array([
        [0.35, 1.25, 0.45, 0.30],
        [0.55, 0.40, 0.98, 0.25],
        [0.30, 0.88, 0.50, 1.38],
        [0.20, 0.42, 0.62, 0.48],
    ])
    ax = s.ax3d(labels=("column", "row", "activation"), elev=30, azim=-56,
                xlim=(-0.7, 3.7), ylim=(-0.7, 3.7), zlim=(0, 1.55), ticks=False)
    ax.set_box_aspect((1.1, 1.1, 0.85), zoom=0.9)

    q_colors = [CLASS_COLORS[0], CLASS_COLORS[1], CLASS_COLORS[2], CLASS_COLORS[4]]
    bars = []
    for qr in range(2):
        for qc in range(2):
            q_id = 2 * qr + qc
            block = vals[2 * qr:2 * qr + 2, 2 * qc:2 * qc + 2]
            win_idx = np.unravel_index(int(np.argmax(block)), (2, 2))
            for u in range(2):
                for v in range(2):
                    r, c = 2 * qr + u, 2 * qc + v
                    x_init, y_init = float(c), float(3 - r)
                    x_pool, y_pool = 1.0 + float(qc), 1.0 + float(1 - qr)
                    bars.append((x_init, y_init, x_pool, y_pool, float(vals[r, c]),
                                 (u, v) == win_idx, q_colors[q_id]))

    _floor_grid(ax, 0.0, 0.0, 4, 4, face=(0.96, 0.97, 0.99, 0.45))
    for qr in range(2):
        for qc in range(2):
            _wire_rect(ax, 2 * qc - 0.48, 2 * qc + 1.48, (2 - 2 * qr) - 0.48,
                       (3 - 2 * qr) + 0.48, 0.005, q_colors[2 * qr + qc], lw=1.2)

    mid = n // 2

    def update(i):
        s.clear()
        if i <= mid:
            t_drop = ease(i / mid)
            t_move = 0.0
            msg = f"drop non-maxima in each 2x2 ({int(round((1 - 0.75 * t_drop) * 16))} active)"
        else:
            t_drop = 1.0
            t_move = ease((i - mid) / (n - 1 - mid))
            msg = "compact 4 winners -> 2x2 pooled map"
        if i == 0:
            msg = "input 4x4 feature map (16 bars)"
        elif i == n - 1:
            msg = "output 2x2 max-pooled map (0 params)"

        if t_move > 0.05:
            s.add(_wire_rect(ax, 0.52, 2.48, 0.52, 2.48, 0.012, C["dark"], lw=2.0, ls="--"))

        xs, ys, hs, cols = [], [], [], []
        for x0, y0, x1, y1, h, is_win, col in bars:
            if is_win:
                xs.append(lerp(x0, x1, t_move))
                ys.append(lerp(y0, y1, t_move))
                hs.append(h)
                cols.append(to_rgba(col, 0.92))
            elif t_drop < 0.98:
                xs.append(x0)
                ys.append(y0)
                hs.append(max(0.015, h * (1.0 - t_drop)))
                cols.append(to_rgba(col, 0.42 * (1.0 - 0.75 * t_drop)))
        xs, ys, hs = np.array(xs), np.array(ys), np.array(hs)
        w = lerp(0.66, 0.76, t_move)
        s.add(ax.bar3d(xs - w / 2, ys - w / 2, np.zeros_like(hs), w, w, hs,
                       color=cols, shade=True, edgecolor=(0, 0, 0, 0.32), linewidth=0.35))
        s.status(msg)
        s.spin(i, n, sweep=36)

    return update


# ------------------------------------------------------------------ 4. CNN 3D volumes

@animation("CNN volumes: narrower and deeper",
           caption=r"""यहाँ पूरा CNN 3D volumes की शृंखला के रूप में दिख रहा है: हर पारदर्शी बक्से की ऊँचाई-चौड़ाई
           स्थानिक आकार $n_H \times n_W$ है और मोटाई channels $n_C$ है। बाएँ $32 \times 32 \times 3$ की चौड़ी पर
           पतली RGB चादर से शुरू होकर हर `[Conv -> Pool]` के बाद बक्सा ऊँचाई-चौड़ाई में सिकुड़ता जाता है और
           गहराई ($8 \to 16 \to 32 \to 64$) में मोटा होता जाता है। अंत में यह $10$ class scores में बदल जाता है।""",
           frames=42, keys=(0.2, 0.58, 1.0))
def cnn_volumes(s, n):
    # (label, shape_str, x_start, dx_depth, hw_size, color)
    specs = [
        ("Input", "32x32x3", 0.0, 0.30, 3.0, C["blue"]),
        ("Conv1", "32x32x8", 1.25, 0.55, 3.0, C["teal"]),
        ("Pool1", "16x16x8", 2.75, 0.55, 1.75, C["green"]),
        ("Conv2", "16x16x16", 4.20, 0.95, 1.75, C["purple"]),
        ("Pool2", "8x8x16", 6.00, 0.95, 0.98, C["orange"]),
        ("Conv3", "4x4x32", 7.75, 1.55, 0.56, C["red"]),
        ("FC", "1x1x10", 10.15, 0.42, 0.28, C["dark"]),
    ]
    ax = s.ax3d(labels=("layers (depth C)", "width W", "height H"), elev=23, azim=-63,
                xlim=(-0.4, 11.1), ylim=(-2.1, 2.1), zlim=(-2.1, 2.1), ticks=False)
    ax.set_box_aspect((2.35, 1.0, 0.95), zoom=0.94)

    for name, shp, x0, dx, hw, col in specs:
        ax.text(x0 + dx / 2, 0.0, -1.92, name, fontsize=7.5, ha="center", color=col)

    n_stages = len(specs)

    def update(i):
        s.clear()
        prog = (i / (n - 1)) * (n_stages - 0.15)
        active = min(n_stages - 1, int(prog))

        faces, fcols, ecols = [], [], []
        for idx, (name, shp, x0, dx, hw, col) in enumerate(specs):
            if idx > active:
                continue
            grow = 1.0 if idx < active else ease((prog - active) / 0.85)
            cur_dx = max(0.05, dx * grow)
            cur_hw = hw * (0.5 + 0.5 * grow) / 2.0
            alpha = 0.38 if idx == active else 0.26
            bf = _box_faces(x0, x0 + cur_dx, -cur_hw, cur_hw, -cur_hw, cur_hw)
            faces.extend(bf)
            fcols.extend([to_rgba(col, alpha)] * 6)
            ecols.extend([to_rgba(col, 0.92)] * 6)

            # connect consecutive volumes with a 3D receptive-field frustum
            if idx > 0 and grow > 0.2:
                _, _, px0, pdx, phw, _ = specs[idx - 1]
                px = px0 + pdx
                p_r = min(phw * 0.28, cur_hw * 1.35)
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        s.add(ax.plot([px, x0], [sy * p_r, sy * cur_hw * 0.55],
                                      [sz * p_r, sz * cur_hw * 0.55],
                                      color=C["gray"], lw=0.75, alpha=0.65))

        s.add(ax.add_collection3d(Poly3DCollection(faces, facecolors=fcols,
                                                   edgecolors=ecols, linewidths=0.95)))
        nm, shp = specs[active][0], specs[active][1]
        s.status(f"stage {active + 1}/{n_stages}: {nm} volume {shp}")
        s.spin(i, n, sweep=30)

    return update


# ------------------------------------------------------------------ 5. receptive field pyramid

@animation("Receptive field widens as a pyramid",
           caption=r"""चार क्षैतिज जाले CNN की चार क्रमिक layers हैं ($z = 0$ पर $9 \times 9$ input से लेकर $z = 3$ पर
           गहरी layer तक, हर जगह $3 \times 3$ filter और $s = 1$)। ऊपर के एक अकेले लाल neuron से नीचे झाँकने पर
           उसका प्रभाव-क्षेत्र हर कदम पर चौड़ा होता पिरामिड बनाता है: $1 \times 1 \to 3 \times 3 \to 5 \times 5 \to 7 \times 7$।
           अंत में जब ऊपरी neuron खिसकता है, तो नीचे $7 \times 7$ का पूरा आधार एक साथ खिसकता है।""",
           frames=42, keys=(0.15, 0.55, 0.95))
def receptive_pyramid(s, n):
    ax = s.ax3d(labels=("image x", "image y", "layer depth l"), elev=25, azim=-55,
                xlim=(-4.8, 4.8), ylim=(-4.8, 4.8), zlim=(-0.15, 3.35), ticks=False)
    ax.set_box_aspect((1.25, 1.25, 0.95), zoom=0.9)

    # Layer l at height z = l has grid size (9 - 2*l) x (9 - 2*l), centred at (0, 0)
    layer_sizes = [9, 7, 5, 3]
    lcols = [C["blue"], C["teal"], C["purple"], C["red"]]
    for l, sz in enumerate(layer_sizes):
        half = (sz - 1) / 2.0
        _floor_grid(ax, -half, -half, sz, sz, z=float(l),
                    face=(0.95, 0.96, 0.99, 0.22), edge=to_rgba(lcols[l], 0.42), lw=0.5)
        ax.text(-4.6, -4.4, float(l) + 0.06, f"L{l} ({sz}x{sz})", fontsize=7.5, color=lcols[l])

    grow_frames = int(0.62 * n)
    apex_path = [(0.0, 0.0), (-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0), (0.0, 0.0)]

    def update(i):
        s.clear()
        if i < grow_frames:
            depth = min(3, int((i / max(1, grow_frames - 1)) * 3.99))
            ax_x, ax_y = 0.0, 0.0
        else:
            depth = 3
            u = (i - grow_frames) / max(1, n - 1 - grow_frames) * (len(apex_path) - 1)
            k = min(len(apex_path) - 2, int(u))
            ax_x, ax_y = lerp(apex_path[k], apex_path[k + 1], ease(u - k))

        s.add(ax.scatter([ax_x], [ax_y], [3.0], color=C["red"], s=65,
                         depthshade=False, edgecolor="white"))

        frustum_faces, f_cols = [], []
        for d in range(1, depth + 1):
            l_hi, l_lo = 3 - (d - 1), 3 - d
            r_hi = d - 1 + 0.45                          # half-width on upper plane
            r_lo = d + 0.45                              # half-width on lower plane
            s.add(_wire_rect(ax, ax_x - r_lo, ax_x + r_lo, ax_y - r_lo, ax_y + r_lo,
                             float(l_lo) + 0.01, lcols[l_lo], lw=2.0))
            c_hi = [(ax_x - r_hi, ax_y - r_hi, l_hi), (ax_x + r_hi, ax_y - r_hi, l_hi),
                    (ax_x + r_hi, ax_y + r_hi, l_hi), (ax_x - r_hi, ax_y + r_hi, l_hi)]
            c_lo = [(ax_x - r_lo, ax_y - r_lo, l_lo), (ax_x + r_lo, ax_y - r_lo, l_lo),
                    (ax_x + r_lo, ax_y + r_lo, l_lo), (ax_x - r_lo, ax_y + r_lo, l_lo)]
            for side in range(4):
                nxt = (side + 1) % 4
                frustum_faces.append([c_hi[side], c_hi[nxt], c_lo[nxt], c_lo[side]])
                f_cols.append(to_rgba(C["orange"], 0.18))
            frustum_faces.append(c_lo)
            f_cols.append(to_rgba(lcols[l_lo], 0.22))

        if frustum_faces:
            s.add(ax.add_collection3d(Poly3DCollection(frustum_faces, facecolors=f_cols,
                                                       edgecolors=to_rgba(C["orange"], 0.55),
                                                       linewidths=0.7)))
        rf = 2 * depth + 1
        s.status(f"trace down {depth} layers: RF = {rf}x{rf} ({rf * rf} pixels)")
        s.spin(i, n, sweep=35)

    return update


# ------------------------------------------------------------------ 6. stride and padding

@animation("Stride and padding set output size",
           caption=r"""एक ही $5 \times 5$ तस्वीर और $3 \times 3$ filter ($f = 3$) पर तीन विन्यास बारी-बारी से चलते हैं।
           पहले **valid** ($p = 0, s = 1$) से $3 \times 3$ output बनता है। फिर चारों ओर नीली zero-padding पट्टी
           ($p = 1, s = 1$, यानी **same**) जोड़ते ही $7 \times 7$ क्षेत्र पर खिसककर पूरा $5 \times 5$ आकार सुरक्षित
           रहता है। अंत में $p = 0, s = 2$ रखने पर खिड़की दो-दो कदम कूदती है और सिर्फ़ $2 \times 2$ संक्षिप्त नक्शा देती है।""",
           frames=45, keys=(0.24, 0.62, 0.96))
def stride_and_padding(s, n):
    img5 = np.array([
        [0.2, 0.8, 0.9, 0.3, 0.2],
        [0.2, 0.9, 0.4, 0.3, 0.2],
        [0.2, 0.9, 0.8, 0.8, 0.2],
        [0.2, 0.3, 0.3, 0.9, 0.2],
        [0.2, 0.8, 0.9, 0.8, 0.2],
    ])
    ker = np.full((3, 3), 1.0 / 6.0)
    ker[1, 1] = 0.45

    regimes = [
        ("Valid: p=0, s=1 -> 3x3", 0, 1),
        ("Same: p=1, s=1 -> 5x5", 1, 1),
        ("Strided: p=0, s=2 -> 2x2", 0, 2),
    ]
    precomp = []
    for label, p, st in regimes:
        out, padded = _conv2d_single(img5, ker, pad=p, stride=st)
        pts = [(r, c) for r in range(out.shape[0]) for c in range(out.shape[1])]
        precomp.append((label, p, st, out, padded, pts))

    ax = s.ax3d(labels=("space x", "row y", "value"), elev=31, azim=-60,
                xlim=(-1.7, 13.0), ylim=(-1.7, 5.7), zlim=(0, 1.45), ticks=False)
    ax.set_box_aspect((2.35, 1.2, 0.72), zoom=0.93)

    # core 5x5 image lives at x in 0..4, y in 0..4 (padding p=1 surrounds it at -1 and 5)
    _floor_grid(ax, 0.0, 0.0, 5, 5)
    rr, cc = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    ix, iy, ih = cc.ravel().astype(float), (4 - rr.ravel()).astype(float), img5.ravel()
    ax.bar3d(ix - 0.35, iy - 0.35, np.zeros_like(ih), 0.7, 0.7, ih,
             color=plt.cm.Blues(0.28 + 0.62 * ih), alpha=0.66, shade=True,
             edgecolor=(0, 0, 0, 0.22), linewidth=0.25)
    ax.text(2.0, -1.55, 0.0, "n=5, f=3 input", fontsize=8.5, ha="center", color=C["dark"])

    per = n // 3

    def update(i):
        s.clear()
        reg_idx = min(2, i // per)
        local_t = min(1.0, (i - reg_idx * per) / max(1, per - 3))
        label, p, st, out, _, pts = precomp[reg_idx]
        nout = out.shape[0]
        step = min(len(pts) - 1, int(local_t * len(pts)))

        if p == 1:
            ring = []
            for r in range(-1, 6):
                for c in range(-1, 6):
                    if r in (-1, 5) or c in (-1, 5):
                        yc = 4 - r
                        ring.append([(c - 0.48, yc - 0.48, 0.01), (c + 0.48, yc - 0.48, 0.01),
                                     (c + 0.48, yc + 0.48, 0.01), (c - 0.48, yc + 0.48, 0.01)])
            s.add(ax.add_collection3d(Poly3DCollection(ring, facecolors=to_rgba(C["teal"], 0.25),
                                                       edgecolors=to_rgba(C["teal"], 0.8),
                                                       linewidths=0.65)))

        pr, pc = pts[step]
        top_r = pr * st - p
        left_c = pc * st - p
        x_lo, x_hi = left_c - 0.48, left_c + 2.48
        y_hi, y_lo = (4 - top_r) + 0.48, (4 - top_r - 2) - 0.48
        s.add(_wire_rect(ax, x_lo, x_hi, y_lo, y_hi, 1.08, C["red"], lw=2.2))

        ox0 = 7.4 + (5 - nout) * 0.5
        oy0 = (5 - nout) * 0.5
        s.add(_floor_grid(ax, ox0, oy0, nout, nout, face=(0.94, 0.96, 1.0, 0.55)))

        ox, oy, oh, ocols = [], [], [], []
        for k in range(step + 1):
            rr_k, cc_k = pts[k]
            v = out[rr_k, cc_k]
            ox.append(ox0 + cc_k)
            oy.append(oy0 + (nout - 1 - rr_k))
            oh.append(max(0.04, v))
            ocols.append(to_rgba(C["red"], 0.92) if k == step else
                         plt.cm.plasma(0.2 + 0.68 * min(1.0, v / 1.25)))
        ox, oy, oh = np.array(ox), np.array(oy), np.array(oh)
        s.add(ax.bar3d(ox - 0.36, oy - 0.36, np.zeros_like(oh), 0.72, 0.72, oh,
                       color=ocols, alpha=0.88, shade=True, edgecolor=(0, 0, 0, 0.3), linewidth=0.3))
        s.status(f"{label}  ({step + 1}/{nout * nout})")
        s.spin(i, n, sweep=28)

    return update
