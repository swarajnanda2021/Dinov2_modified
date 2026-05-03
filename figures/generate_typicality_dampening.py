"""
Typicality dampening — schematic (standalone DINOv2 + iBOT context).

Three lanes:
  Teacher arm (top)    — EMA copy, no gradient on targets.
  Student arm (middle) — trainable; CLS→DINO head, patches→iBOT head;
                         only the DINO CE is modulated by w(x).
  Typicality  (bottom) — tap on student CLS bottleneck wire, descend,
                         then a right-to-left pipeline:
                            tap → R → s(x) → ‖·‖₁ → d(x) → 1-Φ → t(x)
                         with a return bus at the figure floor that
                         carries w(x) = 1-β·t(x) rightward and rises up
                         to the ⊗ node on the student DINO chain.

R training row at the bottom: L_nn, L_cov → L_repr → updates R.
Bank slab between typicality and R training.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path


def make_figure(output_dir='figures',
                output_name='typicality_dampening'):
    Path(output_dir).mkdir(exist_ok=True)

    plt.rcParams.update({
        'mathtext.fontset': 'cm',
        'font.family':       'serif',
        'font.serif':        ['DejaVu Serif'],
        'axes.unicode_minus': False,
    })

    fig, ax = plt.subplots(figsize=(27.0, 13.0))
    ax.set_xlim(-2.4, 25.8)
    ax.set_ylim(-3.1, 11.6)
    ax.set_aspect('equal')
    ax.axis('off')

    C = {
        'wire':         '#3a3a3a',
        'student':      '#DCD8F5', 'student_b': '#534AB7',
        'teacher':      '#E8E5F2', 'teacher_b': '#928CB6',
        'state':        '#F0EEE6', 'state_b':   '#7A7670',
        'typ':          '#C8E8DC', 'typ_b':     '#0F6E56',
        'loss':         '#F8DCB0', 'loss_b':    '#B07A1E',
        'ibot':         '#F5D4D8', 'ibot_b':    '#A04658',
        'mix':          '#F5E0B4', 'mix_b':     '#8A5A0F',
        'aug':          '#FFE89A', 'aug_b':     '#B07A1E',
        'bank':         '#EAE6DA', 'bank_b':    '#6B6660',
        'ema':          '#534AB7',
        'detach':       '#B43030',
    }

    Z_WIRE, Z_BOX, Z_TXT, Z_DOT, Z_TOP = 1, 3, 4, 5, 6

    # ---------------- helpers ----------------
    def box(x, y, w, h, text, fc, ec, fontsize=18, lw=1.6, rounding=0.10,
            italic=False):
        ax.add_patch(FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle=f"round,pad=0.02,rounding_size={rounding}",
            fc=fc, ec=ec, lw=lw, zorder=Z_BOX))
        if text:
            ax.text(x, y, text, ha='center', va='center',
                    fontsize=fontsize,
                    style='italic' if italic else 'normal',
                    zorder=Z_TXT)

    def circ(x, y, r, text, fc, ec, fontsize=18, lw=1.6):
        ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=Z_BOX))
        if text:
            ax.text(x, y, text, ha='center', va='center',
                    fontsize=fontsize, zorder=Z_TXT)

    def hline(y, x1, x2, color=C['wire'], lw=1.4, dashed=False):
        ls = (0, (5, 3)) if dashed else '-'
        ax.add_line(Line2D([x1, x2], [y, y], color=color, lw=lw,
                           linestyle=ls, solid_capstyle='butt',
                           zorder=Z_WIRE))

    def vline(x, y1, y2, color=C['wire'], lw=1.4, dashed=False):
        ls = (0, (5, 3)) if dashed else '-'
        ax.add_line(Line2D([x, x], [y1, y2], color=color, lw=lw,
                           linestyle=ls, solid_capstyle='butt',
                           zorder=Z_WIRE))

    def vline_with_bumps(x, y1, y2, bumps_at_y=None, bump_radius=0.13,
                         color=C['wire'], lw=1.4, side='right',
                         linestyle='-'):
        bumps_at_y = sorted(b for b in (bumps_at_y or [])
                            if min(y1, y2) < b < max(y1, y2))
        ascending = y2 > y1
        if not ascending:
            bumps_at_y = list(reversed(bumps_at_y))
        cur = y1
        for by in bumps_at_y:
            seg_end = by - bump_radius if ascending else by + bump_radius
            ax.add_line(Line2D([x, x], [cur, seg_end], color=color, lw=lw,
                               linestyle=linestyle, zorder=Z_WIRE))
            theta = (np.linspace(-np.pi/2, np.pi/2, 40) if ascending
                     else np.linspace(np.pi/2, -np.pi/2, 40))
            sgn = 1 if side == 'right' else -1
            xs = x + sgn * bump_radius * np.cos(theta)
            ys = by + bump_radius * np.sin(theta)
            ax.add_line(Line2D(xs, ys, color=color, lw=lw,
                               linestyle=linestyle, zorder=Z_WIRE))
            cur = by + bump_radius if ascending else by - bump_radius
        ax.add_line(Line2D([x, x], [cur, y2], color=color, lw=lw,
                           linestyle=linestyle, zorder=Z_WIRE))

    def hline_with_bumps(y, x1, x2, bumps_at_x=None, bump_radius=0.13,
                         color=C['wire'], lw=1.4):
        bumps_at_x = sorted(b for b in (bumps_at_x or [])
                            if min(x1, x2) < b < max(x1, x2))
        cur = x1
        for bx in bumps_at_x:
            if bx - bump_radius > cur:
                ax.add_line(Line2D([cur, bx - bump_radius], [y, y],
                                   color=color, lw=lw, zorder=Z_WIRE))
            theta = np.linspace(np.pi, 0, 40)
            xs = bx + bump_radius * np.cos(theta)
            ys = y + bump_radius * np.sin(theta)
            ax.add_line(Line2D(xs, ys, color=color, lw=lw, zorder=Z_WIRE))
            cur = bx + bump_radius
        if cur < x2:
            ax.add_line(Line2D([cur, x2], [y, y], color=color, lw=lw,
                               zorder=Z_WIRE))

    def arrowhead(x, y, direction, size=0.16, color=C['wire']):
        if direction == 'right':
            pts = [(x, y), (x - size, y + size*0.55), (x - size, y - size*0.55)]
        elif direction == 'left':
            pts = [(x, y), (x + size, y + size*0.55), (x + size, y - size*0.55)]
        elif direction == 'down':
            pts = [(x, y), (x - size*0.55, y + size), (x + size*0.55, y + size)]
        elif direction == 'up':
            pts = [(x, y), (x - size*0.55, y - size), (x + size*0.55, y - size)]
        ax.add_patch(Polygon(pts, color=color, zorder=Z_DOT))

    def junction_dot(x, y, color=C['wire'], size=0.075):
        ax.add_patch(Circle((x, y), size, fc=color, ec=color, zorder=Z_DOT))

    def otimes_node(x, y, r=0.22, color=C['typ_b'], lw=1.8):
        ax.add_patch(Circle((x, y), r, fc='white', ec=color, lw=lw,
                            zorder=Z_BOX))
        # Draw × as two crossed line segments centered on (x, y).
        # Text-based × glyphs are typically offset by font metrics
        # (baseline ≠ geometric center), so we draw lines for true
        # centering. Cross arm extends r*0.55 in each direction.
        a = r * 0.55
        ax.add_line(Line2D([x - a, x + a], [y - a, y + a],
                           color=color, lw=lw, zorder=Z_TXT,
                           solid_capstyle='round'))
        ax.add_line(Line2D([x - a, x + a], [y + a, y - a],
                           color=color, lw=lw, zorder=Z_TXT,
                           solid_capstyle='round'))

    def patches_strip(x_center, y, n=8, sq=0.18, gap=0.04, fc=None, ec=None):
        total_w = n * sq + (n - 1) * gap
        x0 = x_center - total_w / 2
        for i in range(n):
            xi = x0 + i * (sq + gap)
            ax.add_patch(Rectangle((xi, y - sq/2), sq, sq,
                                   fc=fc, ec=ec, lw=1.0, zorder=Z_BOX))
        return x0, x0 + total_w

    def patches_strip_vertical(x, y_center, n=6, sq=0.16, gap=0.04,
                               fc=None, ec=None):
        """Vertical column of patch tokens. Returns (top_y, bottom_y)."""
        total_h = n * sq + (n - 1) * gap
        y0 = y_center - total_h / 2
        for i in range(n):
            yi = y0 + i * (sq + gap)
            ax.add_patch(Rectangle((x - sq/2, yi), sq, sq,
                                   fc=fc, ec=ec, lw=1.0, zorder=Z_BOX))
        return x - sq/2, x + sq/2, y0, y0 + total_h

    def bank_slab(x_center, y, n=10, sq=0.32, gap=0.04, fc=None, ec=None):
        total_w = n * sq + (n - 1) * gap
        x0 = x_center - total_w / 2
        for i in range(n):
            xi = x0 + i * (sq + gap)
            ax.add_patch(Rectangle((xi, y - sq/2), sq, sq,
                                   fc=fc, ec=ec, lw=1.0, zorder=Z_BOX))
        return x0, x0 + total_w

    def stop_grad_marker(x, y, side='right'):
        dx = 0.10 if side == 'right' else -0.10
        ha = 'left' if side == 'right' else 'right'
        ax.text(x + dx, y, 'stop-grad',
                ha=ha, va='center', fontsize=10, color=C['detach'],
                style='italic',
                bbox=dict(facecolor='white', edgecolor=C['detach'],
                          boxstyle='round,pad=0.18', lw=0.9),
                zorder=Z_TOP)

    def tall_matrix_glyph(x_center, y_center, w, h, fc, ec,
                          n_cols=3, n_rows=8, cell_fc=None):
        """Tall matrix glyph: n_cols × n_rows of filled cells + thick
        outer border. cell_fc defaults to fc. Returns (left_x, right_x).
        """
        if cell_fc is None:
            cell_fc = fc
        x0 = x_center - w/2
        y0 = y_center - h/2
        cell_w = w / n_cols
        cell_h = h / n_rows
        for i in range(n_rows):
            for j in range(n_cols):
                ax.add_patch(Rectangle(
                    (x0 + j*cell_w, y0 + i*cell_h),
                    cell_w, cell_h,
                    fc=cell_fc, ec=ec, lw=0.6, zorder=Z_BOX))
        # thick outer border
        ax.add_patch(Rectangle((x0, y0), w, h,
                               fc='none', ec=ec, lw=2.2, zorder=Z_BOX))
        return x0, x0 + w

    def square_matrix_glyph(x_center, y_center, size, fc, ec,
                            n=8, cell_fc=None):
        """Square matrix glyph: n × n cells + thick outer. Returns
        (left_x, right_x, bot_y, top_y)."""
        if cell_fc is None:
            cell_fc = fc
        x0 = x_center - size/2
        y0 = y_center - size/2
        cell = size / n
        for i in range(n):
            for j in range(n):
                ax.add_patch(Rectangle(
                    (x0 + j*cell, y0 + i*cell),
                    cell, cell,
                    fc=cell_fc, ec=ec, lw=0.5, zorder=Z_BOX))
        ax.add_patch(Rectangle((x0, y0), size, size,
                               fc='none', ec=ec, lw=2.2, zorder=Z_BOX))
        return x0, x0 + size, y0, y0 + size

    def gram_glyph(x_center, y_center, size, n=4,
                   fc_diag='#0F6E56', fc_off='#F3E9D2',
                   fc_pen='#D97A4A', pen_cells=None, ec='#5C5852'):
        """n×n Gram matrix glyph. Diagonal in fc_diag (target=1),
        off-diagonals in fc_off (target=0); cells in pen_cells get
        fc_pen to suggest "currently penalized" off-diagonals."""
        pen_cells = set(pen_cells or [])
        cell = size / n
        x0 = x_center - size/2
        y0 = y_center - size/2
        for i in range(n):
            for j in range(n):
                xi = x0 + j * cell
                yi = y0 + (n - 1 - i) * cell
                if i == j:
                    fc = fc_diag
                elif (i, j) in pen_cells or (j, i) in pen_cells:
                    fc = fc_pen
                else:
                    fc = fc_off
                ax.add_patch(Rectangle((xi, yi), cell, cell,
                                       fc=fc, ec=ec, lw=0.5,
                                       zorder=Z_BOX))
        ax.add_patch(Rectangle((x0, y0), size, size,
                               fc='none', ec=ec, lw=1.2, zorder=Z_BOX))

    # ---------------- vertical layout ----------------
    y_T_cls   = 10.10
    y_T_main  =  9.30
    y_T_patch =  8.50
    y_S_cls   =  6.50
    y_S_main  =  5.70
    y_S_patch =  4.90
    y_aug     =  7.50
    y_loss    =  2.35           # R-training row (sub-island top)
    y_merge   =  1.75           # merge wire below loss boxes (between losses and R)
    y_R       = -0.10           # R, s(x), bank all at same y (per user request)
    y_sx      =  y_R
    y_bank    =  y_R
    y_typ     = -1.70           # typicality lane (‖·‖₁, d, 1-Φ, t)
    y_Rtrain  =  y_loss
    y_return  = -2.45

    # ---------------- horizontal landmarks ----------------
    x_image  =  0.40
    x_aug    =  2.50
    x_PE     =  5.20
    x_back   =  7.50
    x_tokens = 10.00
    x_heads  = 11.80
    x_z      = 13.79
    x_Wp     = 15.10
    x_Wp_iBOT = 16.40
    x_CE_D   = 20.80
    x_CE_I   = 20.80
    x_otimes = 22.40
    x_total  = 24.30

    # ---------------- island backgrounds ----------------
    # Translucent rounded backgrounds group (patch_embed + ViT backbone)
    # per arm, and (DINO head + iBOT head) per arm. Drawn first so they
    # sit behind everything else.
    def island(x_left, y_bot, w, h, fc='#F5F3EE', ec='#9A968B', alpha=0.55):
        ax.add_patch(FancyBboxPatch(
            (x_left, y_bot), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.20",
            fc=fc, ec=ec, lw=1.0,
            linestyle=(0, (4, 2)), alpha=alpha,
            zorder=0))

    # Encoder islands: PE (4.25-6.15) + backbone (6.50-8.50); pad to
    # x ∈ [4.05, 8.65], height covers backbone (1.05) plus padding.
    island(4.05, y_T_main - 0.70, 4.60, 1.40)   # teacher encoder
    island(4.05, y_S_main - 0.70, 4.60, 1.40)   # student encoder

    # Heads islands: DINO (y_cls) + iBOT (y_patch) at x_heads=11.80
    island(10.65, y_T_patch - 0.65, 2.30, 2.90)  # teacher heads
    island(10.65, y_S_patch - 0.65, 2.30, 2.90)  # student heads

    # Single big typicality-module island. y_bot=-1.45 (clears typ lane
    # circles at y=-0.85 spanning [-1.25, -0.45]); y_top=3.725 (clears
    # R glyph top at y=3.45). h = 5.175.
    # Main typ island. +5% top and +5% bottom relative to previous
    # h=5.25 → new h=5.8125 (y_bot=-2.0625, y_top=3.7125).
    island(4.55, -2.5625, 13.20, 5.7356,
           fc='#E5F2EA', ec=C['typ_b'], alpha=0.30)

    # ---------------- input + aug ----------------
    # image x is stacked ABOVE multi-crop aug (vertical chain) so the
    # horizontal real-estate left of the patch-embed column is free for
    # clean branch routing. All three exits (globals→teacher PE,
    # globals→student PE, locals→student PE) enter the patch-embed
    # left-edge centerlines from junction points that sit OUTSIDE the
    # patch-embed boxes (avoiding the previous bug where the locals
    # branch at x=4.30 sat inside the student patch-embed box).

    y_image = y_aug + 1.45

    box(x_aug, y_image, 0.80, 0.60, 'image\n$x$',
        C['state'], C['state_b'], fontsize=13)
    vline(x_aug, y_aug + 0.475, y_image - 0.30)
    arrowhead(x_aug, y_aug + 0.475, 'down')

    box(x_aug, y_aug, 1.7, 0.95, 'multi-crop\naug',
        C['aug'], C['aug_b'], fontsize=14)

    # Branch geometry. PE_left at x_PE - 0.95 = 4.25. Aug right at
    # x_aug + 0.85 = 3.35. Branch zone (3.35, 4.25) ≈ 0.90 wide.
    x_PE_left  = x_PE - 0.95
    x_glob_jn  = 3.55          # globals junction (vline up to T, down to S)
    x_loc_jn   = 3.85          # locals junction (vline down to S only)

    # ----- globals branch: T-junction at (x_glob_jn, y_aug) -----
    hline(y_aug, x_aug + 0.85, x_glob_jn)
    junction_dot(x_glob_jn, y_aug)

    # globals → teacher patch embed (left edge centerline)
    vline(x_glob_jn, y_aug, y_T_main)
    hline(y_T_main, x_glob_jn, x_PE_left)
    arrowhead(x_PE_left, y_T_main, 'right')

    # globals → student patch embed (upper-left of box, offset above
    # centerline so the locals arrow can take the lower-left)
    y_S_glob_in = y_S_main + 0.40
    vline(x_glob_jn, y_S_glob_in, y_aug)
    hline(y_S_glob_in, x_glob_jn, x_PE_left)
    arrowhead(x_PE_left, y_S_glob_in, 'right')

    # globals label on the riser between aug and teacher PE
    ax.text(x_glob_jn + 0.08, (y_aug + y_T_main) / 2, '2 globals',
            ha='left', va='center', fontsize=11, style='italic',
            color=C['aug_b'],
            bbox=dict(facecolor='white', edgecolor='none', pad=1),
            zorder=Z_TOP)

    # ----- locals branch: terminates at student PE only -----
    # Exits aug from 6 o'clock (bottom-center), drops to y_S_loc_in,
    # jogs RIGHT into student PE lower-left.
    y_S_loc_in  = y_S_main - 0.40
    y_aug_bot6  = y_aug - 0.475
    junction_dot(x_aug, y_aug_bot6)
    vline(x_aug, y_S_loc_in, y_aug_bot6)
    hline(y_S_loc_in, x_aug, x_PE_left)
    arrowhead(x_PE_left, y_S_loc_in, 'right')

    # locals label along the descent
    ax.text(x_aug + 0.10, 6.30, r'$n$ locals',
            ha='left', va='center', fontsize=11, style='italic',
            color=C['aug_b'],
            bbox=dict(facecolor='white', edgecolor='none', pad=1),
            zorder=Z_TOP)

    # ---------------- arm drawer ----------------
    def draw_arm(y_main, y_cls, y_patch, fc, ec, expose_z=False,
                 wp_role='S', patch_to_head=True):
        box(x_PE, y_main, 1.9, 0.90, 'patch\nembed',
            fc, ec, fontsize=13, rounding=0.12)
        hline(y_main, x_PE + 0.95, x_back - 1.00)
        arrowhead(x_back - 1.00, y_main, 'right')
        box(x_back, y_main, 2.0, 1.05, 'ViT\nbackbone',
            fc, ec, fontsize=14, lw=2.2, rounding=0.14)
        x_split = x_back + 1.40
        hline(y_main, x_back + 1.00, x_split)
        junction_dot(x_split, y_main)
        vline(x_split, y_main, y_cls)
        hline(y_cls, x_split, x_tokens - 0.36)
        arrowhead(x_tokens - 0.36, y_cls, 'right')
        circ(x_tokens, y_cls, 0.36, 'CLS', fc, ec, fontsize=12)
        vline(x_split, y_patch, y_main)
        hline(y_patch, x_split, x_tokens - 0.10)
        arrowhead(x_tokens - 0.10, y_patch, 'right')
        col_L, col_R, col_T, col_B = patches_strip_vertical(
            x_tokens, y_patch, n=6, fc=fc, ec=ec)

        hline(y_cls, x_tokens + 0.36, x_heads - 0.95)
        arrowhead(x_heads - 0.95, y_cls, 'right')
        if expose_z:
            box(x_heads, y_cls, 1.9, 0.90, 'DINO\nhead',
                fc, ec, fontsize=13, rounding=0.12)
            hline(y_cls, x_heads + 0.95, x_z - 0.36)
            arrowhead(x_z - 0.36, y_cls, 'right')
            circ(x_z, y_cls, 0.36,
                 r"$z^{\,S}(x)$" if wp_role == 'S' else r"$z^{\,T}(x)$",
                 C['state'], C['state_b'], fontsize=12)
            hline(y_cls, x_z + 0.36, x_Wp - 0.367 - 0.05)
            arrowhead(x_Wp - 0.367, y_cls, 'right')
            # W_p as tall matrix glyph: 4 cols × 8 rows of cells + thick
            # outer border. 4 cols (w=0.733) makes room for centered
            # W_p^{S/T} label inside the matrix.
            Wp_left, Wp_right = tall_matrix_glyph(
                x_Wp, y_cls, w=0.733, h=1.60, fc=fc, ec=ec,
                n_cols=4, n_rows=8)
            ax.text(x_Wp, y_cls,
                    r'$W^{\,S}$' if wp_role == 'S' else r'$W^{\,T}$',
                    ha='center', va='center', fontsize=14,
                    fontweight='bold', zorder=Z_TOP,
                    bbox=dict(facecolor='white', edgecolor='none',
                              alpha=0.92, pad=2.0))
            if wp_role == 'S':
                ax.text(x_Wp - 0.50, y_cls + 0.45, '65,536',
                        ha='center', va='center', fontsize=9, style='italic',
                        color=ec, rotation=90, zorder=Z_TXT)
                ax.text(x_Wp + 0.41, y_cls - 0.88, '256',
                        ha='right', va='top', fontsize=9, style='italic',
                        color=ec, zorder=Z_TXT)
            # z'(x) circle between W_p and CE_DINO (post-projection state).
            # Both teacher and student arms expose it (DINO design).
            x_zp = (x_Wp + 0.367 + x_CE_D - 0.55) / 2
            hline(y_cls, x_Wp + 0.367, x_zp - 0.36)
            arrowhead(x_zp - 0.36, y_cls, 'right')
            circ(x_zp, y_cls, 0.36,
                 r"$z'^{\,S}(x)$" if wp_role == 'S' else r"$z'^{\,T}(x)$",
                 C['state'], C['state_b'], fontsize=11)
            x_dino_out = x_zp + 0.36
            x_zp_out = x_zp
        else:
            box(x_heads, y_cls, 2.0, 0.90, 'DINO\nhead',
                fc, ec, fontsize=13, rounding=0.12)
            x_dino_out = x_heads + 1.00
            x_zp_out = None

        if patch_to_head:
            hline(y_patch, col_R, x_heads - 0.95)
            arrowhead(x_heads - 0.95, y_patch, 'right')
        box(x_heads, y_patch, 1.9, 0.90, 'iBOT\nhead',
            fc, ec, fontsize=13, rounding=0.12)

        if expose_z:
            # iBOT z(x) bottleneck (z^patch(x))
            hline(y_patch, x_heads + 0.95, x_z - 0.36)
            arrowhead(x_z - 0.36, y_patch, 'right')
            circ(x_z, y_patch, 0.36,
                 r"$z^{\,S}_{p,i}(x)$" if wp_role == 'S'
                 else r"$z^{\,T}_{p,i}(x)$",
                 C['state'], C['state_b'], fontsize=10)
            # iBOT W_p staggered RIGHT of DINO W_p (different x column)
            hline(y_patch, x_z + 0.36, x_Wp_iBOT - 0.367 - 0.05)
            arrowhead(x_Wp_iBOT - 0.367, y_patch, 'right')
            tall_matrix_glyph(x_Wp_iBOT, y_patch, w=0.733, h=1.60,
                              fc=fc, ec=ec, n_cols=4, n_rows=8)
            ax.text(x_Wp_iBOT, y_patch,
                    r'$W^{\,S}_{p,i}$' if wp_role == 'S'
                    else r'$W^{\,T}_{p,i}$',
                    ha='center', va='center', fontsize=12,
                    fontweight='bold', zorder=Z_TOP,
                    bbox=dict(facecolor='white', edgecolor='none',
                              alpha=0.92, pad=2.0))
            if wp_role == 'S':
                ax.text(x_Wp_iBOT - 0.50, y_patch + 0.45, '65,536',
                        ha='center', va='center', fontsize=9,
                        style='italic', color=ec, rotation=90,
                        zorder=Z_TXT)
                ax.text(x_Wp_iBOT + 0.41, y_patch - 0.88, '256',
                        ha='right', va='top', fontsize=9, style='italic',
                        color=ec, zorder=Z_TXT)
            # iBOT z'(x) (z'^patch(x)) — between iBOT W_p and CE_iBOT
            x_zp_i = (x_Wp_iBOT + 0.367 + x_CE_I - 0.95) / 2 + 0.36
            hline(y_patch, x_Wp_iBOT + 0.367, x_zp_i - 0.36)
            arrowhead(x_zp_i - 0.36, y_patch, 'right')
            circ(x_zp_i, y_patch, 0.36,
                 r"$z'^{\,S}_{p,i}(x)$" if wp_role == 'S'
                 else r"$z'^{\,T}_{p,i}(x)$",
                 C['state'], C['state_b'], fontsize=9)
            x_ibot_out = x_zp_i + 0.36
            x_zp_i_out = x_zp_i
        else:
            x_ibot_out = x_heads + 0.95
            x_zp_i_out = None

        return x_dino_out, x_ibot_out, x_zp_out, x_zp_i_out

    x_T_dino_out, x_T_ibot_out, x_T_zp, x_T_zp_i = draw_arm(
        y_T_main, y_T_cls, y_T_patch,
        C['teacher'], C['teacher_b'], expose_z=True, wp_role='T')
    ax.text(-2.20, y_T_main, 'teacher\n(EMA, no grad)',
            ha='left', va='center', fontsize=13, style='italic',
            color=C['teacher_b'], fontweight='bold')

    x_S_dino_out, x_S_ibot_out, x_S_zp, x_S_zp_i = draw_arm(
        y_S_main, y_S_cls, y_S_patch,
        C['student'], C['student_b'], expose_z=True, wp_role='S',
        patch_to_head=False)
    ax.text(-2.20, y_S_main, 'student\n(trainable)',
            ha='left', va='center', fontsize=13, style='italic',
            color=C['student_b'], fontweight='bold')

    # ---------------- semantic iBOT pipeline -------------------------
    # Frozen mask model receives teacher global crop 1, outputs num_masks=3
    # soft semantic channels. One channel sampled per step → semantic token
    # mask M_sem. Random block sampler (always on) produces M_block. Both
    # are applied at the student PE→backbone wire as [MASK] token sources.
    # The student backbone runs a SECOND forward for the semantic channel,
    # producing an additional patch-token column that feeds CE_iBOT^sem.

    y_mm = 4.05  # mask pipeline row (above typ-island top y=3.71)

    # Island enclosing the mask-source branch (mask model → channels → masks)
    island(0.20, 3.35, 5.65, 1.30,
           fc='#FCEFF1', ec=C['ibot_b'], alpha=0.40)

    ax.text(-2.20, y_mm, 'semantic\niBOT',
            ha='left', va='center', fontsize=13, style='italic',
            color=C['ibot_b'], fontweight='bold')

    # Frozen mask model box
    x_mm = 1.10
    box(x_mm, y_mm, 1.55, 0.65, 'frozen\nmask model',
        C['ibot'], C['ibot_b'], fontsize=11, lw=1.4, rounding=0.10)

    # Aug LEFT (9 o'clock) → mask model TOP (12 o'clock).
    x_aug_left = x_aug - 0.85
    y_mm_top = y_mm + 0.325
    junction_dot(x_aug_left, y_aug, color=C['ibot_b'])
    hline(y_aug, x_mm, x_aug_left, color=C['ibot_b'], lw=1.4)
    vline(x_mm, y_mm_top, y_aug, color=C['ibot_b'], lw=1.4)
    arrowhead(x_mm, y_mm_top, 'down', color=C['ibot_b'])
    ax.text(x_mm + 0.10, (y_aug + y_mm_top) / 2, 'global 1',
            ha='left', va='center', fontsize=9, style='italic',
            color=C['ibot_b'])

    # 3 semantic channels glyph (middle highlighted = sampled this step)
    x_3ch = 2.85
    ch_size, ch_gap = 0.30, 0.06
    ch_total = 3 * ch_size + 2 * ch_gap
    x_3ch_left = x_3ch - ch_total / 2
    for i in range(3):
        cx = x_3ch_left + i * (ch_size + ch_gap)
        fc = C['ibot_b'] if i == 1 else C['ibot']
        ax.add_patch(Rectangle((cx, y_mm - ch_size / 2),
                               ch_size, ch_size,
                               fc=fc, ec=C['ibot_b'], lw=1.0, zorder=Z_BOX))
        txt_color = 'white' if i == 1 else C['ibot_b']
        ax.text(cx + ch_size / 2, y_mm, str(i + 1),
                ha='center', va='center', fontsize=9,
                color=txt_color, fontweight='bold', zorder=Z_BOX + 1)
    ax.text(x_3ch, y_mm - ch_size / 2 - 0.10, r'$\sim\mathrm{Unif}(3)$',
            ha='center', va='top', fontsize=9, style='italic',
            color=C['ibot_b'])
    hline(y_mm, x_mm + 0.775, x_3ch_left, color=C['ibot_b'], lw=1.2)
    arrowhead(x_3ch_left, y_mm, 'right', color=C['ibot_b'])

    # Semantic mask 4×4 grid (selected channel as 2D token mask)
    grid_size, n_grid = 0.65, 4
    cell = grid_size / n_grid
    x_grid_sem = 4.30
    sem_cells = {(0, 0), (0, 1), (1, 0), (1, 1), (2, 2), (3, 2)}
    for i in range(n_grid):
        for j in range(n_grid):
            cx = x_grid_sem - grid_size / 2 + j * cell
            cy = y_mm - grid_size / 2 + (n_grid - 1 - i) * cell
            fc = C['ibot_b'] if (i, j) in sem_cells else C['ibot']
            ax.add_patch(Rectangle((cx, cy), cell, cell,
                                   fc=fc, ec=C['ibot_b'], lw=0.4,
                                   zorder=Z_BOX))
    ax.add_patch(Rectangle(
        (x_grid_sem - grid_size / 2, y_mm - grid_size / 2),
        grid_size, grid_size, fc='none', ec=C['ibot_b'], lw=1.2,
        zorder=Z_BOX))
    ax.text(x_grid_sem, y_mm - grid_size / 2 - 0.10,
            r'$M_{\mathrm{sem}}$',
            ha='center', va='top', fontsize=11, style='italic',
            color=C['ibot_b'])
    hline(y_mm, x_3ch + ch_total / 2, x_grid_sem - grid_size / 2,
          color=C['ibot_b'], lw=1.2)
    arrowhead(x_grid_sem - grid_size / 2, y_mm, 'right', color=C['ibot_b'])

    # Block mask 4×4 grid (random rectangular block) — parallel source.
    # Uses a distinct color (steel-blue) to differentiate block-mask
    # source from the pink semantic-mask source.
    block_color = '#3B6FB0'
    block_color_light = '#CFDDED'
    x_grid_block = 5.40
    block_cells = {(1, 1), (1, 2), (2, 1), (2, 2)}
    for i in range(n_grid):
        for j in range(n_grid):
            cx = x_grid_block - grid_size / 2 + j * cell
            cy = y_mm - grid_size / 2 + (n_grid - 1 - i) * cell
            fc = block_color if (i, j) in block_cells else block_color_light
            ax.add_patch(Rectangle((cx, cy), cell, cell,
                                   fc=fc, ec=block_color, lw=0.4,
                                   zorder=Z_BOX))
    ax.add_patch(Rectangle(
        (x_grid_block - grid_size / 2, y_mm - grid_size / 2),
        grid_size, grid_size, fc='none', ec=block_color, lw=1.2,
        zorder=Z_BOX))
    ax.text(x_grid_block, y_mm - grid_size / 2 - 0.10,
            r'$M_{\mathrm{block}}$',
            ha='center', va='top', fontsize=11, style='italic',
            color=block_color)

    # Both grids flatten upward → student PE→backbone wire at x_inject
    y_route  = 4.85
    x_inject = 6.30
    vline(x_grid_sem,   y_mm + grid_size / 2, y_route,
          color=C['ibot_b'], lw=1.2)
    vline(x_grid_block, y_mm + grid_size / 2, y_route,
          color=C['ibot_b'], lw=1.2)
    junction_dot(x_grid_block, y_route, color=C['ibot_b'])
    hline(y_route, x_grid_sem, x_inject, color=C['ibot_b'], lw=1.2)
    vline(x_inject, y_route, y_S_main - 0.05,
          color=C['ibot_b'], lw=1.2)
    arrowhead(x_inject, y_S_main - 0.05, 'up', color=C['ibot_b'])
    junction_dot(x_inject, y_S_main, color=C['ibot_b'])
    ax.text(x_inject + 0.08, y_route - 0.04, '[MASK] tokens',
            ha='left', va='top', fontsize=8, style='italic',
            color=C['ibot_b'])

    # Second patches column on student (semantic forward output).
    # Highlighted cells = masked positions whose student outputs feed
    # CE_iBOT^sem against teacher unmasked targets at same positions.
    x_tokens_sem = x_tokens + 0.25
    col2_L, col2_R, col2_B, col2_T = patches_strip_vertical(
        x_tokens_sem, y_S_patch, n=6,
        fc=C['student'], ec=C['student_b'])
    sq_p, gap_p, n_p = 0.16, 0.04, 6
    total_h_p = n_p * sq_p + (n_p - 1) * gap_p
    y0_p = y_S_patch - total_h_p / 2
    masked_idx = {1, 4}
    for i in masked_idx:
        yi = y0_p + i * (sq_p + gap_p)
        ax.add_patch(Rectangle((x_tokens_sem - sq_p / 2, yi),
                               sq_p, sq_p,
                               fc=C['ibot_b'], ec=C['student_b'], lw=1.0,
                               zorder=Z_BOX + 1))
    ax.text(x_tokens_sem, col2_T + 0.08, 'sem-masked',
            ha='center', va='bottom', fontsize=8, style='italic',
            color=C['ibot_b'])

    # Overlay block-masked cells on the FIRST patches column (existing
    # student strip at x_tokens). Distinct block color, label below.
    block_masked_idx = {2, 3}
    for i in block_masked_idx:
        yi = y0_p + i * (sq_p + gap_p)
        ax.add_patch(Rectangle((x_tokens - sq_p / 2, yi),
                               sq_p, sq_p,
                               fc=block_color, ec=C['student_b'], lw=1.0,
                               zorder=Z_BOX + 1))
    col1_B = y0_p
    ax.text(x_tokens, col1_B - 0.08, 'block-masked',
            ha='center', va='top', fontsize=8, style='italic',
            color=block_color)

    # Single black arrow from sem-masked (2nd) column into iBOT head.
    # Two mask types feed the same head; arrow source on either column
    # is conventional — sem chosen so the arrow doesn't cross M_block
    # block-mask cells visually.
    hline(y_S_patch, col2_R, x_heads - 0.95)
    arrowhead(x_heads - 0.95, y_S_patch, 'right')

    # ---------------- losses on student rows ----------------
    box(x_CE_D, y_S_cls, 1.9, 0.95, r'$\mathrm{CE}_{\mathrm{DINO}}$',
        C['loss'], C['loss_b'], fontsize=18, lw=1.6, rounding=0.10)
    hline(y_S_cls, x_S_dino_out, x_CE_D - 0.95)
    arrowhead(x_CE_D - 0.95, y_S_cls, 'right')

    box(x_CE_I, y_S_patch, 1.9, 0.95,
        r'$\mathrm{CE}_{\mathrm{iBOT}}^{\mathrm{block}}$',
        C['ibot'], C['ibot_b'], fontsize=16, lw=1.6, rounding=0.10)
    hline(y_S_patch, x_S_ibot_out, x_CE_I - 0.95)
    arrowhead(x_CE_I - 0.95, y_S_patch, 'right')

    # CE_iBOT^sem: parallel sub-row UNDER CE_iBOT^block. Same iBOT head /
    # W_p,i / z'^p,i upstream — branch off the z'^p,i → CE_iBOT^block
    # wire and route DOWN, then RIGHT into CE_iBOT^sem from the left.
    y_CE_sem = 3.85
    box(x_CE_I, y_CE_sem, 1.9, 0.85,
        r'$\mathrm{CE}_{\mathrm{iBOT}}^{\mathrm{sem}}$',
        C['ibot'], C['ibot_b'], fontsize=14, lw=1.6, rounding=0.10)
    x_branch_sem = x_S_ibot_out + 0.30
    y_sem_S = y_CE_sem - 0.10
    junction_dot(x_branch_sem, y_S_patch, color=C['ibot_b'])
    vline(x_branch_sem, y_sem_S, y_S_patch, color=C['ibot_b'], lw=1.4)
    hline(y_sem_S, x_branch_sem, x_CE_I - 0.95,
          color=C['ibot_b'], lw=1.4)
    arrowhead(x_CE_I - 0.95, y_sem_S, 'right', color=C['ibot_b'])

    # teacher target wires (dashed). Junction dots anchor the head exits
    # so the dashed wires read as continuations of the boxes, not as
    # orphaned trails. One consolidated sg marker on the DINO target stem
    # conveys the teacher EMA detach for both targets.

    # DINO target: drop straight down onto top of CE_DINO
    x_tgt_D = x_CE_D + 0.55
    junction_dot(x_T_dino_out, y_T_cls, color=C['teacher_b'])
    hline(y_T_cls, x_T_dino_out, x_tgt_D, color=C['teacher_b'], dashed=True)
    vline(x_tgt_D, y_S_cls + 0.475, y_T_cls,
          color=C['teacher_b'], dashed=True)
    arrowhead(x_tgt_D, y_S_cls + 0.475, 'down', color=C['teacher_b'])

    # Teacher iBOT target: from teacher z'^p(x), jog LEFT at y_T_patch
    # to drop column x_tgt_I=17.40 (in 0.13 gap between iBOT W_p,i right
    # edge 17.32 and CE_DINO left edge 17.45), drop to y_TI_entry,
    # then jog RIGHT into CE_iBOT left edge.
    x_tgt_I    = 19.16
    y_block_top = 4.90 + 0.475   # CE^block top edge = 5.375
    y_sem_bot   = 3.85 - 0.425   # CE^sem bottom edge = 3.425
    y_branch_T_top = y_block_top + 0.30   # 5.675
    y_branch_T_bot = y_sem_bot - 0.30     # 3.125
    junction_dot(x_T_zp_i + 0.36, y_T_patch, color=C['teacher_b'])
    hline(y_T_patch, x_T_zp_i + 0.36, x_tgt_I,
          color=C['teacher_b'], dashed=True)
    vline_with_bumps(x_tgt_I, y_branch_T_bot, y_T_patch,
                     bumps_at_y=[y_S_cls, y_S_patch], bump_radius=0.18,
                     color=C['teacher_b'], lw=1.4, side='right',
                     linestyle=(0, (5, 3)))
    # Block target: branch at y_branch_T_top, hline RIGHT, drop DOWN onto top edge
    junction_dot(x_tgt_I, y_branch_T_top, color=C['teacher_b'])
    ax.add_line(Line2D([x_tgt_I, x_CE_I], [y_branch_T_top, y_branch_T_top],
                       color=C['teacher_b'], lw=1.2,
                       linestyle=(0, (5, 3)), zorder=Z_WIRE))
    ax.add_line(Line2D([x_CE_I, x_CE_I], [y_branch_T_top, y_block_top],
                       color=C['teacher_b'], lw=1.2,
                       linestyle=(0, (5, 3)), zorder=Z_WIRE))
    arrowhead(x_CE_I, y_block_top, 'down', color=C['teacher_b'])
    stop_grad_marker(x_tgt_I, 7.40, side='left')

    # Sem target: at descent bottom (y_branch_T_bot), hline RIGHT, vline UP onto bottom edge
    ax.add_line(Line2D([x_tgt_I, x_CE_I], [y_branch_T_bot, y_branch_T_bot],
                       color=C['teacher_b'], lw=1.2,
                       linestyle=(0, (5, 3)), zorder=Z_WIRE))
    ax.add_line(Line2D([x_CE_I, x_CE_I], [y_branch_T_bot, y_sem_bot],
                       color=C['teacher_b'], lw=1.2,
                       linestyle=(0, (5, 3)), zorder=Z_WIRE))
    arrowhead(x_CE_I, y_sem_bot, 'up', color=C['teacher_b'])

    # Single sg marker for the teacher EMA boundary
    stop_grad_marker(x_tgt_D, y_S_cls + 1.20, side='left')

    # ⊗ on student DINO
    hline(y_S_cls, x_CE_D + 0.95, x_otimes - 0.22)
    otimes_node(x_otimes, y_S_cls)
    hline(y_S_cls, x_otimes + 0.22, x_total - 1.30)
    arrowhead(x_total - 1.30, y_S_cls, 'right')
    # iBOT → total directly
    hline(y_S_patch, x_CE_I + 0.95, x_total - 1.30)
    arrowhead(x_total - 1.30, y_S_patch, 'right')
    # CE_iBOT^sem → total (step UP since sem CE sits below L_total bottom)
    y_into_total_sem = 4.50
    hline(y_CE_sem, x_CE_I + 0.95, x_CE_I + 1.40, color=C['ibot_b'])
    vline(x_CE_I + 1.40, y_CE_sem, y_into_total_sem, color=C['ibot_b'])
    hline(y_into_total_sem, x_CE_I + 1.40, x_total - 1.30,
          color=C['ibot_b'])
    arrowhead(x_total - 1.30, y_into_total_sem, 'right',
              color=C['ibot_b'])

    box(x_total, y_S_main, 2.6, 2.70,
        r'$\mathcal{L}_{\mathrm{total}}$' + '\n\n' +
        r'$=\;w(x)\,\mathrm{CE}_{\mathrm{DINO}}$' + '\n' +
        r'$+\;\mathrm{CE}_{\mathrm{iBOT}}^{\mathrm{block}}$' + '\n' +
        r'$+\;\lambda_{\mathrm{sem}}\,\mathrm{CE}_{\mathrm{iBOT}}^{\mathrm{sem}}$',
        C['mix'], C['mix_b'], fontsize=13, lw=2.2, rounding=0.12)

    # ---------------- EMA arrows ----------------
    def ema_arrow(x, y_from, y_to, bumps_at_y=None, bump_radius=0.18,
                  label_y_offset=0.0):
        bumps = sorted(b for b in (bumps_at_y or [])
                       if min(y_from, y_to) < b < max(y_from, y_to))
        ascending = y_to > y_from
        if not ascending:
            bumps = list(reversed(bumps))
        cur = y_from
        for by in bumps:
            seg_end = by - bump_radius if ascending else by + bump_radius
            ax.add_line(Line2D([x, x], [cur, seg_end], color=C['ema'],
                               lw=2.2, linestyle=(0, (4, 2)),
                               solid_capstyle='butt', zorder=Z_WIRE))
            theta = (np.linspace(-np.pi/2, np.pi/2, 40) if ascending
                     else np.linspace(np.pi/2, -np.pi/2, 40))
            xs = x + bump_radius * np.cos(theta)
            ys = by + bump_radius * np.sin(theta)
            ax.add_line(Line2D(xs, ys, color=C['ema'], lw=2.2,
                               linestyle=(0, (4, 2)), zorder=Z_WIRE))
            cur = by + bump_radius if ascending else by - bump_radius
        ax.add_line(Line2D([x, x], [cur, y_to], color=C['ema'], lw=2.2,
                           linestyle=(0, (4, 2)),
                           solid_capstyle='butt', zorder=Z_WIRE))
        arrowhead(x, y_to, 'up', color=C['ema'], size=0.20)
        ax.text(x + 0.18, (y_from + y_to) / 2 + label_y_offset, 'EMA',
                ha='left', va='center', fontsize=12, style='italic',
                color=C['ema'], fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', pad=1.5),
                zorder=Z_TOP)

    ema_arrow(x_PE,    y_S_main + 0.45, y_T_main - 0.45)        # patch embed
    ema_arrow(x_back,  y_S_main + 0.525, y_T_main - 0.525)      # ViT backbone
    ema_arrow(x_heads, y_S_cls  + 0.45, y_T_patch - 0.45)       # heads (student DINO ↔ teacher iBOT)
    ema_arrow(x_Wp,    y_S_cls  + 0.85, y_T_cls   - 0.85,
              bumps_at_y=[y_T_patch], label_y_offset=-0.55)   # bump at iBOT row crossing
    ema_arrow(x_Wp_iBOT, y_S_patch + 0.85, y_T_patch - 0.85,
              bumps_at_y=[y_S_cls], label_y_offset=0.55)    # bump at DINO row crossing; label shifted up

    # ---------------- typicality lane (inverted layout) ---------------
    # Tap is at z'(x) circle on student arm. From z'(x), drop down into
    # R right edge (R sits below the sub-island, with z'(x) feeding it
    # from above-right).
    x_tap = x_S_zp                          # ≈16.46 (z'(x) on student)

    # right-to-left pipeline coordinates
    x_R     = x_Wp                          # R sits under W_p^S column (15.10)
    x_s     = x_R    - 1.85
    x_dist  = x_s    - 1.95
    x_d     = x_dist - 1.70
    x_phi   = x_d    - 1.70
    x_t     = x_phi  - 1.65

    # z'(x) descent: drops from z'(x) bottom (y_S_cls - 0.36) all the
    # way to y_R, single fat bump at y=5.00 covers iBOT crossings.
    # At y_R, jog LEFT into R right edge (R right at x=15.70, z'(x) at
    # x=16.46 → length 0.76).
    vline_with_bumps(x_tap, y_R, y_S_cls - 0.36,
                     bumps_at_y=[5.00], bump_radius=0.22,
                     color=C['detach'], lw=1.2, side='right',
                     linestyle=(0, (4, 2)))
    hline(y_R, x_R + 0.60, x_tap, color=C['detach'], lw=1.2, dashed=True)
    arrowhead(x_R + 0.60, y_R, 'left', color=C['detach'])
    stop_grad_marker(x_tap, 4.10, side='right')

    # R as square matrix glyph (8×8 cells + thick outer)
    R_left, R_right, R_bot, R_top = square_matrix_glyph(
        x_R, y_R, size=1.20, fc=C['typ'], ec=C['typ_b'], n=8)
    # R label inside, above 256x256
    ax.text(x_R, y_R + 0.18, r'$R$',
            ha='center', va='center', fontsize=14,
            fontweight='bold', color=C['typ_b'], zorder=Z_TOP,
            bbox=dict(facecolor='white', edgecolor='none',
                      alpha=0.92, pad=2.0))
    ax.text(x_R, y_R - 0.18, r'$256 \times 256$',
            ha='center', va='center', fontsize=10,
            color=C['typ_b'], style='italic',
            bbox=dict(facecolor='white', edgecolor='none',
                      alpha=0.92, pad=2.0),
            zorder=Z_TOP)
    # SE annotation split into two lines
    ax.text(R_right + 0.05, R_bot - 0.05,
            'unit-sphere\n$\\approx$ orthonormal',
            ha='left', va='top', fontsize=9, style='italic',
            color=C['typ_b'], linespacing=1.2, zorder=Z_TXT)

    # R → s(x): horizontal at y_R from R left edge to s(x) right edge.
    # Both at SAME y for a clean horizontal connection.
    hline(y_R, x_s + 0.40, x_R - 0.60)
    arrowhead(x_s + 0.40, y_R, 'left')
    circ(x_s, y_R, 0.40, r'$s(x)$',
         C['state'], C['state_b'], fontsize=14)

    # s(x) → ‖·‖₁
    # s(x) → ‖·‖₁: s(x) at y_R, dist box at y_typ. L-path: drop at x_s
    # from s(x) bottom down to y_typ, then horizontal LEFT into dist box
    # right edge.
    vline(x_s, y_typ, y_R - 0.40)
    hline(y_typ, x_dist + 0.85, x_s)
    arrowhead(x_dist + 0.85, y_typ, 'left')
    box(x_dist, y_typ, 1.7, 0.95,
        r'$\|\cdot\|_1$' + '\nvs. bank',
        C['typ'], C['typ_b'], fontsize=13, lw=1.6, rounding=0.10)

    # ‖·‖₁ → d(x)
    hline(y_typ, x_d + 0.40, x_dist - 0.85)
    arrowhead(x_d + 0.40, y_typ, 'left')
    circ(x_d, y_typ, 0.40, r'$d(x)$',
         C['state'], C['state_b'], fontsize=14)

    # d(x) → 1-Φ
    hline(y_typ, x_phi + 0.85, x_d - 0.40)
    arrowhead(x_phi + 0.85, y_typ, 'left')
    box(x_phi, y_typ, 1.7, 0.95,
        r'$1-\Phi\!\left(\frac{d-\mu}{\sigma}\right)$',
        C['typ'], C['typ_b'], fontsize=13, lw=1.6, rounding=0.10)

    # 1-Φ → t(x)
    hline(y_typ, x_t + 0.40, x_phi - 0.85)
    arrowhead(x_t + 0.40, y_typ, 'left')
    circ(x_t, y_typ, 0.40, r'$t(x)$',
         C['state'], C['state_b'], fontsize=14)

    # μ, σ annotation feeding into 1-Φ from below (just text, no wire)
    ax.text(x_phi, y_typ + 0.78, r'$\mu,\sigma$ from within-bank NN',
            ha='center', va='bottom', fontsize=10, style='italic',
            color=C['typ_b'],
            bbox=dict(facecolor='white', edgecolor='none', pad=1.5),
            zorder=Z_TOP)

    # ---------------- bank slab ----------------
    # Bank is at SAME y as s(x) and R. n=6 (slab width ≈1.8) so it fits
    # left of s(x) without overlapping.
    bank_L, bank_R = bank_slab(x_dist, y_bank, n=6,
                               fc=C['bank'], ec=C['bank_b'])
    # bank → dist read: vertical drop from bank bottom to dist box top
    vline(x_dist, y_typ + 0.475, y_bank - 0.16)
    arrowhead(x_dist, y_typ + 0.475, 'down')
    ax.text(x_dist + 0.18, (y_bank + y_typ) / 2, 'query / dist',
            ha='left', va='center', fontsize=9, style='italic',
            color=C['bank_b'])
    # bank label: above the slab now (since lane is below)
    ax.text(x_dist, y_bank + 0.30,
            r'bank  $M\times K^{\prime}$' +
            '\nmost-novel\ndisplaces-nearest',
            ha='center', va='bottom', fontsize=9, style='italic',
            color=C['bank_b'], linespacing=1.2)

    # bank update mechanism is conveyed by the "most-novel-displaces-nearest"
    # annotation below; explicit s(x)→bank wire would visually overlap the
    # slab itself, so we omit it.

    # ---------------- return path: t(x) → ⊗ ----------------
    # t(x) drops to y_return, runs right to x_otimes, rises to ⊗
    vline(x_t, y_return, y_typ - 0.40)
    # along the way, the return bus crosses no labelled obstacles at
    # y_return (it's below everything else in the figure)
    hline(y_return, x_t, x_otimes)
    # rise: bump where it crosses the iBOT→total loss wire at y_S_patch
    vline_with_bumps(x_otimes, y_return, y_S_cls - 0.22,
                     bumps_at_y=[y_S_patch, 4.50], side='right')
    arrowhead(x_otimes, y_S_cls - 0.22, 'up')
    # w(x) label: ABOVE the return line (inside the typ island), shifted
    # RIGHT to sit clear of the ‖·‖₁ vs. bank box (centered in the open
    # horizontal span between dist box right edge and x_otimes).
    ax.text(14.0, y_return + 0.12,
            r'$w(x) \;=\; 1 \;-\; \beta \cdot t(x)$',
            ha='center', va='bottom', fontsize=14, style='italic',
            color=C['typ_b'])

    ax.text(-2.20, y_typ, 'typicality\ndampening',
            ha='left', va='center', fontsize=13, style='italic',
            color=C['typ_b'], fontweight='bold')

    # ---------------- R training row ----------------
    # Both losses sit to the LEFT of R (R drops down INTO the sub-island
    # to feed both losses; their merge L_repr returns UP to R).
    x_Lnn   = x_R - 3.40           # 11.70 (more room left of L_cov)
    x_Lcov  = x_R - 1.40           # 13.70 (gram glyph center)
    x_Lrepr = (x_Lnn + x_Lcov) / 2 # 12.70 merge dot

    # Sub-island encloses the R-training row.
    # Sub-island encloses the R-training row.
    sub_y_bot = 1.05                   # lowered (more breathing room below L_repr label)
    sub_y_top = y_loss  + 0.55         # 3.40
    sub_x_left  = x_Lnn - 1.15         # 10.55
    sub_x_right = x_Lcov + 1.55        # 15.25 (extended right to fit L_cov labels)
    island(sub_x_left, sub_y_bot, sub_x_right - sub_x_left,
           sub_y_top - sub_y_bot,
           fc='#D6EADC', ec=C['typ_b'], alpha=0.55)

    box(x_Lnn, y_Rtrain, 2.09, 0.85,
        r'$\mathcal{L}_{\mathrm{nn}}$' + r' (vs. $W^{\,S}$ rows)',
        C['typ'], C['typ_b'], fontsize=11, lw=1.4, rounding=0.10)

    # L_cov as a Gram-matrix glyph: 5×5 grid. Centered AT x_Lcov so it
    # connects to the merge stub directly below. Lifted up by 0.10 for
    # better balance with L_nn box.
    gram_glyph(x_Lcov, y_Rtrain + 0.10, size=0.72, n=5,
               fc_diag=C['typ_b'], fc_off='#F3E9D2',
               fc_pen='#D97A4A',
               pen_cells=[(0, 2), (1, 4)],
               ec=C['typ_b'])
    ax.text(x_Lcov + 0.45, y_Rtrain + 0.30,
            r'$\mathcal{L}_{\mathrm{cov}}$',
            ha='left', va='center', fontsize=14, fontweight='bold',
            color=C['typ_b'], zorder=Z_TXT)
    # 'off-diag of RR^T' moved BELOW the R→L_cov arrow line (which sits
    # at y_Rtrain+0.10). Place text at y_Rtrain - 0.20 → fully below
    # the arrow / gram glyph.
    ax.text(x_Lcov + 0.45, y_Rtrain - 0.30,
            'off-diag\nof $RR^{\\!\\top}$',
            ha='left', va='top', fontsize=9, style='italic',
            color=C['typ_b'], linespacing=1.2, zorder=Z_TXT)

    # vertical stubs from each loss element BOTTOM down to merge wire
    # (merge sits below losses now, between sub-island and R).
    y_box_bot_nn  = y_Rtrain - 0.425          # L_nn box bottom
    y_box_bot_cov = y_Rtrain - 0.26           # gram glyph bottom (lifted by 0.10)
    x_Lcov_stub   = x_Lcov                    # gram center x
    # NOTE: y_merge is set in the y-stack at top (= 2.25)

    vline(x_Lnn,       y_merge, y_box_bot_nn,  color=C['typ_b'])
    vline(x_Lcov_stub, y_merge, y_box_bot_cov, color=C['typ_b'])
    hline(y_merge, x_Lnn, x_Lcov_stub, color=C['typ_b'])
    junction_dot(x_Lrepr, y_merge, color=C['typ_b'])

    # L_repr -> R (recursive update): merge dot DOWN to R top.
    # Path: from merge dot, drop straight to R top at y=R_top=1.80.
    # x_Lrepr (=12.70) is LEFT of R (15.10), so we need a jog: drop to
    # y=2.00 (just above R top), jog RIGHT to x=R_left+0.20=14.70 (into
    # R top-left corner area), arrowhead pointing DOWN onto R top.
    y_R_top_entry = R_top                      # 1.80
    x_R_top_entry = x_R - 0.40                 # 14.70 (top-left of R)
    vline(x_Lrepr, 1.50, y_merge, color=C['typ_b'])
    hline(1.50, x_R_top_entry, x_Lrepr, color=C['typ_b'])
    vline(x_R_top_entry, y_R_top_entry, 1.50, color=C['typ_b'])
    arrowhead(x_R_top_entry, y_R_top_entry, 'down', color=C['typ_b'])

    # R → L_cov: from R top-center UP to gram-glyph y, then LEFT into
    # gram right edge (now at y_Rtrain+0.10).
    junction_dot(x_R, R_top, color=C['typ_b'])
    vline(x_R, y_Rtrain + 0.10, R_top, color=C['typ_b'])
    hline(y_Rtrain + 0.10, x_Lcov + 0.36, x_R, color=C['typ_b'])
    arrowhead(x_Lcov + 0.36, y_Rtrain + 0.10, 'left', color=C['typ_b'])

    # R → L_nn: branch off the R→L_cov vertical at y=1.75 (lower than
    # before, well inside sub-island, above bank label). Horizontal LEFT
    # to a column just left of L_nn box, then short L-jog UP to enter
    # L_nn at its left-edge vertical center. Bump where the horizontal
    # crosses the L_repr→R vertical at x = x_R - 0.40.
    y_RLnn      = 1.25
    x_Lnn_bot   = x_Lnn - 0.60                    # offset left of center to avoid merge wire (which starts at x_Lnn)
    y_Lnn_bot   = y_Rtrain - 0.425                # L_nn box bottom
    junction_dot(x_R, y_RLnn, color=C['typ_b'])
    hline_with_bumps(y_RLnn, x_Lnn_bot, x_R,
                     bumps_at_x=[x_R - 0.40], bump_radius=0.18,
                     color=C['typ_b'], lw=1.4)
    vline(x_Lnn_bot, y_RLnn, y_Lnn_bot, color=C['typ_b'])
    arrowhead(x_Lnn_bot, y_Lnn_bot, 'up', color=C['typ_b'])

    ax.text(-2.20, y_Rtrain, '$R$ training',
            ha='left', va='center', fontsize=13, style='italic',
            color=C['typ_b'], fontweight='bold')

    # ---------------- W_p^S -> sub-island (target for L_nn) -----------
    # Dashed-grey stop-grad reference. W_p^S rows are the target L_nn
    # pulls R toward. Wire enters sub-island from TOP and lands on
    # L_nn box top. R input to L_nn (and R input to L_cov) is implicit
    # — conveyed by the recursive L_repr→R update wire and the loss
    # labels themselves; no explicit R→loss wires needed.
    y_dash_h = sub_y_top + 0.10            # 3.50, just above sub-island top
    junction_dot(x_Wp, y_S_cls - 0.80, color=C['detach'])
    # drop from W_p^S bottom to y_dash_h, single fat bump at y=5.00
    vline_with_bumps(x_Wp, y_dash_h, y_S_cls - 0.80,
                     bumps_at_y=[5.00], bump_radius=0.22,
                     color=C['detach'], lw=1.2, side='right',
                     linestyle=(0, (4, 2)))
    # horizontal LEFT to x_Lnn column
    ax.add_line(Line2D([x_Lnn, x_Wp], [y_dash_h, y_dash_h],
                       color=C['detach'], lw=1.2,
                       linestyle=(0, (4, 2)), zorder=Z_WIRE))
    # drop into L_nn top
    ax.add_line(Line2D([x_Lnn, x_Lnn], [y_loss + 0.425, y_dash_h],
                       color=C['detach'], lw=1.2,
                       linestyle=(0, (4, 2)), zorder=Z_WIRE))
    arrowhead(x_Lnn, y_loss + 0.425, 'down', color=C['detach'])
    stop_grad_marker(x_Wp, 4.40, side='left')

    # NOTE: R input to L_nn and L_cov is implicit. The loss labels
    # ("L_nn vs. W_p^S rows" and "L_cov off-diag of RR^T") plus the
    # recursive L_repr→R update wire convey the dependence; explicit
    # R→loss arrows would clutter without adding information.

    # ---------------- bank write path ---------------------------------
    # s(x) → bank: horizontal at y_R from s(x) 9 o'clock LEFT to bank
    # right edge (both at SAME y now).
    junction_dot(x_s - 0.40, y_R, color=C['bank_b'])
    hline(y_R, bank_R, x_s - 0.40, color=C['bank_b'])
    arrowhead(bank_R, y_R, 'left', color=C['bank_b'])
    ax.text((bank_R + x_s - 0.40) / 2, y_R - 0.20, 'store',
            ha='center', va='top', fontsize=9, style='italic',
            color=C['bank_b'])

    # ---------------- save ----------------
    base = f'{output_dir}/{output_name}'
    plt.savefig(f'{base}.svg', bbox_inches='tight', pad_inches=0.25)
    plt.savefig(f'{base}.pdf', bbox_inches='tight', pad_inches=0.25)
    plt.savefig(f'{base}.png', bbox_inches='tight', pad_inches=0.25, dpi=180)
    plt.close(fig)
    return base


if __name__ == '__main__':
    out = make_figure()
    print(f'wrote: {out}.svg, {out}.pdf, {out}.png')
