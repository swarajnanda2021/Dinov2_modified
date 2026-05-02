"""
Looped DINOv2 method — schematic (Version B: explicit residual pathway).

Input injection is rendered as a residual-style bypass: z_0 fans out to
both f_theta_1 (along the recursion lane) and to a horizontal bypass
highway that taps into a + (sum) node placed between each f_theta_t and
its z_t.  The bypass + ⊕ pattern makes the residual structure
z_t = f_theta(...) + z_0 visually explicit, mirroring how skip
connections are drawn in ResNet/Transformer figures.
"""

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path


def make_figure(output_dir='figures',
                output_name='looped_dinov2_method_residual'):
    Path(output_dir).mkdir(exist_ok=True)

    plt.rcParams.update({
        'mathtext.fontset': 'cm',
        'font.family':       'serif',
        'font.serif':        ['DejaVu Serif'],
        'axes.unicode_minus': False,
    })

    fig, ax = plt.subplots(figsize=(18.5, 9.5))
    ax.set_xlim(-1.6, 21.2)
    ax.set_ylim( 0.5,  9.5)
    ax.set_aspect('equal')
    ax.axis('off')

    C = {
        'wire':    '#3a3a3a',
        'shared':  '#DCD8F5', 'shared_b':  '#534AB7',
        'state':   '#F0EEE6', 'state_b':   '#7A7670',
        'halt':    '#C8E8DC', 'halt_b':    '#0F6E56',
        'loss':    '#F8DCB0', 'loss_b':    '#B07A1E',
        'tau':     '#FFE89A', 'tau_b':     '#B07A1E',
        'mix':     '#F5E0B4', 'mix_b':     '#8A5A0F',
        'bypass':  '#534AB7',  # purple — input-injection theme
    }

    Z_WIRE = 1
    Z_BOX  = 3
    Z_TXT  = 4
    Z_DOT  = 5
    Z_TOP  = 6

    def box(x, y, w, h, text, fc, ec, fontsize=18, lw=1.6, rounding=0.10):
        p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=f"round,pad=0.02,rounding_size={rounding}",
                           fc=fc, ec=ec, lw=lw, zorder=Z_BOX)
        ax.add_patch(p)
        if text:
            ax.text(x, y, text, ha='center', va='center',
                    fontsize=fontsize, zorder=Z_TXT)

    def circ(x, y, r, text, fc, ec, fontsize=18, lw=1.6):
        ax.add_patch(Circle((x, y), r, fc=fc, ec=ec, lw=lw, zorder=Z_BOX))
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, zorder=Z_TXT)

    def hline(y, x1, x2, color=C['wire'], lw=1.4):
        ax.add_line(Line2D([x1, x2], [y, y], color=color, lw=lw,
                           solid_capstyle='butt', zorder=Z_WIRE))

    def vline(x, y1, y2, color=C['wire'], lw=1.4):
        ax.add_line(Line2D([x, x], [y1, y2], color=color, lw=lw,
                           solid_capstyle='butt', zorder=Z_WIRE))

    def hline_with_bumps(y, x1, x2, bumps_at_x=None, bump_radius=0.13,
                         color=C['wire'], lw=1.4):
        bumps_at_x = sorted(b for b in (bumps_at_x or []) if x1 < b < x2)
        cur = x1
        for bx in bumps_at_x:
            if bx - bump_radius > cur:
                ax.add_line(Line2D([cur, bx - bump_radius], [y, y],
                                   color=color, lw=lw,
                                   solid_capstyle='butt', zorder=Z_WIRE))
            theta = np.linspace(np.pi, 0, 40)
            xs = bx + bump_radius * np.cos(theta)
            ys = y  + bump_radius * np.sin(theta)
            ax.add_line(Line2D(xs, ys, color=color, lw=lw,
                               solid_capstyle='butt', zorder=Z_WIRE))
            cur = bx + bump_radius
        if cur < x2:
            ax.add_line(Line2D([cur, x2], [y, y], color=color, lw=lw,
                               solid_capstyle='butt', zorder=Z_WIRE))

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

    def oplus_node(x, y, r=0.22, color=C['bypass'], lw=1.8):
        """Sum node: white circle with bold + inside."""
        ax.add_patch(Circle((x, y), r, fc='white', ec=color, lw=lw,
                            zorder=Z_BOX))
        ax.text(x, y, '+', ha='center', va='center',
                fontsize=20, color=color, fontweight='bold', zorder=Z_TXT)

    # ---- layout ----
    y_rec       = 6.6
    y_tau       = 7.85
    y_share     = 8.55
    y_bypass    = 5.85   # below recursion, above halts
    y_halt      = 4.65
    y_halt_bus  = 3.95
    y_loss      = 2.95
    y_loss_bus  = 1.95

    x_z0    = 0.5
    step_dx = 3.41                                     # was 3.0 — uniform 0.4 gaps
    x_f     = [1.7 + step_dx * i for i in range(4)]
    x_oplus = [x_f[i] + 1.145 for i in range(4)]       # was +1.025
    x_z     = [x_f[i] + 2.125 for i in range(4)]       # was +1.93

    h_offset = -0.95
    h_x = [x_z[i] + h_offset for i in range(4)]
    L_x = list(x_z)

    mix_left_x = 16.605                                # was 13.6 — gap for bus labels
    mix_w      = 4.2                                   # was 3.6 — fits the equation comfortably
    mix_x      = mix_left_x + mix_w / 2
    mix_y_top  = 4.6
    mix_y_bot  = 1.4
    mix_y      = (mix_y_top + mix_y_bot) / 2
    mix_h      = mix_y_top - mix_y_bot

    R_O = 0.22  # ⊕ radius
    R_Z = 0.36  # z_t circle radius

    # ---- recursion lane ----
    circ(x_z0, y_rec, R_Z, r'$z_0$', C['state'], C['state_b'], fontsize=22)
    hline(y_rec, x_z0 + R_Z, x_f[0] - 0.55)
    arrowhead(x_f[0] - 0.55, y_rec, 'right')

    for i in range(4):
        t = i + 1
        # τ_t and arrow down
        circ(x_f[i], y_tau, 0.30, rf'$\tau_{{{t}}}$',
             C['tau'], C['tau_b'], fontsize=18)
        vline(x_f[i], y_rec + 0.42, y_tau - 0.30, color=C['tau_b'], lw=1.5)
        arrowhead(x_f[i], y_rec + 0.42, 'down', color=C['tau_b'])
        # f_θ box
        box(x_f[i], y_rec, 1.05, 0.82, r'$f_\theta$',
            C['shared'], C['shared_b'], fontsize=26, lw=2.4, rounding=0.14)
        # f_θ → ⊕  (no arrowhead; arrow at z_t entry)
        hline(y_rec, x_f[i] + 0.525, x_oplus[i] - R_O)
        # ⊕ sum node
        oplus_node(x_oplus[i], y_rec)
        # ⊕ → z_t
        hline(y_rec, x_oplus[i] + R_O, x_z[i] - R_Z)
        arrowhead(x_z[i] - R_Z, y_rec, 'right')
        # z_t circle
        circ(x_z[i], y_rec, R_Z, rf'$z_{{{t}}}$',
             C['state'], C['state_b'], fontsize=22)
        # z_t → next f_θ
        if i < 3:
            hline(y_rec, x_z[i] + R_Z, x_f[i + 1] - 0.525)
            arrowhead(x_f[i + 1] - 0.525, y_rec, 'right')

    # bracket above
    bL = x_f[0] - 0.7
    bR = x_f[3] + 0.7
    hline(y_share, bL, bR, color=C['shared_b'], lw=1.6)
    for x_end in (bL, bR):
        vline(x_end, y_share - 0.18, y_share, color=C['shared_b'], lw=1.6)
    ax.text((bL + bR) / 2, y_share + 0.22,
            r'shared parameters $f_\theta$  (applied $T_{\max}$ times)',
            ha='center', va='bottom', fontsize=16,
            style='italic', color=C['shared_b'], zorder=Z_TOP)

    # ---- bypass pathway ----
    # z_0 → bypass: short vertical from z_0 bottom down to highway
    vline(x_z0, y_bypass, y_rec - R_Z, color=C['bypass'], lw=1.6)
    # bypass highway runs right and ends at the last ⊕'s tap
    bypass_bumps = [x_z[i] for i in range(3)]   # bypass ends at x_oplus[3] < x_z[3]
    hline_with_bumps(y_bypass, x_z0, x_oplus[3],
                     bumps_at_x=bypass_bumps, bump_radius=0.14,
                     color=C['bypass'], lw=1.6)
    # taps from highway up to each ⊕ (with junction dot at the highway)
    for i in range(4):
        vline(x_oplus[i], y_bypass, y_rec - R_O, color=C['bypass'], lw=1.6)
        junction_dot(x_oplus[i], y_bypass, color=C['bypass'])

    # bypass label (left, where there's clear space)
    ax.text((x_z0 + x_f[0]) / 2 - 0.1, y_bypass - 0.32,
            r'$z_0$  bypass (input injection)',
            ha='center', va='top', fontsize=12,
            style='italic', color=C['bypass'])

    # ---- per-step branches z_t → h_t and z_t → L_t ----
    for i in range(4):
        t = i + 1
        circ(h_x[i], y_halt, 0.34, rf'$h_{{{t}}}$',
             C['halt'], C['halt_b'], fontsize=20)
        box(L_x[i], y_loss, 1.7, 0.85, rf'$\mathcal{{L}}_{{{t}}}$',
            C['loss'], C['loss_b'], fontsize=22, lw=1.6, rounding=0.10)
        vline(x_z[i], y_loss + 0.425, y_rec - R_Z)
        arrowhead(x_z[i], y_loss + 0.425, 'down')
        junction_dot(x_z[i], y_halt)
        hline(y_halt, h_x[i] + 0.34, x_z[i])
        arrowhead(h_x[i] + 0.34, y_halt, 'left')

    # row tags
    ax.text(x_z0 - 0.85, y_halt,
            'halt heads\n(per image, pooled CLS)',
            ha='right', va='center', fontsize=12,
            style='italic', color=C['halt_b'])
    ax.text(x_z0 - 0.85, y_loss,
            'per-step losses\n(student vs. teacher $z_t^T$)',
            ha='right', va='center', fontsize=12,
            style='italic', color=C['loss_b'])

    # halt and loss buses
    for i in range(4):
        vline(h_x[i], y_halt_bus, y_halt - 0.34)
        junction_dot(h_x[i], y_halt_bus)
    hline_with_bumps(y_halt_bus, h_x[0], mix_left_x,
                     bumps_at_x=list(x_z), bump_radius=0.14)
    arrowhead(mix_left_x, y_halt_bus, 'right')

    for i in range(4):
        vline(L_x[i], y_loss_bus, y_loss - 0.425)
        junction_dot(L_x[i], y_loss_bus)
    hline(y_loss_bus, L_x[0], mix_left_x)
    arrowhead(mix_left_x, y_loss_bus, 'right')

    # bus labels (sit in the gap between rightmost bus activity and mix box)
    halt_lbl_x = (x_z[3] + mix_left_x) / 2
    loss_lbl_x = (L_x[3] + 0.85 + mix_left_x) / 2
    ax.text(halt_lbl_x, y_halt_bus + 0.22,
            r'$\{p_t\}_{t=1}^{T_{\max}}$',
            ha='center', va='bottom', fontsize=13,
            color=C['halt_b'], style='italic',
            bbox=dict(facecolor='white', edgecolor='none', pad=2),
            zorder=Z_TOP)
    ax.text(loss_lbl_x, y_loss_bus + 0.22,
            r'$\{\mathcal{L}_t\}_{t=1}^{T_{\max}}$',
            ha='center', va='bottom', fontsize=13,
            color=C['loss_b'], style='italic',
            bbox=dict(facecolor='white', edgecolor='none', pad=2),
            zorder=Z_TOP)

    # mix box
    box(mix_x, mix_y, mix_w, mix_h, '',
        C['mix'], C['mix_b'], fontsize=18, lw=2.2, rounding=0.12)
    ax.text(mix_x, mix_y + 0.55,
            r'$\sum_{t=1}^{T_{\max}}\, p_t \cdot \mathcal{L}_t$',
            ha='center', va='center', fontsize=22, zorder=Z_TXT)
    ax.text(mix_x, mix_y - 0.55,
            r'$+\;\beta\,\mathrm{KL}\!\left(\,p \,\|\, '
            r'\mathrm{Geom}(\lambda_p)\,\right)$',
            ha='center', va='center', fontsize=18, zorder=Z_TXT)
    ax.text(mix_x, mix_y_top + 0.18, 'total loss',
            ha='center', va='bottom', fontsize=13,
            style='italic', color=C['mix_b'])

    base = f'{output_dir}/{output_name}'
    plt.savefig(f'{base}.svg', bbox_inches='tight', pad_inches=0.18)
    plt.savefig(f'{base}.pdf', bbox_inches='tight', pad_inches=0.18)
    plt.savefig(f'{base}.png', bbox_inches='tight', pad_inches=0.18, dpi=200)
    plt.close(fig)
    return base


if __name__ == '__main__':
    out = make_figure()
    print(f'wrote: {out}.svg, {out}.pdf, {out}.png')
