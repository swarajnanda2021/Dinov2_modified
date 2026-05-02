#!/usr/bin/env python3
"""
Generate a publication-quality schematic of the Looped DINOv2 training pipeline.

The figure shows:
  1. Per-image multi-crop input (2 globals + 6 locals) -> patch embedding -> z_0
  2. Shared L-block stack applied T_max times with sandwich LayerNorm,
     per-step time embedding tau_t, and input injection +z_0
  3. Per-recursion-step output states z_1..z_{T_max}
  4. Per-step halt head h_t (image-pooled CLS -> linear + sigmoid),
     producing PonderNet step-marginals p_t = h_t * prod(1 - h_s)
  5. Per-step DINO and iBOT losses against teacher targets
  6. Teacher branch (EMA, T_max only, no halting) emitting shared
     Sinkhorn-Knopp targets reused across all T_max student steps
  7. Total loss:
        L = sum_t p_t * (DINO_t + ibot_w * iBOT_t)
            + koleo_w * KoLeo(z_{T_max} CLS)
            + beta * KL(p || Geom(lambda_p))

Renders with graphviz to figures/looped_dinov2_training.{svg,pdf}.
"""

from pathlib import Path

import graphviz


OUT_DIR = Path(__file__).parent
OUT_DIR.mkdir(exist_ok=True)

# Okabe-Ito-friendly palette (colorblind-safe)
C = {
    'input_g':   '#56B4E9',  # global crops
    'embed':     '#D1C4E9',  # patch embed
    'z0':        '#EDE7F6',  # z_0 packed
    'stack':     '#E8EAF6',  # shared stack background
    'block':     '#7986CB',  # transformer block
    'ln':        '#FFF59D',  # layernorm
    'zstep':     '#FFE0B2',  # per-step z_t
    'halt':      '#FFB74D',  # halt head
    'pmarg':     '#FF8A65',  # PonderNet marginal
    'dino':      '#A5D6A7',  # DINO_t loss
    'ibot':      '#66BB6A',  # iBOT_t loss
    'teacher_g': '#90A4AE',  # teacher inputs
    'teacher':   '#BDBDBD',  # teacher background
    'tblock':    '#9E9E9E',  # teacher block
    'koleo':     '#CE93D8',  # KoLeo
    'kl':        '#FFAB91',  # KL term
    'total':     '#EF9A9A',  # total loss
    'edge_data': '#37474F',
    'edge_loop': '#AD1457',
    'edge_targ': '#5E35B1',
    'edge_loss': '#9E9E9E',
}


def build():
    dot = graphviz.Digraph(
        'looped_dinov2',
        format='svg',
        graph_attr={
            'rankdir': 'TB',
            'compound': 'true',
            'splines': 'spline',
            'nodesep': '0.30',
            'ranksep': '0.55',
            'bgcolor': 'white',
            'fontname': 'Helvetica',
            'fontsize': '12',
            'labelloc': 't',
            'labeljust': 'c',
            'newrank': 'true',
            'label': (
                'Looped DINOv2 with image-level adaptive halting — training pipeline\n'
                'T_max = 4, L = 3 default. Student in lockstep across all recursion steps; '
                'teacher at T_max only with no halting.'
            ),
            'pad': '0.4',
        },
        node_attr={
            'shape': 'box',
            'style': 'rounded,filled',
            'fillcolor': '#F5F5F5',
            'fontname': 'Helvetica',
            'fontsize': '9',
            'margin': '0.10,0.05',
            'penwidth': '1.0',
        },
        edge_attr={
            'color': C['edge_data'],
            'penwidth': '1.1',
            'arrowsize': '0.7',
        },
    )

    # =========================================================================
    # STUDENT INPUT
    # =========================================================================
    with dot.subgraph(name='cluster_student_input') as g:
        g.attr(
            label='Student input',
            style='rounded,filled', color='#90A4AE',
            fillcolor='#FAFAFA',
            fontsize='11', fontname='Helvetica-Bold',
        )
        g.node('s_input',
               '8 crops per image\n(2 globals 224² + 6 locals 96²)',
               fillcolor=C['input_g'], width='2.6', height='0.6')
        g.node('s_embed',
               'PatchEmbed + pos\n+ CLS + register tokens',
               fillcolor=C['embed'], width='2.6')
        g.node('s_z0',
               'z₀  (packed sequence,\nxformers BlockDiagonalMask)',
               fillcolor=C['z0'], width='2.6')
        g.edge('s_input', 's_embed')
        g.edge('s_embed', 's_z0')

    # =========================================================================
    # STUDENT SHARED STACK (with internal blocks + sandwich LN)
    # tau_t and +z_0 shown as side annotations attached to the cluster label,
    # and the recursion is communicated via the cluster label itself plus a
    # short self-loop edge — no big arc across the figure.
    # =========================================================================
    with dot.subgraph(name='cluster_shared_stack') as g:
        g.attr(
            label=(
                'SharedStack — L = 3 weight-tied blocks, applied T_max = 4 times\n'
                'recurrence: z_t = post_LN( Block_L ∘ ··· ∘ Block_1 ( pre_LN( z_{t-1} + τ_t ) ) ) + z₀'
            ),
            style='rounded,filled', color='#3949AB',
            fillcolor=C['stack'],
            fontsize='11', fontname='Helvetica-Bold',
            penwidth='1.4',
        )
        g.node('pre_ln',  'pre-LayerNorm  +  τ_t   (per-step time embedding)',
               fillcolor=C['ln'], width='4.2', fontsize='9.5')
        g.node('blk1', 'TransformerBlock₁  (shared)',
               fillcolor=C['block'], width='4.2', fontcolor='white', fontsize='9.5')
        g.node('blk2', 'TransformerBlock₂  (shared)',
               fillcolor=C['block'], width='4.2', fontcolor='white', fontsize='9.5')
        g.node('blk3', 'TransformerBlock₃  (shared)',
               fillcolor=C['block'], width='4.2', fontcolor='white', fontsize='9.5')
        g.node('post_ln', 'post-LayerNorm  +  z₀   (input injection)',
               fillcolor=C['ln'], width='4.2', fontsize='9.5')

        g.edge('pre_ln',  'blk1')
        g.edge('blk1',    'blk2')
        g.edge('blk2',    'blk3')
        g.edge('blk3',    'post_ln')

        # short self-loop edge attached to the cluster label, indicating recursion
        g.node('loop_anchor', 'recurse × T_max',
               fillcolor='#FCE4EC', shape='box', style='rounded,filled,dashed',
               fontcolor=C['edge_loop'], fontsize='9.5', fontname='Helvetica-Bold',
               width='2.0', height='0.45', color=C['edge_loop'])
        g.edge('post_ln', 'loop_anchor', color=C['edge_loop'],
               style='dashed', arrowsize='0.6', constraint='false')
        g.edge('loop_anchor', 'pre_ln', color=C['edge_loop'],
               style='dashed', arrowsize='0.6', constraint='false')

    # input flow z_0 -> pre_ln (cluster head)
    dot.edge('s_z0', 'pre_ln', lhead='cluster_shared_stack')

    # =========================================================================
    # PER-RECURSION-STEP OUTPUTS z_t  (a horizontal row of 4)
    # =========================================================================
    with dot.subgraph(name='cluster_zsteps') as g:
        g.attr(
            label='Per-recursion-step output states (lockstep × T_max)',
            style='rounded,filled', color='#FB8C00',
            fillcolor='#FFF8E1',
            fontsize='11', fontname='Helvetica-Bold',
        )
        for t in range(1, 5):
            g.node(f'z{t}', f'z_{t}',
                   fillcolor=C['zstep'], width='1.1', height='0.5',
                   fontsize='12', fontname='Helvetica-Bold')
        with g.subgraph() as same:
            same.attr(rank='same')
            for t in range(1, 5):
                same.node(f'z{t}')
        g.edge('z1', 'z2', style='invis')
        g.edge('z2', 'z3', style='invis')
        g.edge('z3', 'z4', style='invis')

    # post_ln -> all per-step z_t (visualized as a single arrow into the cluster)
    dot.edge('post_ln', 'z1', lhead='cluster_zsteps', ltail='cluster_shared_stack')

    # =========================================================================
    # PER-STEP HALT HEAD ROW
    # =========================================================================
    with dot.subgraph(name='cluster_halt') as g:
        g.attr(
            label=(
                'Image-level halt head (mean-pool the image\'s 8 CLS tokens, h_t = σ(W_halt · z̄_t))'
            ),
            style='rounded,filled', color='#F57C00',
            fillcolor='#FFF3E0',
            fontsize='10.5', fontname='Helvetica-Bold',
        )
        for t in range(1, 5):
            g.node(f'h{t}', f'h_{t}',
                   fillcolor=C['halt'], width='1.1', height='0.5',
                   fontsize='10', fontname='Helvetica-Bold')
        with g.subgraph() as same:
            same.attr(rank='same')
            for t in range(1, 5):
                same.node(f'h{t}')
        g.edge('h1', 'h2', style='invis')
        g.edge('h2', 'h3', style='invis')
        g.edge('h3', 'h4', style='invis')

    for t in range(1, 5):
        dot.edge(f'z{t}', f'h{t}')

    # PonderNet marginals annotation (sits to the right of the halt row)
    dot.node(
        'pmarg',
        ('PonderNet marginals (per image)\n'
         'p_t = h_t · ∏_{s<t} (1 − h_s);   p_{T_max} absorbs remainder;   ∑_t p_t = 1'),
        fillcolor=C['pmarg'], width='4.6', fontsize='9.5',
        fontname='Helvetica-Bold',
    )
    dot.edge('h4', 'pmarg', style='dashed', arrowsize='0.5',
             color=C['edge_loop'])

    # =========================================================================
    # PER-STEP DINO LOSS ROW (above iBOT)
    # =========================================================================
    with dot.subgraph(name='cluster_dino') as g:
        g.attr(
            label='Per-step DINO CLS loss  L^DINO_t  (cross-view CE vs. teacher CLS at T_max)',
            style='rounded,filled', color='#388E3C',
            fillcolor='#E8F5E9',
            fontsize='10.5', fontname='Helvetica-Bold',
        )
        for t in range(1, 5):
            g.node(f'dino{t}', f'L^DINO_{t}',
                   fillcolor=C['dino'], width='1.4', fontsize='9.5',
                   fontname='Helvetica-Bold')
        with g.subgraph() as same:
            same.attr(rank='same')
            for t in range(1, 5):
                same.node(f'dino{t}')
        g.edge('dino1', 'dino2', style='invis')
        g.edge('dino2', 'dino3', style='invis')
        g.edge('dino3', 'dino4', style='invis')

    # z_t -> halt is constrained (above); DINO/iBOT are on a row below halt,
    # connected via halt -> dino (constrained) and dino -> ibot (constrained).
    # That stacks the rows vertically: z_t, halt, dino, ibot.
    for t in range(1, 5):
        dot.edge(f'h{t}', f'dino{t}', style='invis')

    # =========================================================================
    # PER-STEP iBOT LOSS ROW
    # =========================================================================
    with dot.subgraph(name='cluster_ibot') as g:
        g.attr(
            label='Per-step iBOT patch loss  L^iBOT_t  (masked tokens vs. teacher patches at T_max)',
            style='rounded,filled', color='#1B5E20',
            fillcolor='#C8E6C9',
            fontsize='10.5', fontname='Helvetica-Bold',
        )
        for t in range(1, 5):
            g.node(f'ibot{t}', f'L^iBOT_{t}',
                   fillcolor=C['ibot'], width='1.4', fontcolor='white',
                   fontsize='9.5', fontname='Helvetica-Bold')
        with g.subgraph() as same:
            same.attr(rank='same')
            for t in range(1, 5):
                same.node(f'ibot{t}')
        g.edge('ibot1', 'ibot2', style='invis')
        g.edge('ibot2', 'ibot3', style='invis')
        g.edge('ibot3', 'ibot4', style='invis')

    for t in range(1, 5):
        dot.edge(f'dino{t}', f'ibot{t}', style='invis')

    # =========================================================================
    # TEACHER BRANCH (parallel column on the right side)
    # =========================================================================
    with dot.subgraph(name='cluster_teacher') as g:
        g.attr(
            label='Teacher (EMA of student, T_max only, no halt head)',
            style='rounded,filled', color='#616161',
            fillcolor='#FAFAFA',
            fontsize='11', fontname='Helvetica-Bold',
            penwidth='1.2',
        )
        g.node('t_input', '2 global crops',
               fillcolor=C['teacher_g'], width='2.2')
        g.node('t_embed', 'PatchEmbed + tokens',
               fillcolor=C['embed'], width='2.2')
        g.node('t_stack',
               'SharedStack (EMA)\n— L = 3 blocks × T_max\n— in lockstep, no halting\n— gradients detached',
               fillcolor=C['teacher'], width='2.6', height='1.05', fontsize='9')
        g.node('t_zfinal', 'z_{T_max}^{teacher}',
               fillcolor=C['tblock'], width='2.2', fontcolor='white', fontsize='10')
        g.node('t_targets',
               'Sinkhorn–Knopp\n(once per batch)\n→ shared targets',
               fillcolor='#E0E0E0', width='2.6', fontsize='9',
               fontname='Helvetica-Bold')
        g.edge('t_input', 't_embed')
        g.edge('t_embed', 't_stack')
        g.edge('t_stack', 't_zfinal')
        g.edge('t_zfinal', 't_targets')

    # Teacher target arrows: aim them at the *cluster* heads (lhead) so
    # graphviz can collapse the fan-out into a single tidy bus.
    dot.edge('t_targets', 'dino1', lhead='cluster_dino',
             color=C['edge_targ'], style='dashed',
             arrowsize='0.6', constraint='false',
             label='shared targets', fontcolor=C['edge_targ'],
             fontsize='8.5', fontname='Helvetica-Oblique')
    dot.edge('t_targets', 'ibot1', lhead='cluster_ibot',
             color=C['edge_targ'], style='dashed',
             arrowsize='0.6', constraint='false')

    # =========================================================================
    # KoLeo on final-step CLS  +  KL  +  TOTAL LOSS
    # =========================================================================
    dot.node(
        'koleo',
        'L^KoLeo / KDE\n(applied on z_{T_max}^{student} CLS only)',
        fillcolor=C['koleo'], width='3.0', fontsize='9',
        fontname='Helvetica-Bold',
    )
    dot.edge('z4', 'koleo', constraint='false', color=C['edge_data'],
             style='solid')

    dot.node(
        'kl',
        'β · KL( p ∥ Geom(λ_p) )\nλ_p annealed 0.9 → 0.3 over first 30%',
        fillcolor=C['kl'], width='3.2', fontsize='9',
        fontname='Helvetica-Bold',
    )
    dot.edge('pmarg', 'kl', style='dashed', arrowsize='0.5',
             color=C['edge_loop'])

    dot.node(
        'total',
        ('Total student loss\n'
         'L = Σ_t  p_t · ( L^DINO_t + w_iBOT · L^iBOT_t )\n'
         '    + w_KoLeo · L^KoLeo(z_{T_max} CLS)\n'
         '    + β · KL( p ∥ Geom(λ_p) )'),
        fillcolor=C['total'], width='6.8', height='1.4',
        fontsize='11.5', fontname='Helvetica-Bold', penwidth='1.4',
    )

    # Per-step loss heads feed total via a single per-cluster bus (lhead/ltail)
    dot.edge('dino4', 'total',
             ltail='cluster_dino', color=C['edge_loss'],
             style='dashed', arrowsize='0.5')
    dot.edge('ibot4', 'total',
             ltail='cluster_ibot', color=C['edge_loss'],
             style='dashed', arrowsize='0.5')
    dot.edge('h4', 'total',
             ltail='cluster_halt', color=C['edge_loss'],
             style='dashed', arrowsize='0.5')
    dot.edge('koleo', 'total', color=C['edge_loss'],
             style='dashed', arrowsize='0.5')
    dot.edge('kl', 'total', color=C['edge_loss'],
             style='dashed', arrowsize='0.5')

    return dot


def main():
    dot = build()
    base = OUT_DIR / 'looped_dinov2_training'
    dot.format = 'svg'
    dot.render(str(base), cleanup=True)
    dot.format = 'pdf'
    dot.render(str(base), cleanup=True)
    print('Wrote', base.with_suffix('.svg'))
    print('Wrote', base.with_suffix('.pdf'))


if __name__ == '__main__':
    main()
