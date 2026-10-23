"""Assemble Figure 11: Blender cut-away renders of actual ReaxFF-MD snapshots + original SEM/TEM micrographs."""
import os, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import argparse
ap = argparse.ArgumentParser()
ap.add_argument('--renders', default='results/figure11_renders')
ap.add_argument('--micrographs', required=True, help="SEM/TEM source image (Figure 10 of the submitted manuscript; not distributed)")
ap.add_argument('--out', default='results/Figure11')
A = ap.parse_args(); R = A.renders + '/'
INK = '#222222'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 7.5})

def md(key, pad=0.04):
    im = Image.open(R + key + '.png'); bg = Image.new('RGBA', im.size, 'white'); bg.alpha_composite(im)
    a = np.asarray(bg.convert('RGB')); m = np.where((a < 250).any(2)); y0, y1, x0, x1 = m[0].min(), m[0].max(), m[1].min(), m[1].max()
    s = max(y1 - y0, x1 - x0); p = int(s * pad); cy, cx = (y0 + y1) // 2, (x0 + x1) // 2; h = s // 2 + p
    canvas = Image.new('RGB', (a.shape[1] + 2 * h, a.shape[0] + 2 * h), 'white'); canvas.paste(bg.convert('RGB'), (h, h))
    return np.asarray(canvas.crop((cx, cy, cx + 2 * h, cy + 2 * h)))

old = Image.open(A.micrographs).convert('RGB')     # SEM/TEM panels are cropped from the authors' micrograph figure
sem, tem = np.asarray(old.crop((984, 205, 1205, 481))), np.asarray(old.crop((1214, 205, 1419, 480)))

fig = plt.figure(figsize=(7.2, 3.9))
W, H = 0.155, 0.40
top = [('a_c1_6.8_final', '(a)', 'Case 1, 6.8 nm'), ('b_c2_6.8_final', '(b)', 'Case 2, 6.8 nm'),
       ('c_c2_10_final', '(c)', 'Case 2, 10.0 nm'), ('d_c2_15_final', '(d)', 'Case 2, 15.0 nm')]
for i, (k, l, t) in enumerate(top):
    ax = fig.add_axes([0.005 + i * 0.16, 0.50, W, H]); ax.imshow(md(k)); ax.axis('off')
    ax.set_title(f'$\\bf{{{l}}}$ {t}\n100 ps, 2500 K', fontsize=7.2, color=INK, pad=2)
for i, (img, l, t) in enumerate([(sem, '(e)', 'SEM'), (tem, '(f)', 'TEM')]):
    ax = fig.add_axes([0.655 + i * 0.172, 0.515, 0.165, 0.37]); ax.imshow(img); ax.axis('off')
    ax.set_title(f'$\\bf{{{l}}}$ {t}\ncombustion products', fontsize=7.2, color=INK, pad=2)
bot = [('g_c2_10_init', '(g)', 'Initial structure', '0 ps'), ('h_c2_10_melt', '(h)', 'End of Stage 1', '4000 K, 10 ps'),
       ('i_c2_10_equil', '(i)', 'End of Stage 3', '2500 K, 30 ps'), ('j_c2_10_final', '(j)', 'End of production', '2500 K, 130 ps')]
for i, (k, l, t, s) in enumerate(bot):
    ax = fig.add_axes([0.005 + i * 0.16, 0.02, W, H]); ax.imshow(md(k)); ax.axis('off')
    ax.set_title(f'$\\bf{{{l}}}$ {t}\n{s}', fontsize=7.2, color=INK, pad=2)
fig.text(0.325, 0.475, 'Case 2, 10.0 nm: evolution through the simulation protocol', ha='center', fontsize=7.2, color=INK, style='italic')
fig.text(0.325, 0.975, 'ReaxFF-MD snapshots (octant cut-away of the actual atomic coordinates)', ha='center', fontsize=7.2, color=INK, style='italic')
leg = [Line2D([], [], marker='o', ls='', ms=7, mfc=c, mec='#555555', mew=0.4, label=n) for c, n in
       (('#40b04a', 'B'), ('#e32b21', 'O'), ('#f2f2f2', 'H'))]
fig.legend(handles=leg, loc='center', bbox_to_anchor=(0.83, 0.36), frameon=False, fontsize=8, handletextpad=0.3, ncol=3, columnspacing=1.2)
for ext in ('png', 'pdf', 'tiff'):
    fig.savefig(f'{A.out}.{ext}', dpi=600 if ext == 'tiff' else 400, facecolor='white')
