"""Figure S9: FCC-type vs alpha-rhombohedral starting core (10 nm).
Requires results/void_{fcc,aB}_c{1,2}_10.json, results/uptake_{c,aB_c}{1,2}_10.npz and the bulk-energy logs in BORON_DATA."""
import sys, os, re, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'analysis'))
from common import RESULTS, data_path
from style import plt, C1, C2, INK, MUTED
from matplotlib.lines import Line2D

def fixed_box_energy(log):           # 'FIXEDBOX <name> pe/atom= <value>' printed by lammps/controls/in.bulk_energy_*
    return float(re.findall(r'^FIXEDBOX \S+ pe/atom= ([-0-9.eE+]+)', open(log).read(), re.M)[-1])

E = np.array([fixed_box_energy(data_path('controls/alpha_rhombohedral_core', f'e_{n}.log')) for n in ('amorph', 'alpha', 'fcc')])
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5)); plt.subplots_adjust(wspace=0.45)
ax = axs[0]; ax.bar(range(3), E - E[0], color=['#9e9e9e', '#4d9221', '#8c6bb1'], width=0.6)
for i, v in enumerate(E - E[0]):
    ax.text(i, v + 4, f'+{v:.0f}' if v else '0', ha='center', fontsize=7, color=INK)
ax.set_xticks(range(3)); ax.set_xticklabels(['Amorphous', 'α-B\n(B$_{12}$)', 'FCC-type'], fontsize=6.5)
ax.set_ylabel('E − E$_{amorphous}$ (kcal mol$^{-1}$ atom$^{-1}$)')
ax = axs[1]
for c, col in ((1, C1), (2, C2)):
    for tag, ls in (('fcc', '-'), ('aB', '--')):
        r = [w for w in json.load(open(os.path.join(RESULTS, f'void_{tag}_c{c}_10.json')))['rows'] if w['t_ps'] <= 10.0]
        ax.plot([w['t_ps'] for w in r], [w['r_void'] for w in r], ls, color=col, lw=1.5)
ax.set(xlabel='Time in Stage 1, 4000 K (ps)', ylabel='Central void radius (Å)')
ax = axs[2]
for c, col in ((1, C1), (2, C2)):
    for tag, ls in (('c', '-'), ('aB_c', '--')):
        d = np.load(os.path.join(RESULTS, f'uptake_{tag}{c}_10.npz')); ax.plot(d['t'], d['N'][:, 0] - d['N'][0, 0], ls, color=col, lw=1.4)
ax.set(xlabel='Production time (ps)', ylabel='ΔN$_O$ inside core boundary')
fig.legend(handles=[Line2D([], [], color=C1, label='Case 1'), Line2D([], [], color=C2, label='Case 2'),
                    Line2D([], [], color=MUTED, label='FCC-type start'), Line2D([], [], color=MUTED, ls='--', label='α-rhombohedral start')],
           loc='upper center', ncol=4, bbox_to_anchor=(0.62, 1.06), fontsize=7)
for a, l in zip(axs, 'abc'):
    a.text(-0.32, 1.02, f'({l})', transform=a.transAxes, fontweight='bold', fontsize=10, color=INK)
fig.savefig(os.path.join(RESULTS, 'FigureS9.png'), dpi=300, bbox_inches='tight', facecolor='white')
print('E - E_amorphous (kcal/mol/atom):', np.round(E - E[0], 1))
