"""Plot the analytic population-risk bounds from main.tex, Section IV.C.

Invoked by reproduce.py; --out selects the output directory. No experiments are run.
"""
from pathlib import Path
import csv
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=True)
n, mu, h, epsilon, r = 1024, 1.0, 1.0, 1.0, 1
q = np.array([8, 16, 32, 64, 128])
directional_stat = h / (mu * (mu + h / 2) * np.sqrt(n))
directional_priv = 2 * h / (mu**2 * n * epsilon)
directional = np.full(q.shape, directional_stat + directional_priv, dtype=float)
frobenius_diag = np.full(q.shape, h / (mu**2 * np.sqrt(n))
                         + 2 * h / (mu**2 * n * epsilon), dtype=float)
frobenius_two = np.sqrt(q) * frobenius_diag

# Check the plotted expressions against the primitive constants in Section IV.C.
sigma_h = h * np.sqrt(q) / 2
diameter_f = h * np.sqrt(q)
for stability, curve in [(1 / (mu**2 * np.sqrt(q)), frobenius_diag),
                         (np.full(q.shape, 1 / mu**2), frobenius_two)]:
    np.testing.assert_allclose(curve, 2 * stability * sigma_h / np.sqrt(n)
                               + 2 * r * stability * diameter_f / (n * epsilon))
v_q = h / (2 * mu * (mu + h / 2))
omega_q = 2 * h / mu**2
np.testing.assert_allclose(directional, 2 * v_q / np.sqrt(n) + r * omega_q / (n * epsilon))

FONT = next((name for name in ('Times New Roman', 'Liberation Serif', 'DejaVu Serif')
             if any(font.name == name for font in font_manager.fontManager.ttflist)), 'DejaVu Serif')
plt.rcParams.update({
    'font.family': FONT, 'mathtext.fontset': 'stix',
    'font.size': 10, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'legend.fontsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.axisbelow': True,
})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=False)
fig.subplots_adjust(left=.095, right=.98, bottom=.235, top=.84, wspace=.42)
for ax, global_curve, title, ticks, ymax in zip(
        axes, [frobenius_diag, frobenius_two],
        ['(a) Diagonal', '(b) Two-basis'],
        [[.02, .03, .04, .05], [.02, .05, .1, .2, .5]], [.05, .5]):
    ax.plot(q, directional, 'o-', color='#2166A0', label='Directional bound',
            linewidth=1.6, markersize=5, markerfacecolor='white')
    ax.plot(q, global_curve, 's--', color='#CC702A', label='Optimized Frobenius bound',
            linewidth=1.6, markersize=5, markerfacecolor='white')
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.xaxis.set_major_locator(FixedLocator(q))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) for x in q]))
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FixedFormatter([f'{tick:.2f}' for tick in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylim(.015, ymax)
    ax.set_xlim(6.5, 158)
    ax.grid(axis='y', color='#DDE2E6', linewidth=.6)
    ax.set_xlabel(r'Lower dimension $q$')
    ax.set_ylabel('Population-risk upper bound')
    ax.text(.5, -.29, title, transform=ax.transAxes,
            ha='center', va='top', fontsize=11)
fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
           bbox_to_anchor=(.54, .985), ncol=2, frameon=False)
fig.savefig(OUT / 'theoretical_bounds.pdf')
fig.savefig(OUT / 'theoretical_bounds.png', dpi=300)
plt.close(fig)

with (OUT / 'theoretical_bounds.csv').open('w', encoding='utf-8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['family', 'q', 'directional_bound', 'optimized_frobenius_bound',
                     'frobenius_over_directional'])
    for family, curve in [('diagonal', frobenius_diag), ('two_basis', frobenius_two)]:
        for dimension, direct, global_bound in zip(q, directional, curve):
            writer.writerow([family, int(dimension), direct, global_bound, global_bound/direct])
metadata = {
    'kind': 'Analytical population-risk upper-bound comparison; not empirical risk',
    'source': 'main.tex, Section IV.C, ext:eq:directrisk and thm:publicupper',
    'parameters': {'n': n, 'mu': mu, 'h': h, 'epsilon': epsilon, 'r': r, 'q': q.tolist()},
    'directional': 'h/(mu*(mu+h/2)*sqrt(n)) + 2*h/(mu^2*n*epsilon)',
    'frobenius_diagonal': 'h/(mu^2*sqrt(n)) + 2*h/(mu^2*n*epsilon)',
    'frobenius_two_basis': 'sqrt(q) * frobenius_diagonal',
    'axes': 'Both panels use logarithmic axes, with independent y-axis ranges: diagonal [0.015, 0.05], two-basis [0.015, 0.5].',
    'scope': 'Explicit, uncapped bounds in the manuscript; no uncertainty intervals.',
    'interpretation': 'The comparison concerns upper-bound analyses with their respective Gibbs calibrations, not measured superiority of an algorithm.',
}
(OUT / 'theoretical_bounds_settings.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
print('Directional bound:', directional[0])
print('Diagonal Frobenius bound:', frobenius_diag[0])
print('Two-basis Frobenius bounds:', frobenius_two.tolist())
print('Two-basis bound ratios:', (frobenius_two / directional).tolist())
