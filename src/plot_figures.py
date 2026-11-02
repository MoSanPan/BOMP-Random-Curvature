"""Restyle the three ICDE figures from saved results; no experiments are run.

Invoked by reproduce.py; --results and --out specify independent directories.
Writes vector PDFs, PNG previews, and a manifest of exact data and source hashes.
"""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator, FixedLocator, ScalarFormatter

import argparse
from matplotlib import font_manager
parser = argparse.ArgumentParser()
parser.add_argument('--results', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
ROOT = args.results.resolve()
OUT = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = [ROOT / 'geometry/response_summary.csv', ROOT / 'geometry/block_summary.csv',
           ROOT / 'calibration/synthetic_summary.csv', ROOT / 'calibration/em_summary.csv']
FONT = next((f for f in ('Times New Roman', 'Liberation Serif', 'DejaVu Serif')
             if any(x.name == f for x in font_manager.fontManager.ttflist)), 'DejaVu Serif')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

manifest = {'source_hashes': {str(p.relative_to(ROOT)): digest(p) for p in SOURCES},
            'new_experiments': False, 'horizontal_jitter': False, 'figures': {}}
plt.rcParams.update({
    'font.family': FONT, 'mathtext.fontset': 'stix',
    'font.size': 8.5, 'axes.labelsize': 9, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 7.6,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#444444', 'axes.linewidth': .65,
    'xtick.major.width': .65, 'ytick.major.width': .65,
    'xtick.major.size': 2.8, 'ytick.major.size': 2.8,
    'axes.labelpad': 3, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})
BLUE, ORANGE, GREEN = '#2166A0', '#CC702A', '#398575'

def canvas(grid, xlabel, ylabel, logy=False):
    fig = plt.figure(figsize=(3.5, 2.22))
    ax = fig.add_axes([.17, .195, .80, .665])
    ax.set_xscale('log', base=2)
    if logy:
        ax.set_yscale('log')
    ax.set_xticks(grid, [str(x) for x in grid])
    span = np.log2(grid[-1] / grid[0])
    ax.set_xlim(grid[0] / 2**(.045*span), grid[-1] * 2**(.045*span))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_axisbelow(True)
    ax.grid(axis='y', color='#DDE2E6', linewidth=.5, alpha=.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(pad=2)
    return fig, ax

def finish(fig, ax, name, points):
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.56, .985),
                     ncol=3, frameon=False, handlelength=1.7, columnspacing=.95,
                     handletextpad=.45, borderaxespad=0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    lb, ab = leg.get_window_extent(renderer), ax.get_window_extent(renderer)
    assert lb.y0 > ab.y1 and lb.x0 >= 0 and lb.x1 <= fig.bbox.width, name
    visible_ylabels = [label for tick,label in zip(ax.get_yticks(),ax.get_yticklabels())
                       if ax.get_ylim()[0] <= tick <= ax.get_ylim()[1]]
    for label in [ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + visible_ylabels:
        bb = label.get_window_extent(renderer)
        assert bb.x0 >= -.5 and bb.y0 >= -.5 and bb.x1 <= fig.bbox.width+.5 and bb.y1 <= fig.bbox.height+.5, (name, label.get_text())
    # Fixed canvas dimensions preserve identical type size after single-column inclusion.
    fig.savefig(OUT / (name+'.pdf'))
    fig.savefig(OUT / (name+'.png'), dpi=400)
    manifest['figures'][name] = points
    plt.close(fig)

def errors(ax, rows, xfield, field, label, color, marker, line, size=3.5):
    x = [int(r[xfield]) for r in rows]
    y = np.array([float(r[field+'_mean']) for r in rows])
    e = np.array([1.96*float(r[field+'_se']) for r in rows])
    plotted = ax.errorbar(x, y, yerr=e, label=label, color=color,
        marker=marker, markersize=size, markerfacecolor='white', markeredgewidth=.9,
        linestyle=line, linewidth=1.1, elinewidth=.65, capsize=1.8, capthick=.65)
    assert np.array_equal(plotted.lines[0].get_ydata(), y)
    return {'x': x, 'mean': y.tolist(), 'lo95': (y-e).tolist(), 'hi95': (y+e).tolist()}

response = read(SOURCES[0])
grid = [8,16,32,64,128]
fig, ax = canvas(grid, r'Lower dimension $q$', r'Scaled response error $\sqrt{n}\,E_S$')
points = {}
for model,label,color,marker,line in [('diagonal','Diagonal',BLUE,'o','-'),
                                      ('two_basis','Two-basis',ORANGE,'s','--')]:
    rows = sorted([r for r in response if r['sweep']=='q' and r['model']==model], key=lambda r:int(r['q']))
    assert [int(r['q']) for r in rows] == grid
    points[model] = errors(ax,rows,'q','scaled_response',label,color,marker,line)
ax.axhline(1/3, color='#737373', linewidth=.9, linestyle=(0,(4,3)), label=r'Bound $V_Q$')
ax.set_ylim(.16,.36)
ax.set_yticks([.16,.20,.24,.28,.32,.36])
points['bound'] = 1/3
finish(fig,ax,'actual_response',points)

blocks = read(SOURCES[1])
grid = [1,4,16,64,256]
fig, ax = canvas(grid, r'Number of blocks $\ell$', 'Normalized excess risk')
points = {}
for method,label,color,marker,line in [('block','Block noise',BLUE,'o','-'),
    ('ambient','Ambient noise',ORANGE,'s','--'),('nonprivate','Non-private',GREEN,'^','-.')]:
    rows = sorted([r for r in blocks if r['method']==method], key=lambda r:int(r['ell']))
    assert [int(r['ell']) for r in rows] == grid
    points[method] = errors(ax,rows,'ell','normalized_risk',label,color,marker,line)
ax.set_ylim(-.015,.55)
ax.set_yticks([0,.1,.2,.3,.4,.5])
finish(fig,ax,'block_geometry_normalized',points)

utility = [r for p in SOURCES[2:] for r in read(p)]
grid = [128,512,2048,8192,32768]
fig, ax = canvas(grid, r'Sample size $n$', 'Population excess risk', logy=True)
points = {}
for key,label,color,marker,line,size in [
    ('coefficient','Algorithm 1',BLUE,'o','-',4.6),
    ('output_tight','Output pert.',ORANGE,'s','--',2.8),
    ('em_tight_normalized','Exponential',GREEN,'^','-.',3.6)]:
    rows = sorted([r for r in utility if r['method']==key and r['family']=='dense_signs'
        and int(r['r'])==16 and float(r['eta'])==.3 and float(r['epsilon'])==1
        and int(r['n']) in grid], key=lambda r:int(r['n']))
    assert [int(r['n']) for r in rows] == grid
    y,lo,hi = [np.array([float(r[f]) for r in rows]) for f in
               ['excess_risk_mean','excess_risk_lo95','excess_risk_hi95']]
    plotted, = ax.plot(grid,y,label=label,color=color,marker=marker,linestyle=line,
                       linewidth=1.1,markersize=size,markerfacecolor='white',markeredgewidth=.9)
    assert np.array_equal(plotted.get_ydata(), y)
    ax.fill_between(grid,lo,hi,color=color,alpha=.10,linewidth=0)
    points[key] = {'x':grid,'mean':y.tolist(),'lo95':lo.tolist(),'hi95':hi.tolist()}
finish(fig,ax,'validation_utility',points)
assert manifest['source_hashes'] == {str(p.relative_to(ROOT)): digest(p) for p in SOURCES}
(OUT/'restyle_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Saved three vector figures and previews; original means, 95% intervals, and source CSV hashes preserved.')
