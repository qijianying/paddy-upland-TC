"""
Nature-style publication figure: SOC, SIC and TC comparison (Paddy vs Upland)
- 3×2 panel: row = SOC / SIC / TC, col = 0-30 cm / 30-100 cm
- TC = SOC + SIC (note in figure caption)
- Nature double-column width: 183 mm
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib import rcParams
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ── 0. Font / style ──────────────────────────────────────────────────────────
rcParams.update({
    'font.family'      : 'Arial',
    'font.size'        : 8,
    'axes.labelsize'   : 9,
    'axes.titlesize'   : 9,
    'xtick.labelsize'  : 8,
    'ytick.labelsize'  : 8,
    'legend.fontsize'  : 8.5,
    'axes.linewidth'   : 0.8,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size' : 3.0,
    'ytick.major.size' : 3.0,
    'xtick.direction'  : 'in',
    'ytick.direction'  : 'in',
    'pdf.fonttype'     : 42,
    'ps.fonttype'      : 42,
})

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_excel(r'C:\Users\Admin\Desktop\MS\China_profile_dataset_TC_added.xlsx')
print(f"Loaded: {len(df)} rows")

def lu_clean(x):
    x = str(x).lower().strip()
    if 'paddy' in x:  return 'Paddy'
    if 'upland' in x: return 'Upland'
    return np.nan

def depth_grp(up, lo):
    if pd.isna(up) or pd.isna(lo): return np.nan
    mid = (float(up) + float(lo)) / 2
    if mid <= 30:  return '0\u201330 cm'
    if mid <= 100: return '30\u2013100 cm'
    return np.nan

df['lu']    = df['landuse_tol1km_2019'].apply(lu_clean)
df['upper'] = pd.to_numeric(df['upper_dept'], errors='coerce')
df['lower'] = pd.to_numeric(df['lower_dept'], errors='coerce')
df['dg']    = df.apply(lambda r: depth_grp(r['upper'], r['lower']), axis=1)
df['SIC']   = pd.to_numeric(df['sic_g_kg_ovl'], errors='coerce')
df['SOC']   = pd.to_numeric(df['SOC'], errors='coerce')
df['TC']    = pd.to_numeric(df['TC'],  errors='coerce')

# ── 2. IQR clean ──────────────────────────────────────────────────────────────
def iqr_clean(s):
    s = s.dropna()
    if len(s) == 0: return np.array([])
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return s[(s >= q1 - 1.5*iqr) & (s <= q3 + 1.5*iqr)].values

indicators = ['SOC', 'SIC', 'TC']
depths     = ['0\u201330 cm', '30\u2013100 cm']
landuses   = ['Paddy', 'Upland']

cell = {}
for ind in indicators:
    for dep in depths:
        sub = df[df['dg'] == dep]
        for lu in landuses:
            cell[(ind, dep, lu)] = iqr_clean(sub[sub['lu'] == lu][ind])

# ── 3. Colors ─────────────────────────────────────────────────────────────────
COL  = {'Paddy': '#C0392B', 'Upland': '#2980B9'}
ABOX, AJIT, JSIZ = 0.76, 0.20, 1.8

# ── 4. Figure layout  (3 rows × 2 cols) ──────────────────────────────────────
# Nature double-col = 183 mm wide; 3-row height ~185 mm
fig, axes = plt.subplots(3, 2, figsize=(183/25.4, 185/25.4))
fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.10,
                    wspace=0.35, hspace=0.55)

panel_labels = [['a','b'], ['c','d'], ['e','f']]

# ── 5. Draw each panel ────────────────────────────────────────────────────────
for ri, ind in enumerate(indicators):
    for ci, dep in enumerate(depths):
        ax = axes[ri, ci]

        # per-group box + jitter
        for xi, lu in zip([1, 2], landuses):
            vals = cell[(ind, dep, lu)]
            c    = COL[lu]
            if len(vals) == 0: continue

            ax.boxplot(
                vals, positions=[xi], widths=0.40,
                patch_artist=True, showfliers=False, whis=1.5,
                medianprops=dict(color='white', linewidth=2.2, solid_capstyle='round'),
                boxprops=dict(facecolor=c, alpha=ABOX, linewidth=0.6, edgecolor=c),
                whiskerprops=dict(color=c, linewidth=0.8, linestyle=(0, (3, 2))),
                capprops=dict(color=c, linewidth=0.9)
            )
            np.random.seed(0)
            jx = np.random.normal(xi, 0.075, len(vals))
            ax.scatter(jx, vals, color=c, s=JSIZ, alpha=AJIT,
                       linewidths=0, zorder=3)

        # autoscale first, then compute bracket position
        ax.set_xlim(0.45, 2.55)
        ax.autoscale(enable=True, axis='y')
        ylo, yhi = ax.get_ylim()
        yr = yhi - ylo

        # Wilcoxon test + significance bracket
        vp = cell[(ind, dep, 'Paddy')]
        vu = cell[(ind, dep, 'Upland')]
        if len(vp) > 1 and len(vu) > 1:
            _, pv = stats.mannwhitneyu(vp, vu, alternative='two-sided')
            stars = ('***' if pv < 0.001 else '**'  if pv < 0.01
                     else '*'   if pv < 0.05  else 'ns')
            pstr  = 'p < 0.001' if pv < 0.001 else f'p = {pv:.3f}'

            def whisker_top(v):
                q1, q3 = np.percentile(v, 25), np.percentile(v, 75)
                return min(q3 + 1.5*(q3-q1), v.max())

            y_top = max(whisker_top(vp), whisker_top(vu))
            bk_y  = y_top + yr * 0.07
            txt_y = y_top + yr * 0.14

            ax.plot([1, 1, 2, 2],
                    [bk_y*0.975, bk_y, bk_y, bk_y*0.975],
                    color='#333', lw=0.7, clip_on=False)
            ax.text(1.5, txt_y, f'{stars}   {pstr}',
                    ha='center', va='bottom',
                    fontsize=6.5, color='#222')
            ax.set_ylim(ylo, txt_y + yr * 0.06)

        # n values at bottom of each box column
        ylo2, yhi2 = ax.get_ylim()
        for xi, lu in zip([1, 2], landuses):
            vals = cell[(ind, dep, lu)]
            if len(vals) > 0:
                ax.text(xi, ylo2 + (yhi2-ylo2)*0.015,
                        f'n = {len(vals)}',
                        ha='center', va='bottom',
                        fontsize=5.8, color='#666', style='italic')

        # axes formatting
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Paddy', 'Upland'], fontsize=8.5)
        ax.set_ylabel(r'%s (g kg$^{-1}$)' % ind, fontsize=9, labelpad=3)
        ax.set_title(dep, fontsize=9, pad=5, fontweight='bold', color='#222')
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.tick_params(which='minor', length=1.8, width=0.5, direction='in')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.yaxis.grid(True, linewidth=0.3, linestyle='--', color='#ccc', zorder=0)
        ax.set_axisbelow(True)

        # panel letter (a–f)
        ax.text(-0.17, 1.08, panel_labels[ri][ci],
                transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='left')

# ── 6. Row labels (SOC / SIC / TC) on left ───────────────────────────────────
row_y = [0.82, 0.52, 0.21]
for ri, (ind, yf) in enumerate(zip(indicators, row_y)):
    label = ind if ind != 'TC' else 'TC†'   # dagger marks TC as derived
    fig.text(0.013, yf, label,
             ha='center', va='center',
             fontsize=11, fontweight='bold', rotation=90, color='#111')

# ── 7. Legend ─────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(facecolor=COL['Paddy'],  alpha=ABOX, edgecolor=COL['Paddy'],  label='Paddy'),
    mpatches.Patch(facecolor=COL['Upland'], alpha=ABOX, edgecolor=COL['Upland'], label='Upland'),
]
fig.legend(handles=handles, loc='lower center', ncol=2,
           frameon=False, fontsize=8.5,
           bbox_to_anchor=(0.54, 0.022),
           handlelength=1.3, handletextpad=0.5, columnspacing=1.5)

# † footnote (below legend)
fig.text(0.50, 0.004,
         '† TC = SOC + SIC (derived, not independently measured)',
         ha='center', va='bottom', fontsize=6.5, color='#555', style='italic')

# ── 8. Save ───────────────────────────────────────────────────────────────────
out = r'C:\Users\Admin\Desktop\MS\修改后代码与图片'
fig.savefig(f'{out}\\Fig_SOC_SIC_TC_Boxplot_Nature.png',
            dpi=600, bbox_inches='tight', format='png')
fig.savefig(f'{out}\\Fig_SOC_SIC_TC_Boxplot_Nature.pdf',
            bbox_inches='tight', format='pdf')

print(f"Saved: Fig_SOC_SIC_TC_Boxplot_Nature.png / .pdf")
