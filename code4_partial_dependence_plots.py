import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
import warnings
warnings.filterwarnings('ignore')

# 1. PARAMETERS & AESTHETICS
DATA_PATH = r'C:\Users\Admin\Desktop\MS\原始数据\China_profile_dataset_TC_added.xlsx'
OUT_PNG = r'C:\Users\Admin\Desktop\MS\修改后代码与图片\Fig_S2_PDP_Nature.png'
OUT_PDF = r'C:\Users\Admin\Desktop\MS\修改后代码与图片\Fig_S2_PDP_Nature.pdf'

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 8, 'axes.labelsize': 9,
    'axes.titlesize': 9.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.linewidth': 0.8,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'pdf.fonttype': 42, 'ps.fonttype': 42
})

ENV_COLS = ['BD', 'DEM', 'NDVI', 'MAT', 'MAP', 'PET', 'AI', 'pH', 'Sand', 'Silt', 'Clay']

# 2. LOAD DATA
df_raw = pd.read_excel(DATA_PATH)

def find_col(df, pattern):
    pattern_l = pattern.lower().replace(' ', '_')
    for c in df.columns:
        cn = c.strip().lower().replace(' ', '_').replace('-', '_')
        if pattern_l in cn or cn in pattern_l: return c
    for c in df.columns:
        if pattern_l[:6] in c.strip().lower(): return c
    return None

lu_col = find_col(df_raw, 'landuse_tol1km_2019')
soc_col = find_col(df_raw, 'SOC')
sic_col = find_col(df_raw, 'sic_g_kg_ovl')
up_col = find_col(df_raw, 'upper_dept')
lo_col = find_col(df_raw, 'lower_dept')

def classify_lu(x):
    x = str(x).lower()
    if 'paddy' in x: return 'Paddy'
    if 'upland' in x: return 'Upland'
    return np.nan

def depth_group(up, lo):
    try:
        mid = (float(up) + float(lo)) / 2
        if mid <= 30: return '0-30 cm'
        if mid <= 100: return '30-100 cm'
    except: pass
    return np.nan

df = pd.DataFrame()
df['lu'] = df_raw[lu_col].apply(classify_lu)
df['depth'] = df_raw.apply(lambda r: depth_group(r[up_col], r[lo_col]), axis=1)
df['SOC'] = pd.to_numeric(df_raw[soc_col], errors='coerce')
df['SIC'] = pd.to_numeric(df_raw[sic_col], errors='coerce')

for c in ENV_COLS:
    ec = find_col(df_raw, c)
    if ec: df[c] = pd.to_numeric(df_raw[ec], errors='coerce')

df = df.dropna(subset=['lu', 'depth'] + ENV_COLS)
df_030 = df[df['depth'] == '0-30 cm']

# 3. TRAIN RF & GET PDP
def get_pdp(data, target_col, lu, predictors, top_k=2):
    sub = data[(data['lu'] == lu) & (data[target_col].notna())]
    X = sub[predictors].values
    y = np.log1p(sub[target_col].values)
    
    rf = RandomForestRegressor(n_estimators=300, max_features=int(len(predictors)/3)+1,
                               min_samples_leaf=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[-top_k:][::-1]
    
    pdps = []
    for idx in top_indices:
        feat_name = predictors[idx]
        res = partial_dependence(rf, X, features=[idx], grid_resolution=50)
        grid = res['average'][0] if 'grid_values' not in res else res['grid_values'][0]
        values = res['average'][0] if 'average' in res else res['values'][0]
        # values shape is (1, n_grid_points) usually, so we take [0]
        pdps.append((feat_name, grid, np.expm1(values)))
    
    return pdps

targets = [('SOC', 'Paddy'), ('SOC', 'Upland'), ('SIC', 'Paddy'), ('SIC', 'Upland')]
results = {}

print("Calculating Partial Dependences for 0-30 cm...")
for t_col, lu in targets:
    results[(t_col, lu)] = get_pdp(df_030, t_col, lu, ENV_COLS, top_k=2)
    print(f"Finished {t_col} - {lu}")

# 4. PLOT Nature Style
fig, axes = plt.subplots(2, 4, figsize=(183/25.4, 110/25.4))
plt.subplots_adjust(wspace=0.35, hspace=0.45, left=0.08, right=0.98, top=0.9, bottom=0.1)

colors = {'Paddy': '#2166AC', 'Upland': '#B2182B'}

for col_idx, (t_col, lu) in enumerate(targets):
    top_pdps = results[(t_col, lu)]
    
    for row_idx, (feat_name, x_grid, y_pdp) in enumerate(top_pdps):
        ax = axes[row_idx, col_idx]
        
        # Plot PDP line
        ax.plot(x_grid, y_pdp, color=colors[lu], lw=2)
        
        # Aesthetics
        ax.set_title(f"{t_col} ({lu})\nTop {row_idx+1}: {feat_name}", fontsize=8.5, pad=4)
        ax.set_xlabel(feat_name, fontsize=8)
        
        if col_idx == 0 and row_idx == 0:
            ax.set_ylabel('SOC Predicted (g/kg)', fontsize=8)
        if col_idx == 2 and row_idx == 0:
            ax.set_ylabel('SIC Predicted (g/kg)', fontsize=8)
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle=':', alpha=0.5)

# Text description for summary
summary_lines = []
for (t_col, lu), top_pdps in results.items():
    drivers = [p[0] for p in top_pdps]
    summary_lines.append(f"{t_col} in {lu} top drivers: {', '.join(drivers)}")

with open(r'C:\Users\Admin\Desktop\MS\修改后代码与图片\PDP_Analysis_Summary.txt', 'w') as f:
    f.write("\n".join(summary_lines))

fig.savefig(OUT_PNG, dpi=600, bbox_inches='tight')
fig.savefig(OUT_PDF, bbox_inches='tight')
print("PDP figures and summary generated successfully.")
