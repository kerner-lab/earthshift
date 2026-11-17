"""
Complete Effective Robustness Analysis Script
==============================================
Generates:
1. Comprehensive figure: rows = shifts with data (temporal, scale, sensor, geographic, processing),
   cols = settings provided (frozen and/or full)
2. CSV with effective robustness values per model and shift
3. CSV with rankings per model and shift
4. Summary statistics CSV
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.linear_model import LinearRegression
from scipy.stats import kendalltau
import os

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True,
                    help='Root directory containing input CSV files and where outputs will be saved')
parser.add_argument('--frozen_filename', type=str, default=None,
                    help='File of frozen-backbone (head fine-tuning) data to plot (without .csv)')
parser.add_argument('--full_filename', type=str, default=None,
                    help='File of full finetuning to plot (without .csv)')
parser.add_argument('--out_dir', type=str, default=None,
                    help='Where outputs are saved (default: <root_dir>/effective)')
args = parser.parse_args()
if args.frozen_filename is None and args.full_filename is None:
    parser.error('Provide at least one of --frozen_filename / --full_filename')
root_dir = args.root_dir
frozen_filename = args.frozen_filename
full_filename = args.full_filename
# ============================================================================
# CONFIGURATION
# ============================================================================

FROZEN_CSV = '{}/{}.csv'.format(root_dir, frozen_filename) if frozen_filename else None
FULL_CSV = '{}/{}.csv'.format(root_dir, full_filename) if full_filename else None
OUTPUT_DIR = args.out_dir or '{}/effective'.format(root_dir)

# Task groupings by shift type
SHIFT_GROUPS = {
    'Temporal Shift': ['ftw-all-window', 'SA Temporal', 'Germany Temporal', 'France RPG 5year'],
    'Scale Shift': ['RESISC45-UCMerced', 'UCMerced-RESISC45', 'Chesapeake-NAIP-Landsat', 'Chesapeake-Landsat-NAIP'],
    'Sensor Shift': ['BenV2-S2-S1', 'Sen1Floods11-S2-S1', 'RGB - RGE1'],
    'Geographic Shift': ['Geo Germany-Cambodia', 'Geo Germany-Denmark', 'BRIGHT', 'DeepGlobe-DFC2022'],
    'Processing Shift': ['CloudSEN-L2A-L1C', 'BenV2-L2A-L1C', 'FTW-AI4Boundaries']
}

# Color of the y=x (no degradation) reference line
YX_COLOR = '#d62728'

# Display names for task legend labels (shift type is given by the panel, so just source→target)
TASK_DISPLAY_NAMES = {
    'ftw-all-window':       'FTW Season→Season',
    'SA Temporal':          'SA Season→Season',
    'Germany Temporal':     'Germany Year→Year',
    'France RPG 5year':     'France RPG Year→Year+5',
    'RESISC45-UCMerced':    'RESISC45→UCMerced',
    'UCMerced-RESISC45':    'UCMerced→RESISC45',
    'Chesapeake-NAIP-Landsat': 'Chesapeake NAIP→Landsat',
    'Chesapeake-Landsat-NAIP': 'Chesapeake Landsat→NAIP',
    'BenV2-S2-S1':          'BenV2 S2→S1',
    'Sen1Floods11-S2-S1':   'Sen1Floods11 S2→S1',
    'RGB - RGE1':           'RGB→RGE1',
    'Geo Germany-Cambodia': 'Germany→Cambodia',
    'Geo Germany-Denmark':  'Germany→Denmark',
    'BRIGHT':               'BRIGHT Turkey→Morocco',
    'DeepGlobe-DFC2022':    'DeepGlobe→DFC2022',
    'CloudSEN-L2A-L1C':     'CloudSEN12 L2A→L1C',
    'BenV2-L2A-L1C':        'BenV2 L2A→L1C',
    'FTW-AI4Boundaries':    'FTW→AI4Boundaries',
}

# Task colours are only distinct within a shift type: the same few colours are reused in every panel.
# Okabe-Ito colours (colourblind-safe, no red/green pair): blue, orange, sky blue, reddish purple.
TASK_PALETTE = ['#0072B2', '#E69F00', '#56B4E9', '#CC79A7']
TASK_COLORS = {}  # filled in main() once we know which tasks have data


def assign_task_colors(present_tasks):
    """Number the tasks with data within each shift group and give them palette colours in that order."""
    for tasks in SHIFT_GROUPS.values():
        with_data = [t for t in tasks if t in present_tasks]
        if len(with_data) > len(TASK_PALETTE):
            raise ValueError(f'{len(with_data)} tasks in one shift group but only {len(TASK_PALETTE)} colours')
        for i, t in enumerate(with_data):
            TASK_COLORS[t] = TASK_PALETTE[i]

# One marker per model; unknown models fall back to a circle
MODEL_MARKERS = {
    'clay': 'o',
    'clip': 's',
    'croma': '^',
    'dinov3_large_sat': 'v',
    'dofa': 'D',
    'galileo': 'd',
    'prithvi': '*',
    'resnet50_imgnet': 'P',
    'resnet_random': 'X',
    'terrafm': 'p',
    'terramind': 'h',
    'vit_imgnet': '<',
    'vit_random': '>',
}
# Kept in the fits and CSVs, but not drawn or listed in the legend
HIDDEN_MODELS = {'olmo-earth'}
# Not in the paper: dropped when the CSVs are loaded, so they are left out of the fits, rho, tau, ranks and CSVs
EXCLUDED_MODELS = {'galileo', 'olmo-earth'}


def marker_size(marker, base):
    """Stars read smaller than other markers at the same size."""
    return base * 1.4 if marker == '*' else base


def task_handle(task, markersize):
    return mlines.Line2D([], [], color=TASK_COLORS.get(task, 'gray'), marker='o', linestyle='None',
                         markersize=markersize, label=TASK_DISPLAY_NAMES.get(task, task))


def model_handle(model, markersize):
    m = MODEL_MARKERS.get(model, 'o')
    return mlines.Line2D([], [], color='0.4', marker=m, linestyle='None', markeredgecolor='black',
                         markeredgewidth=0.3, markersize=marker_size(m, markersize), label=model)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def convert_to_float(val):
    """Convert percentage strings or decimal strings to float"""
    if pd.isna(val) or val == '':
        return np.nan
    if isinstance(val, str):
        if '%' in val:
            return float(val.strip('%')) / 100.0
        else:
            return float(val)
    return float(val)


# A run counts as a failed adaptation if its ID score is below this fraction of the task's median ID score
FAILED_ID_FRACTION = 0.5
# (task, model) runs that fall under the threshold but were checked and are genuine results
VERIFIED_RUNS = {
    ('RGB - RGE1', 'galileo'),
    ('RESISC45-UCMerced', 'galileo'),
    ('UCMerced-RESISC45', 'galileo'),
    ('BenV2-S2-S1', 'galileo'),
    ('ftw-all-window', 'dinov3_large_sat'),
    ('FTW-AI4Boundaries', 'dinov3_large_sat'),
    ('RESISC45-UCMerced', 'resnet_random'),
    ('UCMerced-RESISC45', 'resnet_random'),
}
SHOW_FAILED_LEGEND = False  # set in main(): only list the hollow-marker entry if a run is actually flagged


def mark_failed_runs(df):
    """Add a 'Failed' column: ID score < FAILED_ID_FRACTION x median ID score of all models on that task,
    except runs listed in VERIFIED_RUNS."""
    median_id = df.groupby('Task')['Avg Test ID Acc'].transform('median')
    verified = pd.Series(list(zip(df['Task'], df['Model'])), index=df.index).isin(VERIFIED_RUNS)
    df['Failed'] = (df['Avg Test ID Acc'] < FAILED_ID_FRACTION * median_id) & ~verified
    return df


def ranking_agreement(df_shift, shift_name):
    """
    Kendall's tau between models' ID and OOD scores, computed separately for each task in the shift
    (one seed-averaged point per model, failed runs and hidden models excluded), then averaged over tasks.
    Tasks are not pooled: differences in task difficulty would inflate a pooled tau.
    Returns (mean tau over tasks, list of per-task rows).
    """
    rows = []
    for task in SHIFT_GROUPS[shift_name]:
        d = df_shift[(df_shift['Task'] == task) & ~df_shift['Model'].isin(HIDDEN_MODELS)]
        d = d.dropna(subset=['Avg Test ID Acc', 'Avg Test OOD Acc'])
        if d.empty:
            continue
        kept = d[~d['Failed']]
        # tau needs at least 3 models to mean anything (with 2 it is always +-1)
        tau = kendalltau(kept['Avg Test ID Acc'], kept['Avg Test OOD Acc'])[0] if len(kept) >= 3 else np.nan
        rows.append({'Task': task, 'n_models': len(kept), 'n_failed': int(d['Failed'].sum()), 'tau': tau})
    taus = [r['tau'] for r in rows if not np.isnan(r['tau'])]
    return (float(np.mean(taus)) if taus else np.nan), rows


def fmt_tau(tau):
    return '\u03c4(ID, OOD) = ' + ('n/a' if np.isnan(tau) else f'{tau:.2f}'.replace('-', '\u2212'))


def scatter_model(ax, df_sub, color, marker, size, linewidth, **kw):
    """Filled markers for normal runs, hollow markers (task colour outline) for failed adaptations."""
    ok = ~df_sub['Failed'].astype(bool)
    ax.scatter(df_sub.loc[ok, 'Avg Test ID Acc'], df_sub.loc[ok, 'Avg Test OOD Acc'], c=[color], marker=marker,
               s=size, edgecolors='black', linewidth=linewidth, **kw)
    if (~ok).any():
        ax.scatter(df_sub.loc[~ok, 'Avg Test ID Acc'], df_sub.loc[~ok, 'Avg Test OOD Acc'], facecolors='none',
                   edgecolors=[color], marker=marker, s=size, linewidth=max(1.0, 3 * linewidth), **kw)


def failed_handle(markersize):
    return mlines.Line2D([], [], color='0.4', marker='o', linestyle='None', markerfacecolor='none',
                         markeredgewidth=1.2, markersize=markersize, label='Failed adaptation (excl. from \u03c4)')


def calculate_effective_robustness(df_data):
    """
    Calculate effective robustness for a dataset
    Returns: slope, intercept, df_clean with ρ values
    """
    df_clean = df_data.dropna(subset=['Avg Test ID Acc', 'Avg Test OOD Acc']).copy()

    id_scores = df_clean['Avg Test ID Acc'].values.reshape(-1, 1)
    ood_scores = df_clean['Avg Test OOD Acc'].values

    # Fit linear regression: OOD = slope * ID + intercept
    reg = LinearRegression()
    reg.fit(id_scores, ood_scores)

    slope = reg.coef_[0]
    intercept = reg.intercept_

    # Calculate baseline prediction
    baseline_ood = reg.predict(id_scores)

    # Effective robustness ρ = actual OOD - baseline OOD
    rho = ood_scores - baseline_ood

    # Add to dataframe
    df_clean['Effective Robustness (ρ)'] = rho
    df_clean['Baseline OOD'] = baseline_ood
    df_clean['Slope'] = slope
    df_clean['Intercept'] = intercept

    return slope, intercept, df_clean


def bootstrap_confidence_band(id_vals, ood_vals, id_range, n_bootstrap=1000):
    """Calculate bootstrap confidence band"""
    np.random.seed(42)
    n_points = len(id_vals)
    bootstrap_predictions = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(n_points, size=n_points, replace=True)
        id_boot = id_vals[indices].reshape(-1, 1)
        ood_boot = ood_vals[indices]

        reg_boot = LinearRegression()
        reg_boot.fit(id_boot, ood_boot)
        pred_boot = reg_boot.predict(id_range.reshape(-1, 1))
        bootstrap_predictions.append(pred_boot)

    bootstrap_predictions = np.array(bootstrap_predictions)
    lower_bound = np.percentile(bootstrap_predictions, 2.5, axis=0)
    upper_bound = np.percentile(bootstrap_predictions, 97.5, axis=0)

    return lower_bound, upper_bound


def plot_shift(ax, df_data, shift_name, setting_name, models, show_legend=False, show_task_legend=False, xlim=None,
               legends_outside=False):
    """Plot a single shift type on given axis"""

    slope, intercept, df_clean = calculate_effective_robustness(df_data)

    # Scatter: colour by task (shift combination), shape by model
    task_handles = []
    present = set(df_clean['Task'])
    for task_name in [t for t in SHIFT_GROUPS[shift_name] if t in present]:
        df_task = df_clean[df_clean['Task'] == task_name]
        for model in df_task['Model'].unique():
            if model in HIDDEN_MODELS:
                continue
            df_sub = df_task[df_task['Model'] == model]
            marker = MODEL_MARKERS.get(model, 'o')
            scatter_model(ax, df_sub, TASK_COLORS.get(task_name, 'gray'), marker,
                          marker_size(marker, 80), 0.5, alpha=0.8)
        task_handles.append(task_handle(task_name, 8))

    # Bootstrap confidence band
    id_vals = df_clean['Avg Test ID Acc'].values
    ood_vals = df_clean['Avg Test OOD Acc'].values
    id_range = np.linspace(id_vals.min(), id_vals.max(), 100)
    lower_bound, upper_bound = bootstrap_confidence_band(id_vals, ood_vals, id_range)

    ax.fill_between(id_range, lower_bound, upper_bound, alpha=0.2, color='black')

    # Baseline and y=x lines
    baseline_range = slope * id_range + intercept
    ax.plot(id_range, baseline_range, 'k--', linewidth=2)

    max_val = max(id_vals.max(), ood_vals.max())
    min_val = min(id_vals.min(), ood_vals.min())
    # y=x (no degradation) reference: solid red so it stands out from the dashed black fit
    ax.plot([min_val, max_val], [min_val, max_val], color=YX_COLOR, linestyle='-', linewidth=2.2,
            alpha=0.9, zorder=1)

    # Labels and styling
    ax.set_xlabel('In-Distribution Performance', fontsize=11, fontweight='bold')
    ax.set_ylabel('Out-of-Distribution Performance', fontsize=11, fontweight='bold')
    ax.set_title(f'{shift_name} ({setting_name})', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    if xlim is not None:
        ax.set_xlim(xlim)

    # Equation annotation
    tau, _ = ranking_agreement(df_data, shift_name)
    ax.text(0.02, 0.98, f'OOD = {slope:.3f}×ID + {intercept:.3f}\n{fmt_tau(tau)}',
            transform=ax.transAxes, fontsize=11, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8,
                      edgecolor='black', linewidth=1))

    legends = []

    # Task shape legend — only on full fine-tuning (right) column
    if show_task_legend:
        task_loc = dict(loc='upper left', bbox_to_anchor=(1.02, 1.0)) if legends_outside else dict(loc='lower right')
        leg_tasks = ax.legend(handles=task_handles, fontsize=11, title='Task',
                              frameon=True, fancybox=True, framealpha=0.9, **task_loc)
        ax.add_artist(leg_tasks)
        legends.append(leg_tasks)

    # Model color legend — only on top-left subplot, includes CI and y=x entries
    if show_legend:
        # List every model (not just those present in this panel)
        model_handles = [model_handle(m, 7) for m in models]
        extra_handles = [
            mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Best Fit'),
            mlines.Line2D([], [], color=YX_COLOR, linestyle='-', linewidth=2.2, alpha=0.9,
                          label='y=x (no degradation)'),
            plt.Rectangle((0, 0), 1, 1, fc='black', alpha=0.2, label='95% CI'),
        ] + ([failed_handle(7)] if SHOW_FAILED_LEGEND else [])
        if legends_outside:
            # Model legend goes above the figure so it doesn't collide with the task legend
            legends.append(ax.legend(handles=model_handles + extra_handles, loc='lower center',
                                     bbox_to_anchor=(0.5, 1.12), fontsize=9, ncol=4,
                                     frameon=True, fancybox=True, framealpha=0.9))
        else:
            ax.legend(handles=model_handles + extra_handles, loc='lower right', fontsize=9,
                      ncol=2, frameon=True, fancybox=True, shadow=True, framealpha=0.9)

    return legends



# ============================================================================
# FULL-PAGE FIGURE (single setting)
# ============================================================================

# Physical size of the figure in inches. 6.875in = CVPR/ICCV \textwidth; height leaves room for a caption.
PAGE_FIG_SIZE = (6.875, 4.9)  # full text width, 2 rows x 3 columns


def _fmt_line(slope, intercept):
    m = f'{slope:.3f}'.replace('-', '\u2212')
    sign = '\u2212' if intercept < 0 else '+'
    return f'OOD = {m}\u00b7ID {sign} {abs(intercept):.3f}'


def _emptiest_corner(ax, x, y):
    """Pick the corner (excluding upper-left, used by the task legend) with the fewest data points."""
    ax.figure.canvas.draw()  # finalize limits
    pts = ax.transAxes.inverted().transform(ax.transData.transform(np.column_stack([x, y])))
    corners = {  # name: (x0, x1, y0, y1) region in axes fraction, (text x, y, ha, va)
        'lower right': ((0.50, 1.0, 0.0, 0.14), (0.98, 0.03, 'right', 'bottom')),
        'lower left':  ((0.0, 0.50, 0.0, 0.14), (0.02, 0.03, 'left', 'bottom')),
        'upper right': ((0.50, 1.0, 0.86, 1.0), (0.98, 0.97, 'right', 'top')),
    }
    def count(r):
        x0, x1, y0, y1 = r
        return int(np.sum((pts[:, 0] >= x0) & (pts[:, 0] <= x1) & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)))
    best = min(corners, key=lambda k: count(corners[k][0]))  # dict order => ties go to lower right
    return corners[best][1]


def _clear_boxes(ax, x, y, legend, fit_box, step=0.06, max_steps=12):
    """Widen the y-range until no data point sits under the task legend or the fit box.
    The legend is anchored to the top of the axes, so overlaps are cleared by raising the top limit;
    a fit box at the bottom is cleared by lowering the bottom limit (one at the top: raising the top)."""
    xy = np.column_stack([x, y])
    fig = ax.figure

    def n_under(artist):
        fig.canvas.draw()
        # pad by about half a marker so points poking out from under the box also count
        bb = artist.get_window_extent().padded(10).transformed(ax.transAxes.inverted())
        pts = ax.transAxes.inverted().transform(ax.transData.transform(xy))
        return int(np.sum((pts[:, 0] >= bb.x0) & (pts[:, 0] <= bb.x1) &
                          (pts[:, 1] >= bb.y0) & (pts[:, 1] <= bb.y1)))

    fit_at_top = fit_box.get_va() == 'top'
    for _ in range(max_steps):
        lo, hi = ax.get_ylim()
        r = hi - lo
        top_hit = n_under(legend) > 0 or (fit_at_top and n_under(fit_box) > 0)
        bottom_hit = not fit_at_top and n_under(fit_box) > 0
        if not (top_hit or bottom_hit):
            break
        ax.set_ylim(lo - step * r * bottom_hit, hi + step * r * top_hit)


def plot_shift_page(ax, df_data, shift_name, xlim=None):
    """Compact panel for the full-page figure. Placeholder panel if the shift has no data yet."""
    tasks = SHIFT_GROUPS[shift_name]
    task_handles = [task_handle(t, 4.5) for t in tasks]
    ax.set_title(shift_name, fontsize=10, fontweight='bold', pad=4)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=7, length=2.5, pad=2)

    df_clean = df_data.dropna(subset=['Avg Test ID Acc', 'Avg Test OOD Acc'])
    if len(df_clean) < 2:
        # Dummy panel: keeps the final layout while results are pending
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.plot([0, 1], [0, 1], color=YX_COLOR, linestyle='-', linewidth=1.5, alpha=0.35, zorder=1)
        ax.text(0.5, 0.5, 'Results pending', transform=ax.transAxes, ha='center', va='center',
                fontsize=9, style='italic', color='0.45')
    else:
        slope, intercept, df_clean = calculate_effective_robustness(df_data)
        present = [t for t in tasks if t in set(df_clean['Task'])]
        task_handles = [h for h, t in zip(task_handles, tasks) if t in present]

        for task_name in present:
            df_task = df_clean[df_clean['Task'] == task_name]
            for model, df_sub in df_task.groupby('Model'):
                if model in HIDDEN_MODELS:
                    continue
                marker = MODEL_MARKERS.get(model, 'o')
                scatter_model(ax, df_sub, TASK_COLORS.get(task_name, 'gray'), marker,
                              marker_size(marker, 26), 0.3, alpha=0.85, zorder=3)

        id_vals = df_clean['Avg Test ID Acc'].values
        ood_vals = df_clean['Avg Test OOD Acc'].values
        id_range = np.linspace(id_vals.min(), id_vals.max(), 100)
        lower, upper = bootstrap_confidence_band(id_vals, ood_vals, id_range)
        ax.fill_between(id_range, lower, upper, alpha=0.18, color='black', linewidth=0, zorder=1)
        ax.plot(id_range, slope * id_range + intercept, 'k--', linewidth=1.2, zorder=2)

        lo = min(id_vals.min(), ood_vals.min())
        hi = max(id_vals.max(), ood_vals.max())
        ax.plot([lo, hi], [lo, hi], color=YX_COLOR, linestyle='-', linewidth=1.6, alpha=0.9, zorder=1)

        if xlim is not None:
            ax.set_xlim(xlim)

        # Fit equation in whichever free corner has the fewest points
        tx, ty, ha, va = _emptiest_corner(ax, id_vals, ood_vals)
        tau, _ = ranking_agreement(df_data, shift_name)
        fit_box = ax.text(tx, ty, _fmt_line(slope, intercept) + '\n' + fmt_tau(tau), transform=ax.transAxes, fontsize=7, fontweight='bold',
                ha=ha, va=va, zorder=5,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9,
                          edgecolor='0.6', linewidth=0.5))

    # Task (colour) legend in the upper-left, the region above y=x that is empty when models degrade
    leg = ax.legend(handles=task_handles, loc='upper left', fontsize=6.5, frameon=True, framealpha=0.9,
                    edgecolor='0.7', handletextpad=0.3, borderpad=0.35, labelspacing=0.3, borderaxespad=0.4)
    if len(df_clean) >= 2:
        _clear_boxes(ax, df_clean['Avg Test ID Acc'].values, df_clean['Avg Test OOD Acc'].values,
                     leg, fit_box)


def make_page_figure(data, key, models, out_stem):
    """2x3 full-width figure: five shift panels + one legend panel (bottom right)."""
    shifts = ['Temporal Shift', 'Scale Shift', 'Sensor Shift', 'Geographic Shift', 'Processing Shift']
    df = data[key]

    fig, axes = plt.subplots(2, 3, figsize=PAGE_FIG_SIZE, layout='constrained')
    fig.get_layout_engine().set(w_pad=0.04, h_pad=0.04, hspace=0.04, wspace=0.04)
    axes = axes.ravel()

    for ax, shift_name in zip(axes, shifts):
        tasks = SHIFT_GROUPS[shift_name]
        df_shift = df[df['Task'].isin(tasks)]
        ids = df_shift['Avg Test ID Acc'].dropna()
        xlim = None
        if len(ids) >= 2:
            # Tight x-range around the ID data (y=x would otherwise stretch it to the min OOD value)
            pad = 0.05 * (ids.max() - ids.min())
            xlim = (ids.min() - pad, ids.max() + pad)
        plot_shift_page(ax, df_shift, shift_name, xlim=xlim)

    # Legend panel
    leg_ax = axes[-1]
    leg_ax.axis('off')
    model_handles = [model_handle(m, 5.5) for m in models]
    line_handles = [
        mlines.Line2D([], [], color=YX_COLOR, linestyle='-', linewidth=1.6, label='y = x (no degradation)'),
        mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.2, label='Linear fit'),
        plt.Rectangle((0, 0), 1, 1, fc='black', alpha=0.18, label='95% CI (bootstrap)'),
    ] + ([failed_handle(5.5)] if SHOW_FAILED_LEGEND else [])
    leg_models = leg_ax.legend(handles=model_handles, title='Model', loc='upper center',
                               bbox_to_anchor=(0.42, 1.02), ncol=2, fontsize=6.5, title_fontsize=7.5,
                               frameon=False, columnspacing=0.6, handletextpad=0.1, handlelength=1.2,
                               labelspacing=0.35)
    leg_models._legend_box.align = 'left'
    leg_ax.add_artist(leg_models)
    leg_ax.legend(handles=line_handles, loc='upper center', bbox_to_anchor=(0.5, 0.30), ncol=1,
                  fontsize=6.5, frameon=False, handlelength=2.2, labelspacing=0.35)

    fig.supxlabel('In-distribution performance', fontsize=9.5)
    fig.supylabel('Out-of-distribution performance', fontsize=9.5)

    for ext in ('png', 'pdf'):
        out = f'{out_stem}.{ext}'
        fig.savefig(out, dpi=300)  # no bbox_inches='tight': keep exact physical size for the page
        print(f"Saved: {out}")
    plt.close(fig)

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 80)
    print("EFFECTIVE ROBUSTNESS ANALYSIS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    # (key, column title, csv path) for each setting that was provided
    settings = [(key, title, path) for key, title, path in [
        ('Frozen', 'Frozen Backbone', FROZEN_CSV),
        ('Full', 'Full Fine-tuning', FULL_CSV),
    ] if path is not None]

    data = {}
    for key, _, path in settings:
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            # CSVs saved from Excel on macOS can be Mac Roman encoded (e.g. a '—' placeholder)
            df = pd.read_csv(path, encoding='mac_roman')
        df = df.dropna(subset=['Task', 'Model'])
        df = df[~df['Model'].isin(EXCLUDED_MODELS)]
        df['Avg Test ID Acc'] = df['Avg Test ID Acc'].apply(convert_to_float)
        df['Avg Test OOD Acc'] = df['Avg Test OOD Acc'].apply(convert_to_float)
        data[key] = mark_failed_runs(df)
        if data[key]['Failed'].any():
            global SHOW_FAILED_LEGEND
            SHOW_FAILED_LEGEND = True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_models = sorted(set().union(*[set(df['Model'].unique()) for df in data.values()]))
    assign_task_colors(set().union(*[set(df['Task']) for df in data.values()]))
    print(f"Found {len(all_models)} models: {', '.join(all_models)}")
    unstyled = sorted(set(all_models) - set(MODEL_MARKERS) - HIDDEN_MODELS)
    if unstyled:
        print(f"  WARNING: no marker for model(s) {unstyled}; drawn as circles")

    # Report tasks in SHIFT_GROUPS with no data, and tasks in the CSV not in any group
    grouped_tasks = {t for tasks in SHIFT_GROUPS.values() for t in tasks}
    for key, df in data.items():
        csv_tasks = set(df['Task'].unique())
        missing = sorted(grouped_tasks - csv_tasks)
        unused = sorted(csv_tasks - grouped_tasks)
        if missing:
            print(f"  [{key}] tasks in SHIFT_GROUPS with no data: {missing}")
        if unused:
            print(f"  [{key}] tasks in CSV not in any shift group (ignored): {unused}")

    # ========================================================================
    # CALCULATE EFFECTIVE ROBUSTNESS FOR ALL SHIFTS
    # ========================================================================

    results = {key: [] for key in data}

    candidate_shifts = ['Temporal Shift', 'Scale Shift', 'Sensor Shift',
                        'Geographic Shift', 'Processing Shift']

    # Only keep shifts that have enough data (>=2 points) in every provided setting
    shift_order = []
    for shift_name in candidate_shifts:
        tasks = SHIFT_GROUPS[shift_name]
        n_pts = [data[k][data[k]['Task'].isin(tasks)]
                 .dropna(subset=['Avg Test ID Acc', 'Avg Test OOD Acc']).shape[0] for k in data]
        if min(n_pts) < 2:
            print(f"Skipping {shift_name}: no data yet")
        else:
            shift_order.append(shift_name)

    print("\n" + "=" * 80)
    print("CALCULATING EFFECTIVE ROBUSTNESS")
    print("=" * 80)

    for shift_name in shift_order:
        tasks = SHIFT_GROUPS[shift_name]

        print(f"\n{shift_name}:")

        for key, df in data.items():
            df_shift = df[df['Task'].isin(tasks)]
            slope, intercept, df_clean = calculate_effective_robustness(df_shift)
            df_clean['Setting'] = key
            df_clean['Shift Type'] = shift_name
            results[key].append(df_clean)

            print(f"  {key}: OOD = {slope:.3f} × ID + {intercept:.3f}")
            print(f"          Mean ρ = {df_clean['Effective Robustness (ρ)'].mean():.4f}")

    # ========================================================================
    # RANKING AGREEMENT: Kendall's tau(ID, OOD) per task, mean per shift
    # ========================================================================

    print("\n" + "=" * 80)
    print(f"RANKING AGREEMENT (Kendall's tau per task; failed = ID < {FAILED_ID_FRACTION} x task median ID)")
    print("=" * 80)

    mean_tau = {}
    for key, df in data.items():
        print(f"\n[{key}]")
        print(f"  {'Shift':<18} {'Task':<26} {'n_models':>8} {'n_failed':>8} {'tau':>7}")
        for shift_name in shift_order:
            tau, rows = ranking_agreement(df[df['Task'].isin(SHIFT_GROUPS[shift_name])], shift_name)
            mean_tau[(key, shift_name)] = tau
            for r in rows:
                print(f"  {shift_name:<18} {r['Task']:<26} {r['n_models']:>8} {r['n_failed']:>8} {r['tau']:>7.3f}")
            print(f"  {shift_name:<18} {'MEAN over tasks':<26} {'':>8} {'':>8} {tau:>7.3f}\n")

    # Combine all results
    df_all = {key: pd.concat(res, ignore_index=True) for key, res in results.items()}
    df_all_combined = pd.concat(list(df_all.values()), ignore_index=True)

    # ========================================================================
    # SAVE EFFECTIVE ROBUSTNESS VALUES CSV
    # ========================================================================

    print("\n" + "=" * 80)
    print("SAVING EFFECTIVE ROBUSTNESS VALUES")
    print("=" * 80)

    # Save combined results
    output_file = os.path.join(OUTPUT_DIR, 'effective_robustness_all_values.csv')
    df_all_combined.to_csv(output_file, index=False)
    print(f"All values saved to: {output_file}")

    # Save per-setting values
    for key, df_setting in df_all.items():
        output_file_setting = os.path.join(OUTPUT_DIR, f'effective_robustness_values_{key.lower()}.csv')
        df_setting.to_csv(output_file_setting, index=False)
        print(f"{key} values saved to: {output_file_setting}")

    # ========================================================================
    # CALCULATE AND SAVE RANKINGS
    # ========================================================================

    print("\n" + "=" * 80)
    print("CALCULATING RANKINGS")
    print("=" * 80)

    # Calculate mean ρ per model per shift per setting
    rankings_data = []

    for setting, df_setting in df_all.items():
        for shift_name in shift_order:
            df_shift = df_setting[df_setting['Shift Type'] == shift_name]

            # Calculate mean ρ per model
            mean_rho = df_shift.groupby('Model')['Effective Robustness (ρ)'].mean()

            # Rank (1 = highest ρ = most robust)
            ranks = mean_rho.rank(ascending=False, method='average')

            for model in mean_rho.index:
                rankings_data.append({
                    'Model': model,
                    'Shift Type': shift_name,
                    'Setting': setting,
                    'Mean Effective Robustness (ρ)': mean_rho[model],
                    'Rank': int(ranks[model])
                })

    df_rankings = pd.DataFrame(rankings_data)

    # Save rankings CSV
    output_rankings = os.path.join(OUTPUT_DIR, 'effective_robustness_rankings_per_shift.csv')
    df_rankings.to_csv(output_rankings, index=False)
    print(f"Rankings saved to: {output_rankings}")

    # Create pivot table for easier viewing
    for key in df_all:
        pivot = df_rankings[df_rankings['Setting'] == key].pivot(
            index='Shift Type', columns='Model', values='Rank'
        )
        pivot.to_csv(os.path.join(OUTPUT_DIR, f'rankings_pivot_{key.lower()}.csv'))
        print(f"Pivot table saved to: rankings_pivot_{key.lower()}.csv")

    # Calculate average ranks
    avg_ranks_data = []
    for setting in df_all:
        df_setting_ranks = df_rankings[df_rankings['Setting'] == setting]
        avg_ranks = df_setting_ranks.groupby('Model')['Rank'].mean().sort_values()

        for model, avg_rank in avg_ranks.items():
            avg_ranks_data.append({
                'Model': model,
                'Setting': setting,
                'Average Rank': avg_rank
            })

    df_avg_ranks = pd.DataFrame(avg_ranks_data)
    output_avg_ranks = os.path.join(OUTPUT_DIR, 'average_ranks_by_model.csv')
    df_avg_ranks.to_csv(output_avg_ranks, index=False)
    print(f"Average ranks saved to: {output_avg_ranks}")

    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================

    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    summary_stats = []

    for setting, df_setting in df_all.items():
        for shift_name in shift_order:
            df_shift = df_setting[df_setting['Shift Type'] == shift_name]

            summary_stats.append({
                'Shift Type': shift_name,
                'Setting': setting,
                'Baseline Slope': df_shift['Slope'].iloc[0],
                'Baseline Intercept': df_shift['Intercept'].iloc[0],
                'Mean ρ': df_shift['Effective Robustness (ρ)'].mean(),
                'Std ρ': df_shift['Effective Robustness (ρ)'].std(),
                'Min ρ': df_shift['Effective Robustness (ρ)'].min(),
                'Max ρ': df_shift['Effective Robustness (ρ)'].max(),
                'N Models': df_shift['Model'].nunique(),
                'N Data Points': len(df_shift),
                'Mean Kendall tau (ID, OOD)': mean_tau[(setting, shift_name)]
            })

    df_summary = pd.DataFrame(summary_stats)
    output_summary = os.path.join(OUTPUT_DIR, 'effective_robustness_summary_statistics.csv')
    df_summary.to_csv(output_summary, index=False)
    print(f"Summary statistics saved to: {output_summary}")

    print("\n" + df_summary.to_string(index=False))

    # ========================================================================
    # FULL-PAGE FIGURE(S): one per setting, 3x2 layout incl. placeholder panels
    # ========================================================================

    for key in data:
        print(f"\nCreating full-page figure ({key})...")
        make_page_figure(data, key, [m for m in all_models if m not in HIDDEN_MODELS],
                         os.path.join(OUTPUT_DIR, f'effective_robustness_page_{key.lower()}'))

    # ========================================================================
    # CREATE COMPREHENSIVE FIGURE (rows = shifts, cols = settings)
    # ========================================================================

    n_rows, n_cols = len(shift_order), len(settings)
    fig_tag = f'{n_rows}x{n_cols}'

    print("\n" + "=" * 80)
    print(f"CREATING {fig_tag} COMPREHENSIVE FIGURE")
    print("=" * 80)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 4.4 * n_rows), squeeze=False)

    extra_artists = []  # legends placed outside the axes, so bbox_inches='tight' keeps them
    for row_idx, shift_name in enumerate(shift_order):
        print(f"Plotting {shift_name}...")

        tasks = SHIFT_GROUPS[shift_name]
        xlim = None
        if shift_name in ('Geographic Shift', 'Sensor Shift', 'Processing Shift'):
            # Fit x-axis tightly to the ID data (the y=x line otherwise stretches it down to the min OOD value)
            ids = pd.concat([data[k][data[k]['Task'].isin(tasks)]['Avg Test ID Acc'] for k in data]).dropna()
            pad = 0.05 * (ids.max() - ids.min())
            xlim = (ids.min() - pad, ids.max() + pad)

        for col_idx, (key, title, _) in enumerate(settings):
            df_shift = data[key][data[key]['Task'].isin(tasks)]
            extra_artists += plot_shift(axes[row_idx, col_idx], df_shift, shift_name, title, [m for m in all_models if m not in HIDDEN_MODELS],
                       show_legend=(row_idx == 0 and col_idx == 0),     # model legend top-left
                       show_task_legend=(col_idx == n_cols - 1),        # task legend right column
                       xlim=xlim, legends_outside=(n_cols == 1))

    plt.tight_layout()

    # Save figure
    output_fig = os.path.join(OUTPUT_DIR, f'effective_robustness_comprehensive_{fig_tag}.png')
    plt.savefig(output_fig, dpi=300, bbox_inches='tight', bbox_extra_artists=extra_artists)
    print(f"Figure saved to: {output_fig}")

    output_pdf = os.path.join(OUTPUT_DIR, f'effective_robustness_comprehensive_{fig_tag}.pdf')
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight', bbox_extra_artists=extra_artists)
    print(f"PDF saved to: {output_pdf}")

    plt.close()

    # ========================================================================
    # PRINT TOP PERFORMERS
    # ========================================================================

    print("\n" + "=" * 80)
    print("TOP PERFORMERS")
    print("=" * 80)

    for setting in df_all:
        print(f"\n{setting.upper()}:")
        df_setting_ranks = df_rankings[df_rankings['Setting'] == setting]
        avg_ranks = df_setting_ranks.groupby('Model')['Rank'].mean().sort_values()

        print(f"Top 5 by Average Rank:")
        for i, (model, avg_rank) in enumerate(avg_ranks.head(5).items(), 1):
            print(f"  {i}. {model}: {avg_rank:.2f}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("\nFiles created:")
    print("  1. effective_robustness_all_values.csv - All ρ values")
    for key in df_all:
        print(f"  - effective_robustness_values_{key.lower()}.csv, rankings_pivot_{key.lower()}.csv")
    print("  - effective_robustness_all_values.csv, effective_robustness_rankings_per_shift.csv")
    print("  - average_ranks_by_model.csv, effective_robustness_summary_statistics.csv")
    print(f"  - effective_robustness_comprehensive_{fig_tag}.png / .pdf - Main figure")
    print("=" * 80)


if __name__ == "__main__":
    main()