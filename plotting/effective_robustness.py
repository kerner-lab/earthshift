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
import seaborn as sns
import os

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True,
                    help='Root directory containing input CSV files and where outputs will be saved')
parser.add_argument('--frozen_filename', type=str, default=None,
                    help='File of frozen-backbone (head fine-tuning) data to plot (without .csv)')
parser.add_argument('--full_filename', type=str, default=None,
                    help='File of full finetuning to plot (without .csv)')
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
OUTPUT_DIR = '{}/effective'.format(root_dir)

# Task groupings by shift type
SHIFT_GROUPS = {
    'Temporal Shift': ['ftw-all-window', 'SA Temporal', 'Germany Temporal', 'France RPG 5year'],
    'Scale Shift': ['Chesapeake-CVPR-Landsat', 'Chesapeake-CVPR-NAIP'],
    'Sensor Shift': ['BenV2-S2-S1', 'Sen1Floods11-S2-S1', 'RGB - RGE1', 'RESISC45-UCMerced', 'UCMerced-RESISC45',
                     'DeepGlobe-DFC2022'],
    'Geographic Shift': ['Geo Germany-Cambodia', 'Geo Germany-Denmark', 'BRIGHT'],
    'Processing Shift': ['CloudSEN-L2A-L1C', 'FTW-France-AI4Boundaries']
}

# Color of the y=x (no degradation) reference line
YX_COLOR = '#d62728'

# Display names for task legend labels
TASK_DISPLAY_NAMES = {
    'ftw-all-window':  'Season-to-Season',
    'SA Temporal':     'SA Season-to-Season',
    'Germany Temporal':'Germany Year-to-Year',
    'BRIGHT':          'BRIGHT Turkey-Morocco',
}

# Marker shapes per task for visual differentiation
TASK_MARKERS = {
    'ftw-all-window':      'o',
    'SA Temporal':         's',
    'Germany Temporal':    '^',
    'RESISC45-UCMerced':   'D',
    'UCMerced-RESISC45':   'P',
    'BenV2-S2-S1':         'X',
    'Sen1Floods11-S2-S1':  '*',
    'RGB - RGE1':          'h',
    'Geo Germany-Cambodia':'v',
    'Geo Germany-Denmark': '>',
    'DeepGlobe-DFC2022':   'p',
    'France RPG 5year':    '<',
    'Chesapeake-CVPR-Landsat': 'd',
    'Chesapeake-CVPR-NAIP':    'H',
    'BRIGHT':              '8',
    'CloudSEN-L2A-L1C':    'o',
    'FTW-France-AI4Boundaries': 's',
}


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


def plot_shift(ax, df_data, shift_name, setting_name, model_colors, show_legend=False, show_task_legend=False, xlim=None,
               legends_outside=False):
    """Plot a single shift type on given axis"""

    slope, intercept, df_clean = calculate_effective_robustness(df_data)

    # Scatter: color by model, shape by task
    seen_models = set()
    model_handles = []
    task_handles = []
    seen_tasks = set()

    for task_name in df_clean['Task'].unique():
        marker = TASK_MARKERS.get(task_name, 'o')
        df_task = df_clean[df_clean['Task'] == task_name]
        for model in df_task['Model'].unique():
            df_sub = df_task[df_task['Model'] == model]
            ax.scatter(df_sub['Avg Test ID Acc'],
                       df_sub['Avg Test OOD Acc'],
                       c=[model_colors[model]],
                       marker=marker,
                       s=80,
                       alpha=0.7,
                       edgecolors='black',
                       linewidth=0.5)
            if model not in seen_models:
                model_handles.append(mlines.Line2D(
                    [], [], color=model_colors[model], marker='o', linestyle='None',
                    markersize=7, label=model))
                seen_models.add(model)
        if task_name not in seen_tasks:
            display_name = TASK_DISPLAY_NAMES.get(task_name, task_name)
            task_handles.append(mlines.Line2D(
                [], [], color='black', marker=marker, linestyle='None',
                markersize=7, label=display_name))
            seen_tasks.add(task_name)

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
    ax.text(0.02, 0.98, f'OOD = {slope:.3f}×ID + {intercept:.3f}',
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
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
        model_handles = [mlines.Line2D([], [], color=c, marker='o', linestyle='None', markersize=7, label=m)
                         for m, c in model_colors.items()]
        extra_handles = [
            mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Best Fit'),
            mlines.Line2D([], [], color=YX_COLOR, linestyle='-', linewidth=2.2, alpha=0.9,
                          label='y=x (no degradation)'),
            plt.Rectangle((0, 0), 1, 1, fc='black', alpha=0.2, label='95% CI'),
        ]
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
PAGE_FIG_SIZE = (6.875, 8.25)


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


def plot_shift_page(ax, df_data, shift_name, model_colors, xlim=None):
    """Compact panel for the full-page figure. Placeholder panel if the shift has no data yet."""
    tasks = SHIFT_GROUPS[shift_name]
    task_handles = [mlines.Line2D([], [], color='black', marker=TASK_MARKERS.get(t, 'o'), linestyle='None',
                                  markersize=4.5, label=TASK_DISPLAY_NAMES.get(t, t)) for t in tasks]
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
            ax.scatter(df_task['Avg Test ID Acc'], df_task['Avg Test OOD Acc'],
                       c=[model_colors[m] for m in df_task['Model']],
                       marker=TASK_MARKERS.get(task_name, 'o'), s=26, alpha=0.85,
                       edgecolors='black', linewidth=0.3, zorder=3)

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
        ax.text(tx, ty, _fmt_line(slope, intercept), transform=ax.transAxes, fontsize=7,
                ha=ha, va=va, zorder=5,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.9,
                          edgecolor='0.6', linewidth=0.5))

    # Task (marker) legend in the upper-left, the region above y=x that is empty when models degrade
    ax.legend(handles=task_handles, loc='upper left', fontsize=6.5, frameon=True, framealpha=0.9,
              edgecolor='0.7', handletextpad=0.3, borderpad=0.35, labelspacing=0.3, borderaxespad=0.4)


def make_page_figure(data, key, model_colors, out_stem):
    """3x2 full-page figure: five shift panels + one legend panel."""
    shifts = ['Temporal Shift', 'Scale Shift', 'Sensor Shift', 'Geographic Shift', 'Processing Shift']
    df = data[key]

    fig, axes = plt.subplots(3, 2, figsize=PAGE_FIG_SIZE, layout='constrained')
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
        plot_shift_page(ax, df_shift, shift_name, model_colors, xlim=xlim)

    # Legend panel
    leg_ax = axes[-1]
    leg_ax.axis('off')
    model_handles = [mlines.Line2D([], [], color=c, marker='o', linestyle='None', markersize=5.5,
                                   markeredgecolor='black', markeredgewidth=0.3, label=m)
                     for m, c in model_colors.items()]
    line_handles = [
        mlines.Line2D([], [], color=YX_COLOR, linestyle='-', linewidth=1.6, label='y = x (no degradation)'),
        mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.2, label='Linear fit'),
        plt.Rectangle((0, 0), 1, 1, fc='black', alpha=0.18, label='95% CI (bootstrap)'),
    ]
    leg_models = leg_ax.legend(handles=model_handles, title='Model', loc='upper center',
                               bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5, title_fontsize=8.5,
                               frameon=False, columnspacing=1.0, handletextpad=0.3, labelspacing=0.45)
    leg_models._legend_box.align = 'left'
    leg_ax.add_artist(leg_models)
    leg_ax.legend(handles=line_handles, loc='upper center', bbox_to_anchor=(0.5, 0.36), ncol=1,
                  fontsize=7.5, frameon=False, handlelength=2.2, labelspacing=0.45)

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
        df = pd.read_csv(path)
        df = df.dropna(subset=['Task', 'Model'])
        df['Avg Test ID Acc'] = df['Avg Test ID Acc'].apply(convert_to_float)
        df['Avg Test OOD Acc'] = df['Avg Test OOD Acc'].apply(convert_to_float)
        data[key] = df

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Get model colors
    all_models = sorted(set().union(*[set(df['Model'].unique()) for df in data.values()]))
    colors = sns.color_palette("husl", len(all_models))
    model_colors = dict(zip(all_models, colors))

    print(f"Found {len(all_models)} models: {', '.join(all_models)}")

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
                'N Data Points': len(df_shift)
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
        make_page_figure(data, key, model_colors,
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
            extra_artists += plot_shift(axes[row_idx, col_idx], df_shift, shift_name, title, model_colors,
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