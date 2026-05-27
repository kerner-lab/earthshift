"""
Complete Effective Robustness Analysis Script
==============================================
Generates:
1. 5x2 comprehensive figure (temporal, scale, sensor, geographic, source shifts)
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
parser.add_argument('--frozen_filename', type=str, required=True,
                    help='File of frozen data to plot')
parser.add_argument('--full_filename', type=str, required=True,
                    help='File of full finetuning to plot')
args = parser.parse_args()
root_dir = args.root_dir
frozen_filename = args.frozen_filename
full_filename = args.full_filename
# ============================================================================
# CONFIGURATION
# ============================================================================

FROZEN_CSV = '{}/{}.csv'.format(root_dir, frozen_filename)
FULL_CSV = '{}/{}.csv'.format(root_dir, frozen_filename)
OUTPUT_DIR = '{}/effective'.format(root_dir)

# Task groupings by shift type
SHIFT_GROUPS = {
    'Temporal Shift': ['ftw-all-window', 'SA Temporal', 'Germany Temporal'],
    'Scale Shift': ['RESISC45-UCMerced', 'UCMerced-RESISC45'],
    'Sensor Shift': ['BenV2-S2-S1', 'Sen1Floods11-S2-S1', 'RGB - RGE1'],
    'Geographic Shift': ['Geo Germany-Cambodia', 'Geo Germany-Denmark'],
    'Source Shift': ['DeepGlobe-DFC2022']
}

# Display names for task legend labels
TASK_DISPLAY_NAMES = {
    'ftw-all-window':  'Season-to-Season',
    'SA Temporal':     'SA Season-to-Season',
    'Germany Temporal':'Germany Year-to-Year',
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


def plot_shift(ax, df_data, shift_name, setting_name, model_colors, show_legend=False, show_task_legend=False, xlim=None):
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
    ax.plot([min_val, max_val], [min_val, max_val], 'gray', linestyle=':', linewidth=1, alpha=0.5)

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

    # Task shape legend — only on full fine-tuning (right) column
    if show_task_legend:
        leg_tasks = ax.legend(handles=task_handles, loc='lower right', fontsize=11,
                              frameon=True, fancybox=True, framealpha=0.9)
        ax.add_artist(leg_tasks)

    # Model color legend — only on top-left subplot, includes CI and y=x entries
    if show_legend:
        extra_handles = [
            mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Best Fit'),
            mlines.Line2D([], [], color='gray', linestyle=':', linewidth=1, alpha=0.5, label='y=x'),
            plt.Rectangle((0, 0), 1, 1, fc='black', alpha=0.2, label='95% CI'),
        ]
        ax.legend(handles=model_handles + extra_handles, loc='lower right', fontsize=9,
                  ncol=2, frameon=True, fancybox=True, shadow=True, framealpha=0.9)


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print("=" * 80)
    print("EFFECTIVE ROBUSTNESS ANALYSIS")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df_frozen = pd.read_csv(FROZEN_CSV)
    df_full = pd.read_csv(FULL_CSV)

    df_frozen['Avg Test ID Acc'] = df_frozen['Avg Test ID Acc'].apply(convert_to_float)
    df_frozen['Avg Test OOD Acc'] = df_frozen['Avg Test OOD Acc'].apply(convert_to_float)
    df_full['Avg Test ID Acc'] = df_full['Avg Test ID Acc'].apply(convert_to_float)
    df_full['Avg Test OOD Acc'] = df_full['Avg Test OOD Acc'].apply(convert_to_float)

    # Get model colors
    all_models = sorted(set(df_frozen['Model'].dropna().unique()) | set(df_full['Model'].dropna().unique()))
    colors = sns.color_palette("husl", len(all_models))
    model_colors = dict(zip(all_models, colors))

    print(f"Found {len(all_models)} models")

    # ========================================================================
    # CALCULATE EFFECTIVE ROBUSTNESS FOR ALL SHIFTS
    # ========================================================================

    all_results_frozen = []
    all_results_full = []

    shift_order = ['Temporal Shift', 'Scale Shift', 'Sensor Shift',
                   'Geographic Shift', 'Source Shift']

    print("\n" + "=" * 80)
    print("CALCULATING EFFECTIVE ROBUSTNESS")
    print("=" * 80)

    for shift_name in shift_order:
        tasks = SHIFT_GROUPS[shift_name]

        print(f"\n{shift_name}:")

        # Frozen
        df_frozen_shift = df_frozen[df_frozen['Task'].isin(tasks)]
        slope_f, intercept_f, df_clean_f = calculate_effective_robustness(df_frozen_shift)
        df_clean_f['Setting'] = 'Frozen'
        df_clean_f['Shift Type'] = shift_name
        all_results_frozen.append(df_clean_f)

        print(f"  Frozen: OOD = {slope_f:.3f} × ID + {intercept_f:.3f}")
        print(f"          Mean ρ = {df_clean_f['Effective Robustness (ρ)'].mean():.4f}")

        # Full
        df_full_shift = df_full[df_full['Task'].isin(tasks)]
        slope_full, intercept_full, df_clean_full = calculate_effective_robustness(df_full_shift)
        df_clean_full['Setting'] = 'Full'
        df_clean_full['Shift Type'] = shift_name
        all_results_full.append(df_clean_full)

        print(f"  Full:   OOD = {slope_full:.3f} × ID + {intercept_full:.3f}")
        print(f"          Mean ρ = {df_clean_full['Effective Robustness (ρ)'].mean():.4f}")

    # Combine all results
    df_all_frozen = pd.concat(all_results_frozen, ignore_index=True)
    df_all_full = pd.concat(all_results_full, ignore_index=True)
    df_all_combined = pd.concat([df_all_frozen, df_all_full], ignore_index=True)

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

    # Save frozen only
    output_file_frozen = os.path.join(OUTPUT_DIR, 'effective_robustness_values_frozen.csv')
    df_all_frozen.to_csv(output_file_frozen, index=False)
    print(f"Frozen values saved to: {output_file_frozen}")

    # Save full only
    output_file_full = os.path.join(OUTPUT_DIR, 'effective_robustness_values_full.csv')
    df_all_full.to_csv(output_file_full, index=False)
    print(f"Full values saved to: {output_file_full}")

    # ========================================================================
    # CALCULATE AND SAVE RANKINGS
    # ========================================================================

    print("\n" + "=" * 80)
    print("CALCULATING RANKINGS")
    print("=" * 80)

    # Calculate mean ρ per model per shift per setting
    rankings_data = []

    for setting, df_setting in [('Frozen', df_all_frozen), ('Full', df_all_full)]:
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
    pivot_frozen = df_rankings[df_rankings['Setting'] == 'Frozen'].pivot(
        index='Shift Type', columns='Model', values='Rank'
    )
    pivot_full = df_rankings[df_rankings['Setting'] == 'Full'].pivot(
        index='Shift Type', columns='Model', values='Rank'
    )

    # Save pivot tables
    pivot_frozen.to_csv(os.path.join(OUTPUT_DIR, 'rankings_pivot_frozen.csv'))
    pivot_full.to_csv(os.path.join(OUTPUT_DIR, 'rankings_pivot_full.csv'))
    print(f"Pivot tables saved to: rankings_pivot_frozen.csv and rankings_pivot_full.csv")

    # Calculate average ranks
    avg_ranks_data = []
    for setting in ['Frozen', 'Full']:
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

    for setting, df_setting in [('Frozen', df_all_frozen), ('Full', df_all_full)]:
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
    # CREATE 5x2 COMPREHENSIVE FIGURE
    # ========================================================================

    print("\n" + "=" * 80)
    print("CREATING 5x2 COMPREHENSIVE FIGURE")
    print("=" * 80)

    fig, axes = plt.subplots(5, 2, figsize=(14, 22))

    for row_idx, shift_name in enumerate(shift_order):
        print(f"Plotting {shift_name}...")

        tasks = SHIFT_GROUPS[shift_name]

        # Left column: Frozen
        df_frozen_shift = df_frozen[df_frozen['Task'].isin(tasks)]
        show_legend = (row_idx == 0)  # Only show legend on top-left
        xlim = (0.45, 0.60) if shift_name == 'Geographic Shift' else None
        plot_shift(axes[row_idx, 0], df_frozen_shift, shift_name,
                   'Frozen Backbone', model_colors, show_legend=show_legend, xlim=xlim)

        # Right column: Full
        df_full_shift = df_full[df_full['Task'].isin(tasks)]
        plot_shift(axes[row_idx, 1], df_full_shift, shift_name,
                   'Full Fine-tuning', model_colors, show_legend=False, show_task_legend=True, xlim=xlim)

    plt.tight_layout()

    # Save figure
    output_fig = os.path.join(OUTPUT_DIR, 'effective_robustness_comprehensive_5x2.png')
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_fig}")

    output_pdf = os.path.join(OUTPUT_DIR, 'effective_robustness_comprehensive_5x2.pdf')
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight')
    print(f"PDF saved to: {output_pdf}")

    plt.close()

    # ========================================================================
    # PRINT TOP PERFORMERS
    # ========================================================================

    print("\n" + "=" * 80)
    print("TOP PERFORMERS")
    print("=" * 80)

    for setting in ['Frozen', 'Full']:
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
    print("  2. effective_robustness_values_frozen.csv - Frozen only")
    print("  3. effective_robustness_values_full.csv - Full only")
    print("  4. effective_robustness_rankings_per_shift.csv - Rankings")
    print("  5. rankings_pivot_frozen.csv - Pivot table (frozen)")
    print("  6. rankings_pivot_full.csv - Pivot table (full)")
    print("  7. average_ranks_by_model.csv - Overall rankings")
    print("  8. effective_robustness_summary_statistics.csv - Summary stats")
    print("  9. effective_robustness_comprehensive_5x2.png - Main figure")
    print(" 10. effective_robustness_comprehensive_5x2.pdf - PDF version")
    print("=" * 80)


if __name__ == "__main__":
    main()