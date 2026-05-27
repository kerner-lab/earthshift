import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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

# Load both sheets
df_head = pd.read_csv('{}/{}.csv'.format(root_dir, frozen_filename))
df_full = pd.read_csv('{}/{}.csv'.format(root_dir, full_filename))

# Remove resnet50 (keep only resnet50_imgnet)
df_head = df_head[df_head['Model'] != 'resnet50']
df_full = df_full[df_full['Model'] != 'resnet50']

# Define ALL tasks
tasks = ['RESISC45-UCMerced', 'UCMerced-RESISC45', 'DeepGlobe-DFC2022', 'Geo Germany-Cambodia',
         'Geo Germany-Denmark', 'FTW Seasonal All', 'Germany Temporal', 'SA Temporal',
         'RGB - RGE1', 'Sen1Floods11-S2-S1', 'BenV2-S2-S1']
task_labels = ['RESISC45-\nUCMerced', 'UCMerced-\nRESISC45', 'DeepGlobe-\nDFC2022', 'Germany-\nCambodia',
               'Germany-\nDenmark', 'FTW \nSeasonal All', 'Germany\nTemporal', 'South Africa\nSeasonal', 'RGB -\nRGE1',
               'Sen1Floods11-\nS2-S1', 'BenV2-\nS2-S1',]

# Calculate average and std across all models for each task
results = []
for task in tasks:
    # Head fine-tuning
    head_data = df_head[df_head['Task'] == task]['Delta']
    head_mean = head_data.mean()
    head_std = head_data.std()

    # Full fine-tuning
    full_data = df_full[df_full['Task'] == task]['Delta']
    full_mean = full_data.mean()
    full_std = full_data.std()

    results.append({
        'task': task,
        'head_mean': head_mean,
        'head_std': head_std,
        'full_mean': full_mean,
        'full_std': full_std,
    })

results_df = pd.DataFrame(results)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 7))

# Set up bar positions
x = np.arange(len(tasks))
width = 0.35

# Define colors (matching the reference image style)
full_color = '#5b9aa9'  # Teal/cyan
head_color = '#c685a0'  # Pink/mauve

# Plot bars with error bars (going downward from 0)
bars1 = ax.bar(x - width / 2, results_df['full_mean'], width,
               yerr=results_df['full_std'],
               label='Full Fine-tuning',
               color=full_color, edgecolor='black', linewidth=0.8,
               error_kw={'linewidth': 2, 'ecolor': 'black', 'capsize': 4})

bars2 = ax.bar(x + width / 2, results_df['head_mean'], width,
               yerr=results_df['head_std'],
               label='Head Fine-tuning',
               color=head_color, edgecolor='black', linewidth=0.8,
               error_kw={'linewidth': 2, 'ecolor': 'black', 'capsize': 4})


# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height - 0.015,
                f'{abs(height):.3f}',
                ha='center', va='top', fontsize=11, fontweight='bold')


add_value_labels(bars1)
add_value_labels(bars2)

# Customize plot
ax.set_ylabel('Average Performance Gap (Δ = ID - OOD)', fontsize=14, fontweight='bold')
ax.set_xlabel('Task', fontsize=14, fontweight='bold')
ax.set_title('Distribution Shift Impact Across Tasks', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(task_labels, rotation=0, ha='center', fontsize=11)
ax.legend(loc='lower center', fontsize=12, frameon=True, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Set y-axis to go from 0 downward
y_min = min(results_df[['full_mean', 'head_mean']].min()) * 1.15
ax.set_ylim([y_min, 0.05])

# Adjust tick label size
ax.tick_params(axis='both', which='major', labelsize=11)

plt.tight_layout()
plt.savefig('{}/averaged_distribution_shift_all_tasks.png'.format(root_dir), dpi=300, bbox_inches='tight')
plt.show()

print("Figure saved!")
print("\nAveraged results across all models:")
for i, row in results_df.iterrows():
    print(f"\n{tasks[i]}:")
    print(f"  Full fine-tuning: {row['full_mean']:.3f} ± {row['full_std']:.3f}")
    print(f"  Head fine-tuning: {row['head_mean']:.3f} ± {row['head_std']:.3f}")

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(f"Tasks included: {len(tasks)}")
print(f"Models per task: ~{len(df_head['Model'].unique())}")
print(f"\nFull fine-tuning average across all tasks: {results_df['full_mean'].mean():.3f}")
print(f"Head fine-tuning average across all tasks: {results_df['head_mean'].mean():.3f}")