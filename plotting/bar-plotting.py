
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

# Define model groups
gfms = ['clay', 'croma', 'dinov3_large_sat', 'dofa', 'galileo', 'prithvi', 'terrafm', 'terramind']
vfms = ['clip', 'vit_imgnet', 'resnet50_imgnet']
supervised = ['resnet_random', 'vit_random']

# Define ALL tasks in order
tasks = ['RESISC45-UCMerced', 'UCMerced-RESISC45', 'DeepGlobe-DFC2022', 'Geo Germany-Cambodia',
         'Geo Germany-Denmark', 'FTW Seasonal All', 'Germany Temporal', 'SA Temporal',
         'RGB - RGE1', 'Sen1Floods11-S2-S1', 'BenV2-S2-S1']
task_labels = ['RESISC45-\nUCMerced', 'UCMerced-\nRESISC45', 'DeepGlobe-\nDFC2022', 'Germany-\nCambodia',
               'Germany-\nDenmark', 'FTW \nSeasonal All', 'Germany\nTemporal', 'South Africa\nSeasonal', 'RGB -\nRGE1',
               'Sen1Floods11-\nS2-S1', 'BenV2-\nS2-S1',]


def calculate_group_stats(df, task, models):
    """Calculate average and std dev of delta for a group of models on a specific task"""
    task_data = df[df['Task'] == task]
    model_data = task_data[task_data['Model'].isin(models)]
    if len(model_data) > 0:
        return model_data['Delta'].mean(), model_data['Delta'].std()
    return 0, 0


# Calculate averages and std devs for each group and task
results = []
for task in tasks:
    gfm_head_mean, gfm_head_std = calculate_group_stats(df_head, task, gfms)
    gfm_full_mean, gfm_full_std = calculate_group_stats(df_full, task, gfms)
    vfm_head_mean, vfm_head_std = calculate_group_stats(df_head, task, vfms)
    vfm_full_mean, vfm_full_std = calculate_group_stats(df_full, task, vfms)
    sup_head_mean, sup_head_std = calculate_group_stats(df_head, task, supervised)
    sup_full_mean, sup_full_std = calculate_group_stats(df_full, task, supervised)

    results.append({
        'task': task,
        'gfm_head': gfm_head_mean,
        'gfm_head_std': gfm_head_std,
        'gfm_full': gfm_full_mean,
        'gfm_full_std': gfm_full_std,
        'vfm_head': vfm_head_mean,
        'vfm_head_std': vfm_head_std,
        'vfm_full': vfm_full_mean,
        'vfm_full_std': vfm_full_std,
        'sup_head': sup_head_mean,
        'sup_head_std': sup_head_std,
        'sup_full': sup_full_mean,
        'sup_full_std': sup_full_std,
    })

results_df = pd.DataFrame(results)

# Create the plot (wider to accommodate 7 tasks)
fig, ax = plt.subplots(figsize=(16, 7))

# Set up bar positions
x = np.arange(len(tasks))
width = 0.12  # Slightly narrower bars to fit 7 tasks

# Define colors (darker for full, lighter for head)
gfm_color_full = '#2d5f3f'  # Dark green
gfm_color_head = '#7db88e'  # Light green
vfm_color_full = '#c4590f'  # Dark orange
vfm_color_head = '#f9b384'  # Light orange
sup_color_full = '#1f4788'  # Dark blue
sup_color_head = '#6b9bd1'  # Light blue

# Plot bars with error bars
bars1 = ax.bar(x - 2.5 * width, results_df['gfm_full'], width,
               yerr=results_df['gfm_full_std'],
               label='GFM - Full Fine-tuning',
               color=gfm_color_full, edgecolor='black', linewidth=0.5,
               error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 3})

bars2 = ax.bar(x - 1.5 * width, results_df['gfm_head'], width,
               yerr=results_df['gfm_head_std'],
               label='GFM - Head Fine-tuning',
               color=gfm_color_head, edgecolor='black', linewidth=0.5,
               error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 3})

bars3 = ax.bar(x - 0.5 * width, results_df['vfm_full'], width,
               yerr=results_df['vfm_full_std'],
               label='VFM - Full Fine-tuning',
               color=vfm_color_full, edgecolor='black', linewidth=0.5,
               error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 3})

bars4 = ax.bar(x + 0.5 * width, results_df['vfm_head'], width,
               yerr=results_df['vfm_head_std'],
               label='VFM - Head Fine-tuning',
               color=vfm_color_head, edgecolor='black', linewidth=0.5,
               error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 3})

bars5 = ax.bar(x + 1.5 * width, results_df['sup_full'], width,
               yerr=results_df['sup_full_std'],
               label='Supervised - Full Fine-tuning',
               color=sup_color_full, edgecolor='black', linewidth=0.5,
               error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 3})

bars6 = ax.bar(x + 2.5 * width, results_df['sup_head'], width,
               yerr=results_df['sup_head_std'],
               label='Supervised - Head Fine-tuning',
               color=sup_color_head, edgecolor='black', linewidth=0.5,
               error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 3})


# Add value labels on bars
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height - 0.015,
                f'{abs(height):.3f}',
                ha='center', va='top', fontsize=8, rotation=0, fontweight='bold')


#add_value_labels(bars1)
#add_value_labels(bars2)
#add_value_labels(bars3)
#add_value_labels(bars4)
#add_value_labels(bars5)
#add_value_labels(bars6)

# Customize plot
ax.set_ylabel('Average Performance Gap (Δ = ID - OOD)', fontsize=13, fontweight='bold')
ax.set_xlabel('Task', fontsize=13, fontweight='bold')
ax.set_title('Distribution Shift Impact: Model Type Comparison', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(task_labels, rotation=0, ha='center', fontsize=10)
ax.legend(loc='lower center', ncol=2, fontsize=10, frameon=True, edgecolor='black')
ax.axhline(y=0, color='black', linestyle='-', linewidth=1.2)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Set y-axis to go from 0 downward
y_min = min(results_df[['gfm_full', 'gfm_head', 'vfm_full', 'vfm_head', 'sup_full', 'sup_head']].min()) * 1.1
ax.set_ylim([y_min, 0.05])

plt.tight_layout()
plt.savefig('{}/model_comparison_barplot_all_tasks.png'.format(root_dir), dpi=300, bbox_inches='tight')
plt.show()

print("Figure saved!")
print("\nAverage deltas by group (with std dev):")
print(results_df)

# Print summary statistics
print("\n" + "=" * 60)
print("TASKS INCLUDED")
print("=" * 60)
for i, task in enumerate(tasks):
    print(f"{i + 1}. {task}")

print("\n" + "=" * 60)
print("MODEL GROUP COMPOSITION")
print("=" * 60)
print(f"GFMs ({len(gfms)} models): {', '.join(gfms)}")
print(f"VFMs ({len(vfms)} models): {', '.join(vfms)}")
print(f"Supervised ({len(supervised)} models): {', '.join(supervised)}")