import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True,
                    help='Root directory containing input CSV files and where outputs will be saved')
parser.add_argument('--filename', type=str, required=True,
                    help='File to plot')
args = parser.parse_args()
root_dir = args.root_dir
filename = args.filename

# Read the data
file_path = '{}/{}'.format(root_dir, filename)
df = pd.read_csv(file_path)

# Convert percentage strings to floats
df['Avg Test ID Acc'] = pd.to_numeric(df['Avg Test ID Acc'], errors='coerce')
df['Avg Test OOD Acc'] = pd.to_numeric(df['Avg Test OOD Acc'], errors='coerce')

# Rename ftw-all-window to FTW season-to-season
df['Task'] = df['Task'].replace('ftw-all-window', 'FTW season-to-season')

# Define colors for each model
model_colors = {
    'clay': '#1f77b4',  # blue
    'clip': '#ff7f0e',  # orange
    'dinov3_large_sat': '#2ca02c',  # green (DINOv3)
    'dofa': '#d62728',  # red
    'galileo': '#9467bd',  # purple
    'prithvi': '#8c564b',  # brown (Prithvi v2)
    'resnet50_imgnet': '#e377c2',  # pink (ResNet50_imgnet)
    'resnet_random': '#7f7f7f',  # gray (ResNet50_rnd)
    'terrafm': '#bcbd22',  # yellow-green
    'terramind': '#17becf',  # cyan
    'vit_imgnet': '#aec7e8',  # light blue
    'vit_random': '#c5b0d5',  # light purple
    'croma': '#98df8a',  # light green
    'resnet50': '#ff9896'  # light red
}

# Define markers for grouped tasks
task_markers = {
    'RESISC45-UCMerced': 'D',  # diamond
    'UCMerced-RESISC45': 'D',  # diamond
    'RGB - RGE1': 'P',  # plus (filled)
    'Sen1Floods11-S2-S1': 'P',  # plus (filled)
    'BenV2-S2-S1': 'P',  # plus (filled)
    'FTW season-to-season': '^',  # triangle up
    'Germany Temporal': '^',  # triangle up
    'SA Temporal': '^',  # triangle up
    'Geo Germany-Cambodia': 's',  # square
    'Geo Germany-Denmark': 's',  # square
    'DeepGlobe-DFC2022': 'o',  # circle
}

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plot each data point (no error bars)
for idx, row in df.iterrows():
    model = row['Model']
    task = row['Task']
    x = row['Avg Test ID Acc']
    y = row['Avg Test OOD Acc']

    # Skip if either value is NaN
    if pd.isna(x) or pd.isna(y):
        continue

    color = model_colors.get(model, 'gray')
    marker = task_markers.get(task, 'o')

    # Plot point without error bars
    ax.plot(x, y, marker=marker, color=color, markersize=8,
            alpha=0.7, linestyle='None')

# Add y=x line (light gray dotted)
ax.plot([0, 1], [0, 1], 'lightgray', linestyle=':', linewidth=1.5, zorder=0)

# Calculate and plot line of best fit
# Filter out NaN values
df_clean = df.dropna(subset=['Avg Test ID Acc', 'Avg Test OOD Acc'])
x_data = df_clean['Avg Test ID Acc'].values.reshape(-1, 1)
y_data = df_clean['Avg Test OOD Acc'].values
lr = LinearRegression()
lr.fit(x_data, y_data)
x_line = np.linspace(0, 1, 100).reshape(-1, 1)
y_line = lr.predict(x_line)
ax.plot(x_line, y_line, 'k--', linewidth=2, label='Line of best fit', zorder=1)

# Set axis limits and labels with larger font sizes
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xlabel('In-Distribution (ID) Performance', fontsize=14)
ax.set_ylabel('Out-of-Distribution (OOD) Performance', fontsize=14)
ax.set_title('ID vs OOD Performance (Full Finetuning)', fontsize=16, fontweight='bold')

# Increase tick label font size
ax.tick_params(axis='both', which='major', labelsize=12)

# Add grid
ax.grid(True, alpha=0.3)

# Create model legend
model_legend_elements = [Line2D([0], [0], marker='o', color='w',
                                markerfacecolor=color, markersize=8, label=model)
                         for model, color in sorted(model_colors.items())]

# Create task legend with grouped ordering
task_legend_ordered = [
    ('RESISC45-UCMerced', 'D'),
    ('UCMerced-RESISC45', 'D'),
    ('RGB - RGE1', 'P'),
    ('Sen1Floods11-S2-S1', 'P'),
    ('BenV2-S2-S1', 'P'),
    ('FTW season-to-season', '^'),
    ('Germany Temporal', '^'),
    ('SA Temporal', '^'),
    ('Geo Germany-Cambodia', 's'),
    ('Geo Germany-Denmark', 's'),
    ('DeepGlobe-DFC2022', 'o'),
]

task_legend_elements = [Line2D([0], [0], marker=marker, color='gray',
                               markerfacecolor='gray', markersize=8,
                               linestyle='None', label=task)
                        for task, marker in task_legend_ordered]

# Add legends with larger font
legend1 = ax.legend(handles=model_legend_elements, title='Models',
                    loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10, title_fontsize=11)
legend2 = ax.legend(handles=task_legend_elements, title='Tasks',
                    loc='upper left', bbox_to_anchor=(1.02, 0.35), fontsize=10, title_fontsize=11)
ax.add_artist(legend1)

plt.tight_layout()
plt.savefig('{}/id_vs_ood_performance_full_updated.png'.format(root_dir), dpi=300, bbox_inches='tight')
print("Full finetuning ID vs OOD plot saved")