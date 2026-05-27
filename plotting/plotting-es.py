import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LinearRegression

parser = argparse.ArgumentParser()
parser.add_argument('--root_dir', type=str, required=True,
                    help='Root directory containing input files and where outputs will be saved')
args = parser.parse_args()
root_dir = args.root_dir

# Read the Excel file
df = pd.read_excel('{}/latex-table-results.xlsx'.format(root_dir))

# Parse the data structure
# Extract full fine-tuning data (rows 3-15, 0-indexed)
full_ft_data = df.iloc[3:16].copy()
full_ft_data.columns = ['Model', 'RESISC45_ID', 'RESISC45_OOD', 'RESISC45_Delta',
                        'DeepGlobe_ID', 'DeepGlobe_OOD', 'DeepGlobe_Delta',
                        'Germany_ID', 'Germany_OOD', 'Germany_Delta',
                        'SouthAfrica_ID', 'SouthAfrica_OOD', 'SouthAfrica_Delta',
                        'Sensor_ID', 'Sensor_OOD', 'Sensor_Delta']
full_ft_data['Finetune'] = 'Full'

# Extract head fine-tuning data (rows 21-33, 0-indexed)
head_ft_data = df.iloc[21:34].copy()
head_ft_data.columns = ['Model', 'RESISC45_ID', 'RESISC45_OOD', 'RESISC45_Delta',
                        'DeepGlobe_ID', 'DeepGlobe_OOD', 'DeepGlobe_Delta',
                        'Germany_ID', 'Germany_OOD', 'Germany_Delta',
                        'SouthAfrica_ID', 'SouthAfrica_OOD', 'SouthAfrica_Delta',
                        'Sensor_ID', 'Sensor_OOD', 'Sensor_Delta']
head_ft_data['Finetune'] = 'Head'

# Clean model names (remove * and whitespace)
full_ft_data['Model'] = full_ft_data['Model'].str.strip().str.replace('*', '', regex=False)
head_ft_data['Model'] = head_ft_data['Model'].str.strip().str.replace('*', '', regex=False)

# Convert numeric columns
numeric_cols = [col for col in full_ft_data.columns if col not in ['Model', 'Finetune']]
for col in numeric_cols:
    full_ft_data[col] = pd.to_numeric(full_ft_data[col], errors='coerce')
    head_ft_data[col] = pd.to_numeric(head_ft_data[col], errors='coerce')

# Define tasks and their columns
all_tasks = {
    'RESISC45-UCMerced': ('RESISC45_ID', 'RESISC45_OOD'),
    'DeepGlobe-DFC2022': ('DeepGlobe_ID', 'DeepGlobe_OOD'),
    'Germany-Cambodia': ('Germany_ID', 'Germany_OOD'),
    'South Africa Seasonal': ('SouthAfrica_ID', 'SouthAfrica_OOD'),
    'RGB - RGE1': ('Sensor_ID', 'Sensor_OOD')
}

# Get unique models from both datasets
all_models = sorted(set(list(full_ft_data['Model'].unique()) + list(head_ft_data['Model'].unique())))
colors = plt.cm.tab10(np.linspace(0, 1, len(all_models)))
model_colors = dict(zip(all_models, colors))

# Define shapes for tasks
shapes = ['o', 's', '^', 'D', 'v']
task_shapes = dict(zip(all_tasks.keys(), shapes))


def create_plot(data, finetune_type, output_path):
    """
    Create scatter plot of ID vs OOD performance with line of best fit

    Parameters:
    -----------
    data : DataFrame
        Data containing model performance across tasks
    finetune_type : str
        'Full' or 'Head' fine-tuning
    output_path : str
        Path to save the output plot
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Collect points for line of best fit (excluding South Africa)
    fit_x = []
    fit_y = []

    # Plot data points
    for _, row in data.iterrows():
        model = row['Model']

        for task_name, (id_col, ood_col) in all_tasks.items():
            id_val = row[id_col]
            ood_val = row[ood_col]

            if pd.notna(id_val) and pd.notna(ood_val):
                marker = task_shapes[task_name]
                color = model_colors[model]

                ax.scatter(id_val, ood_val, marker=marker, c=[color],
                           s=200, edgecolors='black', linewidths=2, alpha=0.8, zorder=3)

                # Add to fit data if not South Africa
                if task_name != 'South Africa Seasonal':
                    fit_x.append(id_val)
                    fit_y.append(ood_val)

    # Calculate line of best fit
    if len(fit_x) > 0:
        fit_x = np.array(fit_x).reshape(-1, 1)
        fit_y = np.array(fit_y)

        # Fit linear regression
        reg = LinearRegression()
        reg.fit(fit_x, fit_y)

        # Calculate R-squared
        r_squared = reg.score(fit_x, fit_y)

        # Create line extending from y-intercept to x=1
        x_line = np.array([0, 1]).reshape(-1, 1)
        y_line = reg.predict(x_line)

        # Plot line of best fit
        ax.plot(x_line, y_line, color='gray', linestyle='-', linewidth=3,
                alpha=0.6, zorder=2,
                label=f'Line of best fit (excl. temporal)\n$y = {reg.coef_[0]:.3f}x + {reg.intercept_:.3f}$\n$R^2 = {r_squared:.3f}$')

        print(f"\n{finetune_type} Fine-tuning:")
        print(f"  Slope: {reg.coef_[0]:.4f}")
        print(f"  Y-intercept: {reg.intercept_:.4f}")
        print(f"  R²: {r_squared:.4f}")
        print(f"  Number of points used: {len(fit_x)}")

    # Plot y=x line (no shift)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2.5, label='No shift (y=x)', zorder=1)

    # Set labels and title
    ax.set_xlabel('In-Distribution', fontsize=16, fontweight='bold')
    ax.set_ylabel('Out-of-Distribution', fontsize=16, fontweight='bold')
    ax.set_title(f'ID vs OOD Performance - {finetune_type} Fine-tuning',
                 fontsize=18, fontweight='bold', pad=20)

    # Set axis limits - extend y-axis to -0.5
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linewidth=1, alpha=0.3, zorder=1)

    # Create legends
    # Legend 1: Models (colors)
    model_legend_elements = []
    for model in sorted(data['Model'].unique()):
        model_legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                            markerfacecolor=model_colors[model],
                                            markersize=12, label=model,
                                            markeredgecolor='black', markeredgewidth=1.5))

    # Legend 2: Tasks (shapes)
    task_legend_elements = []
    for task, shape in task_shapes.items():
        task_legend_elements.append(Line2D([0], [0], marker=shape, color='w',
                                           markerfacecolor='gray',
                                           markersize=12, label=task,
                                           markeredgecolor='black', markeredgewidth=1.5))

    # Add y=x line and line of best fit to task legend
    task_legend_elements.append(Line2D([0], [0], color='black', linestyle='--',
                                       linewidth=2.5, label='No shift (y=x)'))
    task_legend_elements.append(Line2D([0], [0], color='gray', linestyle='-',
                                       linewidth=3, alpha=0.6, label='Line of best fit'))

    # Add legends
    legend1 = ax.legend(handles=model_legend_elements, loc='upper left',
                        title='Models', fontsize=10, title_fontsize=12,
                        framealpha=0.95, ncol=1, borderpad=1)
    ax.add_artist(legend1)

    legend2 = ax.legend(handles=task_legend_elements, loc='lower right',
                        title='Tasks & Fit', fontsize=9, title_fontsize=11,
                        framealpha=0.95, ncol=1, borderpad=1)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_path}")

    plt.close()


# Create both plots
print("=" * 80)
create_plot(full_ft_data, 'Full', '{}/id_vs_ood_full_finetuning.png'.format(root_dir))
print("=" * 80)
create_plot(head_ft_data, 'Head', '{}/id_vs_ood_head_finetuning.png'.format(root_dir))
print("=" * 80)

print(f"\n✓ Created separate plots with line of best fit (excluding South Africa Seasonal)")
print(f"  - Y-axis extended to -0.5 to show y-intercept")
print(f"  - Line of best fit shown in gray")