import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from daam.experiment import GenerationExperiment


sns.set_theme("paper", palette="muted")
sns.set_style("ticks")


def plot_box_distribution(data_df):
    fig = sns.boxplot(data=data_df, x="word", y="iou_score", hue="mode", gap=.1)
    fig.set_xlabel("Class")
    fig.set_ylabel("Intersection-over-Union score")
    fig.tick_params(axis="both", direction="in")
    fig.legend(loc="lower left", title="Mode")

    plt.show()


def plot_histogram_distribution(data_df):
    fig = sns.displot(data_df, x="iou_score", hue="mode", col="word", kde=True, fill=True, height=4, aspect=.8)
    fig.set_axis_labels("Intersection-over-Union score", "Count")
    fig.set_titles("{col_name}")
    fig.set_yticklabels([str(num) for num in range(0, 20, 2)])
    fig.tick_params(axis="both", direction="in")

    plt.show()


def plot_images_grid_alternative(output_dir, figsize=(20, 24)):
    """Plot 6x5 grid with different words and modes"""
    rows_config = [
        ("cat", "raw"),
        ("cat", "finetuned"),
        ("dog", "raw"),
        ("dog", "finetuned"),
        ("cow", "raw"),
        ("cow", "finetuned")
    ]
    
    fig, axes = plt.subplots(6, 5, figsize=figsize)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.05, hspace=0.05)
    
    for row_idx, (word, mode) in enumerate(rows_config):
        for col_idx in range(5):
            temp_fig = plt.figure(figsize=(4, 4))
            
            try:
                experiment_name = f"{mode}-photo-of-a-{word}-{col_idx}"
                experiment_path = os.path.join(output_dir, mode, experiment_name)
                experiment = GenerationExperiment.load(experiment_path)
                
                image = experiment.image
                heat_map = experiment.heat_map().compute_word_heat_map(word, word_idx=3)
                heat_map.plot_overlay(image)
                plt.axis("off")
                plt.title("")
                
                temp_fig.canvas.draw()
                buf = temp_fig.canvas.buffer_rgba()
                
                plot_image = np.asarray(buf).copy()
                
                axes[row_idx, col_idx].imshow(plot_image)
                axes[row_idx, col_idx].axis("off")
                
            except Exception as e:
                print(f"Error loading {word} {mode} image {col_idx}: {e}")
                axes[row_idx, col_idx].text(
                    0.5, 0.5, f"Error\n{word} {mode}\nImage {col_idx}", 
                    ha='center', va='center', transform=axes[row_idx, col_idx].transAxes
                )
                axes[row_idx, col_idx].axis("off")
            
            finally:
                plt.close(temp_fig)
    
    plt.tight_layout()
    plt.show()