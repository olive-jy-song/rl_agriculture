import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import sys

def save_analysis_plots(config_file, path="./config_analysis"):
    """
    Generates and saves bar plots based on a JSON configuration file.
    
    Parameters:
    - config_file (str): Path to the JSON file containing the plot configuration.
    - path (str): Directory where the plots will be saved. Defaults to "./config_analysis".

    The JSON configuration file should have the following structure:
    {
        "titles": ["Title 1", "Title 2", ...],
        "y_labels": ["Y-Label 1", "Y-Label 2", ...],
        "x_labels": ["X-Label 1", "X-Label 2", ...],
        "colors": ["Color 1", "Color 2", ...],
        "x_ticks": ["Tick 1", "Tick 2", ...],
        "file_names": ["Filename 1", "Filename 2", ...],
        "mean_profit": [value1, value2, ...],
        "mean_yield": [value1, value2, ...],
        "mean_water": [value1, value2, ...]
    }

    Each plot configuration should have the same number of elements in lists 
    for consistency across titles, labels, colors, ticks, and data.
    """
    with open(config_file, 'r') as file:
        config = json.load(file)
    
    os.makedirs(path, exist_ok=True)
    sns.set(style="whitegrid")

    titles = config["titles"]
    y_labels = config["y_labels"]
    x_labels = config["x_labels"]
    colors = config["colors"]
    x_ticks = config["x_ticks"]
    file_names = config["file_names"]
    plots = [config["mean_profit"], config["mean_yield"], config["mean_water"]]

    for i, plot_data in enumerate(plots):
        plt.figure()
        bar_plot = sns.barplot(x=x_ticks, y=plot_data, color=colors[i], alpha=0.8)
        for p in bar_plot.patches:
            bar_plot.annotate(format(p.get_height(), '.2f'), 
                                (p.get_x() + p.get_width() / 2., p.get_height()), 
                                ha = 'center', va = 'center', 
                                size=10, xytext = (0, 10), 
                                textcoords = 'offset points')
        plt.title(titles[i], fontsize=16)
        plt.ylabel(y_labels[i], fontsize=14)
        plt.xlabel(x_labels[i], fontsize=14)
        plt.xticks(ticks=np.arange(len(x_ticks)), labels=x_ticks)
        plt.savefig(f'{path}/{file_names[i]}')
        plt.close()

if __name__ == "__main__":
    config_file = sys.argv[1]
    save_analysis_plots(config_file)

