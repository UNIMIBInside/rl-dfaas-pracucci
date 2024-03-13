import pathlib

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

def ppo_fitting(csv_file):
    x = ['Test S1', 'Test S2', 'Test S3']

    # Read data from CSV file. The CSV file has three rows: one is the header
    # and two the value for PPO trained with scenario S3 and one for PPO trained
    # with scenario S2. Each row has four values, the first is the label.
    csv = pd.read_csv(csv_file)

    # The first value is the label, we must ignored it.
    fit = csv.iloc[0].iloc[1:]
    nofit = csv.iloc[1].iloc[1:]

    # This figure has two subfigures:
    #   1. Average reward of PPO trained with scenario S3,
    #   2. Average reward of PPO trained width scenario S2.
    # Note sharey=True is required to have zero aligned in both figures.
    fig, (ax_fit, ax_nofit) = plt.subplots(ncols=2, sharey=True)

    # First, set unique properties of each subfigure, then use a loop for common
    # settings.

    # Set subfigures titles.
    ax_fit.set_title('PPO trained with scenario 3')
    ax_nofit.set_title('PPO trained with scenario 2')

    # Add data to both figures.
    bar_width = 0.5
    ax_fit.bar(x, fit, bar_width, color='red')
    ax_nofit.bar(x, nofit, bar_width, color='blue')

    # In the Y axis show ticks with interval of 5000 (0, 5000, 10000...).
    loc = plticker.MultipleLocator(base=5000.0)
    ax_fit.yaxis.set_major_locator(loc)
    ax_nofit.yaxis.set_major_locator(loc)

    for ax in (ax_fit, ax_nofit):
        # Set Y axis label.
        ax.set_ylabel('Average Reward', fontsize='large')

        # Set Y axis granularity (ticks with interval of 5000).
        loc = plticker.MultipleLocator(base=5000.0)
        ax.yaxis.set_major_locator(loc)

        # Set background.
        background_color = '#e5e5e5'
        ax.set_axisbelow(True)
        ax.set_facecolor(background_color)
        ax.grid(color='white', linestyle='-', linewidth=.5)

        # Show Y label only on the left subplot.
        ax.label_outer()

    fig.tight_layout()

    # Save figure as PDF, to be imported in LaTeX.
    plt.savefig('ppo_fitting.pdf')

if __name__ == '__main__':
    ppo_fitting('ppo_fitting.csv')
