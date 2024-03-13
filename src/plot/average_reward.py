import pathlib

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

def average_reward(algorithm, csv_file):
    x = ['Train S1\nTest S1',
         'Train S1\nTest S2',
         'Train S1\nTest S3',
         'Train S2\nTest S1',
         'Train S2\nTest S2',
         'Train S2\nTest S3',
         'Train S3\nTest S1',
         'Train S3\nTest S2',
         'Train S3\nTest S3']

    # Read data from CSV file and extract the standard and tuned columns. The
    # first column can be ignored, it is just the label.
    csv = pd.read_csv(csv_file)
    standard = csv[csv.columns[1]]
    tuned = csv[csv.columns[2]]

    # Set dar width and colors.
    bar_width = 0.35
    standard_bar_color = '#99ccff'
    tuned_bar_color = '#ff9999'

    fig, ax = plt.subplots()

    # Add data. We have two inputs: one for standard hyperparameters and one
    # with tuned hyperparameters. We use an integer interval, one step for each
    # label.
    x_ticks = np.arange(len(x))
    standard_bars = ax.bar(x_ticks - bar_width/2, standard, bar_width,
                           color=standard_bar_color)
    tuned_bars = ax.bar(x_ticks + bar_width/2, tuned, bar_width,
                        color=tuned_bar_color)

    # Set label for y axis.
    ax.set_ylabel('Average Reward', fontsize='large')

    # In the Y axis show ticks with interval of 5000 (0, 5000, 10000...).
    loc = plticker.MultipleLocator(base=5000.0)
    ax.yaxis.set_major_locator(loc)

    # Set legend.
    ax.legend((standard_bars, tuned_bars), (f'{algorithm} Standard', f'{algorithm} Tuned'))

    # Set X axis labels (one for each pair of standard and tuned reward).
    plt.xticks(x_ticks, x, fontsize='medium', fontstretch='condensed')

    # Set background.
    background_color = '#e5e5e5'
    ax.set_axisbelow(True)
    ax.set_facecolor(background_color)
    ax.grid(color='white', linestyle='-', linewidth=.5)

    fig.tight_layout()

    # Save figure as PDF, to be imported in LaTeX.
    plt.savefig(f'{algorithm.lower()}_average_reward.pdf')

if __name__ == '__main__':
    average_reward('PPO', 'ppo.csv')
    average_reward('SAC', 'sac.csv')
    average_reward('NEAT', 'neat.csv')
