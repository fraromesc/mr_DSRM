#!/usr/bin/env python
# encoding: utf-8
"""
A number of miscellaneous tools
@Authors: Arturo Gil
@Time: April 2021

"""
import matplotlib.pyplot as plt
import numpy as np


def plot(x, title='Untitled', block=False):
    plt.figure()
    x = np.array(x)
    plt.plot(x)
    plt.title(title)
    plt.show(block=block)


def plot_path(q_path):
    plt.figure()
    q_path = np.array(q_path)
    sh = q_path.shape
    for i in range(0, sh[1]):
        plt.plot(q_path[:, i], label='q' + str(i + 1), linewidth=5)
    plt.legend()
    plt.show(block=True)
