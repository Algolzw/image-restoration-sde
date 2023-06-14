from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


def clean_ax(ax):
    # 2D or 1D axes are of type np.ndarray
    if isinstance(ax, np.ndarray):
        for one_ax in ax:
            clean_ax(one_ax)
        return

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.tick_params(left=False, right=False, top=False, bottom=False)


def add_text(ax, text, img_shape, place='TOP_LEFT'):
    """
    Adding text on image
    """
    assert place in ['TOP_LEFT', 'BOTTOM_RIGHT']
    if place == 'TOP_LEFT':
        ax.text(img_shape[1] * 20 / 500, img_shape[0] * 35 / 500, text, bbox=dict(facecolor='white', alpha=0.9))
    elif place == 'BOTTOM_RIGHT':
        s0 = img_shape[1]
        s1 = img_shape[0]
        ax.text(s0 - s0 * 150 / 500, s1 - s1 * 35 / 500, text, bbox=dict(facecolor='white', alpha=0.9))


def plot_one_batch_twinnoise(imgs, plot_width=20):
    batch_size = len(imgs)
    ncols = batch_size // 2
    img_sz = plot_width // ncols
    _, ax = plt.subplots(figsize=(ncols * img_sz, 2 * img_sz), ncols=ncols, nrows=2)
    for i in range(ncols):
        ax[0, i].imshow(imgs[i, 0])
        ax[1, i].imshow(imgs[i + batch_size // 2, 0])

        ax[1, i].set_title(f'{i + 1 + batch_size // 2}.')
        ax[0, i].set_title(f'{i + 1}.')

        ax[0, i].tick_params(left=False, right=False, top=False, bottom=False)
        ax[0, i].axis('off')
        ax[1, i].tick_params(left=False, right=False, top=False, bottom=False)
        ax[1, i].axis('off')


def get_k_largest_indices(arr: np.ndarray, K: int):
    """
    Returns the index for K largest elements, in the order small->large.
    """
    ind = np.argpartition(arr, -1 * K)[-1 * K:]
    return ind[np.argsort(arr[ind])]


def add_subplot_axes(ax, rect: List[float], facecolor: str = 'w', min_labelsize: int = 5):
    """
    Add an axes inside an axes. This can be used to create an inset plot.
    Adapted from https://stackoverflow.com/questions/17458580/embedding-small-plots-inside-subplots-in-matplotlib
    Args:
        ax: matplotblib.axes
        rect: Array with 4 elements describing where to position the new axes inside the current axes ax.
            eg: [0.1,0.1,0.4,0.2]
        facecolor: what should be the background color of the new axes
        min_labelsize: what should be the minimum labelsize in the new axes
    """
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    # transAxes: co-ordinate system of the axes: 0,0 is bottomleft and 1,1 is top right.
    # With below command, we want to get to a position which would be the position of new plot in the axes coordinate
    # system
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    # with below command, we now have a position of the new plot in the figure coordinate system. we need this because
    # we can create a new axes in the figure coordinate system. so we want to get everything in that system.
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]  # <= Typo was here
    # subax = fig.add_axes([x,y,width,height],facecolor=facecolor)  # matplotlib 2.0+
    subax = fig.add_axes([x, y, width, height], facecolor=facecolor)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.5
    y_labelsize *= rect[3]**0.5
    subax.xaxis.set_tick_params(labelsize=max(min_labelsize, x_labelsize))
    subax.yaxis.set_tick_params(labelsize=max(min_labelsize, y_labelsize))
    return subax


def clean_for_xaxis_plot(inset_ax):
    """
    For an xaxis plot, the y axis values don't matter. Neither the axes borders save the bottom one.
    """
    # Removing y-axes ticks and text
    inset_ax.set_yticklabels([])
    inset_ax.tick_params(left=False, right=False)
    inset_ax.set_ylabel('')

    # removing the axes border lines.
    inset_ax.spines['top'].set_visible(False)
    inset_ax.spines['right'].set_visible(False)
    inset_ax.spines['left'].set_visible(False)


def add_pixel_kde(ax,
                  rect: List[float],
                  data1: np.ndarray,
                  data2: Union[np.ndarray, None],
                  min_labelsize: int,
                  color1='r',
                  color2='black',
                  color_xtick='white',
                  label1='Target',
                  label2='Predicted'):
    """
    Adds KDE (density plot) of data1(eg: target) and data2(ex: predicted) image pixel values as an inset
    """
    inset_ax = add_subplot_axes(ax, rect, facecolor="None", min_labelsize=min_labelsize)

    inset_ax.tick_params(axis='x', colors=color_xtick)

    sns.kdeplot(data=data1.reshape(-1, ), ax=inset_ax, color=color1, label=label1, clip=(0, None))
    if data2 is not None:
        sns.kdeplot(data=data2.reshape(-1, ), ax=inset_ax, color=color2, label=label2, clip=(0, None))

    xmin, xmax = inset_ax.get_xlim()
    xmax_data = data1.max()
    if data2 is not None:
        xmax_data = max(xmax_data, data2.max())

    inset_ax.set_xlim(0, xmax_data)

    xticks = inset_ax.get_xticks()
    inset_ax.set_xticks([xticks[0], xticks[-1]])
    clean_for_xaxis_plot(inset_ax)
    return inset_ax


# Adding arrows.
def add_left_arrow(ax, xy_location, arrow_length=20, color='red', arrowstyle='->'):
    xy_start = (xy_location[0] + arrow_length, xy_location[1])
    return add_arrow(ax, xy_start, xy_location, color='red', arrowstyle=arrowstyle)


def add_right_arrow(ax, xy_location, arrow_length=20, color='red', arrowstyle='->'):
    xy_start = (xy_location[0] - arrow_length, xy_location[1])
    return add_arrow(ax, xy_start, xy_location, color='red', arrowstyle=arrowstyle)


def add_top_arrow(ax, xy_location, arrow_length=20, color='red', arrowstyle='->'):
    xy_start = (xy_location[0], xy_location[1] + arrow_length)
    return add_arrow(ax, xy_start, xy_location, color='red', arrowstyle=arrowstyle)


def add_bottom_arrow(ax, xy_location, arrow_length=20, color='red', arrowstyle='->'):
    xy_start = (xy_location[0], xy_location[1] - arrow_length)
    return add_arrow(ax, xy_start, xy_location, color='red', arrowstyle=arrowstyle)


def get_start_vector(xy_start, xy_end, arrow_length):
    """
    Given an arrow_length, return a xy_start such that xy_start => xy_end vector has  this length.
    """
    direction = (xy_end[0] - xy_start[0], xy_end[1] - xy_start[1])
    norm = np.linalg.norm(direction)
    direction = (direction[0] / norm, direction[1] / norm)
    direction = (direction[0] * arrow_length, direction[1] * arrow_length)
    xy_start = (xy_end[0] - direction[0], xy_end[1] - direction[1])
    return xy_start


def add_arrow(ax, xy_start, xy_end, arrow_length=None, color='red', arrowstyle="->"):
    if arrow_length is not None:
        xy_start = get_start_vector(xy_start, xy_end, arrow_length)
    ax.annotate("", xy=xy_end, xytext=xy_start, arrowprops=dict(arrowstyle=arrowstyle, color=color, linewidth=1))
