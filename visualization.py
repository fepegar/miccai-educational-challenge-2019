import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.exposure import rescale_intensity

sns.set(context='notebook')

def plot_parameters(model, name, title=None, axis=None, kde=True, bw=None):
    for name_, params in model.named_parameters():
        if name_ == name:
            tensor = params.data
            break
    else:
        raise ValueError(f'{name} not found in model')
    array = tensor.numpy().ravel()
    if axis is None:
        fig, axis = plt.subplots()
    if bw is None:
        sns.distplot(array, ax=axis, kde=kde)
    else:
        sns.kdeplot(array, ax=axis, bw=bw)
    if title is not None:
        ax.set_title(title)

        
def plot_all_parameters(model, labelsize=6, kde=True, bw=None):
    fig, axes = plt.subplots(3, 7, figsize=(11, 5))
    axes = list(reversed(axes.ravel()))
    for name, params in model.named_parameters():
        if len(params.data.shape) < 2:
            continue
        axis = axes.pop()
        plot_parameters(model, name, axis=axis, kde=kde, bw=bw)
        axis.xaxis.set_tick_params(labelsize=labelsize)
    plt.tight_layout()


def turn(s):
    return np.flipud(np.rot90(s))


def rescale_array(array, cutoff=(2, 98)):
    percentiles = tuple(np.percentile(array, cutoff))
    array = rescale_intensity(array, in_range=percentiles)
    return array


def plot_volume(array, enhance=True, colors_path=None, title=None):
    """
    Expects an isotropic-spacing volume in RAS orientation
    """
    if array.ndim > 5:
        array = array[0]
    if array.ndim == 5:
        array = array[..., 0, 0]  # 5D to 3D
    if enhance:
        array = rescale_array(array)
    si, sj, sk = array.shape[:3]
    slices = [
        array[si//2, ...],
        array[:, sj//2, ...],
        array[:, :, sk//2, ...],
    ]
    if colors_path is not None:
        color_table = ColorTable(colors_path)
        slices = [color_table.colorize(s) for s in slices]
    cmap = 'gray' if array.ndim == 3 else None
    labels = ('AS', 'RS', 'RA')
    
    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[256/160, 1, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    axes = ax1, ax2, ax3
    for (slice_, axis, label) in zip(slices, axes, labels):
        axis.imshow(turn(slice_), cmap=cmap)
        axis.grid(False)
        axis.invert_xaxis()
        axis.invert_yaxis()
        x, y = label
        axis.set_xlabel(x)
        axis.set_ylabel(y)
        axis.set_aspect('equal')
    if title is not None:
        plt.gcf().suptitle(title)
    plt.tight_layout()

    
def plot_histogram(array, kde=True, ylim=None, labels=False):
    sns.distplot(array.ravel(), kde=kde)
    if ylim is not None:
        plt.ylim(ylim)
    if labels:
        plt.xlabel('Intensity')
        plt.ylabel('Count')


class ColorTable:
    def __init__(self, colors_path):
        self.df = self.read_color_table(colors_path)
    
    def read_color_table(self, colors_path):
        df = pd.read_csv(
            colors_path,
            sep=' ',
            header=None,
            names=[
                'Label',
                'Name',
                'R',
                'G',
                'B',
                'A',
            ],
            index_col='Label'
        )
        return df
    
    def get_color(self, label):
        """
        There must be nicer ways of doing this
        """
        try:
            rgb = (
                self.df.loc[label].R,
                self.df.loc[label].G,
                self.df.loc[label].B,
            )
        except KeyError:
            rgb = 0, 0, 0
        return rgb
    
    def colorize(self, label_map):
        rgb = np.stack(3 * [label_map], axis=-1)
        for label in np.unique(label_map):
            mask = label_map == label
            rgb[mask] = self.get_color(label)
        return rgb