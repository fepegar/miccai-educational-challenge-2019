import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.exposure import rescale_intensity

from ipywidgets import fixed
import ipywidgets as widgets
from IPython.display import display

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
        axis.set_title(title)


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


def to_rgb(array):
    if array.shape[-1] == 3:  # assume already RGB
        return array
    array = array.astype(float)
    array -= array.min()
    array /= array.max()
    array *= 255
    array = array.astype(np.uint8)
    rgb = np.stack(3 * [array], axis=-1)
    return rgb


def turn(s):
    return np.flipud(np.rot90(s))


def rescale_array(array, cutoff=(2, 98)):
    percentiles = tuple(np.percentile(array, cutoff))
    array = rescale_intensity(array, in_range=percentiles)
    return array


def add_intersections(slices, i, j, k):
    """
    Colors from 3D Slicer
    """
    sag, cor, axi = slices
    red = 255, 131, 114
    green = 143, 229, 97
    yellow = 255, 237, 135
    sag[j, :] = green
    sag[:, k] = red
    cor[i, :] = yellow
    cor[:, k] = red
    axi[i, :] = yellow
    axi[:, j] = green
    return sag, cor, axi


def plot_volume(
        array,
        enhance=True,
        colors_path=None,
        title=None,
        idx_sag=None,
        idx_cor=None,
        idx_axi=None,
        return_figure=False,
        intersections=True,
):
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
    i = idx_sag if idx_sag is not None else si // 2
    j = idx_cor if idx_cor is not None else sj // 2
    k = idx_axi if idx_axi is not None else sk // 2
    slices = [
        array[i, ...],
        array[:, j, ...],
        array[:, :, k, ...],
    ]

    if colors_path is not None:
        color_table = ColorTable(colors_path)
        slices = [color_table.colorize(s) for s in slices]
    if intersections:
        slices = [to_rgb(s) for s in slices]
        slices = add_intersections(slices, i, j, k)
    cmap = 'gray' if array.ndim == 3 else None
    labels = 'AS', 'RS', 'RA'
    titles = 'Sagittal', 'Coronal', 'Axial'

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[256 / 160, 1, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    axes = ax1, ax2, ax3
    for (slice_, axis, label, stitle) in zip(slices, axes, labels, titles):
        axis.imshow(turn(slice_), cmap=cmap)
        axis.grid(False)
        axis.invert_xaxis()
        axis.invert_yaxis()
        x, y = label
        axis.set_xlabel(x)
        axis.set_ylabel(y)
        axis.set_title(stitle)
        axis.set_aspect('equal')
    if title is not None:
        plt.gcf().suptitle(title)
    plt.tight_layout()
    if return_figure:
        return fig


def plot_volume_interactive(array, **kwargs):
    def get_widget(size, description):
        w = widgets.IntSlider(
            min=0,
            max=size-1,
            step=1,
            value=size//2,
            continuous_update=False,
            description=description,
        )
        return w
    shape = array.shape[:3]
    names = 'Sagittal L-R', 'Coronal P-A', 'Axial I-S'
    widget_sag, widget_cor, widget_axi = [
        get_widget(s, n) for (s, n) in zip(shape, names)]
    ui = widgets.HBox([widget_sag, widget_cor, widget_axi])
    args_dict = {
        'array': fixed(array),
        'idx_sag': widget_sag,
        'idx_cor': widget_cor,
        'idx_axi': widget_axi,
        'return_figure': fixed(True),
    }
    kwargs = {key: fixed(value) for (key, value) in kwargs.items()}
    args_dict.update(kwargs)
    out = widgets.interactive_output(plot_volume, args_dict)
    display(ui, out)

    
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
