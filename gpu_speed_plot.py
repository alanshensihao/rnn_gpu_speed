import pylab
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import interp1d


def set_plot_environ():
    plt.rcParams['font.sans-serif'] = ['Calibri', 'Tahoma', 'DejaVu Sans', 'Lucida Grande', 'Verdana']
    plt.rcParams['axes.facecolor'] = '#000000'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 26
    plt.rcParams['axes.titlesize'] = 28
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['axes.titlepad'] = 12
    plt.rcParams['axes.edgecolor'] = '#cccccc'

    grid_color = 1.
    plt.grid(color=(grid_color, grid_color, grid_color, 0.1), linestyle='--', linewidth=0.2)


def line_plot(data_sets, n_epochs, save_name=None, legend_loc=1, ylabel=None):
    set_plot_environ()
    colors = ('#7DF9FFff', '#FF00FFff', '#33bb22ff')

    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    smooth_points = 500
    for d, data in enumerate(data_sets):
        mat = data['data']
        x = np.arange(0, mat.shape[1])
        x_new = np.linspace(x.min(), x.max(), smooth_points)

        for y_ct, y in enumerate(mat):
            f = interp1d(x, y, kind='quadratic')
            y_smooth = f(x_new)
            if y_ct == 0:
                lbl = data['label']
            else:
                lbl = None
            sns.lineplot(data=y_smooth, ci=None, color=colors[d], label=lbl, linewidth=data['linewidth'])

    pylab.xlabel(f'{n_epochs} Epochs')

    if ylabel is not None:
        pylab.ylabel(ylabel)

    pylab.title(f'RNN speed: CUDA, CUDA+Cudnn, and no GPU')

    legnd = plt.legend(fontsize=24, loc=legend_loc)
    for text in legnd.get_texts():
        plt.setp(text, color='w')

    pylab.xlim([0, smooth_points])

    vals = ax.get_xticks()
    ax.set_xticklabels(['' for x in vals])

    figure = plt.gcf()
    height = 10
    aspect = 1920. / 1080.
    figure.set_size_inches(aspect * height, height)

    if save_name is not None:
        pylab.savefig(save_name, format='png', bbox_inches='tight', pad_inches=0.5, dpi=300)

    plt.close()
