import matplotlib as mpl
import numpy as np
import pickle


mpl.rc('font', **{
    'size' : 12,
})
mpl.rcParams.update({
    'font.family': 'normal',
    'font.serif': [],
    'font.sans-serif': [],
    'font.monospace': [],
})
mpl.rcParams['text.usetex'] = False


import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns


sns.set_context('paper', font_scale=2.5)
sns.set_style('white')
sns.mpl.rcParams['legend.frameon'] = 'False'


OPTS = {
    'citeulike': {
        'name': 'CiteULike',
        'marker': 'o',
    },
    'movies': {
        'name': 'TMD',
        'marker': 's',
    },
    'xing': {
        'name': 'Xing',
        'marker': '*',
    },
}

MARKERSIZE = 5
LINEWIDTH = 2


def plot(data, fpath=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(wspace=0.4)

    ax1.set_xlabel('Number of evaluations')
    ax2.set_xlabel('Number of evaluations')

    ax1.set_ylabel('Best $x^T BQM x$ value')
    ax2.set_ylabel('Computation time (s)')

    for name, item in data.items():
        m = np.array(item['m'])
        t = np.array(item['t'])
        y = np.array(item['y'])

        opt = OPTS[name]

        ax1.plot(m, y, label=opt['name'],
            marker=opt['marker'], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        ax2.plot(m,  t, label=opt['name'],
            marker=opt['marker'], markersize=MARKERSIZE, linewidth=LINEWIDTH)

    prep_ax(ax1, xlog=True, ylog=True, leg=False)
    prep_ax(ax2, xlog=True, ylog=True, leg=True)

    ax1.set_xticks([1.E+1, 1.E+2, 1.E+3, 1.E+4, 1.E+5, 1.E+6, 1.E+7])
    ax2.set_xticks([1.E+1, 1.E+2, 1.E+3, 1.E+4, 1.E+5, 1.E+6, 1.E+7])

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def plot_ext(data, fpath=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    plt.subplots_adjust(wspace=0.4)

    ax1.set_xlabel('Number of evaluations')
    ax2.set_xlabel('Number of evaluations')
    ax3.set_xlabel('Number of evaluations')

    ax1.set_ylabel('Best $x^T BQM x$ value')
    ax2.set_ylabel('Relative error')
    ax3.set_ylabel('Computation time (s)')

    for name, item in data.items():
        m = np.array(item['m'])
        t = np.array(item['t'])
        y = np.array(item['y'])

        y_real = y[-1]
        e = abs((y - y_real) / y_real)

        opt = OPTS[name]

        ax1.plot(m, y, label=opt['name'],
            marker=opt['marker'], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        ax2.plot(m,  e, label=opt['name'],
            marker=opt['marker'], markersize=MARKERSIZE, linewidth=LINEWIDTH)
        ax3.plot(m,  t, label=opt['name'],
            marker=opt['marker'], markersize=MARKERSIZE, linewidth=LINEWIDTH)

    prep_ax(ax1, xlog=True, ylog='symlog', leg=False)
    prep_ax(ax2, xlog=True, ylog=True, leg=False)
    prep_ax(ax3, xlog=True, ylog=True, leg=True)

    ax1.set_xticks([1.E+1, 1.E+2, 1.E+3, 1.E+4, 1.E+5, 1.E+6, 1.E+7])
    ax2.set_xticks([1.E+1, 1.E+2, 1.E+3, 1.E+4, 1.E+5, 1.E+6, 1.E+7])
    ax3.set_xticks([1.E+1, 1.E+2, 1.E+3, 1.E+4, 1.E+5, 1.E+6, 1.E+7])

    if fpath:
        plt.savefig(fpath, bbox_inches='tight')
    else:
        plt.show()


def prep_ax(ax, xlog=False, ylog=False, leg=False, xint=False, xticks=None):
    if xlog:
        ax.semilogx()
    if ylog is True:
        ax.semilogy()
    elif ylog is 'symlog':
        ax.set_yscale("symlog")

    if leg:
        ax.legend(loc='best', frameon=True)

    ax.grid(ls=":")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    if xint:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if xticks is not None:
        ax.set(xticks=xticks, xticklabels=xticks)


if __name__ == '__main__':
    data = np.load('./result/result.npz', allow_pickle=True).get('res').item()
    plot(data, './result/result.png')
    plot_ext(data, './result/result_ext.png')
