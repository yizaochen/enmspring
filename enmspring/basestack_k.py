import numpy as np
import matplotlib.pyplot as plt

class StackResidPlot:
    n_bp = 21
    strand_id_lst = ['STRAND1', 'STRAND2']
    pair_lst = ['ADE:C4---THY:C5', 'ADE:N1---THY:N3', 'ADE:C6---THY:C4']

    tickfz = 6
    lbfz = 8

    def __init__(self, host):
        self.host = host

    def plot_two_strands(self, figsize):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(axes)
        for strand_id in self.strand_id_lst:
            self.plot_lines(d_axes[strand_id])
            self.set_ylabel_xlabel_xticks(d_axes[strand_id])
        return fig, d_axes

    def plot_lines(self, ax):
        xarray = self.get_xarray()
        for pair_name in self.pair_lst:
            yarray = np.random.rand(xarray.shape[0])
            ax.plot(xarray, yarray, '-o', label=pair_name)

    def set_ylabel_xlabel_xticks(self, ax):
        ax.set_ylabel('k (kcal/mol/Å$^2$)', fontsize=self.lbfz)
        ax.set_xlabel('Resid', fontsize=self.lbfz)
        ax.set_xticks(range(1, self.n_bp+1))
        ax.tick_params(axis='both', labelsize=self.tickfz)

    def get_xarray(self):
        interval = 0.5
        return np.arange(1+interval, 21, 1)

    def get_d_axes(self, axes):
        d_axes = dict()
        for idx, strand_id in enumerate(self.strand_id_lst):
            d_axes[strand_id] = axes[idx]
        return d_axes