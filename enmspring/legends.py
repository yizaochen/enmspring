from os import path
import numpy as np
import matplotlib.pyplot as plt

class ErrorBarLegend:
    lgfz = 5

    def __init__(self, output_folder):
        self.output_folder = output_folder
        self.xarray, self.yarray_mean, self.yarray_std = self.get_pseudo_data()

    def get_pseudo_data(self):
        n_size = 10
        xarray = range(n_size)
        yarray_mean = np.random.rand(n_size)
        yarray_std = np.random.rand(n_size)
        return xarray, yarray_mean, yarray_std

    def get_err(self, color, label):
        err = plt.errorbar(self.xarray, self.yarray_mean, yerr=self.yarray_std, marker='.', color=color, linewidth=0.5, markersize=2, label=label)
        plt.show()
        return err

    def get_legend(self, figsize, color, label, fig_id):
        err = self.get_err(color, label)
        fig_ax = plt.subplots(figsize=figsize)
        ax = fig_ax[1]
        ax.legend(handles=[err], fontsize=self.lgfz, frameon=False)
        plt.axis('off')
        plt.savefig(path.join(self.output_folder, f'{fig_id}.svg'), dpi=300, transparent=True)
        plt.show()