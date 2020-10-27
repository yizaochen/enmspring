import matplotlib.pyplot as plt
import numpy as np
from enmspring.na_seq import sequences
from enmspring.spring import Spring
from enmspring.k_b0_util import get_df_by_filter_st

class StackAgent:
    cutoff = 4.7
    type_na = 'bdna+bdna'
    strands = ['STRAND1', 'STRAND2']

    def __init__(self, host, rootfolder, n_bp):
        self.rootfolder = rootfolder
        self.host = host
        self.n_bp = n_bp

        self.d_result = self.__initialize_d_result()
        self.df_st = self.__read_df_st()
        self.d_seq = {'STRAND1': sequences[host]['guide'], 'STRAND2': sequences[host]['target']}

    def update_d_result(self):
        for strand in self.strands:
            mask = (self.df_st['Strand_i'] == strand) & (self.df_st['Strand_j'] == strand)
            df1 = self.df_st[mask]
            for resid in range(1, self.n_bp):
                key = (resid, resid+1)
                mask = (df1['Resid_i'] == resid) & (df1['Resid_j'] == resid+1)
                df2 = df1[mask]
                self.d_result[strand][key] = df2['k']

    def boxplot(self, figsize, lbfz):
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)
        for i, strand in enumerate(self.strands):
            ax = axes[i]
            data = self.__get_data_for_boxplot(strand)
            maxlist = self.__get_data_max(strand)
            positions = self.__get_positions()
            ax.boxplot(data, positions=positions, manage_ticks=False)
            ax.plot(positions, maxlist, 'o', color='red', label="Maximum")
            ax.set_xticks(range(1, self.n_bp+1))
            self.__set_xticklabels(ax, strand)
            ax.tick_params(axis='x', labelsize=lbfz)
            ax.set_xlabel(f'Resid in {strand}  (5\'->3\')', fontsize=lbfz)
            ax.set_ylabel('k (kcal/mol/Å$^2$)', fontsize=lbfz)
            ax.legend(fontsize=lbfz, frameon=False)
            ax.set_ylim(-0.1, 3.6)
            ax.axhline(2.5, color='grey', alpha=0.5)
            ax.axhline(1, color='grey', alpha=0.5)
        return fig, axes

    def barplot(self, figsize, lbfz):
        width = 0.4
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True)
        for i, strand in enumerate(self.strands):
            ax = axes[i]
            mean_list, std_list = self.__get_data_for_barplot(strand)
            positions = self.__get_positions()
            ax.bar(positions, mean_list, width, yerr=std_list, ecolor='black', capsize=2)
            ax.set_xticks(range(1, self.n_bp+1))
            self.__set_xticklabels(ax, strand)
            ax.tick_params(axis='x', labelsize=lbfz)
            ax.set_xlabel(f'Resid in {strand}  (5\'->3\')', fontsize=lbfz)
            ax.set_ylabel('Mean of k (kcal/mol/Å$^2$)', fontsize=lbfz)
            ax.set_ylim(0, 2.6)
            ax.axhline(0.2, color='grey', alpha=0.5)
            ax.axhline(0.7, color='grey', alpha=0.5)
        return fig, axes

    def __set_xticklabels(self, ax, strand):
        labels = [nt for nt in self.d_seq[strand]]
        ax.set_xticklabels(labels)

    def __get_data_for_boxplot(self, strand):
        data = list()
        for resid in range(1, self.n_bp):
            key = (resid, resid+1)
            data.append(self.d_result[strand][key])
        return data

    def __get_data_for_barplot(self, strand):
        mean_list = list()
        std_list = list()
        for resid in range(1, self.n_bp):
            key = (resid, resid+1)
            data = self.d_result[strand][key]
            mean_list.append(np.mean(data))
            std_list.append(np.std(data))
        return mean_list, std_list

    def __get_data_max(self, strand):
        maxlist = list()
        for resid in range(1, self.n_bp):
            key = (resid, resid+1)
            maxlist.append(max(self.d_result[strand][key]))
        return maxlist

    def __get_positions(self):
        positions = list()
        for resid in range(1, self.n_bp):
            positions.append((2*resid+1)/2)
        return positions

    def __initialize_d_result(self):
        d_result = dict()
        for strand in self.strands:
            d_result[strand] = dict()
        return d_result

    def __read_df_st(self):
        criteria = 1e-3
        spring_obj = Spring(self.rootfolder, self.host, self.type_na, self.n_bp)
        df = spring_obj.read_k_b0_pairtype_df_given_cutoff(self.cutoff)
        df1 = get_df_by_filter_st(df, 'st')
        mask = df1['k'] > criteria
        return df1[mask]


class SingleBoxPlot:
    strands = ['STRAND1', 'STRAND2']

    def __init__(self, figsize, rootfolder, lbfz, lgfz, ttfz, title_pos):
        self.figsize = figsize
        self.rootfolder = rootfolder
        self.fig, self.d_axes = self.__make_layout()

        self.lbfz = lbfz
        self.lgfz = lgfz
        self.ttfz = ttfz
        self.title_pos = title_pos

    def __make_layout(self):
        d_axes = dict()
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=self.figsize)
        for i, strand in enumerate(self.strands):
            d_axes[strand] = axes[i]
        return fig, d_axes