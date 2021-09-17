from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from enmspring.graphs_bigtraj import BackboneMeanModeAgent
from enmspring import pairtype
from enmspring.k_b0_util import get_df_by_filter_PP, get_df_by_filter_R, get_df_by_filter_RB
from enmspring.basestack_k import StackResidPlot
from enmspring.kappa_mat_backbone import KappaBackbone, KappaBackboneWithNext

class BackboneRiboseK:
    interval_time = 500
    column_lst = ['PairType', 'Big_Category', 'Strand_i', 'Resid_i',  
                  'Atomname_i', 'Strand_j', 'Resid_j', 'Atomname_j', 'k-mean', 'k-std', 'b0', 'b0-std']
    n_bp = 21
    strand_lst = ['STRAND1', 'STRAND2']
    resid_lst = list(range(4, 19))

    def __init__(self, host, big_traj_folder, df_folder):
        self.host = host
        self.big_traj_folder = big_traj_folder
        self.df_folder = df_folder

        self.mean_mode_agent = None
        self.n_node = None
        self.d_idx_inverse = None

        self.f_df = path.join(self.df_folder, f'{self.host}.csv')
        self.df = None

    def plot_kmean_by_resid(self, figsize, category, ylim=None):
        df_filter = self.get_df_category(category)
        fig, axes = plt.subplots(nrows=2, figsize=figsize)
        for idx, strand_id in enumerate(self.strand_lst):
            ax = axes[idx]
            yarray, y_std_array = self.get_yarray_ystd_array(df_filter, strand_id)
            ax.errorbar(self.resid_lst, yarray, yerr=y_std_array)
            ax.set_xticks(self.resid_lst)
            ax.set_xlabel('Resid')
            ax.set_ylabel(f'{category} k (kcal/mol/Å$^2$)')
            ax.set_title(f'{strand_id}')
            if ylim is not None:
                ax.set_ylim(ylim)
        return fig, axes

    def get_df_category(self, category):
        if category in ['PP0', 'PP1', 'PP2']:
            return get_df_by_filter_PP(self.df, category)
        elif category in ['R0', 'R1']:
            return get_df_by_filter_R(self.df, category)
        elif category in ['RB0', 'RB1', 'RB2']:
            return get_df_by_filter_RB(self.df, category)
        else:
            print('Only PP0 PP1 PP2 R0 R1 RB0 RB1 RB2 are allowed here')
            return None

    def get_yarray_ystd_array(self, df, strand_id):
        yarray = np.zeros(len(self.resid_lst))
        y_std_array = np.zeros(len(self.resid_lst))
        for idx, resid in enumerate(self.resid_lst):
            mask = (df['Resid_i'] == resid) & (df['Resid_j'] == resid) & (df['Strand_i'] == strand_id)
            df_sele = df[mask]
            if df_sele.shape[0] == 0:
                continue
            yarray[idx] = df_sele['k-mean'].mean()
            y_std_array[idx] = df_sele['k-mean'].std()
        return yarray, y_std_array

    def initialize_mean_mode_agent(self):
        self.mean_mode_agent = BackboneMeanModeAgent(self.host, self.big_traj_folder, self.interval_time)
        self.mean_mode_agent.load_mean_mode_laplacian_from_npy()
        self.mean_mode_agent.load_mean_mode_std_laplacian_from_npy()
        self.mean_mode_agent.load_b0_mean_std_from_npy()
        self.mean_mode_agent.process_first_small_agent()
        self.mean_mode_agent.initialize_all_maps()
        self.n_node = len(self.mean_mode_agent.node_list)
        self.d_idx_inverse = {y:x for x,y in self.mean_mode_agent.d_idx.items()}

    def get_df_filter(self):
        d_result = {column_key: list() for column_key in ['i', 'j', 'k-mean', 'k-std', 'b0-mean', 'b0-std']}
        for i in range(self.n_node):
            for j in range(self.n_node):
                if i == j:
                    continue
                d_result['i'].append(i)
                d_result['j'].append(j)
                d_result['k-mean'].append(self.mean_mode_agent.laplacian_mat[i,j])
                d_result['k-std'].append(self.mean_mode_agent.laplacian_std_mat[i,j])
                d_result['b0-mean'].append(self.mean_mode_agent.b0_mean_mat[i,j])
                d_result['b0-std'].append(self.mean_mode_agent.b0_std_mat[i,j])
        df_result = pd.DataFrame(d_result)
        mask = df_result['k-mean'] > 1e-1
        return df_result[mask]

    def make_df(self):
        df_filter = self.get_df_filter()
        d_result = {column_key: list() for column_key in self.column_lst}
        for i, j in zip(df_filter['i'], df_filter['j']):
            cgname_i = self.d_idx_inverse[i]
            cgname_j = self.d_idx_inverse[j]
            strandid1 = self.mean_mode_agent.strandid_map[cgname_i]
            strandid2 = self.mean_mode_agent.strandid_map[cgname_j]
            resid1 = self.mean_mode_agent.resid_map[cgname_i]
            resid2 = self.mean_mode_agent.resid_map[cgname_j]
            atomname1 = self.mean_mode_agent.atomname_map[cgname_i]
            atomname2 = self.mean_mode_agent.atomname_map[cgname_j]
            temp = pairtype.Pair(strandid1, resid1, atomname1, strandid2, resid2, atomname2, n_bp=self.n_bp)
            d_result['PairType'].append(temp.pair_type)
            d_result['Big_Category'].append(temp.big_category)
            d_result['Strand_i'].append(strandid1)
            d_result['Resid_i'].append(resid1)
            d_result['Atomname_i'].append(atomname1)
            d_result['Strand_j'].append(strandid2)
            d_result['Resid_j'].append(resid2)
            d_result['Atomname_j'].append(atomname2)
        d_result['k-mean'] = df_filter['k-mean'].tolist()
        d_result['k-std'] = df_filter['k-std'].tolist()
        d_result['b0'] = df_filter['b0-mean'].tolist()
        d_result['b0-std'] = df_filter['b0-std'].tolist()
        df_result = pd.DataFrame(d_result)
        df_result.to_csv(self.f_df, index=False)
        print(f'Write DataFrame to {self.f_df}')

    def read_df(self):
        self.df = pd.read_csv(self.f_df)
        print(f'Read DataFrame from {self.f_df}')

    def get_pair_container(self, category):
        df_sele = self.get_df_category(category)
        return PairContainer(category, df_sele)

class BackboneResidPlot(StackResidPlot):
    resid_lst = list(range(4, 19))
    tickfz = 8
    lbfz = 12

    def plot_two_strands(self, figsize, start_mode, end_mode, ylims, d_pair, yticks=None, assist_hlines=None):
        big_k_mat = self.kmat_agent.get_K_mat(start_mode, end_mode)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(axes)
        for strand_id in self.strand_id_lst:
            self.plot_lines(d_axes[strand_id], strand_id, big_k_mat, d_pair)
            self.set_xticks(d_axes)
            self.set_ylabel_xlabel(d_axes[strand_id])
            self.set_ylims(d_axes[strand_id], ylims)
            if yticks is not None:
                d_axes[strand_id].set_yticks(yticks)
            if assist_hlines is not None:
                for hline in assist_hlines:
                    d_axes[strand_id].axhline(hline, color='grey', alpha=0.2)
        return fig, d_axes

    def plot_lines(self, ax, strand_id, big_k_mat, d_pair):
        xarray = self.get_xarray()
        yarray = self.get_yarray(strand_id, big_k_mat, d_pair)
        #ax.plot(xarray, yarray, marker='.', linewidth=0.5, markersize=2)
        ax.plot(xarray, yarray, marker='o')
    
    def get_yarray(self, strand_id, big_k_mat, d_pair):
        k_array = np.zeros(len(self.resid_lst))
        for idx, resid_i in enumerate(self.resid_lst):
            basetype_i = self.d_kappa[strand_id][resid_i].get_basetype_i()
            basetype_j = self.d_kappa[strand_id][resid_i].get_basetype_j()
            atomname_i = d_pair[basetype_i]['atomname_i']
            atomname_j = d_pair[basetype_j]['atomname_j']
            k_array[idx] = self.d_kappa[strand_id][resid_i].get_k_by_atomnames(big_k_mat, atomname_i, atomname_j)
        return k_array

    def get_xarray(self):
        return self.resid_lst

    def get_d_kappa(self):
        d_kappa = dict()
        for strand_id in self.strand_id_lst:
            d_kappa[strand_id] = dict()
            seq = self.d_seq[strand_id]
            for resid_i in self.resid_lst:
                d_kappa[strand_id][resid_i] = KappaBackbone(self.host, strand_id, resid_i, self.s_agent, self.map_idx_from_strand_resid_atomname, seq)
        return d_kappa

    def set_ylabel_xlabel(self, ax):
        ax.set_ylabel('k (kcal/mol/Å$^2$)', fontsize=self.lbfz)
        #ax.set_xlabel('Resid', fontsize=self.lbfz)
        ax.tick_params(axis='both', labelsize=self.tickfz)

class BackboneResidPlotWithNext(BackboneResidPlot):
    resid_lst = list(range(4, 18))
    
    def get_d_kappa(self):
        d_kappa = dict()
        for strand_id in self.strand_id_lst:
            d_kappa[strand_id] = dict()
            seq = self.d_seq[strand_id]
            for resid_i in self.resid_lst:
                d_kappa[strand_id][resid_i] = KappaBackboneWithNext(self.host, strand_id, resid_i, self.s_agent, self.map_idx_from_strand_resid_atomname, seq)
        return d_kappa

    def get_xarray(self):
        interval = 0.5
        return np.arange(4+interval, 18, 1)

class BarPlot:
    strand_id_lst = ['STRAND1', 'STRAND2']
    d_diff = {'PP0': [0, 1], 'PP1': [0, 1], 'PP2': [0, 1], 'R0': [0], 'R1': [0], 'RB0': [0], 'RB1': [0], 'RB2': [0]}
    color_lst = ['royalblue', 'orange', 'seagreen']
    d_abbr = {'a_tract_21mer': 'A-tract', 'atat_21mer': 'TATA', 'g_tract_21mer': 'G-tract', 'gcgc_21mer': 'CpG'}

    def __init__(self, host, category, df_sele, width=0.8):
        self.host = host
        self.category = category
        self.df_sele = df_sele
        self.width = width
        self.abbr = self.d_abbr[self.host]

    def bar_two_strands(self, figsize, ylims, assit_hlines):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(axes)
        for strand_id in self.strand_id_lst:
            self.bar_plot(d_axes[strand_id], strand_id)
            if ylims is not None:
                d_axes[strand_id].set_ylim(ylims)
            if assit_hlines is not None:
                for hline in assit_hlines:
                    d_axes[strand_id].axhline(hline, color='grey', alpha=0.1)
        return fig, d_axes

    def bar_two_strands_by_d_pair_type(self, figsize, ylims, d_pair_type, assit_hlines):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(axes)
        for strand_id in self.strand_id_lst:
            self.bar_plot_by_d_pair_type(d_axes[strand_id], strand_id, d_pair_type)
            if ylims is not None:
                d_axes[strand_id].set_ylim(ylims)
            if assit_hlines is not None:
                for hline in assit_hlines:
                    d_axes[strand_id].axhline(hline, color='grey', alpha=0.1)
        return fig, d_axes

    def bar_plot(self, ax, strand_id):
        pair_container = self.get_pair_container_by_strandid(strand_id)
        xticks, xticklabels = self.ini_xtick_xticklabel()
        i = 0
        for idx, diff in enumerate(self.d_diff[self.category]):
            sub_xticks, sub_xticklabels, y_mean_list, y_std_list = self.get_y_mean_std_list_xticklabels(i, diff, pair_container)
            label = f'Resid_j - Resid_i = {diff}'
            ax.bar(sub_xticks, y_mean_list, self.width, yerr=y_std_list, color=self.color_lst[idx], label=label)
            xticklabels += sub_xticklabels
            xticks += sub_xticks
            i += len(sub_xticks)
        self.set_tick_label(ax, xticks, xticklabels, strand_id)

    def bar_plot_by_d_pair_type(self, ax, strand_id, d_pair_type):
        pair_container = self.get_pair_container_by_strandid(strand_id)
        xticks, xticklabels = self.ini_xtick_xticklabel()
        i = 0
        for idx, diff in enumerate(self.d_diff[self.category]):
            sub_xticks, sub_xticklabels, y_mean_list, y_std_list = self.get_y_mean_std_list_xticklabels_by_d_pair_type(i, diff, pair_container, d_pair_type)
            label = f'Resid_j - Resid_i = {diff}'
            ax.bar(sub_xticks, y_mean_list, self.width, yerr=y_std_list, color=self.color_lst[idx], label=label)
            xticklabels += sub_xticklabels
            xticks += sub_xticks
            i += len(sub_xticks)
        self.set_tick_label(ax, xticks, xticklabels, strand_id)

    def ini_xtick_xticklabel(self):
        return list(), list()

    def get_pair_container_by_strandid(self, strand_id):
        mask = self.df_sele['Strand_i'] == strand_id
        df = self.df_sele[mask]
        return PairContainer(self.category, df)

    def set_tick_label(self, ax, xticks, xticklabels, strand_id):
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.tick_params(labelsize=12)
        ax.set_ylabel(f'{self.category} k (kcal/mol/Å$^2$)', fontsize=12)
        ax.set_title(f'{self.abbr} {strand_id}', fontsize=12)
        ax.legend(fontsize=8)

    def get_y_mean_std_list_xticklabels(self, i, diff, pair_container):
        d_mean_std = pair_container.get_d_mean_std()[diff]
        xticklabels = d_mean_std.keys()
        y_mean_list = [d_mean_std[key]['mean'] for key in xticklabels]
        y_std_list = [d_mean_std[key]['std'] for key in xticklabels]
        xticks = list(range(i, i+len(xticklabels)))
        return xticks, xticklabels, y_mean_list, y_std_list

    def get_y_mean_std_list_xticklabels_by_d_pair_type(self, i, diff, pair_container, d_pair_type):
        d_mean_std = pair_container.get_d_mean_std()[diff]
        d_mean_std = self.filter_d_mean_std_by_d_pair_type(d_mean_std, d_pair_type[diff])
        xticklabels = d_mean_std.keys()
        y_mean_list = [d_mean_std[key]['mean'] for key in xticklabels]
        y_std_list = [d_mean_std[key]['std'] for key in xticklabels]
        xticks = list(range(i, i+len(xticklabels)))
        return xticks, xticklabels, y_mean_list, y_std_list

    def filter_d_mean_std_by_d_pair_type(self, d_mean_std, pair_type):
        d_mean_std_new = dict()
        for key in d_mean_std.keys():
            if key in pair_type:
                d_mean_std_new[key] = d_mean_std[key]
        return d_mean_std_new

    def get_d_axes(self, axes):
        d_axes = dict()
        for idx, strand_id in enumerate(self.strand_id_lst):
            d_axes[strand_id] = axes[idx]
        return d_axes
class Pair:
    def __init__(self, atomname_i, atomname_j, resid_i, resid_j, k_mean, k_std):
        diff = resid_j - resid_i # Let diff always >= 0
        if diff < 0:
            self.atomname_i = atomname_j
            self.atomname_j = atomname_i
            self.resid_i = resid_j
            self.resid_j = resid_i
        else:
            self.atomname_i = atomname_i
            self.atomname_j = atomname_j
            self.resid_i = resid_i
            self.resid_j = resid_j

        self.diff = self.resid_j - self.resid_i
        self.k_mean = k_mean
        self.k_std = k_std

    def __eq__(self, other):
        if self.diff == other.diff:
            if ((self.atomname_i == other.atomname_i) and (self.atomname_j == other.atomname_j)) or ((self.atomname_i == other.atomname_j) and (self.atomname_j == other.atomname_i)):
                return True
            else:
                return False
        else:
            return False

    def __repr__(self):
        return f'Diff:{self.diff} Pair:{self.atomname_i} - {self.atomname_j}'

class PairContainer:
    def __init__(self, category, df):
        self.category = category
        self.df = df
        self.lst = self.get_lst_by_df()
        self.n_pairs = len(self.lst)
        self.pair_type_lst = self.get_pair_type_lst()
        self.diff_lst = self.get_diff_lst()
        self.d_pair_type = self.get_d_pair_type()

    def get_lst_by_df(self):
        result_lst = list()
        for atomname_i, atomname_j, resid_i, resid_j, k_mean, k_std in zip(self.df['Atomname_i'], self.df['Atomname_j'], self.df['Resid_i'], self.df['Resid_j'], self.df['k-mean'], self.df['k-std']):
            if (resid_i < 4) or (resid_i > 18) or (resid_j < 4) or (resid_j > 18):
                continue
            result_lst.append(Pair(atomname_i, atomname_j, resid_i, resid_j, k_mean, k_std))
        return result_lst

    def get_pair_type_lst(self):
        pair_type_lst = list()
        for pair in self.lst:
            if pair not in pair_type_lst:
                pair_type_lst.append(pair)
        return pair_type_lst

    def get_diff_lst(self):
        diff_lst = [pair.diff for pair in self.lst]
        return list(set(diff_lst))

    def get_d_pair_type(self):
        d_pair_type = {diff: list() for diff in self.diff_lst}
        for pair in self.pair_type_lst:
            d_pair_type[pair.diff].append(pair)
        return d_pair_type

    def get_d_pair_type_keys(self):
        d_keys = {diff: list() for diff in self.diff_lst}
        for diff in self.diff_lst:
            pair_type_lst = self.d_pair_type[diff]
            for pair in pair_type_lst:
                key = f'{pair.atomname_i}-{pair.atomname_j}'
                d_keys[diff].append(key)
        return d_keys

    def get_d_mean_std(self):
        d_mean_std = {diff: dict() for diff in self.diff_lst}
        for diff in self.diff_lst:
            pair_type_lst = self.d_pair_type[diff]
            for pair in pair_type_lst:
                key1 = f'{pair.atomname_i}-{pair.atomname_j}'
                d_mean_std[diff][key1] = dict()
                mean, std = self.get_mean_std_by_pair(pair)
                d_mean_std[diff][key1]['mean'] = mean
                d_mean_std[diff][key1]['std'] = std
        return d_mean_std

    def get_mean_std_by_pair(self, pair_sele):
        k_list = list()
        for pair in self.lst:
            if pair == pair_sele:
                k_list.append(pair.k_mean)
        k_array = np.array(k_list)
        if k_array.shape[0] == 0:
            return 0., 0.
        else:
            return k_array.mean(), k_array.std()

    def __repr__(self):
        return f'{self.category} Number of Pairs: {self.n_pairs}'


    
