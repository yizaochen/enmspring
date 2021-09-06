import numpy as np
import matplotlib.pyplot as plt
from enmspring.kappa_mat import Kappa
from enmspring.na_seq import sequences

class StackResidPlot:
    n_bp = 21
    resid_lst = list(range(4, 18))
    strand_id_lst = ['STRAND1', 'STRAND2']
    

    tickfz = 4
    lbfz = 6

    def __init__(self, host, s_agent, kmat_agent):
        self.host = host
        self.s_agent = s_agent
        self.kmat_agent = kmat_agent

        self.node_list = s_agent.node_list
        self.d_idx = s_agent.d_idx
        self.strandid_map = s_agent.strandid_map
        self.resid_map = s_agent.resid_map
        self.atomname_map = s_agent.atomname_map
        self.map_idx_from_strand_resid_atomname = self.get_map_idx_from_strand_resid_atomname()

        self.d_seq = {'STRAND1': sequences[host]['guide'], 'STRAND2': sequences[host]['target']}
        self.d_kappa = self.get_d_kappa()

    def plot_two_strands(self, figsize, start_mode, end_mode):
        big_k_mat = self.kmat_agent.get_K_mat(start_mode, end_mode)
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(axes)
        for strand_id in self.strand_id_lst:
            self.plot_lines(d_axes[strand_id], strand_id, big_k_mat)
            self.set_xticks(d_axes)
            self.set_ylabel_xlabel(d_axes[strand_id])
        return fig, d_axes

    def plot_lines(self, ax, strand_id, big_k_mat):
        #pair_d_lst = [{'A': 'C4', 'T': 'C5'}, {'A': 'N1', 'T': 'N3'}, {'A': 'C6', 'T': 'C4'}]
        #pair_lst = ['ADE:C4---THY:C5', 'ADE:N1---THY:N3', 'ADE:C6---THY:C4']
        pair_d_lst = [{'A': 'C4', 'T': 'C5'}]
        pair_lst = ['ADE:C4---THY:C5']
        xarray = self.get_xarray()
        for idx, pair_dict in enumerate(pair_d_lst):
            pair_name = pair_lst[idx]
            yarray = self.get_yarray(pair_dict, strand_id, big_k_mat)
            #yarray = np.random.rand(xarray.shape[0])
            ax.plot(xarray, yarray, marker='.', label=pair_name, linewidth=0.5, markersize=2)

    def get_xarray(self):
        interval = 0.5
        return np.arange(4+interval, 18, 1)

    def get_yarray(self, pair_dict, strand_id, big_k_mat):
        k_array = np.zeros(len(self.resid_lst))
        for idx, resid_i in enumerate(self.resid_lst):
            basetype_i = self.d_kappa[strand_id][resid_i].get_basetype_i()
            basetype_j = self.d_kappa[strand_id][resid_i].get_basetype_j()
            atomname_i = pair_dict[basetype_i]
            atomname_j = pair_dict[basetype_j]
            k_array[idx] = self.d_kappa[strand_id][resid_i].get_k_by_atomnames(big_k_mat, atomname_i, atomname_j)
        return k_array

    def get_d_axes(self, axes):
        d_axes = dict()
        for idx, strand_id in enumerate(self.strand_id_lst):
            d_axes[strand_id] = axes[idx]
        return d_axes

    def get_d_kappa(self):
        d_kappa = dict()
        for strand_id in self.strand_id_lst:
            d_kappa[strand_id] = dict()
            seq = self.d_seq[strand_id]
            for resid_i in self.resid_lst:
                d_kappa[strand_id][resid_i] = Kappa(self.host, strand_id, resid_i, self.s_agent, self.map_idx_from_strand_resid_atomname, seq)
        return d_kappa

    def get_map_idx_from_strand_resid_atomname(self):
        d_result = dict()
        for node_name in self.node_list:
            idx = self.d_idx[node_name]
            strand_id = self.strandid_map[node_name]
            resid = self.resid_map[node_name]
            atomname = self.atomname_map[node_name]
            d_result[(strand_id, resid, atomname)] = idx
        return d_result

    def set_ylabel_xlabel(self, ax):
        #ax.set_ylabel('k (kcal/mol/Å$^2$)', fontsize=self.lbfz)
        #ax.set_xlabel('Resid', fontsize=self.lbfz)
        ax.tick_params(axis='y', labelsize=self.tickfz, length=1, pad=1)
        ax.tick_params(axis='x', labelsize=self.tickfz, length=1, pad=0.6)

    def set_xticks(self, d_axes):
        for strand_id in self.strand_id_lst:
            d_axes[strand_id].set_xticks(self.resid_lst)
            seq = self.d_seq[strand_id]
            xticklabels = [seq[resid-1] for resid in self.resid_lst]
            d_axes[strand_id].set_xticklabels(xticklabels)
