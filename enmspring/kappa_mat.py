import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
from enmspring.na_seq import sequences

CMAP = 'Reds'

class KMat:
    def __init__(self, s_agent):
        self.s_agent = s_agent
        self.pre_process()

        self.n_node = self.s_agent.n_node
        self.eigvector_mat = self.s_agent.v

    def get_K_mat(self, m, n):
        K_mat = np.zeros((self.n_node, self.n_node))
        for eigv_id in range(m, n+1):
            lambda_mat = np.zeros((self.n_node, self.n_node))
            lambda_mat[eigv_id-1, eigv_id-1] = self.s_agent.get_eigenvalue_by_id(eigv_id)
            K_mat += np.dot(self.eigvector_mat, np.dot(lambda_mat, self.eigvector_mat.transpose()))
        return K_mat

    def pre_process(self):
        self.s_agent.load_mean_mode_laplacian_from_npy()
        self.s_agent.eigen_decompose()
        self.s_agent.initialize_nodes_information()
        self.s_agent.split_node_list_into_two_strand()
        self.s_agent.set_benchmark_array()
        self.s_agent.set_strand_array()

class Kappa:
    d_atomlist = {'A': ['N9', 'C8', 'N7', 'C5', 'C4', 'N3', 'C2', 'N1', 'C6', 'N6'],
                  'T': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2', 'O2', 'O4', 'C7'],
                  'C': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2', 'O2', 'N4'],
                  'G': ['N9', 'C8', 'N7', 'C5', 'C4', 'N3', 'C2', 'N1', 'C6', 'O6', 'N2']}
    lbfz = 12

    def __init__(self, host, strand_id, resid_i, s_agent, d_map, seq):
        self.host = host
        self.strand_id = strand_id
        self.s_agent = s_agent
        self.map_idx_from_strand_resid_atomname = d_map
        self.seq = seq

        self.resid_i = resid_i
        self.resid_j = resid_i + 1

        self.atomlst_i, self.atomlst_j = self.get_atomlst()
        self.n_atom_i = len(self.atomlst_i)
        self.n_atom_j = len(self.atomlst_j)


    def heatmap(self, ax, big_k_mat, norm):
        data_mat = self.get_data_mat(big_k_mat)
        im = ax.imshow(data_mat, cmap=CMAP, norm=norm)
        self.set_xticks_yticks(ax)
        self.set_xlabel_ylabel(ax)
        return im

    def get_data_mat(self, big_k_mat):
        data_mat = np.zeros((self.n_atom_j, self.n_atom_i))
        for idx_j, atomname_j in enumerate(self.atomlst_j):
            atomid_j = self.get_atomid_by_resid_atomname(self.resid_j, atomname_j)
            for idx_i, atomname_i in enumerate(self.atomlst_i):
                atomid_i = self.get_atomid_by_resid_atomname(self.resid_i, atomname_i)
                data_mat[idx_j, idx_i] = big_k_mat[atomid_j, atomid_i]
        return data_mat

    def get_atomid_by_resid_atomname(self, resid, atomname):
        key = (self.strand_id, resid, atomname)
        return self.map_idx_from_strand_resid_atomname[key]

    def set_xticks_yticks(self, ax):
        ax.set_xticks(range(self.n_atom_i))
        ax.set_yticks(range(self.n_atom_j))
        ax.set_xticklabels(self.atomlst_i)
        ax.set_yticklabels(self.atomlst_j)
        ax.xaxis.tick_top()
    
    def set_xlabel_ylabel(self, ax):
        ax.set_xlabel(f'Resid {self.resid_i}', fontsize=self.lbfz)
        ax.set_ylabel(f'Resid {self.resid_j}', fontsize=self.lbfz)
        ax.xaxis.set_label_position('top')

    def get_basetype_by_resid(self, resid):
        return self.seq[resid-1]

    def get_atomlst(self):
        basetype_i = self.get_basetype_by_resid(self.resid_i)
        basetype_j = self.get_basetype_by_resid(self.resid_j)
        atomlst_i = self.d_atomlist[basetype_i]
        atomlst_j = self.d_atomlist[basetype_j]
        return atomlst_i, atomlst_j

class KappaStrand:
    resid_lst = list(range(4, 18))

    def __init__(self, host, strand_id, s_agent, kmat_agent):
        self.host = host
        self.strand_id = strand_id
        self.s_agent = s_agent
        self.kmat_agent = kmat_agent

        self.node_list = s_agent.node_list
        self.d_idx = s_agent.d_idx
        self.strandid_map = s_agent.strandid_map
        self.resid_map = s_agent.resid_map
        self.atomname_map = s_agent.atomname_map
        self.map_idx_from_strand_resid_atomname = self.get_map_idx_from_strand_resid_atomname()

        self.n_row = 2
        self.n_col = 7

        self.d_seq = {'STRAND1': sequences[host]['guide'], 'STRAND2': sequences[host]['target']}
        self.seq = self.d_seq[strand_id]

        self.d_kappa = self.get_d_kappa()

    def plot_all_heatmap(self, figsize, start_mode, end_mode, vmin, vmax):
        fig, axes = plt.subplots(nrows=self.n_row, ncols=self.n_col, figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(axes)
        norm = Normalize(vmin=vmin, vmax=vmax)
        K_mat = self.kmat_agent.get_K_mat(start_mode, end_mode)
        for resid in self.resid_lst:
            self.d_kappa[resid].heatmap(d_axes[resid], K_mat, norm)
        return fig, d_axes

    def plot_colorbar(self, figsize, vmin, vmax):
        fig = plt.figure(figsize=figsize, facecolor='white')
        ax1 = fig.add_axes([0.05, 0.80, 0.9, 0.15])
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(CMAP)
        cb1 = ColorbarBase(ax1, cmap=cmap, norm=norm, orientation='horizontal', label='k (kcal/mol/Å$^2$)')
        return cb1

    def get_d_kappa(self):
        d_kappa = dict()
        for resid in self.resid_lst:
            d_kappa[resid] = Kappa(self.host, self.strand_id, resid, self.s_agent, self.map_idx_from_strand_resid_atomname, self.seq)
        return d_kappa

    def get_kmin_kmax(self, start_mode, end_mode):
        K_mat = self.kmat_agent.get_K_mat(start_mode, end_mode)
        for idx, resid in enumerate(self.resid_lst):
            data_mat = self.d_kappa[resid].get_data_mat(K_mat)
            if idx == 0:
                minimum = data_mat.min()
                maximum = data_mat.max()
            else:
                minimum = np.minimum(minimum, data_mat.min())
                maximum = np.maximum(maximum, data_mat.max())
        print(f'Min: {minimum:.3f}  Max: {maximum:.3f}')
        return minimum, maximum

    def get_d_axes(self, axes):
        d_axes = dict()
        idx = 0
        for row_id in range(self.n_row):
            for col_id in range(self.n_col):
                resid = self.resid_lst[idx]
                d_axes[resid] = axes[row_id, col_id]
                idx += 1
        return d_axes

    def get_map_idx_from_strand_resid_atomname(self):
        d_result = dict()
        for node_name in self.node_list:
            idx = self.d_idx[node_name]
            strand_id = self.strandid_map[node_name]
            resid = self.resid_map[node_name]
            atomname = self.atomname_map[node_name]
            d_result[(strand_id, resid, atomname)] = idx
        return d_result