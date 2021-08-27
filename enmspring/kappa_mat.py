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

class KappaUpperDown(Kappa):
    def __init__(self, host, strand_id, resid_i, s_agent, d_map, seq):
        self.host = host
        self.strand_id = strand_id
        self.s_agent = s_agent
        self.map_idx_from_strand_resid_atomname = d_map
        self.seq = seq

        self.resid_i = resid_i
        self.resid_j = resid_i + 1 # 3'
        self.resid_k = resid_i - 1 # 5'

        self.atomlst_i, self.atomlst_j, self.atomlst_k = self.get_atomlst()
        self.n_atom_i = len(self.atomlst_i)
        self.n_atom_j = len(self.atomlst_j)
        self.n_atom_k = len(self.atomlst_k)

    def heatmap_j(self, ax, big_k_mat, norm):
        data_mat = self.get_data_mat_j(big_k_mat)
        im = ax.imshow(data_mat, cmap=CMAP, norm=norm)
        self.set_xticks_yticks_j(ax)
        self.set_xlabel_ylabel_j(ax)
        #ax.xaxis.tick_top()
        #ax.xaxis.set_label_position('top')
        return im

    def heatmap_k(self, ax, big_k_mat, norm):
        data_mat = self.get_data_mat_k(big_k_mat)
        im = ax.imshow(data_mat, cmap=CMAP, norm=norm)
        self.set_xticks_yticks_k(ax)
        self.set_xlabel_ylabel_k(ax)
        return im

    def get_data_mat_j(self, big_k_mat):
        data_mat = np.zeros((self.n_atom_j, self.n_atom_i))
        for idx_j, atomname_j in enumerate(self.atomlst_j):
            atomid_j = self.get_atomid_by_resid_atomname(self.resid_j, atomname_j)
            for idx_i, atomname_i in enumerate(self.atomlst_i):
                atomid_i = self.get_atomid_by_resid_atomname(self.resid_i, atomname_i)
                data_mat[idx_j, idx_i] = big_k_mat[atomid_j, atomid_i]
        return data_mat

    def get_data_mat_k(self, big_k_mat):
        data_mat = np.zeros((self.n_atom_k, self.n_atom_i))
        for idx_k, atomname_k in enumerate(self.atomlst_k):
            atomid_k = self.get_atomid_by_resid_atomname(self.resid_k, atomname_k)
            for idx_i, atomname_i in enumerate(self.atomlst_i):
                atomid_i = self.get_atomid_by_resid_atomname(self.resid_i, atomname_i)
                data_mat[idx_k, idx_i] = big_k_mat[atomid_k, atomid_i]
        return data_mat

    def set_xticks_yticks_j(self, ax):
        ax.set_xticks(range(self.n_atom_i))
        ax.set_yticks(range(self.n_atom_j))
        #ax.set_xticklabels(self.atomlst_i)
        ax.set_yticklabels(self.atomlst_j)
        ax.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False)

    def set_xticks_yticks_k(self, ax):
        ax.set_xticks(range(self.n_atom_i))
        ax.set_yticks(range(self.n_atom_j))
        ax.set_xticklabels(self.atomlst_k)
        ax.set_yticklabels(self.atomlst_k)
    
    def set_xlabel_ylabel_j(self, ax):
        ax.set_xlabel(f'Resid {self.resid_i}', fontsize=self.lbfz)
        ax.set_ylabel(f'Resid {self.resid_j}', fontsize=self.lbfz)

    def set_xlabel_ylabel_k(self, ax):
        #ax.set_xlabel(f'Resid {self.resid_i}', fontsize=self.lbfz)
        ax.set_ylabel(f'Resid {self.resid_k}', fontsize=self.lbfz)

    def get_atomlst(self):
        basetype_i = self.get_basetype_by_resid(self.resid_i)
        basetype_j = self.get_basetype_by_resid(self.resid_j)
        basetype_k = self.get_basetype_by_resid(self.resid_k)
        atomlst_i = self.d_atomlist[basetype_i]
        atomlst_j = self.d_atomlist[basetype_j]
        atomlst_k = self.d_atomlist[basetype_k]
        return atomlst_i, atomlst_j, atomlst_k


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

class MeanKappaStrand(KappaStrand):
    resid_lst = list(range(5, 18))
    d_basetype = {'a_tract_21mer': {'STRAND1': {'i': 'A', 'j': 'A', 'k': 'A'},
                                    'STRAND2': {'i': 'T', 'j': 'T', 'k': 'T'}},
                  'g_tract_21mer': {'STRAND1': {'i': 'G', 'j': 'G', 'k': 'G'},
                                    'STRAND2': {'i': 'C', 'j': 'C', 'k': 'C'}}
                 }
    lbfz = 12

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

        self.d_seq = {'STRAND1': sequences[host]['guide'], 'STRAND2': sequences[host]['target']}
        self.seq = self.d_seq[strand_id]

        self.d_kappa = self.get_d_kappa()

        self.atomlst_i, self.atomlst_j, self.atomlst_k = self.get_atomlst()
        self.n_atom_i = len(self.atomlst_i)
        self.n_atom_j = len(self.atomlst_j)
        self.n_atom_k = len(self.atomlst_k)

    def plot_mean_heatmap(self, figsize, start_mode, end_mode, vmin, vmax):
        fig = plt.figure(figsize=figsize, facecolor='white')
        axes = self.make_axes(fig)
        norm = Normalize(vmin=vmin, vmax=vmax)
        mean_data_mat_j, mean_data_mat_k = self.get_mean_data_mat_j_k(start_mode, end_mode)

        im_k, im_j = self.heatmap(axes, mean_data_mat_j, mean_data_mat_k, norm)
        self.set_yticks_yticklabels(axes)
        self.set_xticks_xticklabels(axes)
        return fig, im_k, im_j, axes

    def set_yticks_yticklabels(self, axes):
        axes[0].set_yticks(range(self.n_atom_k))
        axes[0].set_yticklabels(self.atomlst_k)
        axes[0].set_ylabel('Resid I-1', fontsize=self.lbfz)
        axes[1].set_yticks(range(self.n_atom_j))
        axes[1].set_yticklabels(self.atomlst_j)
        axes[1].set_ylabel('Resid I+1', fontsize=self.lbfz)

    def set_xticks_xticklabels(self, axes):
        axes[0].set_xticks(range(self.n_atom_i))
        axes[0].set_xticklabels(self.atomlst_i)
        axes[1].set_xlabel('Resid I', fontsize=self.lbfz)
        axes[1].set_xticks(range(self.n_atom_i))
        axes[1].tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False)

    def heatmap(self, axes, data_mat_j, data_mat_k, norm):
        im_k = axes[0].imshow(data_mat_k, cmap=CMAP, norm=norm)
        im_j = axes[1].imshow(data_mat_j, cmap=CMAP, norm=norm)
        return im_k, im_j

    def make_axes(self, fig):
        gs = fig.add_gridspec(21, 1, hspace=0)
        ax1 = fig.add_subplot(gs[0:10])
        ax2 = fig.add_subplot(gs[11:])
        return [ax1, ax2]

    def get_mean_data_mat_j_k(self, start_mode, end_mode):
        K_mat = self.kmat_agent.get_K_mat(start_mode, end_mode)
        mean_data_mat_j = self.get_mean_data_mat_j(K_mat)
        mean_data_mat_k = self.get_mean_data_mat_k(K_mat)
        return mean_data_mat_j, mean_data_mat_k

    def get_mean_data_mat_j(self, K_mat):
        d_data_mat_j = dict()
        for resid in self.resid_lst:
            d_data_mat_j[resid] = self.d_kappa[resid].get_data_mat_j(K_mat)
        mean_data_mat_j = np.zeros(d_data_mat_j[resid].shape)
        for row_id in range(d_data_mat_j[resid].shape[0]):
            for col_id in range(d_data_mat_j[resid].shape[1]):
                mean_data_mat_j[row_id, col_id] = self.get_mean_matrix_element(d_data_mat_j, row_id, col_id)
        return mean_data_mat_j

    def get_mean_data_mat_k(self, K_mat):
        d_data_mat_k = dict()
        for resid in self.resid_lst:
            d_data_mat_k[resid] = self.d_kappa[resid].get_data_mat_k(K_mat)
        mean_data_mat_k = np.zeros(d_data_mat_k[resid].shape)
        for row_id in range(d_data_mat_k[resid].shape[0]):
            for col_id in range(d_data_mat_k[resid].shape[1]):
                mean_data_mat_k[row_id, col_id] = self.get_mean_matrix_element(d_data_mat_k, row_id, col_id)
        return mean_data_mat_k

    def get_mean_matrix_element(self, d_data_mat, row_id, col_id):
        temp_array = np.zeros(len(self.resid_lst))
        for idx, resid in enumerate(self.resid_lst):
            temp_array[idx] = d_data_mat[resid][row_id, col_id]
        return temp_array.mean()

    def get_d_kappa(self):
        d_kappa = dict()
        for resid in self.resid_lst:
            d_kappa[resid] = KappaUpperDown(self.host, self.strand_id, resid, self.s_agent, self.map_idx_from_strand_resid_atomname, self.seq)
        return d_kappa

    def get_atomlst(self):
        basetype_i = self.d_basetype[self.host][self.strand_id]['i']
        basetype_j = self.d_basetype[self.host][self.strand_id]['j']
        basetype_k = self.d_basetype[self.host][self.strand_id]['k']
        atomlst_i = Kappa.d_atomlist[basetype_i]
        atomlst_j = Kappa.d_atomlist[basetype_j]
        atomlst_k = Kappa.d_atomlist[basetype_k]
        return atomlst_i, atomlst_j, atomlst_k


class MeanKappaStrandHetreo(MeanKappaStrand):
    d_basetype_jk = {'atat_21mer': {'A': 'T', 'T': 'A'},
                     'gcgc_21mer': {'G': 'C', 'C': 'G'}}
    strand_id_lst = ['STRAND1', 'STRAND2']
    d_resid_lst = {'atat_21mer': {'A': {'STRAND1': list(range(5, 18, 2)), 'STRAND2': list(range(6, 17, 2))},
                                  'T': {'STRAND1': list(range(6, 17, 2)), 'STRAND2': list(range(5, 18, 2))}},
                   'gcgc_21mer': {'G': {'STRAND1': list(range(5, 18, 2)), 'STRAND2': list(range(6, 17, 2))},
                                  'C': {'STRAND1': list(range(6, 17, 2)), 'STRAND2': list(range(5, 18, 2))}}
                   }

    resid_lst = list(range(5, 18))

    def __init__(self, host, basetype_i, s_agent, kmat_agent):
        self.host = host
        self.basetype_i = basetype_i
        self.s_agent = s_agent
        self.kmat_agent = kmat_agent

        self.node_list = s_agent.node_list
        self.d_idx = s_agent.d_idx
        self.strandid_map = s_agent.strandid_map
        self.resid_map = s_agent.resid_map
        self.atomname_map = s_agent.atomname_map
        self.map_idx_from_strand_resid_atomname = self.get_map_idx_from_strand_resid_atomname()

        self.d_seq = {'STRAND1': sequences[host]['guide'], 'STRAND2': sequences[host]['target']}

        self.basetype_j = self.d_basetype_jk[self.host][self.basetype_i]
        self.basetype_k = self.d_basetype_jk[self.host][self.basetype_i]

        self.d_kappa = self.get_d_kappa()

        self.atomlst_i, self.atomlst_j, self.atomlst_k = self.get_atomlst()
        self.n_atom_i = len(self.atomlst_i)
        self.n_atom_j = len(self.atomlst_j)
        self.n_atom_k = len(self.atomlst_k)

    def get_atomlst(self):
        atomlst_i = Kappa.d_atomlist[self.basetype_i]
        atomlst_j = Kappa.d_atomlist[self.basetype_j]
        atomlst_k = Kappa.d_atomlist[self.basetype_k]
        return atomlst_i, atomlst_j, atomlst_k

    def get_d_kappa(self):
        d_kappa = {strand_id: dict() for strand_id in self.strand_id_lst}
        for strand_id in self.strand_id_lst:
            resid_lst = self.d_resid_lst[self.host][self.basetype_i][strand_id]
            seq = self.d_seq[strand_id]
            for resid in resid_lst:
                d_kappa[strand_id][resid] = KappaUpperDown(self.host, strand_id, resid, self.s_agent, self.map_idx_from_strand_resid_atomname, seq)
        return d_kappa

    def get_mean_data_mat_j(self, K_mat):
        d_data_mat_j = {strand_id: dict() for strand_id in self.strand_id_lst}
        for strand_id in self.strand_id_lst:
            resid_lst = self.d_resid_lst[self.host][self.basetype_i][strand_id]
            for resid in resid_lst:
                d_data_mat_j[strand_id][resid] = self.d_kappa[strand_id][resid].get_data_mat_j(K_mat)
        mean_data_mat_j = np.zeros(d_data_mat_j[strand_id][resid].shape)
        for row_id in range(d_data_mat_j[strand_id][resid].shape[0]):
            for col_id in range(d_data_mat_j[strand_id][resid].shape[1]):
                mean_data_mat_j[row_id, col_id] = self.get_mean_matrix_element(d_data_mat_j, row_id, col_id)
        return mean_data_mat_j

    def get_mean_data_mat_k(self, K_mat):
        d_data_mat_k = {strand_id: dict() for strand_id in self.strand_id_lst}
        for strand_id in self.strand_id_lst:
            resid_lst = self.d_resid_lst[self.host][self.basetype_i][strand_id]
            for resid in resid_lst:
                d_data_mat_k[strand_id][resid] = self.d_kappa[strand_id][resid].get_data_mat_k(K_mat)
        mean_data_mat_k = np.zeros(d_data_mat_k[strand_id][resid].shape)
        for row_id in range(d_data_mat_k[strand_id][resid].shape[0]):
            for col_id in range(d_data_mat_k[strand_id][resid].shape[1]):
                mean_data_mat_k[row_id, col_id] = self.get_mean_matrix_element(d_data_mat_k, row_id, col_id)
        return mean_data_mat_k

    def get_mean_matrix_element(self, d_data_mat, row_id, col_id):
        n_mat = len(self.d_resid_lst[self.host][self.basetype_i]['STRAND1']) + len(self.d_resid_lst[self.host][self.basetype_i]['STRAND2'])
        temp_array = np.zeros(n_mat)
        idx = 0
        for strand_id in self.strand_id_lst:
            resid_lst = self.d_resid_lst[self.host][self.basetype_i][strand_id]
            for resid in resid_lst:
                temp_array[idx] = d_data_mat[strand_id][resid][row_id, col_id]
                idx += 1
        return temp_array.mean()