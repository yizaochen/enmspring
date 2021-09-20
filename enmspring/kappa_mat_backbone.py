import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from enmspring.kappa_mat import Kappa, KappaStrand, MeanKappaStrand
from enmspring.na_seq import sequences


class KappaBackbone(Kappa):
    backbone_atomlist = ['P', 'O1P', 'O2P', "O5'", "C5'"]
    ribose_atomlist = ["C4'", "O4'", "C1'", "C2'", "C3'", "O3'"]
    base_atomlist = {'A': ['N9', 'C8', 'N7', 'C5', 'C4', 'N3', 'C2', 'N1', 'C6', 'N6'],
                     'T': ['N1', 'C2', 'N3', 'C4', 'C5', 'C7', 'C6', 'O2', 'O4'],
                     'C': ['N1', 'C2', 'N3', 'C4', 'N4', 'C5', 'C6', 'O2'],
                     'G': ['N9', 'C8', 'N7', 'C5', 'C4', 'N3', 'C2', 'N1', 'C6', 'O6', 'N2']}
    d_atomlist = {'A': backbone_atomlist+ribose_atomlist+base_atomlist['A'],
                  'T': backbone_atomlist+ribose_atomlist+base_atomlist['T'],
                  'C': backbone_atomlist+ribose_atomlist+base_atomlist['C'],
                  'G': backbone_atomlist+ribose_atomlist+base_atomlist['G']}

    def __init__(self, host, strand_id, resid_i, s_agent, d_map, seq):
        self.host = host
        self.strand_id = strand_id
        self.s_agent = s_agent
        self.map_idx_from_strand_resid_atomname = d_map
        self.seq = seq

        self.resid_i = resid_i
        self.resid_j = self.set_resid_j()

        self.atomlst_i, self.atomlst_j = self.get_atomlst()
        self.n_atom_i = len(self.atomlst_i)
        self.n_atom_j = len(self.atomlst_j)

    def set_resid_j(self):
        return self.resid_i

    def get_data_mat_j(self, big_k_mat):
        data_mat = np.zeros((self.n_atom_j, self.n_atom_i))
        for idx_j, atomname_j in enumerate(self.atomlst_j):
            atomid_j = self.get_atomid_by_resid_atomname(self.resid_j, atomname_j)
            for idx_i, atomname_i in enumerate(self.atomlst_i):
                atomid_i = self.get_atomid_by_resid_atomname(self.resid_i, atomname_i)
                data_mat[idx_j, idx_i] = big_k_mat[atomid_j, atomid_i]
        return data_mat

class KappaBackboneWithNext(KappaBackbone):
    def set_resid_j(self):
        return self.resid_i + 1

class KappaStrandBackbone(KappaStrand):

    def plot_all_heatmap(self, figsize, start_mode, end_mode, vmin, vmax, lbfz=8, tickfz=6):
        fig, axes = plt.subplots(nrows=self.n_row, ncols=self.n_col, figsize=figsize, facecolor='white')
        d_axes = self.get_d_axes(axes)
        norm = Normalize(vmin=vmin, vmax=vmax)
        mode_list = list(range(start_mode, end_mode+1))
        K_mat = self.kmat_agent.get_K_mat_by_strandid_modelist(self.strand_id, mode_list)
        K_mat = self.kmat_agent.set_diagonal_zero(K_mat)
        for resid in self.resid_lst:
            self.d_kappa[resid].heatmap(d_axes[resid], K_mat, norm, lbfz, tickfz)
        return fig, d_axes

    def get_d_kappa(self):
        d_kappa = dict()
        for resid in self.resid_lst:
            d_kappa[resid] = KappaBackbone(self.host, self.strand_id, resid, self.s_agent, self.map_idx_from_strand_resid_atomname, self.seq)
        return d_kappa

class MeanKappaStrandBackbone(MeanKappaStrand):
    resid_lst = list(range(4, 19))
    d_basetype = {'a_tract_21mer': {'STRAND1': {'i': 'A', 'j': 'A'},
                                    'STRAND2': {'i': 'T', 'j': 'T'}},
                  'g_tract_21mer': {'STRAND1': {'i': 'G', 'j': 'G'},
                                    'STRAND2': {'i': 'C', 'j': 'C'}}
                 }

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

        self.atomlst_i, self.atomlst_j = self.get_atomlst()
        self.n_atom_i = len(self.atomlst_i)
        self.n_atom_j = len(self.atomlst_j)

    def plot_mean_heatmap_single(self, figsize, start_mode, end_mode, vmin, vmax, dot_criteria):
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        norm = Normalize(vmin=vmin, vmax=vmax)
        K_mat = self.kmat_agent.get_K_mat(start_mode, end_mode)
        K_mat = self.kmat_agent.set_diagonal_zero(K_mat)
        mean_data_mat_j = self.get_mean_data_mat_j(K_mat)
        im_j = self.heatmap_single(ax, mean_data_mat_j, norm)
        self.scatter_center_over_criteria(ax, mean_data_mat_j, dot_criteria)
        self.set_yticks_yticklabels_single(ax)
        self.set_xticks_xticklabels_single(ax)
        return fig, im_j, ax

    def get_d_kappa(self):
        d_kappa = dict()
        for resid in self.resid_lst:
            d_kappa[resid] = KappaBackbone(self.host, self.strand_id, resid, self.s_agent, self.map_idx_from_strand_resid_atomname, self.seq)
        return d_kappa

    def get_atomlst(self):
        basetype_i = self.d_basetype[self.host][self.strand_id]['i']
        basetype_j = self.d_basetype[self.host][self.strand_id]['j']
        atomlst_i = KappaBackbone.d_atomlist[basetype_i]
        atomlst_j = KappaBackbone.d_atomlist[basetype_j]
        return atomlst_i, atomlst_j

class MeanKappaStrandBackboneWithNext(MeanKappaStrandBackbone):
    def get_d_kappa(self):
        d_kappa = dict()
        for resid in self.resid_lst:
            d_kappa[resid] = KappaBackboneWithNext(self.host, self.strand_id, resid, self.s_agent, self.map_idx_from_strand_resid_atomname, self.seq)
        return d_kappa

    def plot_mean_heatmap_single(self, figsize, start_mode, end_mode, vmin, vmax, dot_criteria):
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        norm = Normalize(vmin=vmin, vmax=vmax)
        K_mat = self.kmat_agent.get_K_mat(start_mode, end_mode)
        mean_data_mat_j = self.get_mean_data_mat_j(K_mat)
        im_j = self.heatmap_single(ax, mean_data_mat_j, norm)
        self.scatter_center_over_criteria(ax, mean_data_mat_j, dot_criteria)
        self.set_yticks_yticklabels_single(ax)
        self.set_xticks_xticklabels_single(ax)
        return fig, im_j, ax


class MeanKappaStrandHetreoBackbone(MeanKappaStrandBackbone):
    strand_id_lst = ['STRAND1', 'STRAND2']
    d_resid_lst = {'atat_21mer': {'A': {'STRAND1': list(range(5, 18, 2)), 'STRAND2': list(range(4, 19, 2))},
                                  'T': {'STRAND1': list(range(4, 19, 2)), 'STRAND2': list(range(5, 18, 2))}},
                   'gcgc_21mer': {'G': {'STRAND1': list(range(5, 18, 2)), 'STRAND2': list(range(4, 19, 2))},
                                  'C': {'STRAND1': list(range(4, 19, 2)), 'STRAND2': list(range(5, 18, 2))}}
                   }

    resid_lst = list(range(4, 19))

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

        self.basetype_j = self.set_basetype_j()

        self.d_kappa = self.get_d_kappa()

        self.atomlst_i, self.atomlst_j = self.get_atomlst()
        self.n_atom_i = len(self.atomlst_i)
        self.n_atom_j = len(self.atomlst_j)

    def set_basetype_j(self):
        return self.basetype_i

    def get_atomlst(self):
        atomlst_i = KappaBackbone.d_atomlist[self.basetype_i]
        atomlst_j = KappaBackbone.d_atomlist[self.basetype_j]
        return atomlst_i, atomlst_j

    def get_d_kappa(self):
        d_kappa = {strand_id: dict() for strand_id in self.strand_id_lst}
        for strand_id in self.strand_id_lst:
            resid_lst = self.d_resid_lst[self.host][self.basetype_i][strand_id]
            seq = self.d_seq[strand_id]
            for resid in resid_lst:
                d_kappa[strand_id][resid] = KappaBackbone(self.host, strand_id, resid, self.s_agent, self.map_idx_from_strand_resid_atomname, seq)
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

class MeanKappaStrandHetreoBackboneWithNext(MeanKappaStrandHetreoBackbone):
    d_basetype_j = {'atat_21mer': {'A': 'T', 'T': 'A'},
                    'gcgc_21mer': {'G': 'C', 'C': 'G'}}

    def set_basetype_j(self):
        return self.d_basetype_j[self.host][self.basetype_i]

    def plot_mean_heatmap_single(self, figsize, start_mode, end_mode, vmin, vmax, dot_criteria):
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        norm = Normalize(vmin=vmin, vmax=vmax)
        K_mat = self.kmat_agent.get_K_mat(start_mode, end_mode)
        mean_data_mat_j = self.get_mean_data_mat_j(K_mat)
        im_j = self.heatmap_single(ax, mean_data_mat_j, norm)
        self.scatter_center_over_criteria(ax, mean_data_mat_j, dot_criteria)
        self.set_yticks_yticklabels_single(ax)
        self.set_xticks_xticklabels_single(ax)
        return fig, im_j, ax

    def get_d_kappa(self):
        d_kappa = {strand_id: dict() for strand_id in self.strand_id_lst}
        for strand_id in self.strand_id_lst:
            resid_lst = self.d_resid_lst[self.host][self.basetype_i][strand_id]
            seq = self.d_seq[strand_id]
            for resid in resid_lst:
                d_kappa[strand_id][resid] = KappaBackboneWithNext(self.host, strand_id, resid, self.s_agent, self.map_idx_from_strand_resid_atomname, seq)
        return d_kappa
