import numpy as np
from enmspring.kappa_mat import Kappa, MeanKappaStrand
from enmspring.na_seq import sequences


class KappaBB2(Kappa):
    backbone_atomlist = ['O2P', 'O1P', 'P', "O5'", "C5'"]
    ribose_atomlist = ["C4'", "O4'", "C1'", "C2'", "C3'", "O3'"]

    def __init__(self, host, strand_id, resid_i, s_agent, d_map, seq):
        self.host = host
        self.strand_id = strand_id
        self.s_agent = s_agent
        self.map_idx_from_strand_resid_atomname = d_map
        self.seq = seq

        self.resid_i = resid_i
        self.resid_j = resid_i + 1 # 3'
        self.resid_k = resid_i - 1 # 5'

        self.atomlst_i = self.backbone_atomlist + self.ribose_atomlist
        self.atomlst_j = self.backbone_atomlist + self.ribose_atomlist
        self.atomlst_k = self.backbone_atomlist + self.ribose_atomlist

        self.n_atom_i = len(self.atomlst_i)
        self.n_atom_j = len(self.atomlst_j)
        self.n_atom_k = len(self.atomlst_k)

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

class MeanKappaStrandBB2(MeanKappaStrand):
    resid_lst = list(range(4, 19))

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

        self.atomlst_i = KappaBB2.backbone_atomlist + KappaBB2.ribose_atomlist
        self.atomlst_j = KappaBB2.backbone_atomlist + KappaBB2.ribose_atomlist
        self.n_atom_i = len(self.atomlst_i)
        self.n_atom_j = len(self.atomlst_j)

    def get_d_kappa(self):
        d_kappa = dict()
        for resid in self.resid_lst:
            d_kappa[resid] = KappaBB2(self.host, self.strand_id, resid, self.s_agent, self.map_idx_from_strand_resid_atomname, self.seq)
        return d_kappa

"""
class MeanKappaStrandHetreoBB1(MeanKappaStrandBB1):
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
        atomlst_i = KappaBB1.d_atomlist[self.basetype_i]
        atomlst_j = KappaBB1.d_atomlist[self.basetype_j]
        return atomlst_i, atomlst_j

    def get_d_kappa(self):
        d_kappa = {strand_id: dict() for strand_id in self.strand_id_lst}
        for strand_id in self.strand_id_lst:
            resid_lst = self.d_resid_lst[self.host][self.basetype_i][strand_id]
            seq = self.d_seq[strand_id]
            for resid in resid_lst:
                d_kappa[strand_id][resid] = KappaBB1(self.host, strand_id, resid, self.s_agent, self.map_idx_from_strand_resid_atomname, seq)
        return d_kappa

    def get_mean_data_mat_j(self, K_mat):
        d_data_mat_j = {strand_id: dict() for strand_id in self.strand_id_lst}
        for strand_id in self.strand_id_lst:
            resid_lst = self.d_resid_lst[self.host][self.basetype_i][strand_id]
            for resid in resid_lst:
                d_data_mat_j[strand_id][resid] = self.d_kappa[strand_id][resid].get_data_mat_j(K_mat)
        mean_data_mat_j = np.zeros((self.n_atom_i, self.n_atom_i))
        for row_id in range(self.n_atom_i):
            for col_id in range(self.n_atom_i):
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
"""