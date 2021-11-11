import numpy as np
from enmspring.kappa_mat import Kappa, MeanKappaStrand
from enmspring.na_seq import sequences


class KappaBB1(Kappa):
    backbone_atomlist = ['O2P', 'O1P', 'P', "O5'", "C5'"]
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


class MeanKappaStrandBB1(MeanKappaStrand):
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

    def get_d_kappa(self):
        d_kappa = dict()
        for resid in self.resid_lst:
            d_kappa[resid] = KappaBB1(self.host, self.strand_id, resid, self.s_agent, self.map_idx_from_strand_resid_atomname, self.seq)
        return d_kappa

    def get_atomlst(self):
        basetype_i = self.d_basetype[self.host][self.strand_id]['i']
        basetype_j = self.d_basetype[self.host][self.strand_id]['j']
        atomlst_i = KappaBB1.d_atomlist[basetype_i]
        atomlst_j = KappaBB1.d_atomlist[basetype_j]
        return atomlst_i, atomlst_j

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