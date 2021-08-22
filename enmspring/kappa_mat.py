import numpy as np

class Kappa:
    d_atomlist = {'A': ['N9', 'C8', 'N7', 'C5', 'C4', 'N3', 'C2', 'N1', 'C6', 'N6'],
                  'T': ['C4', 'C5', 'C6', 'N1', 'C2', 'N3', 'C7', 'O2', 'O4'],
                  'C': ['C4', 'C5', 'C6', 'N1', 'C2', 'N3', 'O2', 'N4'],
                  'G': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2', 'O6', 'N2', 'N7', 'C8', 'N9']}
    d_base_stack_types = {
        'a_tract_21mer': {'STRAND1': ('A', 'A'), 'STRAND2': ('T', 'T')}
    }
    lbfz = 12

    def __init__(self, host, strand_id, resid_i, s_agent):
        self.host = host
        self.strand_id = strand_id
        self.s_agent = s_agent

        self.resid_i = resid_i
        self.resid_j = resid_i + 1

        self.atomlst_i, self.atomlst_j = self.get_atomlst()
        self.n_atom_i = len(self.atomlst_i)
        self.n_atom_j = len(self.atomlst_j)

        self.node_list = s_agent.node_list
        self.d_idx = s_agent.d_idx
        self.strandid_map = s_agent.strandid_map
        self.resid_map = s_agent.resid_map
        self.atomname_map = s_agent.atomname_map

        self.map_idx_from_strand_resid_atomname = self.get_map_idx_from_strand_resid_atomname()

    def heatmap(self, ax, big_k_mat):
        data_mat = self.get_data_mat(big_k_mat)
        im = ax.imshow(data_mat, cmap='Reds')
        self.set_xticks_yticks(ax)
        self.set_xlabel_ylabel(ax)
        return im

    def get_data_mat(self, big_k_mat):
        data_mat = np.zeros((self.n_atom_j, self.n_atom_i))
        #data_mat = np.random.rand(self.n_atom_j, self.n_atom_i)
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

    def get_atomlst(self):
        basetype_1 = self.d_base_stack_types[self.host][self.strand_id][0]
        basetype_2 = self.d_base_stack_types[self.host][self.strand_id][1]
        atomlst_i = self.d_atomlist[basetype_1]
        atomlst_j = self.d_atomlist[basetype_2]
        return atomlst_i, atomlst_j

    def get_map_idx_from_strand_resid_atomname(self):
        d_result = dict()
        for node_name in self.node_list:
            idx = self.d_idx[node_name]
            strand_id = self.strandid_map[node_name]
            resid = self.resid_map[node_name]
            atomname = self.atomname_map[node_name]
            d_result[(strand_id, resid, atomname)] = idx
        return d_result