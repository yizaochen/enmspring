from os import path
from shutil import copyfile
import numpy as np
import pandas as pd
import MDAnalysis
import matplotlib.pyplot as plt
import networkx as nx
from enmspring import pairtype
from enmspring.spring import Spring
from enmspring.k_b0_util import get_df_by_filter_st, get_df_by_filter_PP, get_df_by_filter_R, get_df_by_filter_RB, get_df_by_filter_PB
from enmspring.hb_util import HBAgent
from enmspring.na_seq import sequences
from enmspring.networkx_display import THY_Base, CYT_Base, ADE_Base, GUA_Base, THY_Right_Base, CYT_Right_Base, ADE_Right_Base, GUA_Right_Base

hosts = ['a_tract_21mer', 'gcgc_21mer', 'tgtg_21mer',
         'atat_21mer', 'ctct_21mer', 'g_tract_21mer']

class GraphAgent:
    type_na = 'bdna+bdna'
    n_bp = 21
    cutoff = 4.7

    def __init__(self, host, rootfolder):
        self.host = host
        self.rootfolder = rootfolder

        self.host_folder = path.join(rootfolder, host)
        self.na_folder = path.join(self.host_folder, self.type_na)
        self.input_folder = path.join(self.na_folder, 'input')

        self.spring_obj = Spring(self.rootfolder, self.host, self.type_na, self.n_bp)
        self.df_all_k = self.spring_obj.read_k_b0_pairtype_df_given_cutoff(self.cutoff)

        self.crd = path.join(self.input_folder, '{0}.nohydrogen.avg.crd'.format(self.type_na))
        self.npt4_crd = path.join(self.input_folder, '{0}.nohydrogen.crd'.format(self.type_na))
        self.u = MDAnalysis.Universe(self.crd, self.crd)
        self.map, self.inverse_map, self.residues_map, self.atomid_map,\
        self.atomid_map_inverse, self.atomname_map, self.strandid_map,\
        self.resid_map, self.mass_map = self.build_map()

        self.node_list = None
        self.d_idx = None
        self.n_node = None
        self.adjacency_mat = None
        self.degree_mat = None
        self.laplacian_mat = None
        self.b0_mat = None

        self.w = None  # Eigenvalue array
        self.v = None  # Eigenvector matrix, the i-th column is the i-th eigenvector
        self.strand1_array = list() # 0: STRAND1, 1: STRAND2
        self.strand2_array = list() #
        self.strand1_benchmark = None
        self.strand2_benchmark = None

        self.d_seq = {'STRAND1': sequences[host]['guide'], 'STRAND2': sequences[host]['target']}
        
    def build_node_list(self):
        node_list = list()
        d_idx = dict()
        idx = 0
        for cgname, atomname in self.atomname_map.items():
            atom_type = pairtype.d_atomcgtype[atomname]
            if atom_type == 'B':
                node_list.append(cgname)
                d_idx[cgname] = idx
                idx += 1
        self.node_list = node_list
        self.d_idx = d_idx
        self.n_node = len(self.node_list)
        print(f"Thare are {self.n_node} nodes.")

    def initialize_three_mat(self):
        self.adjacency_mat = np.zeros((self.n_node, self.n_node))
        self.degree_mat = np.zeros((self.n_node, self.n_node))
        self.laplacian_mat = np.zeros((self.n_node, self.n_node))
        self.b0_mat = np.zeros((self.n_node, self.n_node))
        print('Initialize adjacency, degree and Laplacian matrices... Done.')

    def build_degree_from_adjacency(self):
        for idx in range(self.n_node):
            self.degree_mat[idx, idx] = self.adjacency_mat[idx, :].sum()

    def build_laplacian_by_adjacency_degree(self):
        self.laplacian_mat = self.degree_mat + self.adjacency_mat
        print("Finish the setup for Laplaican matrix.")

    def get_networkx_graph(self, df, key='k'):
        # key: 'k', 'b0'
        node1_list = df['Atomid_i'].tolist()
        node2_list = df['Atomid_j'].tolist()
        weight_list = df[key].tolist()
        edges_list = [(node1, node2, {'weight': weight}) for node1, node2, weight in zip(node1_list, node2_list, weight_list)]
        G = nx.Graph()
        G.add_nodes_from(self.get_node_list_by_id())
        G.add_edges_from(edges_list)
        return G

    def get_node_list_by_id(self):
        return [self.atomid_map[name] for name in self.node_list]

    def get_networkx_d_pos(self, radius, dist_bw_base, dist_bw_strand):
        d_atcg = {'A': {'STRAND1': ADE_Base, 'STRAND2': ADE_Right_Base}, 
                  'T': {'STRAND1': THY_Base, 'STRAND2': THY_Right_Base}, 
                  'C': {'STRAND1': CYT_Base, 'STRAND2': CYT_Right_Base}, 
                  'G': {'STRAND1': GUA_Base, 'STRAND2': GUA_Right_Base}
                  }
        d_strandid_resid = self.get_d_strandid_resid()
        d_pos = dict()
        x_move = 0
        y_move = 0
        for strand_id in ['STRAND1', 'STRAND2']:
            for resid in range(1, self.n_bp+1):
                resname = self.d_seq[strand_id][resid-1]
                nucleobase = d_atcg[resname][strand_id](radius)
                nucleobase.translate_xy(x_move, y_move)
                for name in d_strandid_resid[strand_id][resid]:
                    atomid = self.atomid_map[name]
                    atomname = self.atomname_map[name]
                    d_pos[atomid] = nucleobase.d_nodes[atomname]
                if strand_id == 'STRAND1' and (resid != self.n_bp):
                    y_move += dist_bw_base
                elif (strand_id == 'STRAND1') and (resid == self.n_bp):
                    y_move -= 0
                else:
                    y_move -= dist_bw_base
            x_move -= dist_bw_strand
        return d_pos

    def get_d_strandid_resid(self):
        d_strandid_resid = self.initialize_d_strandid_resid()
        for name in self.node_list:
            strandid = self.strandid_map[name]
            resid = self.resid_map[name]
            d_strandid_resid[strandid][resid].append(name)
        return d_strandid_resid

    def initialize_d_strandid_resid(self):
        d_strandid_resid = dict()
        for strand_id in ['STRAND1', 'STRAND2']:
            d_strandid_resid[strand_id] = dict()
            for resid in range(1, self.n_bp+1):
                d_strandid_resid[strand_id][resid] = list()
        return d_strandid_resid

    def get_D_by_atomname_strandid(self, sele_name, sele_strandid):
        sele_resid_list = list(range(4, 19))
        sele_idx_list = list()
        for idx, name in enumerate(self.node_list):
            if (self.atomname_map[name] == sele_name) and (self.strandid_map[name] == sele_strandid) and (self.resid_map[name] in sele_resid_list):
                sele_idx_list.append(idx)
        sele_D = np.zeros((self.n_node, self.n_node))
        for idx in sele_idx_list:
            sele_D[idx, idx] = self.degree_mat[idx, idx]
        return sele_D

    def get_D_by_atomname_strandid_resname(self, sele_name, sele_strandid, sele_resname):
        sele_resid_list = self.get_sele_resid_list_by_resname(sele_resname, sele_strandid)
        sele_idx_list = list()
        for idx, name in enumerate(self.node_list):
            if (self.atomname_map[name] == sele_name) and (self.strandid_map[name] == sele_strandid) and (self.resid_map[name] in sele_resid_list):
                sele_idx_list.append(idx)
        sele_D = np.zeros((self.n_node, self.n_node))
        for idx in sele_idx_list:
            sele_D[idx, idx] = self.degree_mat[idx, idx]
        return sele_D

    def get_sele_resid_list_by_resname(self, resname, strandid):
        sele_resid_list = list()
        central_resids = list(range(4, 19))
        #central_resids = list(range(1, 22))
        for idx, nt_name in enumerate(self.d_seq[strandid]):
            resid = idx + 1
            if (resid in central_resids) and (nt_name == resname):
                sele_resid_list.append(resid)
        return sele_resid_list

    def get_A_by_atomname1_atomname2(self, atomname_i, atomname_j, sele_strandid):
        sele_idx_list = list()
        for resid_i in range(4, 18):
            resid_j = resid_i + 1
            idx_i = self.d_idx[self.map[self.get_key_by_atomname_resid_strandid(atomname_i, resid_i, sele_strandid)]]
            idx_j = self.d_idx[self.map[self.get_key_by_atomname_resid_strandid(atomname_j, resid_j, sele_strandid)]]
            sele_idx_list.append((idx_i, idx_j))
        sele_A = np.zeros((self.n_node, self.n_node))
        for idx_i, idx_j in sele_idx_list:
            sele_A[idx_i, idx_j] = self.adjacency_mat[idx_i, idx_j]
        i_lower = np.tril_indices(self.n_node, -1)
        sele_A[i_lower] = sele_A.transpose()[i_lower]  # make the matrix symmetric
        return sele_A

    def get_A_by_atomname1_atomname2_by_resnames(self, atomname_i, atomname_j, resname_i, resname_j, sele_strandid):
        sele_idx_list = list()
        resid_i_list, resid_j_list = self.get_resid_i_resid_j_list(resname_i, resname_j, sele_strandid)
        for resid_i, resid_j in zip(resid_i_list, resid_j_list):
            idx_i = self.d_idx[self.map[self.get_key_by_atomname_resid_strandid(atomname_i, resid_i, sele_strandid)]]
            idx_j = self.d_idx[self.map[self.get_key_by_atomname_resid_strandid(atomname_j, resid_j, sele_strandid)]]
            sele_idx_list.append((idx_i, idx_j))
        sele_A = np.zeros((self.n_node, self.n_node))
        for idx_i, idx_j in sele_idx_list:
            sele_A[idx_i, idx_j] = self.adjacency_mat[idx_i, idx_j]
        i_lower = np.tril_indices(self.n_node, -1)
        sele_A[i_lower] = sele_A.transpose()[i_lower]  # make the matrix symmetric
        return sele_A

    def get_resid_i_resid_j_list(self, resname_i, resname_j, sele_strandid):
        seq = self.d_seq[sele_strandid]
        central_resids = range(4, 19)
        resid_i_list = list()
        resid_j_list = list()
        for resid in central_resids:
            if (seq[resid-1] == resname_i) and (seq[resid] == resname_j):
                resid_i_list.append(resid)
                resid_j_list.append(resid+1)
        return resid_i_list, resid_j_list

    def get_atomidpairs_atomname1_atomname2(self, atomname_i, atomname_j, sele_strandid):
        atomidpairs = list()
        for resid_i in range(4, 18):
            resid_j = resid_i + 1
            idx_i = self.atomid_map[self.map[self.get_key_by_atomname_resid_strandid(atomname_i, resid_i, sele_strandid)]]
            idx_j = self.atomid_map[self.map[self.get_key_by_atomname_resid_strandid(atomname_j, resid_j, sele_strandid)]]
            atomidpairs.append((idx_i, idx_j))
        return atomidpairs

    def get_key_by_atomname_resid_strandid(self, atomname, resid, strandid):
        return f'segid {strandid} and resid {resid} and name {atomname}'

    def get_filter_by_atomname_strandid(self, sele_name, sele_strandid):
        sele_resid_list = list(range(4, 19))
        sele_idx_list = list()
        for idx, name in enumerate(self.node_list):
            if (self.atomname_map[name] == sele_name) and (self.strandid_map[name] == sele_strandid) and (self.resid_map[name] in sele_resid_list):
                sele_idx_list.append(idx)

        y = np.zeros(self.n_node)
        y[sele_idx_list] = 1
        return y / np.linalg.norm(y)

    def get_filter_by_atomname_for_YR(self, sele_name, sele_resname, sele_strandid):
        sele_resid_list = list(range(4, 19))
        sele_idx_list = list()
        for idx, name in enumerate(self.node_list):
            resid = self.resid_map[name]
            if resid not in sele_resid_list:
                continue
            strandid = self.strandid_map[name]
            if strandid != sele_strandid:
                continue
            resname = self.d_seq[strandid][resid-1]
            if resname != sele_resname:
                continue
            if self.atomname_map[name] == sele_name:
                sele_idx_list.append(idx)
        y = np.zeros(self.n_node)
        y[sele_idx_list] = 1
        return y / np.linalg.norm(y)

    def eigen_decompose(self):
        w, v = np.linalg.eig(self.laplacian_mat)
        idx = w.argsort()[::-1] # sort from big to small
        self.w = w[idx]
        self.v = v[:, idx]

    def get_eigenvalue_by_id(self, sele_id):
        return self.w[sele_id-1]

    def get_eigenvector_by_id(self, sele_id):
        return self.v[:,sele_id-1]

    def get_qtAq(self, sele_id):
        eigvector_sele = self.get_eigenvector_by_id(sele_id)
        return np.dot(eigvector_sele.T, np.dot(self.adjacency_mat, eigvector_sele))

    def get_qtDq(self, sele_id):
        eigvector_sele = self.get_eigenvector_by_id(sele_id)
        return np.dot(eigvector_sele.T, np.dot(self.degree_mat, eigvector_sele))

    def get_qtMq(self, sele_id, M):
        ### M is customized matrix
        eigvector_sele = self.get_eigenvector_by_id(sele_id)
        return np.dot(eigvector_sele.T, np.dot(M, eigvector_sele))

    def vmd_show_crd(self):
        print(f'vmd -cor {self.npt4_crd}')

    def copy_nohydrogen_crd(self):
        allsys_root = '/home/yizaochen/codes/dna_rna/all_systems'
        srt = path.join(allsys_root, self.host, self.type_na, 'input', 'heavyatoms', f'{self.type_na}.nohydrogen.crd')
        dst = self.npt4_crd
        copyfile(srt, dst)
        print(f'cp {srt} {dst}')

    def decide_eigenvector_strand(self, eigv_id):
        eigv = self.get_eigenvector_by_id(eigv_id)
        dot_product = np.dot(eigv, self.strand1_benchmark)
        if np.isclose(dot_product, 0.):
            return True #'STRAND2'
        else:
            return False #'STRAND1'

    def set_strand_array(self):
        for eigv_id in range(1, self.n_node+1):
            if self.decide_eigenvector_strand(eigv_id):
                self.strand2_array.append(eigv_id)
            else:
                self.strand1_array.append(eigv_id)
        print(f'Total number of nodes: {self.n_node}')
        print(f'There are {len(self.strand1_array)} eigenvectors belonging to STRAND1.')
        print(f'There are {len(self.strand2_array)} eigenvectors belonging to STRAND2.')
        print(f'Sum of two strands: {len(self.strand1_array)+len(self.strand2_array)}')

    def get_lambda_by_strand(self, strandid):
        if strandid == 'STRAND1':
            return [self.get_eigenvalue_by_id(eigv_id) for eigv_id in self.strand1_array]
        else:
            return [self.get_eigenvalue_by_id(eigv_id) for eigv_id in self.strand2_array]

    def get_eigvector_by_strand(self, strandid, sele_id):
        if strandid == 'STRAND1':
            real_eigv_id = self.strand1_array[sele_id]
        else:
            real_eigv_id = self.strand2_array[sele_id]
        return self.get_eigenvector_by_id(real_eigv_id), self.get_eigenvalue_by_id(real_eigv_id)

    def set_adjacency_by_df(self, df_sele):
        idx_i_list = self.__get_idx_list(df_sele['Atomid_i'])
        idx_j_list = self.__get_idx_list(df_sele['Atomid_j'])
        k_list = df_sele['k'].tolist()
        for idx_i, idx_j, k in zip(idx_i_list, idx_j_list, k_list):
            self.adjacency_mat[idx_i, idx_j] = k

    def set_b0_mat_by_df(self, df_sele):
        idx_i_list = self.__get_idx_list(df_sele['Atomid_i'])
        idx_j_list = self.__get_idx_list(df_sele['Atomid_j'])
        b0_list = df_sele['b0'].tolist()
        for idx_i, idx_j, b0 in zip(idx_i_list, idx_j_list, b0_list):
            self.b0_mat[idx_i, idx_j] = b0
        i_lower = np.tril_indices(self.n_node, -1)
        self.b0_mat[i_lower] = self.b0_mat.transpose()[i_lower]  # make the matrix symmetric

    def set_adjacency_by_d(self, d_sele):
        idx_i_list = self.__get_idx_list(d_sele['Atomid_i'])
        idx_j_list = self.__get_idx_list(d_sele['Atomid_j'])
        k_list = d_sele['k']
        for idx_i, idx_j, k in zip(idx_i_list, idx_j_list, k_list):
            self.adjacency_mat[idx_i, idx_j] = k
    
    def make_adjacency_symmetry(self):
        i_lower = np.tril_indices(self.n_node, -1)
        self.adjacency_mat[i_lower] = self.adjacency_mat.transpose()[i_lower]  # make the matrix symmetric

    def write_show_nodes_tcl(self, tcl_out, colorid=0, vdw_radius=1.0):
        serials_str = self.__get_serial_nodes()
        f = open(tcl_out, 'w')
        f.write('display resize 362 954\n\n')
        f.write('mol color ColorID 6\n')
        f.write('mol representation Lines 3.000\n')
        f.write('mol selection all\n')
        f.write('mol material Opaque\n')
        f.write('mol addrep 0\n')
        f.write(f'mol color ColorID {colorid}\n')
        f.write(f'mol representation VDW {vdw_radius:.3f} 12.000\n')
        f.write(f'mol selection serial {serials_str}\n')
        f.write('mol material Opaque\n')
        f.write('mol addrep 0\n')
        f.write(f'mol color ColorID 7\n')
        f.write(f'mol representation VDW 0.300 12.000\n')
        f.write(f'mol selection serial 6 7 8 9\n')
        f.write('mol material Opaque\n')
        f.write('mol addrep 0\n')
        f.close()
        print(f'Write tcl to {tcl_out}')
        print(f'source {tcl_out}')

    def process_lines_for_edges_tcl(self, lines, df_sele, radius=0.05):
        u_npt4 = MDAnalysis.Universe(self.npt4_crd, self.npt4_crd)       
        for atomid1, atomid2 in zip(df_sele['Atomid_i'], df_sele['Atomid_j']):
            line = self.__get_draw_edge_line(u_npt4.atoms.positions, atomid1-1, atomid2-1, radius)
            lines.append(line)
        return lines

    def write_lines_to_tcl_out(self, lines, tcl_out):
        f = open(tcl_out, 'w')        
        for line in lines:
            f.write(line)
        f.close()
        print(f'Write tcl to {tcl_out}')
        print(f'source {tcl_out}')
        
    def __get_idx_list(self, df_column):
        cgname_list = [self.atomid_map_inverse[atomid] for atomid in df_column]
        return [self.d_idx[cgname] for cgname in cgname_list]

    def __get_serial_nodes(self):
        serials_list = [str(self.atomid_map[cgname]) for cgname in self.d_idx.keys()]
        return ' '.join(serials_list)

    def __get_draw_edge_line(self, positions, atomid1, atomid2, radius):
        str_0 = 'graphics 0 cylinder {'
        str_1 = f'{positions[atomid1,0]:.3f} {positions[atomid1,1]:.3f} {positions[atomid1,2]:.3f}'
        str_2 = '} {'
        str_3 = f'{positions[atomid2,0]:.3f} {positions[atomid2,1]:.3f} {positions[atomid2,2]:.3f}'
        str_4 = '} '
        str_5 = f'radius {radius:.2f}\n'
        return str_0 + str_1 + str_2 + str_3 + str_4 + str_5
       
    def build_map(self):
        d1 = dict()  # key: selction, value: cgname
        d2 = dict()  # key: cgname,   value: selection
        d3 = dict()
        d4 = dict()  # key: cgname, value: atomid
        d5 = dict()  # key: atomid, value: cgname
        d6 = dict()  # key: cgname, value: atomname
        d7 = dict()  # key: cgname, value: strand_id
        d8 = dict()  # key: cgname, value: resid
        d9 = dict()  # key: cgname, value: mass
        atomid = 1
        segid1 = self.u.select_atoms("segid STRAND1")
        d3['STRAND1'] = dict()
        for i, atom in enumerate(segid1):
            cgname = 'A{0}'.format(i+1)
            selection = self.__get_selection(atom)
            d1[selection] = cgname
            d2[cgname] = selection
            if atom.resid not in d3['STRAND1']:
                d3['STRAND1'][atom.resid] = list()
            d3['STRAND1'][atom.resid].append(cgname)
            d4[cgname] = atomid
            d5[atomid] = cgname
            d6[cgname] = atom.name
            d7[cgname] = 'STRAND1'
            d8[cgname] = atom.resid
            d9[cgname] = atom.mass
            atomid += 1
        segid2 = self.u.select_atoms("segid STRAND2")
        d3['STRAND2'] = dict()
        for i, atom in enumerate(segid2):
            cgname = 'B{0}'.format(i+1)
            selection = self.__get_selection(atom)
            d1[selection] = cgname
            d2[cgname] = selection
            if atom.resid not in d3['STRAND2']:
                d3['STRAND2'][atom.resid] = list()
            d3['STRAND2'][atom.resid].append(cgname)
            d4[cgname] = atomid
            d5[atomid] = cgname
            d6[cgname] = atom.name
            d7[cgname] = 'STRAND2'
            d8[cgname] = atom.resid
            d9[cgname] = atom.mass
            atomid += 1
        return d1, d2, d3, d4, d5, d6, d7, d8, d9
        
    def __get_selection(self, atom):
        return 'segid {0} and resid {1} and name {2}'.format(atom.segid, atom.resid, atom.name)

class Stack(GraphAgent):

    def __init__(self, host, rootfolder):
        super().__init__(host, rootfolder)
        self.df_st = self.read_df_st()

    def pre_process(self):
        self.build_node_list()
        self.initialize_three_mat()
        self.build_adjacency_from_df_st()
        self.build_degree_from_adjacency()
        self.build_laplacian_by_adjacency_degree()
        self.eigen_decompose()
        self.set_benchmark_array()
        self.set_strand_array()

    def build_adjacency_from_df_st(self):
        self.set_adjacency_by_df(self.df_st)
        self.make_adjacency_symmetry()

    def set_benchmark_array(self):
        idx_start_strand2 = self.d_idx['B6']
        strand1 = np.zeros(self.n_node)
        strand2 = np.zeros(self.n_node)
        strand1[:idx_start_strand2] = 1.
        strand2[idx_start_strand2:] = 1.
        self.strand1_benchmark = strand1
        self.strand2_benchmark = strand2

    def write_show_base_edges_tcl(self, tcl_out, radius=0.05):
        lines = ['graphics 0 color 1\n', 'graphics 0 material AOShiny\n']
        lines = self.process_lines_for_edges_tcl(lines, self.df_st, radius=radius)
        self.write_lines_to_tcl_out(lines, tcl_out)

    def get_df_qTAq_for_vmd_draw(self, eigv_id, strandid):
        df = self.df_st
        columns_qTAq = ['Strand_i', 'Resid_i', 'Atomname_i', 'Strand_j', 'Resid_j', 'Atomname_j']
        d_qTAq = {col_name: df[col_name].tolist() for col_name in columns_qTAq}
        d_qTAq['qTAq'] = np.zeros(df.shape[0])
        q = self.get_eigvector_by_strand(strandid, eigv_id)[0]
        for idx, atomids in enumerate(zip(df['Atomid_i'], df['Atomid_j'])):
            atomid_i , atomid_j = atomids
            A = self.get_sele_A_by_idx(atomid_i, atomid_j)
            d_qTAq['qTAq'][idx] = np.dot(q.T, np.dot(A, q))
        df_result = pd.DataFrame(d_qTAq)
        columns_qTAq.append('qTAq')
        return df_result[columns_qTAq]

    def get_sele_A_by_idx(self, atomid_i, atomid_j):
        sele_A = np.zeros((self.n_node, self.n_node))
        idx_i = self.d_idx[self.atomid_map_inverse[atomid_i]]
        idx_j = self.d_idx[self.atomid_map_inverse[atomid_j]]
        sele_A[idx_i, idx_j] = self.adjacency_mat[idx_i, idx_j]
        i_lower = np.tril_indices(self.n_node, -1)
        sele_A[i_lower] = sele_A.transpose()[i_lower]
        return sele_A
        
    def read_df_st(self):
        criteria = 1e-3
        df1 = get_df_by_filter_st(self.df_all_k, 'st')
        mask = (df1['k'] > criteria)
        print("Read Dataframe of stacking: df_st")
        return df1[mask]

class StackHB(Stack):
    def __init__(self, host, rootfolder):
        super().__init__(host, rootfolder)
        self.hb_agent = HBAgent(host, rootfolder, self.n_bp)

    def build_adjacency_from_df_st_df_hb(self):
        self.set_adjacency_by_df(self.df_st)
        d_hb_new = self.hb_agent.get_d_hb_contain_atomid_k_all_basepair()
        self.set_adjacency_by_d(d_hb_new)         
        self.make_adjacency_symmetry()

    def write_show_base_hb_edges_tcl(self, tcl_out, radius=0.05):
        lines = ['graphics 0 color 1\n', 'graphics 0 material AOShiny\n']
        lines = self.process_lines_for_edges_tcl(lines, self.df_st, radius=radius)
        lines += ['graphics 0 color 1\n', 'graphics 0 material AOShiny\n']
        d_hb_new = self.hb_agent.get_d_hb_contain_atomid_k_all_basepair()
        lines = self.process_lines_for_edges_tcl(lines, d_hb_new, radius=radius)
        self.write_lines_to_tcl_out(lines, tcl_out)


class onlyHB(StackHB):
    def pre_process(self):
        self.build_node_list()
        self.initialize_three_mat()
        self.build_adjacency_from_df_hb()
        self.build_degree_from_adjacency()
        self.build_laplacian_by_adjacency_degree()
        self.eigen_decompose()

    def build_adjacency_from_df_hb(self):
        d_hb_new = self.hb_agent.get_d_hb_contain_atomid_k_all_basepair()
        self.set_adjacency_by_d(d_hb_new)         
        self.make_adjacency_symmetry()

    def write_show_base_hb_edges_tcl(self, tcl_out, radius=0.05):
        lines = ['graphics 0 color 1\n', 'graphics 0 material AOShiny\n']
        d_hb_new = self.hb_agent.get_d_hb_contain_atomid_k_all_basepair()
        lines = self.process_lines_for_edges_tcl(lines, d_hb_new, radius=radius)
        self.write_lines_to_tcl_out(lines, tcl_out)

    def get_df_hb_new(self):
        columns = ['Strand_i', 'Resid_i', 'Atomname_i', 'Atomid_i', 'Strand_j', 'Resid_j', 'Atomname_j', 'Atomid_j', 'k']
        d_result = dict()
        d_hb_new = self.hb_agent.get_d_hb_contain_atomid_k_all_basepair()
        cgname_i_list = [self.atomid_map_inverse[atomid_i] for atomid_i in d_hb_new['Atomid_i']]
        cgname_j_list = [self.atomid_map_inverse[atomid_j] for atomid_j in d_hb_new['Atomid_j']]
        d_result['Strand_i'] = [self.strandid_map[cgname_i] for cgname_i in cgname_i_list]
        d_result['Strand_j'] = [self.strandid_map[cgname_j] for cgname_j in cgname_j_list]
        d_result['Resid_i'] = [self.resid_map[cgname_i] for cgname_i in cgname_i_list]
        d_result['Resid_j'] = [self.resid_map[cgname_j] for cgname_j in cgname_j_list]
        d_result['Atomname_i'] = [self.atomname_map[cgname_i] for cgname_i in cgname_i_list]
        d_result['Atomname_j'] = [self.atomname_map[cgname_j] for cgname_j in cgname_j_list]
        d_result['Atomid_i'] = d_hb_new['Atomid_i']
        d_result['Atomid_j'] = d_hb_new['Atomid_j']
        d_result['k'] = d_hb_new['k']

        df_hb_new = pd.DataFrame(d_result)
        criteria = 1e-3
        mask = (df_hb_new['k'] > criteria)
        df_hb_new = df_hb_new[mask]
        return df_hb_new[columns]

    def get_df_qTAq_for_vmd_draw(self, eigv_id):
        df = self.get_df_hb_new()
        columns_qTAq = ['Strand_i', 'Resid_i', 'Atomname_i', 'Strand_j', 'Resid_j', 'Atomname_j']
        d_qTAq = {col_name: df[col_name].tolist() for col_name in columns_qTAq}
        d_qTAq['qTAq'] = np.zeros(df.shape[0])
        q = self.get_eigenvector_by_id(eigv_id)
        for idx, atomids in enumerate(zip(df['Atomid_i'], df['Atomid_j'])):
            atomid_i , atomid_j = atomids
            A = self.get_sele_A_by_idx(atomid_i, atomid_j)
            d_qTAq['qTAq'][idx] = np.dot(q.T, np.dot(A, q))
        df_result = pd.DataFrame(d_qTAq)
        columns_qTAq.append('qTAq')
        return df_result[columns_qTAq]


class BackboneRibose(GraphAgent):
    def pre_process(self):
        self.build_node_list()
        self.initialize_three_mat()
        self.build_adjacency_from_pp_r()
        self.build_degree_from_adjacency()
        self.build_laplacian_by_adjacency_degree()
        self.eigen_decompose()
        self.set_benchmark_array()
        self.set_strand_array()

    def build_node_list(self):
        node_list = list()
        d_idx = dict()
        idx = 0
        for cgname, atomname in self.atomname_map.items():
            atom_type = pairtype.d_atomcgtype[atomname]
            if (atom_type == 'P') or (atom_type == 'S') or (atom_type == 'B'):
                node_list.append(cgname)
                d_idx[cgname] = idx
                idx += 1
        self.node_list = node_list
        self.d_idx = d_idx
        self.n_node = len(self.node_list)
        print(f"Thare are {self.n_node} nodes.")

    def get_df_backbone_ribose(self):
        df_pp_lst = [get_df_by_filter_PP(self.df_all_k, subcategory) for subcategory in ['PP0', 'PP1', 'PP2', 'PP3']]
        df_r_lst = [get_df_by_filter_R(self.df_all_k, subcategory) for subcategory in ['R0', 'R1']]
        df_rb_lst = [get_df_by_filter_RB(self.df_all_k, subcategory) for subcategory in ['RB0', 'RB1', 'RB2', 'RB3']]
        df_pb_lst = [get_df_by_filter_PB(self.df_all_k, subcategory) for subcategory in ['PB']]
        df_pp_r_rb = pd.concat(df_pp_lst+df_r_lst+df_rb_lst+df_pb_lst)
        criteria = 1e-1
        mask = (df_pp_r_rb['k'] > criteria)
        return df_pp_r_rb[mask]

    def build_adjacency_from_pp_r(self):
        df_sele = self.get_df_backbone_ribose()
        self.set_adjacency_by_df(df_sele)
        self.make_adjacency_symmetry()
        self.set_b0_mat_by_df(df_sele)

    def get_sele_A_by_idx(self, atomid_i, atomid_j):
        sele_A = np.zeros((self.n_node, self.n_node))
        idx_i = self.d_idx[self.atomid_map_inverse[atomid_i]]
        idx_j = self.d_idx[self.atomid_map_inverse[atomid_j]]
        sele_A[idx_i, idx_j] = self.adjacency_mat[idx_i, idx_j]
        i_lower = np.tril_indices(self.n_node, -1)
        sele_A[i_lower] = sele_A.transpose()[i_lower]
        return sele_A

    def get_df_qTAq_for_vmd_draw(self, eigv_id, strandid):
        df = self.get_df_backbone_ribose()
        columns_qTAq = ['Strand_i', 'Resid_i', 'Atomname_i', 'Strand_j', 'Resid_j', 'Atomname_j']
        d_qTAq = {col_name: df[col_name].tolist() for col_name in columns_qTAq}
        d_qTAq['qTAq'] = np.zeros(df.shape[0])
        q = self.get_eigvector_by_strand(strandid, eigv_id)[0]
        for idx, atomids in enumerate(zip(df['Atomid_i'], df['Atomid_j'])):
            atomid_i , atomid_j = atomids
            A = self.get_sele_A_by_idx(atomid_i, atomid_j)
            d_qTAq['qTAq'][idx] = np.dot(q.T, np.dot(A, q))
        df_result = pd.DataFrame(d_qTAq)
        columns_qTAq.append('qTAq')
        return df_result[columns_qTAq]

    def set_benchmark_array(self):
        idx_start_strand2 = self.d_idx['B1']
        strand1 = np.zeros(self.n_node)
        strand2 = np.zeros(self.n_node)
        strand1[:idx_start_strand2] = 1.
        strand2[idx_start_strand2:] = 1.
        self.strand1_benchmark = strand1
        self.strand2_benchmark = strand2

    def write_show_backbone_edges_tcl(self, tcl_out, radius=0.05):
        lines = ['graphics 0 color 1\n', 'graphics 0 material AOShiny\n']
        for subcategory in ['PP0', 'PP1', 'PP2', 'PP3']:
            df_sele = get_df_by_filter_PP(self.df_all_k, subcategory)
            lines = self.process_lines_for_edges_tcl(lines, df_sele, radius=radius)
        for subcategory in ['R0', 'R1']:
            df_sele = get_df_by_filter_R(self.df_all_k, subcategory)
            lines = self.process_lines_for_edges_tcl(lines, df_sele, radius=radius)
        self.write_lines_to_tcl_out(lines, tcl_out)

class EigenPlot:
    d_groups = {0: ('a_tract_21mer', 'g_tract_21mer'),
                1: ('atat_21mer', 'gcgc_21mer'),
                2: ('ctct_21mer', 'tgtg_21mer')}
    d_colors = {'a_tract_21mer': 'b', 'g_tract_21mer': 'r',
                'atat_21mer': 'g', 'gcgc_21mer': 'orange',
                'ctct_21mer': 'c', 'tgtg_21mer': 'm'}
    d_labels = {'a_tract_21mer': ('A-Tract: (AA)', 'A-Tract: (TT)'), 'g_tract_21mer': ('G-Tract: (GG)', 'G-Tract: (CC)'),
                'atat_21mer': ('AT: (AT)', 'AT: (AT)'), 'gcgc_21mer':  ('GC: (GC)', 'GC: (GC)'),
                'ctct_21mer':  ('CT: (CT)', 'CT: (GA)'), 'tgtg_21mer': ('TG: (TG)', 'TG: (AC)')}
    strandids = ['STRAND1', 'STRAND2']

    def __init__(self, rootfolder):
        self.rootfolder = rootfolder
        self.d_agent = dict()
        self.d_eigenvalues = dict()
   
    def plot_lambda_six_together(self, figsize):
        fig, ax = plt.subplots(figsize=figsize)
        for host in hosts:
            agent = self.d_agent[host]
            x = range(1, agent.n_node+1)
            y = agent.w
            ax.plot(x, y, '-o', label=host)
        ax.legend()
        ax.set_xlabel("Mode ID")
        ax.set_ylabel("Eigenvalue")
        return fig, ax

    def plot_lambda_separate_strand(self, figsize):
        ncols = 3
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, sharey=True)
        for ax_id in range(ncols):
            ax = axes[ax_id]
            host1 = self.d_groups[ax_id][0]
            host2 = self.d_groups[ax_id][1]
            self.ax_plot_lambda(ax, host1)
            self.ax_plot_lambda(ax, host2)
            self.ax_plot_assistline(ax)
            ax.legend(frameon=False)
            ax.set_xlabel("Eigenvalue ID")
            if ax_id == 0:
                ax.set_ylabel("Eigenvalue")
        return fig, ax

    def ax_plot_lambda(self, ax, host):
        x1, y1, x2, y2 = self.get_lambda_array_by_host(host)
        ax.plot(x1, y1, ls='-', color=self.d_colors[host], alpha=0.7, label=self.d_labels[host][0])
        ax.plot(x2, y2, ls='-.', color=self.d_colors[host], alpha=0.7, label=self.d_labels[host][1])

    def ax_plot_assistline(self, ax):
        for y in [0, 5, 10, 15, 20]:
            ax.axhline(y, color='grey',alpha=0.15)

    def get_lambda_array_by_host(self, host):
        y1 = self.d_agent[host].get_lambda_by_strand('STRAND1')
        y2 = self.d_agent[host].get_lambda_by_strand('STRAND2')
        x1 = range(1, len(y1)+1)
        x2 = range(1, len(y2)+1)
        return x1, y1, x2, y2
        
    def plot_eigenvector(self, figsize, hspace, wspace, eigv_id_list, lw):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(6, 6, hspace=hspace, wspace=wspace)
        d_axes = self.__get_d_axes(fig, gs)
        for col_id, host in enumerate(hosts):
            g_agent = self.d_agent[host]
            x = range(1, g_agent.n_node+1)
            for row_id, eigv_id in enumerate(eigv_id_list):
                ax = d_axes[host][row_id]
                eigvalue = g_agent.get_eigenvalue_by_id(eigv_id)
                y = g_agent.get_eigenvector_by_id(eigv_id)
                ax.vlines(x, 0, y, colors='b', lw=lw)
                ax.set_xlim(1, g_agent.n_node+1)
                if col_id == 0:
                    ax.set_ylabel(r'$e_{' + f'{eigv_id}' +r'}$', fontsize=16)

                if row_id == 0:
                    title = f'{host} ' + r'$\lambda_{' + f'{eigv_id}' + r'}=$' + f'{eigvalue:.2f}'
                else:
                    title = r'$\lambda_{' + f'{eigv_id}' + r'}=$' + f'{eigvalue:.2f}'
                ax.set_title(title, fontsize=10)

                if row_id == 5:
                    ax.set_xlabel('CG-bead ID')
                else:
                    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        gs.tight_layout(fig)        
        return fig, d_axes

    def plot_eigenvector_separate_strand(self, figsize, hspace, wspace, groupid, lw):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(5, 4, hspace=hspace, wspace=wspace)
        d_axes = self.__get_d_axes_by_groupid(fig, gs, groupid)
        col_id = 0
        for host in self.d_groups[groupid]:
            g_agent = self.d_agent[host]
            x = range(1, g_agent.n_node+1)
            for strand_id_int, strand_id in enumerate(self.strandids):
                for row_id in range(5):
                    ax = d_axes[host][strand_id][row_id]
                    y, eigvalue = self.__get_eigv_array_by_host(host, row_id, strand_id)
                    ax.vlines(x, 0, y, colors='b', lw=lw)
                    ax.set_xlim(1, g_agent.n_node+1)
                    title = self.d_labels[host][strand_id_int] + r'  $\lambda=$' + f'{eigvalue:.2f}'
                    ax.set_title(title, fontsize=12)
                    if col_id == 0:
                        ax.set_ylabel('Eigenvector', fontsize=12)
                    if row_id == 4:
                        ax.set_xlabel('CG-bead ID')
                    else:
                        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                col_id += 1
        gs.tight_layout(fig)     
        return fig, d_axes

    def __get_d_axes(self, fig, gs):
        d_axes = {host: list() for host in hosts}
        for row_id in range(6):
            for col_id, host in enumerate(hosts):
                d_axes[host].append(fig.add_subplot(gs[row_id,col_id]))
        return d_axes

    def __get_d_axes_by_groupid(self, fig, gs, groupid):
        d_axes = dict()
        col_id = 0
        for host in self.d_groups[groupid]:
            d_axes[host] = dict()
            for strand_id in self.strandids:
                d_axes[host][strand_id] = list()
                for row_id in range(5):
                    d_axes[host][strand_id].append(fig.add_subplot(gs[row_id,col_id]))
                col_id += 1
        return d_axes

    def __get_eigv_array_by_host(self, host, sele_id, strandid):
        return self.d_agent[host].get_eigvector_by_strand(strandid, sele_id)

class EigenPlotStack(EigenPlot):
    def initailize_six_systems(self):
        for host in hosts:
            g_agent = Stack(host, self.rootfolder)
            g_agent.build_node_list()
            g_agent.initialize_three_mat()
            g_agent.build_adjacency_from_df_st()
            g_agent.build_degree_from_adjacency()
            g_agent.build_laplacian_by_adjacency_degree()
            g_agent.set_benchmark_array()
            g_agent.eigen_decompose()
            g_agent.set_strand_array()
            self.d_agent[host] = g_agent

class EigenPlotStackHB(EigenPlot):
    def initailize_six_systems(self):
        for host in hosts:
            g_agent = StackHB(host, self.rootfolder)
            g_agent.build_node_list()
            g_agent.initialize_three_mat()
            g_agent.build_adjacency_from_df_st_df_hb()
            g_agent.build_degree_from_adjacency()
            g_agent.build_laplacian_by_adjacency_degree()
            g_agent.eigen_decompose()
            g_agent.set_strand_array()
            self.d_agent[host] = g_agent

class EigenPlotHB(EigenPlot):
    def initailize_six_systems(self):
        for host in hosts:
            g_agent = onlyHB(host, self.rootfolder)
            g_agent.build_node_list()
            g_agent.initialize_three_mat()
            g_agent.build_adjacency_from_df_hb()
            g_agent.build_degree_from_adjacency()
            g_agent.build_laplacian_by_adjacency_degree()
            g_agent.eigen_decompose()
            self.d_agent[host] = g_agent

class EigenPlotBackboneRibose(EigenPlot):
    def initailize_six_systems(self):
        for host in hosts:
            g_agent = BackboneRibose(host, self.rootfolder)
            g_agent.build_node_list()
            g_agent.initialize_three_mat()
            g_agent.build_adjacency_from_pp_r()
            g_agent.build_degree_from_adjacency()
            g_agent.build_laplacian_by_adjacency_degree()
            g_agent.eigen_decompose()
            g_agent.set_benchmark_array()
            g_agent.set_strand_array()
            self.d_agent[host] = g_agent

    def plot_lambda_separate_strand(self, figsize):
        ncols = 3
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=figsize, sharey=True)
        for ax_id in range(ncols):
            ax = axes[ax_id]
            host1 = self.d_groups[ax_id][0]
            host2 = self.d_groups[ax_id][1]
            self.ax_plot_lambda(ax, host1)
            self.ax_plot_lambda(ax, host2)
            self.ax_plot_assistline(ax)
            ax.legend(frameon=False)
            ax.set_xlabel("Eigenvalue ID")
            if ax_id == 0:
                ax.set_ylabel("Eigenvalue")
        return fig, ax

    def ax_plot_assistline(self, ax):
        for y in range(100, 801, 100):
            ax.axhline(y, color='grey',alpha=0.15)


class StackHBCoupling:
    abbr_hosts = {'a_tract_21mer': 'A-tract', 'ctct_21mer': 'CTCT', 'gcgc_21mer': 'GCGC',
                  'g_tract_21mer': 'G-tract', 'atat_21mer': 'ATAT', 'tgtg_21mer': 'TGTG'}

    def __init__(self, rootfolder, host):
        self.rootfolder = rootfolder
        self.host = host
        self.abbrhost = self.abbr_hosts[host]

        self.stack = Stack(host, rootfolder)
        self.hb = onlyHB(host, rootfolder)
        self.stackhb = StackHB(host, rootfolder)

        self.n_eigenvector = 20
        self.eigenlist = list(range(1, self.n_eigenvector+1))

    def initialize_eigenvectors(self):
        self.initialize_stack()
        self.initialize_hb()
        self.initialize_stackhb()

    def initialize_stack(self):
        self.stack.build_node_list()
        self.stack.initialize_three_mat()
        self.stack.build_adjacency_from_df_st()
        self.stack.build_degree_from_adjacency()
        self.stack.build_laplacian_by_adjacency_degree()
        self.stack.eigen_decompose()

    def initialize_hb(self):
        self.hb.build_node_list()
        self.hb.initialize_three_mat()
        self.hb.build_adjacency_from_df_hb()
        self.hb.build_degree_from_adjacency()
        self.hb.build_laplacian_by_adjacency_degree()
        self.hb.eigen_decompose() 

    def initialize_stackhb(self):
        self.stackhb.build_node_list()
        self.stackhb.initialize_three_mat()
        self.stackhb.build_adjacency_from_df_st_df_hb()
        self.stackhb.build_degree_from_adjacency()
        self.stackhb.build_laplacian_by_adjacency_degree()
        self.stackhb.eigen_decompose()

    def plot_main(self, figsize, hspace, wspace, width):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(5, 4, hspace=hspace, wspace=wspace)
        d_axes = self.get_d_axes(fig, gs)
        for eigid in self.eigenlist:
            ax = d_axes[eigid]
            xlist_stack, xlist_hb, xticks = self.get_xlist()
            ylist_stack, ylist_hb = self.get_ylist(eigid)
            xticklabels = self.get_xticklabels()
            ax.bar(xlist_stack, ylist_stack, width, color='blue', label='Stack')
            ax.bar(xlist_hb, ylist_hb, width, color='red', label='HB')
            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels)
            if eigid == 1:
                ax.set_title(f'{self.abbrhost} $j$ = Mode {eigid} of Stack-HB')
            else:
                ax.set_title(f'$j$ = Mode {eigid} of Stack-HB')
            if eigid == 1:
                ax.legend()
            if eigid in [1, 5, 9, 13, 17]:
                ax.set_ylabel(r'$|e_{j}^{S+H} \cdot e_{i}^{S,H}|$')
            if eigid in [17, 18, 19, 20]:
                ax.set_xlabel(r'$i$')
        gs.tight_layout(fig)        
        return fig, d_axes

    def get_d_axes(self, fig, gs):
        d_axes = dict()
        eigid = 1
        for row_id in range(5):
            for col_id in range(4):
                d_axes[eigid] = fig.add_subplot(gs[row_id,col_id])
                eigid += 1
        return d_axes

    def get_ylist(self, eigid):
        eig_stackhb = self.stackhb.get_eigenvector_by_id(eigid)
        ylist_stack = np.zeros(self.n_eigenvector)
        ylist_hb = np.zeros(self.n_eigenvector)
        for tempeigid in self.eigenlist:
            eig_stack = self.stack.get_eigenvector_by_id(tempeigid)
            dotproduct = np.dot(eig_stackhb, eig_stack)
            ylist_stack[tempeigid-1] = np.abs(dotproduct)
        for tempeigid in self.eigenlist:
            eig_hb = self.hb.get_eigenvector_by_id(tempeigid)
            dotproduct = np.dot(eig_stackhb, eig_hb)
            ylist_hb[tempeigid-1] = np.abs(dotproduct)
        return ylist_stack, ylist_hb

    def get_xlist(self):
        start = 1
        end = start + self.n_eigenvector
        xlist_stack = list(range(start, end))

        start = end + 1
        end = start + self.n_eigenvector
        xlist_hb = list(range(start, end))

        xticks = xlist_stack[::2] + xlist_hb[::2]
        return xlist_stack, xlist_hb, xticks

    def get_xticklabels(self):
        start = 1
        end = start + self.n_eigenvector
        temp = [f'{i}' for i in range(start,end)]
        return temp[::2] + temp[::2]


class StackHBCommonNodes(StackHBCoupling):

    def __init__(self, rootfolder, host):
        super().__init__(rootfolder, host)
        self.n_eigenvector = 40
        self.eigenlist = list(range(1, 21))

    def plot_main(self, figsize, hspace, wspace, width, ylim):
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(5, 4, hspace=hspace, wspace=wspace)
        d_axes = self.get_d_axes(fig, gs)
        for eigid in self.eigenlist:
            ax = d_axes[eigid]
            xlist = self.get_xlist()
            ylist = self.get_ylist(eigid)
            ax.bar(xlist, ylist, width, color='blue')
            ax.set_xticks(xlist[::5])
            ax.set_ylim(ylim)
            if eigid == 1:
                ax.set_title(f'{self.abbrhost} $j$ = Mode {eigid} of HB')
            else:
                ax.set_title(f'$j$ = Mode {eigid} of HB')
            if eigid in [1, 5, 9, 13, 17]:
                ax.set_ylabel(r'$|e_{j}^{H} \cdot e_{i}^{S}|$')
            if eigid in [17, 18, 19, 20]:
                ax.set_xlabel(r'$i$')
            
        gs.tight_layout(fig)        
        return fig, d_axes

    def get_xlist(self):
        start = 1
        end = start + self.n_eigenvector
        return list(range(start, end))

    def get_ylist(self, eigid):
        eig_hb = self.hb.get_eigenvector_by_id(eigid)
        ylist = np.zeros(self.n_eigenvector)
        for tempeigid in range(1, self.n_eigenvector+1):
            eig_stack = self.stack.get_eigenvector_by_id(tempeigid)
            dotproduct = np.dot(eig_hb, eig_stack)
            ylist[tempeigid-1] = np.abs(dotproduct)
        return ylist
