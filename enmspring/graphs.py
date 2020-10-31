from os import path
from shutil import copyfile
import numpy as np
import MDAnalysis
from enmspring import pairtype
from enmspring.spring import Spring
from enmspring.k_b0_util import get_df_by_filter_st

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

        self.crd = path.join(self.input_folder, '{0}.nohydrogen.avg.crd'.format(self.type_na))
        self.npt4_crd = path.join(self.input_folder, '{0}.nohydrogen.crd'.format(self.type_na))
        self.u = MDAnalysis.Universe(self.crd, self.crd)
        self.map, self.inverse_map, self.residues_map, self.atomid_map,\
        self.atomid_map_inverse, self.atomname_map, self.strandid_map,\
        self.resid_map, self.mass_map = self.__build_map()

        self.node_list, self.d_idx = self.build_node_list_base()
        self.n_node = len(self.node_list)
        self.adjacency_mat, self.degree_mat, self.laplacian_mat = self.__initialize_three_mat()
        self.df_st = self.__read_df_st()

    def build_node_list_base(self):
        node_list = list()
        d_idx = dict()
        idx = 0
        for cgname, atomname in self.atomname_map.items():
            atom_type = pairtype.d_atomcgtype[atomname]
            if atom_type == 'B':
                node_list.append(cgname)
                d_idx[cgname] = idx
                idx += 1
        return node_list, d_idx

    def build_adjacency_from_df_st(self):
        idx_i_list = self.__get_idx_list(self.df_st['Atomid_i'])
        idx_j_list = self.__get_idx_list(self.df_st['Atomid_j'])
        k_list = self.df_st['k'].tolist()
        for idx_i, idx_j, k in zip(idx_i_list, idx_j_list, k_list):
            self.adjacency_mat[idx_i, idx_j] = k
        i_lower = np.tril_indices(self.n_node, -1)
        self.adjacency_mat[i_lower] = self.adjacency_mat.transpose()[i_lower]  # make the matrix symmetric

    def build_degree_from_adjacency(self):
        for idx in range(self.n_node):
            self.degree_mat[idx, idx] = self.adjacency_mat[idx, :].sum()

    def build_laplacian_by_adjacency_degree(self):
        self.laplacian_mat = self.degree_mat + self.adjacency_mat
        print("Finish the setup for Laplaican matrix.")

    def vmd_show_crd(self):
        print(f'vmd -cor {self.npt4_crd}')

    def write_show_base_nodes_tcl(self, tcl_out, colorid=0, vdw_radius=1.0):
        serials_str = self.__get_serial_base_nodes()
        f = open(tcl_out, 'w')
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
        f.close()
        print(f'Write tcl to {tcl_out}')
        print(f'source {tcl_out}')

    def copy_nohydrogen_crd(self):
        allsys_root = '/home/yizaochen/codes/dna_rna/all_systems'
        srt = path.join(allsys_root, self.host, self.type_na, 'input', 'heavyatoms', f'{self.type_na}.nohydrogen.crd')
        dst = self.npt4_crd
        copyfile(srt, dst)
        print(f'cp {srt} {dst}')
        
    def write_show_base_edges_tcl(self, tcl_out, radius=0.05):
        u_npt4 = MDAnalysis.Universe(self.npt4_crd, self.npt4_crd)
        lines = ['graphics 0 color 1\n', 'graphics 0 material AOShiny\n']
        for atomid1, atomid2 in zip(self.df_st['Atomid_i'], self.df_st['Atomid_j']):
            line = self.__get_draw_edge_line(u_npt4.atoms.positions, atomid1-1, atomid2-1, radius)
            lines.append(line)
        f = open(tcl_out, 'w')
        for line in lines:
            f.write(line)
        f.close()
        print(f'Write tcl to {tcl_out}')
        print(f'source {tcl_out}')

    def __initialize_three_mat(self):
        adjacency_mat = np.zeros((self.n_node, self.n_node))
        degree_mat = np.zeros((self.n_node, self.n_node))
        laplacian_mat = np.zeros((self.n_node, self.n_node))
        return adjacency_mat, degree_mat, laplacian_mat

    def __read_df_st(self):
        criteria = 1e-3
        spring_obj = Spring(self.rootfolder, self.host, self.type_na, self.n_bp)
        df = spring_obj.read_k_b0_pairtype_df_given_cutoff(self.cutoff)
        df1 = get_df_by_filter_st(df, 'st')
        mask = df1['k'] > criteria
        return df1[mask]

    def __get_idx_list(self, df_column):
        cgname_list = [self.atomid_map_inverse[atomid] for atomid in df_column]
        return [self.d_idx[cgname] for cgname in cgname_list]

    def __get_serial_base_nodes(self):
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
        
    def __build_map(self):
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