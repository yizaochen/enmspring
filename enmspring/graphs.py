from os import path
from shutil import copyfile
import numpy as np
import MDAnalysis
import matplotlib.pyplot as plt
from enmspring import pairtype
from enmspring.spring import Spring
from enmspring.k_b0_util import get_df_by_filter_st
from enmspring.hb_util import HBAgent

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

        self.hb_agent = HBAgent(host, rootfolder, self.n_bp)

        self.w = None  # Eigenvalue array
        self.v = None  # Eigenvector matrix, the i-th column is the i-th eigenvector
        self.strand1_array = list() # 0: STRAND1, 1: STRAND2
        self.strand2_array = list() #
        self.strand1_benchmark, self.strand2_benchmark = self.__get_benchmark_array_basestack()

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

    def build_adjacency_from_df_st_df_hb(self):
        # For stack
        idx_i_list = self.__get_idx_list(self.df_st['Atomid_i'])
        idx_j_list = self.__get_idx_list(self.df_st['Atomid_j'])
        k_list = self.df_st['k'].tolist()
        for idx_i, idx_j, k in zip(idx_i_list, idx_j_list, k_list):
            self.adjacency_mat[idx_i, idx_j] = k

        # For HB
        d_hb_new = self.hb_agent.get_d_hb_contain_atomid_k_all_basepair()
        idx_i_list = self.__get_idx_list(d_hb_new['Atomid_i'])
        idx_j_list = self.__get_idx_list(d_hb_new['Atomid_j'])
        k_list = d_hb_new['k']
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

    def eigen_decompose(self):
        w, v = np.linalg.eig(self.laplacian_mat)
        idx = w.argsort()[::-1] # sort from big to small
        self.w = w[idx]
        self.v = v[:, idx]

    def get_eigenvalue_by_id(self, sele_id):
        return self.w[sele_id-1]

    def get_eigenvector_by_id(self, sele_id):
        return self.v[:,sele_id-1]

    def vmd_show_crd(self):
        print(f'vmd -cor {self.npt4_crd}')

    def write_show_base_nodes_tcl(self, tcl_out, colorid=0, vdw_radius=1.0):
        serials_str = self.__get_serial_base_nodes()
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

    def write_show_base_hb_edges_tcl(self, tcl_out, radius=0.05):
        u_npt4 = MDAnalysis.Universe(self.npt4_crd, self.npt4_crd)
        lines = ['graphics 0 color 1\n', 'graphics 0 material AOShiny\n']
        for atomid1, atomid2 in zip(self.df_st['Atomid_i'], self.df_st['Atomid_j']):
            line = self.__get_draw_edge_line(u_npt4.atoms.positions, atomid1-1, atomid2-1, radius)
            lines.append(line)
        lines.append('graphics 0 color 10\n')
        d_hb_new = self.hb_agent.get_d_hb_contain_atomid_k_all_basepair()
        for atomid1, atomid2 in zip(d_hb_new['Atomid_i'], d_hb_new['Atomid_j']):
            line = self.__get_draw_edge_line(u_npt4.atoms.positions, atomid1-1, atomid2-1, radius)
            lines.append(line)
        f = open(tcl_out, 'w')
        for line in lines:
            f.write(line)
        f.close()
        print(f'Write tcl to {tcl_out}')
        print(f'source {tcl_out}')

    def decide_eigenvector_strand_basestack(self, eigv_id):
        eigv = self.get_eigenvector_by_id(eigv_id)
        dot_product = np.dot(eigv, self.strand1_benchmark)
        if np.isclose(dot_product, 0.):
            return True #'STRAND2'
        else:
            return False #'STRAND1'

    def set_strand_array(self):
        for eigv_id in range(1, self.n_node+1):
            if self.decide_eigenvector_strand_basestack(eigv_id):
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

    def __get_benchmark_array_basestack(self):
        idx_start_strand2 = self.d_idx['B6']
        strand1 = np.zeros(self.n_node)
        strand2 = np.zeros(self.n_node)
        strand1[:idx_start_strand2] = 1.
        strand2[idx_start_strand2:] = 1.
        return strand1, strand2
        
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

    def __init__(self, rootfolder, hb_include=False):
        self.rootfolder = rootfolder
        self.d_agent = dict()
        self.d_eigenvalues = dict()
        self.hb_include = hb_include

    def initailize_six_systems(self):
        for host in hosts:
            g_agent = GraphAgent(host, self.rootfolder)
            g_agent.build_node_list_base()
            if self.hb_include:
                g_agent.build_adjacency_from_df_st_df_hb()
            else:
                g_agent.build_adjacency_from_df_st()
            g_agent.build_degree_from_adjacency()
            g_agent.build_laplacian_by_adjacency_degree()
            g_agent.eigen_decompose()
            g_agent.set_strand_array()
            self.d_agent[host] = g_agent
    
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
            self.__ax_plot_lambda(ax, host1)
            self.__ax_plot_lambda(ax, host2)
            self.__ax_plot_assistline(ax)
            ax.legend(frameon=False)
            ax.set_xlabel("Eigenvalue ID")
            if ax_id == 0:
                ax.set_ylabel("Eigenvalue")
        return fig, ax

    def __ax_plot_lambda(self, ax, host):
        x1, y1, x2, y2 = self.__get_lambda_array_by_host(host)
        ax.plot(x1, y1, ls='-', color=self.d_colors[host], alpha=0.7, label=self.d_labels[host][0])
        ax.plot(x2, y2, ls='-.', color=self.d_colors[host], alpha=0.7, label=self.d_labels[host][1])

    def __ax_plot_assistline(self, ax):
        for y in [0, 5, 10, 15, 20]:
            ax.axhline(y, color='grey',alpha=0.15)

    def __get_lambda_array_by_host(self, host):
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
                