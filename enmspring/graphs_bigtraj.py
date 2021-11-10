from os import path
import matplotlib.pyplot as plt
import MDAnalysis
import numpy as np
from enmspring.spring import Spring
from enmspring.graphs import Stack, BackboneRibose, onlyHB, HBAgent, BB1
from enmspring.na_seq import sequences
from enmspring.miscell import check_dir_exist_and_make

class StackMeanModeAgent:
    start_time = 0
    end_time = 5000 # 5000 ns
    n_bp = 21
    d_atomlist = {'A': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2', 'N6', 'N7', 'C8', 'N9'],
                  'T': ['C4', 'C5', 'C6', 'N1', 'C2', 'N3', 'C7', 'O2', 'O4'],
                  'C': ['C4', 'C5', 'C6', 'N1', 'C2', 'N3', 'O2', 'N4'],
                  'G': ['N1', 'C6', 'C5', 'C4', 'N3', 'C2', 'O6', 'N2', 'N7', 'C8', 'N9']}

    def __init__(self, host, rootfolder, interval_time):
        self.host = host
        self.rootfolder = rootfolder
        self.interval_time = interval_time
        
        self.host_folder = path.join(rootfolder, host)
        self.npy_folder = path.join(self.host_folder, 'mean_mode_npy')
        self.f_laplacian = self.set_f_laplacian()
        self.f_std_laplacian = self.set_f_std_laplacian()
        self.f_b0_mean, self.f_b0_std = self.set_f_b0_mean_std()
        self.check_folders()

        self.time_list = self.get_time_list()
        self.n_window = len(self.time_list)
        self.d_smallagents = self.get_all_small_agents()

        self.node_list = None
        self.d_idx = None
        self.d_idx_inverse = None
        self.n_node = None

        self.map = None
        self.inverse_map = None
        self.residues_map = None
        self.atomid_map = None
        self.atomid_map_inverse = None
        self.atomname_map = None
        self.strandid_map = None
        self.resid_map = None
        self.mass_map = None

        self.d_node_list_by_strand = None
        self.d_idx_list_by_strand = None
        
        self.adjacency_mat = None
        self.degree_mat = None
        self.laplacian_mat = None
        self.laplacian_std_mat = None
        self.b0_mean_mat = None
        self.b0_std_mat = None

        self.w = None  # Eigenvalue array
        self.v = None  # Eigenvector matrix, the i-th column is the i-th eigenvector

        self.strand1_array = list() # 0: STRAND1, 1: STRAND2
        self.strand2_array = list() #
        self.strand1_benchmark = None
        self.strand2_benchmark = None

        self.d_seq = {'STRAND1': sequences[host]['guide'], 'STRAND2': sequences[host]['target']}

    def check_folders(self):
        for folder in [self.npy_folder]:
            check_dir_exist_and_make(folder)

    def set_f_laplacian(self):
        return path.join(self.npy_folder, 'laplacian.npy')

    def set_f_std_laplacian(self):
        return path.join(self.npy_folder, 'laplacian.std.npy')

    def set_f_b0_mean_std(self):
        return path.join(self.npy_folder, 'b0.mean.npy'), path.join(self.npy_folder, 'b0.std.npy')

    def get_time_list(self):
        middle_interval = int(self.interval_time/2)
        time_list = list()
        for time1 in range(self.start_time, self.end_time, middle_interval):
            time2 = time1 + self.interval_time
            if time2 <= self.end_time:
                time_list.append((time1, time2))
        return time_list

    def get_all_small_agents(self):
        d_smallagents = dict()
        for time1, time2 in self.time_list:
            time_label = f'{time1}_{time2}'
            d_smallagents[(time1,time2)] = StackGraph(self.host, self.rootfolder, time_label)
        return d_smallagents

    def preprocess_all_small_agents(self):
        for time1, time2 in self.time_list:
            self.d_smallagents[(time1,time2)].pre_process()
        self.n_node = self.d_smallagents[(time1,time2)].n_node

    def process_first_small_agent(self):
        time1, time2 = self.time_list[0]
        self.d_smallagents[(time1,time2)].pre_process()
        self.set_d_idx_and_inverse()

    def set_d_idx_and_inverse(self):
        time1, time2 = self.time_list[0]
        self.node_list = self.d_smallagents[(time1,time2)].node_list
        self.d_idx = self.d_smallagents[(time1,time2)].d_idx
        self.d_idx_inverse = {y:x for x,y in self.d_idx.items()}
        self.n_node = self.d_smallagents[(time1,time2)].n_node

    def set_degree_adjacency_from_laplacian(self):
        self.adjacency_mat = np.zeros((self.n_node, self.n_node))
        self.degree_mat = np.zeros((self.n_node, self.n_node))
        for idx in range(self.n_node):
            self.degree_mat[idx, idx] = self.laplacian_mat[idx, idx]
        self.adjacency_mat = self.laplacian_mat - self.degree_mat

    def make_mean_mode_laplacian(self):
        self.laplacian_mat = np.zeros((self.n_node, self.n_node))
        for time1, time2 in self.time_list:
            self.laplacian_mat += self.d_smallagents[(time1,time2)].laplacian_mat
        self.laplacian_mat = self.laplacian_mat / self.n_window
        print("Set laplacian_mat.")

    def save_mean_mode_laplacian_into_npy(self):
        np.save(self.f_laplacian, self.laplacian_mat)
        print(f'Save laplacian_mat into {self.f_laplacian}')

    def load_mean_mode_laplacian_from_npy(self):
        self.laplacian_mat = np.load(self.f_laplacian)
        print(f'Load laplacian_mat from {self.f_laplacian}')

    def make_mean_mode_std_laplacian(self):
        big_mat = np.zeros((self.n_node, self.n_node, self.n_window))
        for k in range(self.n_window):
            time1, time2 = self.time_list[k]
            big_mat[:,:,k] = self.d_smallagents[(time1,time2)].laplacian_mat
        self.laplacian_std_mat = big_mat.std(2)

    def save_mean_mode_std_laplacian_into_npy(self):
        np.save(self.f_std_laplacian, self.laplacian_std_mat)
        print(f'Save laplacian_std_mat into {self.f_std_laplacian}')

    def load_mean_mode_std_laplacian_from_npy(self):
        self.laplacian_std_mat = np.load(self.f_std_laplacian)
        print(f'Load laplacian_std_mat from {self.f_std_laplacian}')

    def make_b0_mean_std(self):
        big_mat = np.zeros((self.n_node, self.n_node, self.n_window))
        for k in range(self.n_window):
            time1, time2 = self.time_list[k]
            big_mat[:,:,k] = self.d_smallagents[(time1,time2)].b0_mat
        self.b0_mean_mat = big_mat.mean(2)
        self.b0_std_mat = big_mat.std(2)

    def save_b0_mean_std_into_npy(self):
        np.save(self.f_b0_mean, self.b0_mean_mat)
        print(f'Save b0_mean_mat into {self.f_b0_mean}')
        np.save(self.f_b0_std, self.b0_std_mat)
        print(f'Save b0_std_mat into {self.f_b0_std}')

    def load_b0_mean_std_from_npy(self):
        self.b0_mean_mat = np.load(self.f_b0_mean)
        print(f'Load b0_mean_mat from {self.f_b0_mean}')
        self.b0_std_mat = np.load(self.f_b0_std)
        print(f'Load b0_std_mat from {self.f_b0_std}')

    def eigen_decompose(self):
        w, v = np.linalg.eig(self.laplacian_mat)
        idx = w.argsort()[::-1] # sort from big to small
        self.w = w[idx]
        self.v = v[:, idx]

    def get_eigenvalue_by_id(self, sele_id):
        return self.w[sele_id-1]

    def get_eigenvector_by_id(self, sele_id):
        return self.v[:,sele_id-1]

    def set_benchmark_array(self):
        idx_start_strand2 = self.d_idx['B6']
        strand1 = np.zeros(self.n_node)
        strand2 = np.zeros(self.n_node)
        strand1[:idx_start_strand2] = 1.
        strand2[idx_start_strand2:] = 1.
        self.strand1_benchmark = strand1
        self.strand2_benchmark = strand2

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

    def decide_eigenvector_strand_by_strand_array(self, eigv_id):
        if eigv_id in self.strand1_array:
            return 'STRAND1'
        else:
            return 'STRAND2'

    def get_eigv_id_by_strandid_modeid(self, strand_id, mode_id):
        d_temp = {'STRAND1': self.strand1_array, 'STRAND2': self.strand2_array}
        return d_temp[strand_id][mode_id-1]

    def get_lambda_by_strand(self, strandid):
        if strandid == 'STRAND1':
            return [self.get_eigenvalue_by_id(eigv_id) for eigv_id in self.strand1_array]
        else:
            return [self.get_eigenvalue_by_id(eigv_id) for eigv_id in self.strand2_array]

    def initialize_nodes_information(self):
        time1_tuple = self.time_list[0]
        self.d_smallagents[time1_tuple].build_node_list()
        self.node_list = self.d_smallagents[time1_tuple].node_list
        self.d_idx = self.d_smallagents[time1_tuple].d_idx
        self.n_node = len(self.node_list)
        self.initialize_all_maps()

    def initialize_all_maps(self):
        time1_tuple = self.time_list[0]
        self.map = self.d_smallagents[time1_tuple].map
        self.inverse_map = self.d_smallagents[time1_tuple].inverse_map
        self.residues_map = self.d_smallagents[time1_tuple].residues_map
        self.atomid_map = self.d_smallagents[time1_tuple].atomid_map
        self.atomid_map_inverse = self.d_smallagents[time1_tuple].atomid_map_inverse
        self.atomname_map = self.d_smallagents[time1_tuple].atomname_map
        self.strandid_map = self.d_smallagents[time1_tuple].strandid_map
        self.resid_map = self.d_smallagents[time1_tuple].resid_map
        self.mass_map = self.d_smallagents[time1_tuple].mass_map

    def split_node_list_into_two_strand(self):
        strandid_list = ['STRAND1', 'STRAND2']
        d_node_list_by_strand = dict()
        d_idx_list_by_strand = dict()
        for strand_id in strandid_list:
            d_node_list_by_strand[strand_id] = [node_id for node_id in self.node_list if self.strandid_map[node_id] == strand_id]
            d_idx_list_by_strand[strand_id] = [idx for idx, node_id in enumerate(self.node_list) if self.strandid_map[node_id] == strand_id]
        self.d_node_list_by_strand = d_node_list_by_strand
        self.d_idx_list_by_strand = d_idx_list_by_strand

    def get_vlines_by_resid_list(self, node_list):
        vlines = list()
        resid_list = [self.resid_map[node_id] for node_id in node_list]
        resid_prev = None
        for idx, resid in enumerate(resid_list):
            if idx == 0:
                resid_prev = resid
                continue
            if resid_prev != resid:
                vlines.append((2 * idx - 1) / 2)
            resid_prev = resid    
        return vlines

    def plot_sele_eigenvector(self, figsize, strand_id, mode_id, show_xticklabel=False):
        fig, ax = plt.subplots(figsize=figsize)
        sele_id = self.get_eigv_id_by_strandid_modeid(strand_id, mode_id)
        idx_list = self.d_idx_list_by_strand[strand_id]
        node_list= self.d_node_list_by_strand[strand_id]

        y = self.get_eigenvector_by_id(sele_id) # eigenvector
        y = y[idx_list]
        x = range(len(y))
        vlines = self.get_vlines_by_resid_list(node_list)
        ax.plot(x, y)

        for vline in vlines:
            ax.axvline(vline, color="grey", alpha=0.2)

        if show_xticklabel:
            xticklabels = [self.atomname_map[node_id] for node_id in node_list]
            ax.set_xticks(x)
            ax.set_xticklabels(xticklabels)
        else:
            ax.set_xticks(x[::50])
        ax.set_xlim(x[0]-1, x[-1]+1)
        ax.set_xlabel('Atom index', fontsize=14)
        ax.set_ylabel(r'$\mathbf{e}_{' + f'{mode_id}' + r'}$', fontsize=14)
        ax.set_title(f'{self.host}-{strand_id}', fontsize=16)
        return fig, ax

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
        
    def get_key_by_atomname_resid_strandid(self, atomname, resid, strandid):
        return f'segid {strandid} and resid {resid} and name {atomname}'

    def get_map_idx_from_strand_resid_atomname(self):
        d_result = dict()
        for node_name in self.node_list:
            idx = self.d_idx[node_name]
            strand_id = self.strandid_map[node_name]
            resid = self.resid_map[node_name]
            atomname = self.atomname_map[node_name]
            d_result[(strand_id, resid, atomname)] = idx
        return d_result

    def get_last_mode_by_strand_id(self, strand_id):
        if strand_id == 'STRAND1':
            return len(self.strand1_array)
        else:
            return len(self.strand2_array)
class ProminentModes:
    def __init__(self, host, rootfolder, interval_time):
        self.host = host
        self.rootfolder = rootfolder
        self.interval_time = interval_time
        self.host_folder = path.join(rootfolder, host)
        self.npy_folder = path.join(self.host_folder, 'mean_mode_npy')
        self.s_agent = None

        self.mean_modes_w = None # eigenvalues
        self.mean_modes_v = None # eigenvectors
        self.time_list = None
        self.d_smallagents = None
        self.initialize_s_agent()

        self.n_eigenvalues = len(self.mean_modes_w)
        self.n_window = len(self.time_list)
        
        self.f_mean_r_alpha_array = self.set_f_mean_r_alpha_array()
        self.mean_r_alpha_array = None

    def initialize_s_agent(self):
        self.s_agent = StackMeanModeAgent(self.host, self.rootfolder, self.interval_time)
        self.s_agent.load_mean_mode_laplacian_from_npy()
        self.s_agent.eigen_decompose()

        self.mean_modes_w = self.s_agent.w # eigenvalues
        self.mean_modes_v = self.s_agent.v # eigenvectors
        self.time_list = self.s_agent.time_list
        self.d_smallagents = self.s_agent.d_smallagents

    def initialize_small_agents(self):
        self.s_agent.preprocess_all_small_agents()
        for time_tuple in self.time_list:
            self.s_agent.d_smallagents[time_tuple].eigen_decompose()
            
    def set_f_mean_r_alpha_array(self):
        return path.join(self.npy_folder, 'mean_r_alpha.npy')

    def get_mean_modes_v_mat(self):
        return self.mean_modes_v

    def get_window_modes_v_mat(self, window_id):
        key = self.time_list[window_id]
        return self.d_smallagents[key].v

    def get_r_n_alpha(self):
        mean_modes_v_mat_T = self.get_mean_modes_v_mat().T
        r_n_alpha_mat = np.zeros((self.n_window, self.n_eigenvalues))
        for window_id in range(self.n_window):
            window_modes_v_mat = self.get_window_modes_v_mat(window_id)
            product_mat = np.abs(np.dot(mean_modes_v_mat_T, window_modes_v_mat))
            for eigv_id in range(self.n_eigenvalues):
                r_n_alpha_mat[window_id, eigv_id] = product_mat[eigv_id,:].max()
        return r_n_alpha_mat

    def set_mean_r_alpha_array(self):
        mean_r_alpha_array = np.zeros(self.n_eigenvalues)
        r_n_alpha_mat = self.get_r_n_alpha()
        for eigv_idx in range(self.n_eigenvalues):
            mean_r_alpha_array[eigv_idx] = r_n_alpha_mat[:, eigv_idx].mean()
        self.mean_r_alpha_array = mean_r_alpha_array

    def save_mean_r_alpha_array(self):
        np.save(self.f_mean_r_alpha_array, self.mean_r_alpha_array)
        print(f'Save mean_r_alpha_array into {self.f_mean_r_alpha_array}')

    def load_mean_r_alpha_array(self):
        self.mean_r_alpha_array = np.load(self.f_mean_r_alpha_array)
        print(f'Load mean_r_alpha_array from {self.f_mean_r_alpha_array}')

    def get_mean_r_alpha_array(self):
        return self.mean_r_alpha_array

class BackboneMeanModeAgent(StackMeanModeAgent):
    def set_f_laplacian(self):
        return path.join(self.npy_folder, 'laplacian_backbone.npy')

    def set_f_std_laplacian(self):
        return path.join(self.npy_folder, 'laplacian_backbone.std.npy')

    def set_f_b0_mean_std(self):
        return path.join(self.npy_folder, 'b0_backbone.mean.npy'), path.join(self.npy_folder, 'b0_backbone.std.npy')
    
    def get_all_small_agents(self):
        d_smallagents = dict()
        for time1, time2 in self.time_list:
            time_label = f'{time1}_{time2}'
            d_smallagents[(time1,time2)] = BackboneGraph(self.host, self.rootfolder, time_label)
        return d_smallagents

    def set_benchmark_array(self):
        idx_start_strand2 = self.d_idx['B1']
        strand1 = np.zeros(self.n_node)
        strand2 = np.zeros(self.n_node)
        strand1[:idx_start_strand2] = 1.
        strand2[idx_start_strand2:] = 1.
        self.strand1_benchmark = strand1
        self.strand2_benchmark = strand2

class BB1MeanModeAgent(StackMeanModeAgent):
    def set_f_laplacian(self):
        return path.join(self.npy_folder, 'laplacian_BB1.npy')

    def set_f_std_laplacian(self):
        return path.join(self.npy_folder, 'laplacian_BB1.std.npy')

    def set_f_b0_mean_std(self):
        return path.join(self.npy_folder, 'b0_BB1.mean.npy'), path.join(self.npy_folder, 'b0_BB1.std.npy')
    
    def get_all_small_agents(self):
        d_smallagents = dict()
        for time1, time2 in self.time_list:
            time_label = f'{time1}_{time2}'
            d_smallagents[(time1,time2)] = BB1Graph(self.host, self.rootfolder, time_label)
        return d_smallagents

    def set_benchmark_array(self):
        idx_start_strand2 = self.d_idx['B1']
        strand1 = np.zeros(self.n_node)
        strand2 = np.zeros(self.n_node)
        strand1[:idx_start_strand2] = 1.
        strand2[idx_start_strand2:] = 1.
        self.strand1_benchmark = strand1
        self.strand2_benchmark = strand2

class HBMeanModeAgent(StackMeanModeAgent):
    def set_f_laplacian(self):
        return path.join(self.npy_folder, 'laplacian_hb.npy')

    def set_f_std_laplacian(self):
        return path.join(self.npy_folder, 'laplacian_hb.std.npy')

    def set_f_b0_mean_std(self):
        return path.join(self.npy_folder, 'b0_hb.mean.npy'), path.join(self.npy_folder, 'b0_hb.std.npy')
    
    def get_all_small_agents(self):
        d_smallagents = dict()
        for time1, time2 in self.time_list:
            time_label = f'{time1}_{time2}'
            d_smallagents[(time1,time2)] = HBGraph(self.host, self.rootfolder, time_label)
        return d_smallagents

class ProminentModesBackbone(ProminentModes):
    def initialize_s_agent(self):
        self.s_agent = BackboneMeanModeAgent(self.host, self.rootfolder, self.interval_time)
        self.s_agent.load_mean_mode_laplacian_from_npy()
        self.s_agent.eigen_decompose()

        self.mean_modes_w = self.s_agent.w # eigenvalues
        self.mean_modes_v = self.s_agent.v # eigenvectors
        self.time_list = self.s_agent.time_list
        self.d_smallagents = self.s_agent.d_smallagents
        
    def set_f_mean_r_alpha_array(self):
        return path.join(self.npy_folder, 'mean_r_alpha_backbone.npy')

class StackGraph(Stack):
    def __init__(self, host, rootfolder, time_label):
        self.host = host
        self.rootfolder = rootfolder
        self.time_label = time_label

        self.host_folder = path.join(rootfolder, host)
        self.na_folder = path.join(self.host_folder, self.type_na, time_label)
        self.input_folder = path.join(self.na_folder, 'input')

        self.spring_obj = Spring(self.rootfolder, self.host, self.type_na, self.n_bp, time_label)
        self.df_all_k = self.spring_obj.read_k_b0_pairtype_df_given_cutoff(self.cutoff)
        self.df_st = self.read_df_st()

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

class HBGraph(onlyHB):
    def __init__(self, host, rootfolder, time_label):
        self.host = host
        self.rootfolder = rootfolder
        self.time_label = time_label

        self.host_folder = path.join(rootfolder, host)
        self.na_folder = path.join(self.host_folder, self.type_na, time_label)
        self.input_folder = path.join(self.na_folder, 'input')

        self.spring_obj = Spring(self.rootfolder, self.host, self.type_na, self.n_bp, time_label)
        self.df_all_k = self.spring_obj.read_k_b0_pairtype_df_given_cutoff(self.cutoff)
        self.df_st = self.read_df_st()

        self.hb_agent = HBAgent(self.host, self.rootfolder, self.n_bp, time_label)

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

class BackboneGraph(BackboneRibose):
    def __init__(self, host, rootfolder, time_label):
        self.host = host
        self.rootfolder = rootfolder
        self.time_label = time_label

        self.host_folder = path.join(rootfolder, host)
        self.na_folder = path.join(self.host_folder, self.type_na, time_label)
        self.input_folder = path.join(self.na_folder, 'input')

        self.spring_obj = Spring(self.rootfolder, self.host, self.type_na, self.n_bp, time_label)
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

class BB1Graph(BB1):
    def __init__(self, host, rootfolder, time_label):
        self.host = host
        self.rootfolder = rootfolder
        self.time_label = time_label

        self.host_folder = path.join(rootfolder, host)
        self.na_folder = path.join(self.host_folder, self.type_na, time_label)
        self.input_folder = path.join(self.na_folder, 'input')

        self.spring_obj = Spring(self.rootfolder, self.host, self.type_na, self.n_bp, time_label)
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