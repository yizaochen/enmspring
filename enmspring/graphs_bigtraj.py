from os import path
import MDAnalysis
import numpy as np
from enmspring.spring import Spring
from enmspring.graphs import Stack, BackboneRibose
from enmspring.na_seq import sequences
from enmspring.miscell import check_dir_exist_and_make

class StackMeanModeAgent:
    start_time = 0
    end_time = 5000 # 5000 ns

    def __init__(self, host, rootfolder, interval_time):
        self.host = host
        self.rootfolder = rootfolder
        self.interval_time = interval_time
        
        self.host_folder = path.join(rootfolder, host)
        self.npy_folder = path.join(self.host_folder, 'mean_mode_npy')
        self.f_laplacian = self.set_f_laplacian()
        self.check_folders()

        self.time_list = self.get_time_list()
        self.n_window = len(self.time_list)
        self.d_smallagents = self.get_all_small_agents()

        self.node_list = None
        self.d_idx = None
        self.n_node = None

        self.laplacian_mat = None

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
        self.node_list = self.d_smallagents[(time1,time2)].node_list
        self.d_idx = self.d_smallagents[(time1,time2)].d_idx
        self.n_node = self.d_smallagents[(time1,time2)].n_node

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

    def get_lambda_by_strand(self, strandid):
        if strandid == 'STRAND1':
            return [self.get_eigenvalue_by_id(eigv_id) for eigv_id in self.strand1_array]
        else:
            return [self.get_eigenvalue_by_id(eigv_id) for eigv_id in self.strand2_array]

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
        mean_modes_v_mat = np.zeros((self.n_eigenvalues, self.n_eigenvalues))
        for eigv_idx in range(self.n_eigenvalues):
            mean_modes_v_mat[:, eigv_idx] = self.mean_modes_v[eigv_idx] # column as eigenvector
        return mean_modes_v_mat

    def get_window_modes_v_mat(self, window_id):
        key = self.time_list[window_id]
        window_modes_v_mat = np.zeros((self.n_eigenvalues, self.n_eigenvalues))
        v_array = self.d_smallagents[key].v
        for eigv_idx in range(self.n_eigenvalues):
            window_modes_v_mat[:, eigv_idx] = v_array[eigv_idx] # column as eigenvector
        return window_modes_v_mat

    def get_r_n_alpha(self):
        mean_modes_v_mat_T = self.get_mean_modes_v_mat().T
        r_n_alpha_mat = np.zeros((self.n_window, self.n_eigenvalues))
        for window_id in range(self.n_window):
            window_modes_v_mat = self.get_window_modes_v_mat(window_id)
            product_mat = np.dot(mean_modes_v_mat_T, window_modes_v_mat)
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

        self.w = None  # Eigenvalue array
        self.v = None  # Eigenvector matrix, the i-th column is the i-th eigenvector
        self.strand1_array = list() # 0: STRAND1, 1: STRAND2
        self.strand2_array = list() #
        self.strand1_benchmark = None
        self.strand2_benchmark = None

        self.d_seq = {'STRAND1': sequences[host]['guide'], 'STRAND2': sequences[host]['target']}