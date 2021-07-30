from os import path
import MDAnalysis
import numpy as np
from enmspring.spring import Spring
from enmspring.graphs import Stack
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
        self.f_laplacian = path.join(self.npy_folder, 'laplacian.npy')
        self.check_folders()

        self.time_list = self.get_time_list()
        self.n_window = len(self.time_list)
        self.d_smallagents = self.get_all_small_agents()

        self.n_node = None
        self.laplacian_mat = None

        self.w = None  # Eigenvalue array
        self.v = None  # Eigenvector matrix, the i-th column is the i-th eigenvector

    def check_folders(self):
        for folder in [self.npy_folder]:
            check_dir_exist_and_make(folder)

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