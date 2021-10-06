from os import path
import numpy as np
import MDAnalysis
from enmspring.graphs_bigtraj import StackMeanModeAgent, BackboneMeanModeAgent, HBMeanModeAgent


class DrawAgent:
    interval_time = 500
    type_na = 'bdna+bdna'
    all_folder = '/home/yizaochen/codes/dna_rna/all_systems'

    def __init__(self, host, big_traj_folder, tcl_folder):
        self.host = host
        self.big_traj_folder = big_traj_folder
        self.tcl_folder = tcl_folder

        self.heavy_folder = path.join(self.all_folder, self.host, self.type_na, 'input', 'heavyatoms')
        self.noh_perfect_gro = path.join(self.heavy_folder, f'{self.type_na}.perfect.noH.gro')
        self.u = MDAnalysis.Universe(self.noh_perfect_gro)

        self.b_agent = None
        self.s_agent = None
        self.h_agent = None

    def ini_b_agent(self):
        if self.b_agent is None:
            self.b_agent = BackboneMeanModeAgent(self.host, self.big_traj_folder, self.interval_time)
            self.b_agent.process_first_small_agent()
            self.b_agent.load_mean_mode_laplacian_from_npy()
            self.b_agent.initialize_all_maps()

    def ini_s_agent(self):
        if self.s_agent is None:
            self.s_agent = StackMeanModeAgent(self.host, self.big_traj_folder, self.interval_time)
            self.s_agent.process_first_small_agent()
            self.s_agent.load_mean_mode_laplacian_from_npy()
            self.s_agent.initialize_all_maps()

    def ini_h_agent(self):
        if self.h_agent is None:
            self.h_agent = HBMeanModeAgent(self.host, self.big_traj_folder, self.interval_time)
            self.h_agent.process_first_small_agent()
            self.h_agent.load_mean_mode_laplacian_from_npy()
            self.h_agent.initialize_all_maps()

    def show_all_atom_model(self):
        print(f'vmd -gro {self.noh_perfect_gro}')

    def add_backbone_edges(self, radius, colorname, k_criteria):
        tcl_lst = list()
        idx_i_array, idx_j_array = self.get_idx_array_by_laplacian_mat(self.b_agent.laplacian_mat, k_criteria)
        fraying_lst = [self.determine_backbone_fraying(idx_i, idx_j) for idx_i, idx_j in zip(idx_i_array, idx_j_array)]
        for idx_i, idx_j, fraying in zip(idx_i_array, idx_j_array, fraying_lst):
            if fraying:
                continue
            positions = self.get_positions_by_ij(idx_i, idx_j)
            tcl_lst += self.get_draw_edge_line(positions, radius, colorname)
        f_tcl_out = path.join(self.tcl_folder, f'add_backbone_edges.tcl')
        self.write_tcl_out(f_tcl_out, tcl_lst)

    def add_backbone_edges_trimer(self, radius, colorname, k_criteria):
        tcl_lst = list()
        idx_i_array, idx_j_array = self.get_idx_array_by_laplacian_mat(self.b_agent.laplacian_mat, k_criteria)
        trimer_lst = [self.determine_backbone_trimer(idx_i, idx_j) for idx_i, idx_j in zip(idx_i_array, idx_j_array)]
        for idx_i, idx_j, trimer in zip(idx_i_array, idx_j_array, trimer_lst):
            if not trimer:
                continue
            positions = self.get_positions_by_ij(idx_i, idx_j)
            tcl_lst += self.get_draw_edge_line(positions, radius, colorname)
        f_tcl_out = path.join(self.tcl_folder, f'add_backbone_edges_trimer.tcl')
        self.write_tcl_out(f_tcl_out, tcl_lst)

    def add_stack_edges(self, radius, colorname, k_criteria):
        tcl_lst = list()
        idx_i_array, idx_j_array = self.get_idx_array_by_laplacian_mat(self.s_agent.laplacian_mat, k_criteria)
        idx_i_lst = [self.convert_stack_idx(idx_i) for idx_i in idx_i_array]
        idx_j_lst = [self.convert_stack_idx(idx_j) for idx_j in idx_j_array]
        fraying_lst = [self.determine_stack_fraying(idx_i, idx_j) for idx_i, idx_j in zip(idx_i_array, idx_j_array)]
        for idx_i, idx_j, fraying in zip(idx_i_lst, idx_j_lst, fraying_lst):
            if fraying:
                continue
            positions = self.get_positions_by_ij(idx_i, idx_j)
            tcl_lst += self.get_draw_edge_line(positions, radius, colorname)
        f_tcl_out = path.join(self.tcl_folder, f'add_stack_edges.tcl')
        self.write_tcl_out(f_tcl_out, tcl_lst)

    def add_stack_edges_trimer(self, radius, colorname, k_criteria):
        tcl_lst = list()
        idx_i_array, idx_j_array = self.get_idx_array_by_laplacian_mat(self.s_agent.laplacian_mat, k_criteria)
        idx_i_lst = [self.convert_stack_idx(idx_i) for idx_i in idx_i_array]
        idx_j_lst = [self.convert_stack_idx(idx_j) for idx_j in idx_j_array]
        trimer_lst = [self.determine_stack_trimer(idx_i, idx_j) for idx_i, idx_j in zip(idx_i_array, idx_j_array)]
        for idx_i, idx_j, trimer in zip(idx_i_lst, idx_j_lst, trimer_lst):
            if not trimer:
                continue
            positions = self.get_positions_by_ij(idx_i, idx_j)
            tcl_lst += self.get_draw_edge_line(positions, radius, colorname)
        f_tcl_out = path.join(self.tcl_folder, f'add_stack_edges_trimer.tcl')
        self.write_tcl_out(f_tcl_out, tcl_lst)

    def add_hb_edges(self, radius, colorname, k_criteria):
        tcl_lst = ['graphics 0 material AOChalky']
        idx_i_array, idx_j_array = self.get_idx_array_by_laplacian_mat(self.h_agent.laplacian_mat, k_criteria)
        idx_i_lst = [self.convert_hb_idx(idx_i) for idx_i in idx_i_array]
        idx_j_lst = [self.convert_hb_idx(idx_j) for idx_j in idx_j_array]
        fraying_lst = [self.determine_hb_fraying(idx_i, idx_j) for idx_i, idx_j in zip(idx_i_array, idx_j_array)]
        for idx_i, idx_j, fraying in zip(idx_i_lst, idx_j_lst, fraying_lst):
            if fraying:
                continue
            positions = self.get_positions_by_ij(idx_i, idx_j)
            tcl_lst += self.get_draw_edge_line(positions, radius, colorname)
        f_tcl_out = path.join(self.tcl_folder, f'add_hb_edges.tcl')
        self.write_tcl_out(f_tcl_out, tcl_lst)

    def add_hb_edges_trimer(self, radius, colorname, k_criteria):
        tcl_lst = ['graphics 0 material AOChalky']
        idx_i_array, idx_j_array = self.get_idx_array_by_laplacian_mat(self.h_agent.laplacian_mat, k_criteria)
        idx_i_lst = [self.convert_hb_idx(idx_i) for idx_i in idx_i_array]
        idx_j_lst = [self.convert_hb_idx(idx_j) for idx_j in idx_j_array]
        trimer_lst = [self.determine_hb_trimer(idx_i, idx_j) for idx_i, idx_j in zip(idx_i_array, idx_j_array)]
        for idx_i, idx_j, trimer in zip(idx_i_lst, idx_j_lst, trimer_lst):
            if not trimer:
                continue
            positions = self.get_positions_by_ij(idx_i, idx_j)
            tcl_lst += self.get_draw_edge_line(positions, radius, colorname)
        f_tcl_out = path.join(self.tcl_folder, f'add_hb_edges_trimer.tcl')
        self.write_tcl_out(f_tcl_out, tcl_lst)

    def determine_backbone_fraying(self, idx_i, idx_j):
        resid_i = self.get_resid_backbone_by_idx(idx_i)
        resid_j = self.get_resid_backbone_by_idx(idx_j)
        fraying_resid_lst = [1, 2, 3, 19, 20, 21]
        if (resid_i in fraying_resid_lst) or (resid_j in fraying_resid_lst):
            return True
        else:
            return False

    def determine_backbone_trimer(self, idx_i, idx_j):
        resid_i = self.get_resid_backbone_by_idx(idx_i)
        resid_j = self.get_resid_backbone_by_idx(idx_j)
        strandid_i = self.get_strand_id_backbone_by_idx(idx_i)
        strandid_j = self.get_strand_id_backbone_by_idx(idx_j)
        trimer_resid_i_lst = [8, 9]
        trimer_resid_j_lst = [13, 14]
        if (resid_i in trimer_resid_i_lst) and (strandid_i == 'STRAND1'):
            return True
        elif (resid_j in trimer_resid_j_lst) and (strandid_j == 'STRAND2'):
            return True
        else:
            return False

    def determine_stack_fraying(self, idx_i, idx_j):
        resid_i = self.get_resid_stack_by_idx(idx_i)
        resid_j = self.get_resid_stack_by_idx(idx_j)
        fraying_resid_lst = [1, 2, 3, 19, 20, 21]
        if (resid_i in fraying_resid_lst) or (resid_j in fraying_resid_lst):
            return True
        else:
            return False

    def determine_stack_trimer(self, idx_i, idx_j):
        resid_i = self.get_resid_stack_by_idx(idx_i)
        resid_j = self.get_resid_stack_by_idx(idx_j)
        strandid_i = self.get_strand_id_stack_by_idx(idx_i)
        strandid_j = self.get_strand_id_stack_by_idx(idx_j)
        trimer_resid_i_lst = [8, 9]
        trimer_resid_j_lst = [13, 14]
        if (resid_i in trimer_resid_i_lst) and (strandid_i == 'STRAND1'):
            return True
        elif (resid_j in trimer_resid_j_lst) and (strandid_j == 'STRAND2'):
            return True
        else:
            return False

    def determine_hb_fraying(self, idx_i, idx_j):
        resid_i = self.get_resid_hb_by_idx(idx_i)
        resid_j = self.get_resid_hb_by_idx(idx_j)
        fraying_resid_lst = [1, 2, 3, 19, 20, 21]
        if (resid_i in fraying_resid_lst) and (resid_j in fraying_resid_lst):
            return True
        else:
            return False

    def determine_hb_trimer(self, idx_i, idx_j):
        resid_i = self.get_resid_hb_by_idx(idx_i)
        resid_j = self.get_resid_hb_by_idx(idx_j)
        trimer_resid_i_lst = [9]
        trimer_resid_j_lst = [13]
        if (resid_i in trimer_resid_i_lst) and (resid_j in trimer_resid_j_lst):
            return True
        else:
            return False

    def convert_stack_idx(self, idx):
        cgname = self.s_agent.d_idx_inverse[idx]
        return self.s_agent.atomid_map[cgname] - 1

    def convert_hb_idx(self, idx):
        cgname = self.h_agent.d_idx_inverse[idx]
        return self.h_agent.atomid_map[cgname] - 1

    def get_strand_id_backbone_by_idx(self, idx):
        cgname = self.b_agent.d_idx_inverse[idx]
        return self.b_agent.strandid_map[cgname]

    def get_strand_id_stack_by_idx(self, idx):
        cgname = self.s_agent.d_idx_inverse[idx]
        return self.s_agent.strandid_map[cgname]

    def get_resid_backbone_by_idx(self, idx):
        cgname = self.b_agent.d_idx_inverse[idx]
        return self.b_agent.resid_map[cgname]

    def get_resid_stack_by_idx(self, idx):
        cgname = self.s_agent.d_idx_inverse[idx]
        return self.s_agent.resid_map[cgname]

    def get_resid_hb_by_idx(self, idx):
        cgname = self.h_agent.d_idx_inverse[idx]
        return self.h_agent.resid_map[cgname]

    def get_positions_by_ij(self, idx_i, idx_j):
        positions = np.zeros((2,3))
        positions[0, :] = self.u.atoms.positions[idx_i, :]
        positions[1, :] = self.u.atoms.positions[idx_j, :]
        return positions

    def get_idx_array_by_laplacian_mat(self, laplacian_mat, k_criteria):
        tri_upper = np.triu(laplacian_mat, 1)
        return np.where(tri_upper > k_criteria)

    def tachyon_take_photo_cmd(self, drawzone_folder, tga_name):
        output = path.join(drawzone_folder, tga_name)
        str_1 = f'render Tachyon {output} '
        str_2 = '"/usr/local/lib/vmd/tachyon_LINUXAMD64" '
        str_3 = '-aasamples 12 %s -format TARGA -o %s.tga'
        print(str_1+str_2+str_3)

    def write_tcl_out(self, tcl_out, container):
        f = open(tcl_out, 'w')
        for line in container:
            f.write(line)
            f.write('\n')
        f.close()
        print(f'source {tcl_out}')

    def get_draw_edge_line(self, positions, radius, colorname):
        color_line = f'graphics 0 color {colorname}'
        str_0 = 'graphics 0 cylinder {'
        str_1 = f'{positions[0,0]:.3f} {positions[0,1]:.3f} {positions[0,2]:.3f}'
        str_2 = '} {'
        str_3 = f'{positions[1,0]:.3f} {positions[1,1]:.3f} {positions[1,2]:.3f}'
        str_4 = '} '
        str_5 = f'radius {radius:.2f}\n'
        return [color_line, str_0 + str_1 + str_2 + str_3 + str_4 + str_5]
