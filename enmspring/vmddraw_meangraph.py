from os import path
import MDAnalysis
from enmspring.miscell import check_dir_exist_and_make
from enmspring.vmddraw import BaseStackImportanceAgent
from enmspring.graphs_bigtraj import StackMeanModeAgent
all_folder = '/home/yizaochen/codes/dna_rna/all_systems'
enmspring_folder = '/home/yizaochen/codes/dna_rna/enmspring'

class ATract:
    d_atomlist = {
        'STRAND1': {
            1: [('C6', 'C6'), ('N1', 'C6')],
            2: [('C6', 'C6'), ('N1', 'C6')],
            3: None,
            4: None,
            5: None,
            6: None,
            7: [('C4', 'C5'), ('C5', 'C5'), ('N3', 'C4')]
        },
        'STRAND2': {
            10: [('N3', 'C4'), ('C2', 'C5'), ('N1', 'C5'), ('C4', 'C4')],
            12: None,
            14: None
        }
    }

    d_resids = {
        'STRAND1': {
            1: (11, 12),
            2: (11, 12),
            3: None,
            4: None,
            5: None,
            6: None,
            7: (11, 12)
        },
        'STRAND2': {
            10: (31, 32),
            12: None,
            14: None
        }
    }

    def get_atomlist_by_strandid_modeid(self, strand_id, mode_id):
        return self.d_atomlist[strand_id][mode_id]

    def get_resid_i_j_by_strandid_modeid(self, strand_id, mode_id):
        return self.d_resids[strand_id][mode_id]

class MeanBaseStackImportanceAgent(BaseStackImportanceAgent):
    d_agents = {'a_tract_21mer': ATract(), 'g_tract_21mer': None}


    def __init__(self, host, rootfolder, interval_time, pic_out_folder):
        self.host = host
        self.rootfolder = rootfolder
        self.interval_time = interval_time

        self.tcl_folder = path.join(enmspring_folder, 'tclscripts')
        self.pic_out_folder = pic_out_folder
        self.mol_stru_folder = path.join(self.pic_out_folder, 'mol_structure')

        self.allatom_folder = path.join(all_folder, host, self.type_na, 'input', 'allatoms')
        self.perferct_gro = path.join(self.allatom_folder, f'{self.type_na}.perfect.gro')

        self.g_agent = self.get_g_agent_and_preprocess()

        self.check_folder()

    def check_folder(self):
        for folder in [self.mol_stru_folder]:
            check_dir_exist_and_make(folder)

    def get_g_agent_and_preprocess(self):
        g_agent = StackMeanModeAgent(self.host, self.rootfolder, self.interval_time)
        g_agent.process_first_small_agent()
        g_agent.initialize_nodes_information()
        g_agent.load_mean_mode_laplacian_from_npy()
        g_agent.set_degree_adjacency_from_laplacian()
        g_agent.eigen_decompose()
        g_agent.set_benchmark_array()
        g_agent.set_strand_array()
        return g_agent

    def get_resid_i_j_by_host_strandid(self, strand_id, mode_id):
        host_agent = self.d_agents[self.host]
        return host_agent.get_resid_i_j_by_strandid_modeid(strand_id, mode_id)
        
    def get_atompair_list_by_host_strandid(self, strand_id, mode_id):
        host_agent = self.d_agents[self.host]
        return host_agent.get_atomlist_by_strandid_modeid(strand_id, mode_id)      

    def vmd_show_bonds_in_dimer(self, strand_id, mode_id):
        u = MDAnalysis.Universe(self.perferct_gro, self.perferct_gro)
        resid_i, resid_j = self.get_resid_i_j_by_host_strandid(strand_id, mode_id)
        atompair_list = self.get_atompair_list_by_host_strandid(strand_id, mode_id)
        radius_list = [0.15] * len(atompair_list)
        color_list = [1] * len(atompair_list)

        self.vmd_open_perfect_gro()
        lines = ['mol delrep 0 0']
        lines += self.vmd_add_resid_cpk_color_by_name(resid_i)
        lines += self.vmd_add_resid_cpk_color_by_name(resid_j)
        for atompair, radius, color in zip(atompair_list, radius_list, color_list):
            positions = self.get_pair_positions_by_resid_names(u, resid_i, resid_j, atompair[0], atompair[1])
            temp_lines = [f'graphics 0 color {color}',
                          self.get_draw_edge_line(positions, 0, 1, radius)]
            lines += temp_lines
        tcl_out = path.join(self.tcl_folder, 'show_basestack_pair.tcl')
        self.write_tcl_out(tcl_out, lines)
        self.print_tga_out(f'{self.host}_{strand_id}_{mode_id}')


